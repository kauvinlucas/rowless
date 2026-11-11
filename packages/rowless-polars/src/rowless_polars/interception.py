"""Library-boundary interception: the Polars calls a dry run has to notice.

This is layer 2 of the sandbox described in `rowless.runtime.sandbox`, the one
that has to live in an adapter because only an adapter knows which calls of its
engine matter. Four jobs, and they are deliberately in one place so that the
full list of what rowless does to Polars can be read in one sitting:

1. **Row dependence.** Everything that turns a frame into Python values -
   `is_empty()`, `item()`, `len(df)`, `iter_rows()`, `Series.max()` - answers
   from zero rows. Each one is reported through
   `rowless.runtime.rowdep.observe`, which downgrades the whole result to
   `Confidence.UNKNOWN`: production would take the other branch. Expression
   methods are *not* intercepted, because `pl.col("amount").sum()` resolves to
   a dtype without touching data; only the frame and series receivers do.

   `collect()` is the deliberate exception, and the one place this adapter
   declines to follow `rowless.runtime.rowdep.ROW_DEPENDENT_APIS`. Collecting
   executes the plan, but the schema it returns is the schema the plan
   resolved to, and a Polars pipeline that ends in `.collect()` is the normal
   way to write one. Downgrading all of them to unverified would fail CI on
   code rowless checked perfectly well, and false positives are how linters
   get switched off. So collecting is recorded as a note, and the *value*
   accessors that a collected frame makes available - every one of which is
   intercepted above - are what cost confidence. The static pass in
   `rowless.analysis` still flags the call in the source, which is the
   conservative half of the same judgment.
2. **Plan validation at the line that broke it.** Polars is lazy, so
   `lf.select("amount")` on a frame with no `amount` resolves successfully and
   fails later, when something asks for the schema. Every builder call resolves
   its result immediately, so `ColumnNotFound` is raised inside the user call
   and the finding points at the user line instead of at the end of the dry
   run. Resolution is schema-only; nothing executes. The same pass refuses two
   expressions in one call that would produce the same column name, which
   Polars 2.x also refuses and Polars 1.x silently collapses.
3. **Column lineage.** The same wrappers record which input column each output
   column came from. See `rowless_polars.lineage`.
4. **Writers.** `write_parquet`, `sink_parquet`, `write_database` and the rest
   are replaced with recorders. The audit hook would block them anyway, but a
   blocked write aborts the dry run, and "this pipeline writes to warehouse.x"
   is information worth keeping rather than a crash.

Two rules the patching follows. It is **installed once per process and gated
per run**: the wrappers check `rowdep.active_log()` and do nothing outside a
dry run, so importing this module does not change how Polars behaves for
anybody else in the process. And every wrapper **calls the original**: these
are observers, not replacements, with the one documented exception of the
writers, which are the side effects the whole sandbox exists to stop.
"""

from __future__ import annotations

import functools
import weakref
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from rowless.adapters.base import DryRunContext
from rowless.errors import DuplicateColumn, UnsupportedDType
from rowless.ir.dtypes import DType
from rowless.ir.schema import Schema
from rowless.runtime.rowdep import active_log, observe

from rowless_polars import dtypes as dtype_map
from rowless_polars import errors, lineage

#: Accessors whose answer depends on there being rows, by receiver type. The
#: canonical list of what rowless knows about lives in
#: `rowless.runtime.rowdep.ROW_DEPENDENT_APIS`; this is the Polars spelling of
#: it, restricted to receivers that produce Python values rather than
#: expressions.
ROW_DEPENDENT: dict[str, tuple[str, ...]] = {
    "DataFrame": (
        "is_empty",
        "item",
        "iter_rows",
        "iter_slices",
        "rows",
        "rows_by_key",
        "to_dicts",
        "to_dict",
        "__len__",
    ),
    "Series": (
        "item",
        "to_list",
        "to_numpy",
        "min",
        "max",
        "sum",
        "mean",
        "median",
        "n_unique",
        "unique",
        "value_counts",
        "first",
        "last",
        "head",
        "tail",
        "__len__",
        "__iter__",
    ),
}

#: Calls that execute the plan. The schema of what comes back is the schema the
#: plan resolved to, so these are noted rather than treated as row-dependent;
#: the module docstring explains the judgment.
EXECUTES: dict[str, tuple[str, ...]] = {"LazyFrame": ("collect", "collect_async")}

#: Writers, replaced with recorders. A write with no destination
#: (`df.write_csv()` returns the CSV as a string) is passed through instead,
#: because it is a conversion rather than a side effect.
WRITERS: dict[str, tuple[str, ...]] = {
    "DataFrame": (
        "write_parquet",
        "write_csv",
        "write_ipc",
        "write_ipc_stream",
        "write_json",
        "write_ndjson",
        "write_avro",
        "write_excel",
        "write_delta",
        "write_database",
    ),
    "LazyFrame": ("sink_parquet", "sink_csv", "sink_ipc", "sink_ndjson"),
}

#: Builder methods that keep their column names: lineage carries straight
#: through, and a column they invent (`with_row_index`, `unpivot`) is left
#: untraced rather than attributed to a name that happens to collide.
CARRY_THROUGH: tuple[str, ...] = (
    "filter",
    "remove",
    "sort",
    "unique",
    "head",
    "tail",
    "limit",
    "slice",
    "reverse",
    "shift",
    "cast",
    "clear",
    "drop_nulls",
    "drop_nans",
    "fill_null",
    "fill_nan",
    "interpolate",
    "set_sorted",
    "top_k",
    "bottom_k",
    "with_row_index",
    "explode",
    "unnest",
    "unpivot",
    "drop",
    "cache",
)

#: Grouped reductions on `LazyGroupBy`: the keys and the reduced columns keep
#: their names, so the carry rule applies.
GROUP_REDUCTIONS: tuple[str, ...] = (
    "all",
    "first",
    "last",
    "len",
    "max",
    "mean",
    "median",
    "min",
    "n_unique",
    "quantile",
    "sum",
    "head",
    "tail",
)

#: True while this module is running its own bookkeeping. Keeps a patched
#: method that Polars calls internally from being reported twice, and keeps the
#: eager schema resolution from re-entering the wrappers.
_INSIDE: ContextVar[bool] = ContextVar("rowless_polars_inside", default=False)

_INSTALLED = False
_CONTEXT: DryRunContext | None = None
_RUN_ID: str | None = None

#: Notes already written for this run, so a pipeline that collects in a loop
#: says so once rather than forty times.
_NOTED: set[str] = set()


@dataclass(frozen=True, slots=True)
class _Grouping:
    """The frame and key expressions a `LazyGroupBy` was built from.

    Polars does not expose the frame behind a `LazyGroupBy`, so `group_by` is
    intercepted to remember it for the `agg` that follows.
    """

    frame: Any
    keys: tuple[Any, ...]


#: `id(LazyGroupBy)` to the grouping it came from, purged when it is collected.
_GROUPINGS: dict[int, _Grouping] = {}
_GROUPINGS_ALIVE: dict[int, weakref.ref[Any]] = {}


def install(ctx: DryRunContext) -> None:
    """Patch the Polars entry points, and bind the wrappers to this dry run.

    Called from `materialize`, which the runtime always calls before user code.
    Patching happens once per process; the context and the lineage tracker are
    reset whenever a new run starts.
    """
    global _INSTALLED, _CONTEXT, _RUN_ID

    _CONTEXT = ctx
    if ctx.run_id != _RUN_ID:
        _RUN_ID = ctx.run_id
        lineage.TRACKER.clear()
        _GROUPINGS.clear()
        _GROUPINGS_ALIVE.clear()
        _NOTED.clear()
    if _INSTALLED:
        return
    _install_patches()
    _INSTALLED = True


def context() -> DryRunContext | None:
    """The dry-run context the wrappers report into, if a run is open."""
    return _CONTEXT


def tracking() -> bool:
    """True while a dry run is executing user code.

    The gate on everything in this module. Outside a dry run the wrappers call
    straight through, so a process that happens to have imported this adapter
    gets unmodified Polars.
    """
    return active_log() is not None and not _INSIDE.get()


def udf_return_dtype(
    function: Callable[..., Any],
    arg_schema: Schema | None = None,
    *,
    ctx: DryRunContext | None = None,
) -> DType:
    """Work out what a UDF returns, in the order `rowless.udf.infer` prefers.

    An explicit `return_dtype=` never reaches here; this is the annotation,
    then - only when `probe_udfs` is configured - one synthetic call. The note
    explaining which method was used goes on the context, so `rowless explain`
    can say where the dtype came from.
    """
    from rowless.udf.infer import infer_return_dtype

    active = ctx if ctx is not None else _CONTEXT
    allow_probe = bool(active.options.get("probe_udfs")) if active is not None else False
    signature = infer_return_dtype(function, arg_schema or Schema(), allow_probe=allow_probe)
    if active is not None:
        detail = f": {signature.note}" if signature.note else ""
        active.note(
            f"UDF {signature.qualname} returns {signature.returns} "
            f"({signature.method.value}){detail}"
        )
    return signature.returns


# --- Patching ----------------------------------------------------------------


def _install_patches() -> None:
    """Wrap every Polars entry point this adapter needs to see."""
    import polars as pl

    for owner_name, methods in ROW_DEPENDENT.items():
        owner = getattr(pl, owner_name)
        for method in methods:
            _patch(owner, owner_name, method, _row_dependent_wrapper)
    for owner_name, methods in EXECUTES.items():
        owner = getattr(pl, owner_name)
        for method in methods:
            _patch(owner, owner_name, method, _execution_wrapper)
    for owner_name, methods in WRITERS.items():
        owner = getattr(pl, owner_name)
        for method in methods:
            _patch(owner, owner_name, method, _writer_wrapper)
    for method in CARRY_THROUGH:
        _patch(pl.LazyFrame, "LazyFrame", method, _builder(rule=_carry_rule))
    for method in ("select", "select_seq", "with_columns", "with_columns_seq"):
        _patch(
            pl.LazyFrame,
            "LazyFrame",
            method,
            _builder(rule=_expression_rule, check=_check_duplicate_outputs),
        )
    for method in ("join", "join_asof", "join_where"):
        _patch(pl.LazyFrame, "LazyFrame", method, _builder(rule=_join_rule))
    _patch(pl.LazyFrame, "LazyFrame", "rename", _builder(rule=_rename_rule))
    grouped = _lazy_group_by_class(pl)
    # `group_by` is patched after the class behind it has been found, because
    # finding it may mean calling the unpatched method.
    _patch(pl.LazyFrame, "LazyFrame", "group_by", _group_by_wrapper)
    if grouped is not None:
        _patch(
            grouped,
            "LazyGroupBy",
            "agg",
            _builder(rule=_aggregate_rule, grouped=True, check=_check_duplicate_outputs),
        )
        for method in GROUP_REDUCTIONS:
            _patch(grouped, "LazyGroupBy", method, _builder(rule=_carry_rule, grouped=True))
    _patch(pl.Expr, "Expr", "map_elements", _map_elements_wrapper)
    _patch_concat(pl)


def _lazy_group_by_class(pl: Any) -> type | None:
    """The `LazyGroupBy` type, which Polars has moved around between versions.

    Polars 1.x re-exported it at the top level, 2.x does not and raises on the
    attribute. The last resort is to build one and look at what came back,
    which is version-proof as long as `group_by` exists at all.
    """
    found = getattr(pl, "LazyGroupBy", None)
    if isinstance(found, type):
        return found
    try:
        from polars.lazyframe.group_by import LazyGroupBy

        return LazyGroupBy
    except Exception:
        pass
    try:
        probe = pl.LazyFrame(schema={"rowless": pl.Int64}).group_by("rowless")
    except Exception:  # pragma: no cover - a Polars with no group_by at all
        return None
    return type(probe)


def _builder(
    *,
    rule: _Rule,
    grouped: bool = False,
    check: _Check | None = None,
) -> Callable[[str, Any], Any]:
    """A `_patch` factory for one builder method, bound to its lineage rule."""
    return functools.partial(_builder_wrapper, rule=rule, grouped=grouped, check=check)


def _patch(
    owner: type,
    owner_name: str,
    method: str,
    factory: Callable[[str, Any], Any],
) -> None:
    """Replace one method with a wrapper, skipping what this Polars lacks.

    Engine versions differ in which methods exist (`collect_async`, `remove`),
    and a missing one is not an error: the adapter supports a range of Polars
    releases, and a method that is not there cannot be called.
    """
    original = owner.__dict__.get(method)
    if original is None or getattr(original, "__rowless_wrapped__", False):
        return
    if not callable(original):
        return
    wrapper = factory(f"polars.{owner_name}.{method}", original)
    wrapper.__rowless_wrapped__ = True
    setattr(owner, method, wrapper)


def _patch_concat(pl: Any) -> None:
    """Wrap `pl.concat`, the only union a lazy pipeline has.

    A module-level function rather than a method, and the one place the
    arguments are normalised before the original is called: `items` may be a
    generator, and the lineage rule has to iterate it too.
    """
    original = pl.concat
    if getattr(original, "__rowless_wrapped__", False):
        return

    @functools.wraps(original)
    def wrapper(items: Any, *args: Any, **kwargs: Any) -> Any:
        frames = list(items) if not isinstance(items, list) else items
        result = original(frames, *args, **kwargs)
        if not _claims(result) or not tracking():
            return result
        with _own_work():
            columns = _resolve(result, parent=frames[0] if frames else None)
            if columns is not None:
                parents = [lineage.TRACKER.of(frame) for frame in frames]
                lineage.TRACKER.record(result, lineage.merged(parents, columns))
        return result

    wrapper.__rowless_wrapped__ = True  # type: ignore[attr-defined]
    pl.concat = wrapper


# --- Wrappers ----------------------------------------------------------------


def _row_dependent_wrapper(api: str, original: Any) -> Any:
    """Report a row-count-sensitive call, then hand back what Polars returned."""

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not tracking():
            return original(self, *args, **kwargs)
        with _own_work():
            result = original(self, *args, **kwargs)
        observe(api, returned=_summarise(result))
        with _own_work():
            _carry_lineage(self, result)
        return result

    return wrapper


def _execution_wrapper(api: str, original: Any) -> Any:
    """Note that the plan was executed, and carry lineage onto the result.

    Executing a rowless plan moves no data, because there is none. It is worth
    saying out loud in `rowless explain`, and it is worth saying once however
    many times the pipeline collects.
    """

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not tracking():
            return original(self, *args, **kwargs)
        with _own_work():
            result = original(self, *args, **kwargs)
            ctx = _CONTEXT
            if ctx is not None and api not in _NOTED:
                _NOTED.add(api)
                ctx.note(
                    f"{api} executed the plan; on a rowless frame that resolves the schema and "
                    "moves no data, so it does not cost confidence on its own"
                )
            _carry_lineage(self, result)
        return result

    return wrapper


def _writer_wrapper(api: str, original: Any) -> Any:
    """Record a write instead of performing it, and say so in the notes."""

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        ctx = _CONTEXT
        target = _write_target(args, kwargs)
        if not tracking() or ctx is None or target is None:
            return original(self, *args, **kwargs)
        ctx.note(f"{api} would write to {target}; a dry run resolves schemas and writes nothing")
        return None

    return wrapper


def _map_elements_wrapper(api: str, original: Any) -> Any:
    """Supply the return dtype of a UDF that was not given one.

    Polars infers it by calling the function with a synthetic value, which is a
    reasonable guess and an unreasonable thing to do to a function that talks
    to a database. A return annotation is both cheaper and more trustworthy, so
    it is used when there is one; when there is not, Polars is left to its own
    inference, and the `InvalidOperationError` it raises when that fails is
    translated into an unverified finding rather than a traceback.
    """
    _ = api

    @functools.wraps(original)
    def wrapper(
        self: Any,
        function: Callable[..., Any],
        return_dtype: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if return_dtype is not None or not tracking():
            return original(self, function, return_dtype, *args, **kwargs)
        with _own_work():
            inferred = _polars_return_dtype(function)
        return original(self, function, inferred, *args, **kwargs)

    return wrapper


@dataclass(frozen=True, slots=True)
class _Plan:
    """One intercepted builder call, with its result already resolved.

    Everything a lineage rule needs, so that the rules stay pure functions of
    the call and the wrapper below stays the only place that knows about
    tracking, reentrancy and plan resolution.
    """

    receiver: Any
    """What the method was called on: a `LazyFrame`, or a `LazyGroupBy`."""

    source: Any
    """The frame the result derives from. The receiver, except for a grouped
    call, where it is the frame the `LazyGroupBy` was built from."""

    result: Any
    args: tuple[Any, ...]
    kwargs: Mapping[str, Any]
    columns: tuple[str, ...]
    """Resolved output column names, in order."""

    @property
    def parent(self) -> lineage.Lineage | None:
        """Lineage of the source frame, or None when it was never tracked."""
        return lineage.TRACKER.of(self.source)


#: Maps one intercepted call onto the lineage of the frame it produced, or None
#: when there is nothing to record.
_Rule = Callable[[_Plan], "lineage.Lineage | None"]

#: Checked before the result is resolved, to report what the engine will not.
_Check = Callable[[tuple[Any, ...], Mapping[str, Any]], None]


def _builder_wrapper(
    api: str,
    original: Any,
    *,
    rule: _Rule,
    grouped: bool = False,
    check: _Check | None = None,
) -> Any:
    """Wrap a builder method: resolve its result, then record its lineage.

    The shared half of every plan-building interception. Resolving the result
    is the validation - a plan that cannot resolve raises here, inside the user
    call, rather than at the end of the dry run - and `rule` is the per-method
    part that says where each output column came from.
    """
    _ = api

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not tracking():
            return original(self, *args, **kwargs)
        with _own_work():
            result = original(self, *args, **kwargs)
        if not _claims(result):
            return result
        with _own_work():
            if check is not None:
                check(args, kwargs)
            source = _grouped_source(self) if grouped else self
            columns = _resolve(result, parent=source)
            if columns is None:
                return result
            recorded = rule(
                _Plan(
                    receiver=self,
                    source=source,
                    result=result,
                    args=args,
                    kwargs=kwargs,
                    columns=columns,
                )
            )
            if recorded is not None:
                lineage.TRACKER.record(result, recorded)
        return result

    return wrapper


def _group_by_wrapper(api: str, original: Any) -> Any:
    """Remember the frame and keys behind a `LazyGroupBy`, for the `agg` to come."""
    _ = api

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not tracking():
            return original(self, *args, **kwargs)
        with _own_work():
            group = original(self, *args, **kwargs)
            _remember_grouping(group, self, args, kwargs)
        return group

    return wrapper


# --- Lineage rules -----------------------------------------------------------


def _carry_rule(plan: _Plan) -> lineage.Lineage | None:
    """`filter`, `sort`, `cast`, `head`: the surviving names keep their lineage."""
    parent = plan.parent
    return None if parent is None else lineage.carry(parent, plan.columns)


def _expression_rule(plan: _Plan) -> lineage.Lineage | None:
    """`select` and `with_columns`: each output comes from its expression roots."""
    parent = plan.parent
    if parent is None:
        return None
    return lineage.from_expressions(
        parent,
        plan.columns,
        lineage.expressions(plan.args, plan.kwargs),
        carry_rest=True,
    )


def _rename_rule(plan: _Plan) -> lineage.Lineage | None:
    """`rename`: the new name inherits what the old name had."""
    parent = plan.parent
    if parent is None:
        return None
    mapping = _rename_mapping(plan.args, plan.kwargs, parent)
    return lineage.renamed(parent, plan.columns, mapping)


def _join_rule(plan: _Plan) -> lineage.Lineage | None:
    """A join: each output column comes from the side that produced it."""
    left = plan.parent
    other = plan.args[0] if plan.args else plan.kwargs.get("other")
    right = lineage.TRACKER.of(other) if other is not None else None
    if left is None and right is None:
        return None
    return lineage.joined(
        left or {},
        right,
        plan.columns,
        suffix=str(plan.kwargs.get("suffix", "_right")),
    )


def _aggregate_rule(plan: _Plan) -> lineage.Lineage | None:
    """`agg`: the group keys and the aggregates, each from its own roots."""
    parent = plan.parent
    grouping = _GROUPINGS.get(id(plan.receiver))
    if parent is None or grouping is None:
        return None
    return lineage.from_expressions(
        parent,
        plan.columns,
        [*grouping.keys, *lineage.expressions(plan.args, plan.kwargs)],
        carry_rest=True,
    )


def _check_duplicate_outputs(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> None:
    """Refuse two expressions in one call that would produce the same column."""
    _refuse_duplicate_outputs(lineage.expressions(args, kwargs))


def _grouped_source(group: Any) -> Any:
    """The frame a `LazyGroupBy` was built from, or None when it was not seen."""
    grouping = _GROUPINGS.get(id(group))
    return grouping.frame if grouping is not None else None


# --- Helpers -----------------------------------------------------------------


@contextmanager
def _own_work() -> Iterator[None]:
    """Mark the frames below as this adapter working, not as user code."""
    token = _INSIDE.set(True)
    try:
        yield
    finally:
        _INSIDE.reset(token)


def _claims(obj: object) -> bool:
    """True when `obj` is a lazy frame, the only thing the builders track."""
    try:
        import polars as pl
    except ImportError:  # pragma: no cover - the wrappers only exist with polars
        return False
    return isinstance(obj, pl.LazyFrame)


def _resolve(frame: Any, *, parent: Any = None) -> tuple[str, ...] | None:
    """Resolve the plan schema now, translating a plan failure into a finding.

    Returns the output column names, or None when Polars failed in a way this
    adapter cannot classify. None means "leave it for `describe`": resolving
    the same plan again at the end of the dry run reports the engine message
    through `PIPELINE_RAISED`, which is better than guessing at a code here.

    `collect_schema` resolves the plan through the Polars type rules and
    executes nothing, which is the whole reason this adapter can afford to do
    it on every builder call.

    `parent` is the frame the operation was called on. Its columns are what a
    missing-column finding suggests from: Polars 1.x reports the bare name of
    the column it could not find, and only later versions list the valid ones,
    so the near miss comes from the frame rather than from the message.
    """
    try:
        schema = frame.collect_schema()
    except Exception as exc:
        translated = errors.translate(exc, available=_columns_of(parent))
        if translated is not None:
            raise translated from exc
        return None
    return tuple(schema)


def _refuse_duplicate_outputs(expressions: Sequence[Any]) -> None:
    """Raise when two expressions in one call produce the same column name.

    Polars 2.x refuses this itself, with a `DuplicateError`. Polars 1.x keeps
    one of the two and says nothing, which is a silently different schema.
    Checking it here means rowless reports `DuplicateColumn` on both, rather
    than a finding that depends on which engine release is installed.

    Only the names this adapter can read off the expressions are checked. A
    wildcard or a selector expands inside the engine, and a collision between
    one of those and a named expression is left to Polars.
    """
    seen: set[str] = set()
    for expression in expressions:
        name = lineage.output_name(expression)
        if name is None:
            continue
        if name in seen:
            raise DuplicateColumn(
                f"two expressions in this call both produce a column named {name!r}; "
                "alias one of them"
            )
        seen.add(name)


def _columns_of(frame: Any) -> tuple[str, ...]:
    """The column names of a frame, or nothing when it will not say."""
    if frame is None:
        return ()
    try:
        return tuple(frame.collect_schema())
    except Exception:
        # The parent plan is broken too, so it has no column list to offer.
        # The finding still names the column that was asked for.
        return ()


def _carry_lineage(source: Any, result: Any) -> None:
    """Copy lineage onto the frame a row-dependent call produced.

    `collect()` turns a tracked `LazyFrame` into a `DataFrame`, and a pipeline
    that returns the collected frame should still get its lineage - flagged
    unverified, because `collect` was observed, but not untraced as well.
    """
    try:
        import polars as pl
    except ImportError:  # pragma: no cover
        return
    if not isinstance(result, pl.DataFrame):
        return
    recorded = lineage.TRACKER.of(source)
    if recorded is not None:
        lineage.TRACKER.record(result, recorded)


def _summarise(result: object) -> object:
    """A short value for the diagnostic: what the call handed back on zero rows."""
    if isinstance(result, bool | int | float | str) or result is None:
        return result
    height = getattr(result, "height", None)
    if isinstance(height, int):
        return f"{height} rows"
    return None


def _write_target(args: Sequence[Any], kwargs: Mapping[str, Any]) -> str | None:
    """Where a writer was pointed, or None when it was not pointed anywhere.

    `df.write_csv()` with no destination returns the CSV as a string, which is
    a conversion and not a side effect, so it is passed through untouched.
    """
    import os

    candidates: list[Any] = [*args]
    for key in ("file", "path", "target", "table_name", "name", "uri"):
        if key in kwargs:
            candidates.append(kwargs[key])
    for candidate in candidates:
        if isinstance(candidate, str | os.PathLike):
            return str(candidate)[:200]
        if candidate is not None and not isinstance(candidate, bool | int | float):
            return f"a {type(candidate).__qualname__}"
    return None


def _rename_mapping(
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    parent: Mapping[str, tuple[str, ...]],
) -> dict[str, str]:
    """The old-to-new mapping of a `rename` call, however it was spelled."""
    if args and isinstance(args[0], Mapping):
        return {str(old): str(new) for old, new in args[0].items()}
    if args and callable(args[0]):
        return {column: str(args[0](column)) for column in parent}
    mapping = kwargs.get("mapping")
    if isinstance(mapping, Mapping):
        return {str(old): str(new) for old, new in mapping.items()}
    if callable(mapping):
        return {column: str(mapping(column)) for column in parent}
    return {}


def _remember_grouping(
    group: Any,
    frame: Any,
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
) -> None:
    """Record the frame and keys a `LazyGroupBy` was built from."""
    key = id(group)
    _GROUPINGS[key] = _Grouping(frame=frame, keys=tuple(lineage.expressions(args, kwargs)))

    def purge(_reference: weakref.ref[Any], key: int = key) -> None:
        """Drop the grouping once the `LazyGroupBy` is gone."""
        _forget_grouping(key)

    with suppress(TypeError):  # a LazyGroupBy is weak-referenceable; be sure anyway
        _GROUPINGS_ALIVE[key] = weakref.ref(group, purge)


def _forget_grouping(key: int) -> None:
    """Drop a collected `LazyGroupBy` entry."""
    _GROUPINGS.pop(key, None)
    _GROUPINGS_ALIVE.pop(key, None)


def _polars_return_dtype(function: Callable[..., Any]) -> Any:
    """A Polars return dtype for a UDF, or None to let Polars infer it itself.

    Returning None costs the run its confidence, and deliberately. What Polars
    does next depends on the release - 1.x types the column `Unknown`, 2.x
    calls the function with an invented value and raises if that fails - and
    neither is a basis rowless will report as verified when it was told
    nothing. Probing a UDF is opt-in here precisely because it runs user code
    with made-up input, so trusting the engine to do the same silently would
    contradict the rest of the tool. `return_dtype=` or a return annotation
    makes the degradation go away, and both are worth having anyway.
    """
    dtype = udf_return_dtype(function)
    if dtype.is_known and not dtype.is_any:
        try:
            return dtype_map.from_ir(dtype)
        except UnsupportedDType:
            pass
    ctx = _CONTEXT
    if ctx is not None:
        qualified = getattr(function, "__qualname__", repr(function))
        ctx.degrade(
            f"the return dtype of the UDF {qualified} could not be determined: no "
            "return_dtype= and no usable return annotation, so whatever Polars reports for "
            "that column rests on nothing rowless can check"
        )
    return None
