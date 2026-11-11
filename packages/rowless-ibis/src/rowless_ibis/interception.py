"""Library-boundary interception: the Ibis calls a dry run has to notice.

This is layer 2 of the sandbox described in `rowless.runtime.sandbox`, the one
that has to live in an adapter because only an adapter knows which calls of its
engine matter. Four jobs, deliberately in one place so that the full list of
what rowless does to ibis can be read in one sitting:

1. **Findings at the line that caused them.** Ibis builds expressions eagerly,
   so `orders.select("amount_cents")` raises inside the user's call rather than
   at the end of the dry run. That is a gift - the traceback already points at
   the right line - but what it raises is an `IbisTypeError`, and CI needs
   `ColumnNotFound` with the column named and a near miss suggested. Every
   builder call is wrapped so the translation in `rowless_ibis.errors` happens
   where the error was raised.

   The important one is `Table.__getattr__`: `orders.amount` is how ibis
   pipelines read a column, and a missing one fails with `AttributeError`
   rather than with an ibis error. It becomes `MissingColumn`, which is both
   findings at once - see `rowless_ibis.errors`.
2. **Row dependence.** Ibis is the strictest engine here about this, and it is
   strict in rowless's favour: an unbound expression *cannot* be executed, and
   the aggregate methods that read data in Polars and pandas (`max`, `sum`,
   `count`) are query building in ibis and cost nothing. So the only calls that
   matter are the handful that turn an expression into Python values -
   `execute`, `to_pandas`, `to_pyarrow`, `to_list`, `cache` - and each one is
   reported through `rowless.runtime.rowdep.observe` and then refused, because
   there is no data to hand back and inventing some would be worse than
   stopping. The result reads as unverified rather than as a broken pipeline:
   the code is very likely correct, and rowless simply cannot check it.
3. **Writers.** `to_parquet`, `to_csv`, `to_delta` and the rest are replaced
   with recorders, so a dry run of a pipeline that ends in a write resolves its
   schema and writes nothing. The write is reported as a note, which doubles as
   output lineage.
4. **Raw SQL inside a pipeline.** `Table.sql(...)` asks the backend what a
   query returns, and a rowless frame has no backend. Rather than report a
   pipeline that mixes ibis and SQL as unresolvable, the wrapper resolves the
   query through ibis's own parser against the schemas already in hand - the
   same seam `run_sql` uses - and hands back an equivalent ibis expression. Its
   columns are traced, where an opaque SQL view's columns would not be.

Two rules the patching follows. It is **installed once per process and gated
per run**: the wrappers check `rowless_ibis.session.tracking()` and do nothing
outside a dry run, so importing this module does not change how ibis behaves
for anybody else in the process. And every wrapper **calls the original**, with
the two documented exceptions - the writers, which are the side effects the
whole sandbox exists to stop, and the executors, which have nothing to call.

Methods a given ibis does not have are skipped rather than failing the install,
so the adapter supports the range of releases its `requires` declares.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from contextlib import suppress
from typing import Any

from rowless.adapters.base import DryRunContext
from rowless.runtime.rowdep import observe

from rowless_ibis import errors, session, sql

#: Methods on `Expr` that hand an expression's contents back as Python values.
#: The canonical list of what rowless knows about lives in
#: `rowless.runtime.rowdep.ROW_DEPENDENT_APIS`; these are the ibis spellings.
EXPR_EXECUTORS: tuple[str, ...] = (
    "execute",
    "to_pandas_batches",
    "to_polars",
    "to_pyarrow",
    "to_pyarrow_batches",
    "to_torch",
)

#: The same, on `Table`. `cache` executes the expression into a temporary
#: table, and `preview` prints rows.
TABLE_EXECUTORS: tuple[str, ...] = ("to_pandas", "to_pyarrow", "to_array", "cache", "preview")

#: On `Column`: `to_list` is the one that bites, because a pipeline that builds
#: a literal from it gets an empty list here and a populated one in production.
COLUMN_EXECUTORS: tuple[str, ...] = ("to_list", "to_pyarrow", "preview")

#: On `Scalar` and on `Value`, for a pipeline that reads one aggregate out.
SCALAR_EXECUTORS: tuple[str, ...] = ("to_pyarrow",)
VALUE_EXECUTORS: tuple[str, ...] = ("to_pandas",)

#: `len(table)` is row-dependent by definition. Patched separately because it
#: is a dunder and looked up on the type.
LENGTH_METHODS: tuple[str, ...] = ("__len__",)

#: Methods replaced with recorders. Each one returns None already, so recording
#: instead of writing does not change what the call evaluates to.
WRITERS: tuple[str, ...] = (
    "to_csv",
    "to_delta",
    "to_json",
    "to_parquet",
    "to_parquet_dir",
    "to_xlsx",
)

#: Table methods that build an expression, wrapped so an ibis error becomes a
#: finding at the user's line. Every one of them resolves column references
#: eagerly, which is what makes the finding land in the right place.
TABLE_BUILDERS: tuple[str, ...] = (
    "__getitem__",
    "agg",
    "aggregate",
    "alias",
    "anti_join",
    "as_scalar",
    "asof_join",
    "bind",
    "cast",
    "cross_join",
    "describe",
    "difference",
    "distinct",
    "drop",
    "dropna",
    "fill_null",
    "filter",
    "group_by",
    "head",
    "inner_join",
    "intersect",
    "join",
    "limit",
    "mutate",
    "nunique",
    "order_by",
    "pivot_longer",
    "pivot_wider",
    "relocate",
    "rename",
    "sample",
    "select",
    "semi_join",
    "topk",
    "try_cast",
    "union",
    "unpack",
    "value_counts",
    "view",
    "window_by",
)

#: The same for a grouped table, which is where half of an aggregation's column
#: references are resolved.
GROUPED_BUILDERS: tuple[str, ...] = (
    "agg",
    "aggregate",
    "count",
    "having",
    "mutate",
    "order_by",
    "over",
    "projection",
    "select",
    "size",
)

#: Column methods and operators that refuse a dtype. `amount + status` is the
#: everyday version of a dtype mismatch, and it raises while the *argument* to
#: `select` is being built - before any Table method is called - so the
#: translation has to sit on the operator itself.
VALUE_BUILDERS: tuple[str, ...] = (
    "__add__",
    "__radd__",
    "__sub__",
    "__rsub__",
    "__mul__",
    "__rmul__",
    "__truediv__",
    "__rtruediv__",
    "__floordiv__",
    "__rfloordiv__",
    "__mod__",
    "__rmod__",
    "__pow__",
    "__rpow__",
    "__and__",
    "__rand__",
    "__or__",
    "__ror__",
    "__xor__",
    "__rxor__",
    "__invert__",
    "__neg__",
    "cast",
    "fill_null",
    "over",
)

_INSTALLED = False


def install(ctx: DryRunContext) -> None:
    """Patch the ibis entry points, and bind the wrappers to this dry run.

    Called from `materialize`, which the runtime always calls before user code.
    Patching happens once per process; the per-run state is reset through
    `rowless_ibis.session` whenever a new run starts.
    """
    global _INSTALLED

    session.begin(ctx)
    if _INSTALLED:
        return
    _install_patches()
    _INSTALLED = True


# --- Patching ----------------------------------------------------------------


def _install_patches() -> None:
    """Wrap every ibis entry point this adapter needs to see."""
    import ibis.expr.types as ir

    for method in EXPR_EXECUTORS:
        _patch(ir.Expr, "Expr", method, _executor_wrapper)
    for method in WRITERS:
        _patch(ir.Expr, "Expr", method, _writer_wrapper)
    for method in TABLE_EXECUTORS:
        _patch(ir.Table, "Table", method, _executor_wrapper)
    for method in LENGTH_METHODS:
        _patch(ir.Table, "Table", method, _executor_wrapper)
    for method in COLUMN_EXECUTORS:
        _patch(ir.Column, "Column", method, _executor_wrapper)
    for method in SCALAR_EXECUTORS:
        _patch(ir.Scalar, "Scalar", method, _executor_wrapper)
    for method in VALUE_EXECUTORS:
        _patch(ir.Value, "Value", method, _executor_wrapper)

    for method in TABLE_BUILDERS:
        _patch(ir.Table, "Table", method, _translating_wrapper)
    for method in VALUE_BUILDERS:
        for owner, owner_name in _value_classes(ir):
            _patch(owner, owner_name, method, _translating_wrapper)

    _patch(ir.Table, "Table", "sql", _sql_wrapper)
    _patch(ir.Table, "Table", "__getattr__", _column_access_wrapper)
    _patch_grouped()


def _value_classes(ir: Any) -> tuple[tuple[type, str], ...]:
    """Classes whose operators are worth translating, where this ibis has them.

    The arithmetic lives on the dtype-specific mixins rather than on `Value`,
    so each one is patched where it is defined; `_patch` skips a class that
    does not define the method at all.
    """
    found: list[tuple[type, str]] = []
    for name in ("Value", "NumericValue", "StringValue", "BooleanValue", "TemporalValue"):
        owner = getattr(ir, name, None)
        if isinstance(owner, type):
            found.append((owner, name))
    return tuple(found)


def _patch_grouped() -> None:
    """Wrap the grouped-table builders, which live in their own module."""
    try:
        from ibis.expr.types.groupby import GroupedTable
    except Exception:  # pragma: no cover - only on an ibis that moved it
        session.note(
            "this ibis does not expose ibis.expr.types.groupby.GroupedTable, so rowless cannot "
            "translate an error raised inside a group_by(...).agg(...)",
            once_for="unpatchable:GroupedTable",
        )
        return
    for method in GROUPED_BUILDERS:
        _patch(GroupedTable, "GroupedTable", method, _translating_wrapper)


def _patch(
    owner: type,
    owner_name: str,
    method: str,
    factory: Callable[[str, Any], Any],
) -> None:
    """Replace one method with a wrapper, skipping what this ibis lacks.

    Releases differ in which methods exist, and a missing one is not an error:
    the adapter supports a range of ibis releases, and a method that is not
    there cannot be called. The class is patched rather than the instances
    because ibis expressions are immutable value objects - every call hands
    back a new one.
    """
    original = owner.__dict__.get(method)
    if original is None or getattr(original, "__rowless_wrapped__", False):
        return
    if not callable(original):
        return
    wrapper = factory(f"ibis.{owner_name}.{method}", original)
    wrapper.__rowless_wrapped__ = True
    try:
        setattr(owner, method, wrapper)
    except TypeError:  # pragma: no cover - an ibis whose types refuse patching
        # Nothing to fall back on: the call simply goes unobserved, and saying
        # so is better than a dry run that quietly reports less than it should.
        session.note(
            f"this ibis does not allow {owner_name}.{method} to be wrapped, so rowless cannot "
            "report what it does",
            once_for=f"unpatchable:{owner_name}.{method}",
        )


# --- Wrappers ----------------------------------------------------------------


def _executor_wrapper(api: str, original: Any) -> Any:
    """Report an attempt to read values out of a rowless frame, and refuse it.

    There is nothing to call: an unbound ibis expression has no backend, so the
    original would raise anyway, with a message about backends rather than
    about the pipeline. Reporting it as row dependence puts the finding at the
    user's line and marks the result unverified instead of broken.
    """

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not session.tracking():
            return original(self, *args, **kwargs)
        observe(api)
        raise errors.ExecutionAttempted(
            f"{api} reads data, and a rowless frame has no rows: it is an unbound ibis "
            "expression carrying a schema and no backend. Whatever this pipeline computes from "
            "the result would be a value production never produces, so rowless reports the "
            "schema as unverified rather than guessing at the branch"
        )

    return wrapper


def _writer_wrapper(api: str, original: Any) -> Any:
    """Record a write instead of performing it, and say so in the notes."""

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        target = _write_target(args, kwargs)
        if not session.tracking() or target is None:
            return original(self, *args, **kwargs)
        session.note(
            f"{api} would write to {target}; a dry run resolves schemas and writes nothing",
            once_for=f"{api}:{target}",
        )
        return None

    return wrapper


def _translating_wrapper(api: str, original: Any) -> Any:
    """Turn an ibis error into a finding, and otherwise stay out of the way."""

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not session.tracking():
            return original(self, *args, **kwargs)
        try:
            return original(self, *args, **kwargs)
        except Exception as exc:
            errors.raise_translated(
                exc,
                relation=_relation_name(self),
                available=_columns(self),
            )

    return wrapper


def _column_access_wrapper(api: str, original: Any) -> Any:
    """Turn `table.no_such_column` into the flagship finding.

    Attribute access is how an ibis pipeline reads a column, and ibis reports a
    miss as a plain `AttributeError` with no column list attached. The
    replacement is a `MissingColumn`, which is a `ColumnNotFound` *and* an
    `AttributeError`: the finding names the column and suggests the near miss,
    while every `hasattr` in the process keeps behaving.

    Two cases are left alone. A private or dunder name is a library probing the
    object, not a pipeline reading a column; and ibis's own "did you mean
    'order_by'" hint for a mistyped *method* is a better message than anything
    this adapter could produce from a column list.
    """

    @functools.wraps(original)
    def wrapper(self: Any, key: str, /) -> Any:
        try:
            return original(self, key)
        except AttributeError as exc:
            if not session.tracking() or str(key).startswith("_") or "did you mean" in str(exc):
                raise
            raise errors.missing_column(
                str(key),
                available=_columns(self),
                relation=_relation_name(self),
            ) from exc

    return wrapper


def _sql_wrapper(api: str, original: Any) -> Any:
    """Resolve a raw SQL string against the schemas already in hand.

    `Table.sql` normally asks the backend what the query returns. A rowless
    frame has no backend, so the wrapper parses the query with ibis's own SQL
    parser against the named tables the expression reads from, and returns the
    equivalent ibis expression. The pipeline carries on against a real ibis
    table, and the columns the SQL produced are traced like any others.
    """

    @functools.wraps(original)
    def wrapper(self: Any, query: str, /, *args: Any, **kwargs: Any) -> Any:
        if not session.tracking():
            return original(self, query, *args, **kwargs)
        with session.own_work():
            catalog, collisions = sql.catalog_for(sql.named_relations(self))
            for collision in collisions:
                session.note(collision, once_for=collision)
            if not catalog:
                raise errors.SqlNotResolvable(
                    "this expression has no named table for a SQL query to read from, so "
                    "rowless cannot resolve Table.sql() without a backend"
                )
            resolved = sql.resolve(
                sql.strip_terminator(query),
                catalog,
                dialect=kwargs.get("dialect"),
            )
        session.note(
            f"{api} was resolved through ibis's own SQL parser against "
            f"{', '.join(sorted(catalog))}, with no backend and nothing executed",
            once_for=f"{api}:{query}",
        )
        return resolved

    return wrapper


# --- Helpers -----------------------------------------------------------------


def _columns(obj: Any) -> tuple[str, ...]:
    """The column names of an ibis expression, for the near-miss suggestion."""
    with session.own_work(), suppress(Exception):
        columns = getattr(obj, "columns", None)
        if columns is not None:
            return tuple(str(name) for name in columns)
    return ()


def _relation_name(obj: Any) -> str | None:
    """The name of the relation a finding is about, when it has one."""
    with session.own_work(), suppress(Exception):
        name = getattr(obj.op(), "name", None)
        if isinstance(name, str) and name:
            return name
    return None


def _write_target(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str | None:
    """Where a writer was pointed, as a string, or None when it cannot be read."""
    candidate = args[0] if args else kwargs.get("path", kwargs.get("target"))
    if candidate is None:
        return None
    try:
        return str(candidate)
    except Exception:  # pragma: no cover - a path that will not render
        return None
