"""Library-boundary interception: the pandas calls a dry run has to notice.

Layer 2 of the sandbox described in `rowless.runtime.sandbox`, the layer that
has to live in an adapter because only an adapter knows which calls of its
engine matter. Four jobs, deliberately in one file so the full list of what
rowless does to pandas can be read in one sitting:

1. **Row dependence.** pandas is eager, so everything that reads a value reads
   it from zero rows: `df.empty` is True, `len(df)` is 0, `s.max()` is NaN,
   `iterrows()` yields nothing. Each one is reported through
   `rowless.runtime.rowdep.observe`, which downgrades the result to
   `Confidence.UNKNOWN`, because production would take the other branch.

   This costs pandas pipelines more confidence than it costs Polars ones, and
   the asymmetry is real rather than a limitation here: `df["amount"].max()` is
   a value read, while the Polars spelling of the same aggregate,
   `pl.col("amount").max()`, is an expression that resolves to a dtype without
   touching data. An engine whose natural spelling of an aggregate reads values
   is an engine where more pipelines come out unverified, and saying so is the
   point.

2. **Drift fixups.** The table is in `rowless_pandas.drift`; the wrappers that
   fire each entry are here. Most of them have one cause: the operation takes a
   function, pandas has no row to call it with, so the output keeps the *input*
   dtype. The correction is the UDF inference in `rowless.udf.infer`, cast onto
   the empty result, and recorded as a note.

3. **Error translation at the line that broke.** pandas raises eagerly, which
   is the one way it is easier to adapt than Polars: the exception arrives
   while the user's own frame is on the stack, so a finding points at their
   line with no plan-resolution machinery in between. What the wrappers add is
   the *columns the frame had*, because pandas does not list them in the
   message and the near-miss suggestion needs them.

4. **Writers.** `to_parquet`, `to_sql` and the rest are replaced with
   recorders. The audit hook would block them, but a blocked write aborts the
   dry run, and "this pipeline writes to warehouse.orders" is worth keeping
   rather than crashing on.

## The rule that makes this safe: only report the user's own calls

pandas calls its own public API constantly. `sort_values` reads `.values`,
`merge` reads `len(self)`, almost everything touches `empty`. A wrapper that
reported every one of those would mark every pipeline unverified within the
first operation, and `rowless.runtime.rowdep.user_frame` cannot sort it out,
because it only walks out of *rowless* frames - and a pandas-internal call has
a pandas frame directly above the wrapper, which is not rowless and would be
reported as the caller.

So every wrapper here is gated on `called_from_user_code()`, which walks out of
rowless frames and then asks whether what is left is inside pandas, numpy or
pyarrow. If it is, the call is the engine's own bookkeeping and nothing is
reported. If it is not, the call came from the pipeline, and it is.

The consequence worth knowing: a value read performed *by* pandas on the
pipeline's behalf is invisible to this layer, and so is one made inside a
callback pandas invokes - `agg(lambda s: s.max())` reads values by definition,
and reporting every such call would mark every aggregation unverified. What
covers those instead is the drift layer, which corrects or degrades the dtype
the callback produced, and the static pass in `rowless.analysis`, which flags
the row-dependent names in the source whatever the dry run reached. Between the
three the gap is small; the alternative - a tool that reports every pandas
pipeline as unverified - would be useless.

## Two rules the patching follows

It is **installed once per process and gated per run**: every wrapper checks
`rowdep.active_log()` and calls straight through outside a dry run, so
importing this module does not change how pandas behaves for anybody else in
the process. And every wrapper **calls the original**: these are observers, not
replacements, with the documented exception of the writers, which are the side
effects the sandbox exists to stop.
"""

from __future__ import annotations

import functools
import os
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from pathlib import Path
from types import FrameType
from typing import Any

from rowless.adapters.base import DryRunContext
from rowless.ir.dtypes import DType
from rowless.ir.schema import Field, Schema
from rowless.runtime.rowdep import active_log, observe

from rowless_pandas import drift, errors
from rowless_pandas import dtypes as dtype_map

#: Methods whose answer depends on there being rows, by receiver. Every one of
#: them returns Python values rather than a frame, which is what distinguishes
#: them from the aggregations that only move dtypes around: `groupby().sum()`
#: is a schema operation and is not here, while `df["x"].sum()` is a value and
#: is. The canonical engine-agnostic list is
#: `rowless.runtime.rowdep.ROW_DEPENDENT_APIS`; this is the pandas spelling.
ROW_DEPENDENT: dict[str, tuple[str, ...]] = {
    "DataFrame": (
        "__len__",
        # `iterrows` and `itertuples` walk rows; `items` walks columns, which
        # is why it is on the Series list below and not on this one.
        "iterrows",
        "itertuples",
        "to_dict",
        "to_records",
        "to_numpy",
        "squeeze",
        "any",
        "all",
        "sum",
        "prod",
        "mean",
        "median",
        "min",
        "max",
        "std",
        "var",
        "sem",
        "skew",
        "kurt",
        "count",
        "nunique",
        "quantile",
        "mode",
        "value_counts",
        "describe",
        "corr",
        "cov",
    ),
    "Series": (
        "__len__",
        "__iter__",
        "items",
        "item",
        "tolist",
        "to_list",
        "to_dict",
        "to_numpy",
        "squeeze",
        "any",
        "all",
        "sum",
        "prod",
        "mean",
        "median",
        "min",
        "max",
        "std",
        "var",
        "sem",
        "skew",
        "kurt",
        "count",
        "nunique",
        "unique",
        "quantile",
        "mode",
        "value_counts",
        "describe",
    ),
}

#: Row-dependent *properties*, which need a property wrapper rather than a
#: function one. `df.empty` is the single most common row-dependent expression
#: in pandas code, so it is worth the extra mechanism.
ROW_DEPENDENT_PROPERTIES: dict[str, tuple[str, ...]] = {
    "DataFrame": ("empty", "values"),
    "Series": ("empty", "values"),
}

#: Reductions that *raise* on zero rows rather than returning something
#: useless. Reported as row dependence, not as a broken pipeline: the code is
#: fine, and failing it as an error in CI would be a false positive.
EMPTY_REDUCTIONS: dict[str, tuple[str, ...]] = {
    "DataFrame": ("idxmax", "idxmin"),
    "Series": ("idxmax", "idxmin", "argmax", "argmin"),
}

#: Writers, replaced with recorders. One whose destination is absent
#: (`df.to_csv()` returns the CSV as a string) is passed through instead,
#: because that is a conversion rather than a side effect.
WRITERS: dict[str, tuple[str, ...]] = {
    "DataFrame": (
        "to_parquet",
        "to_csv",
        "to_json",
        "to_sql",
        "to_feather",
        "to_orc",
        "to_stata",
        "to_excel",
        "to_hdf",
        "to_pickle",
        "to_xml",
        "to_clipboard",
    ),
    "Series": (
        "to_csv",
        "to_json",
        "to_sql",
        "to_excel",
        "to_hdf",
        "to_pickle",
        "to_clipboard",
    ),
}

#: Writers with no path argument to read a destination off. Recorded on the
#: strength of the call alone.
ALWAYS_WRITES: frozenset[str] = frozenset({"to_sql", "to_clipboard", "to_gbq"})

#: Calls whose pandas exception is translated into a rowless finding, with the
#: receiver's columns supplied for the near-miss suggestion.
TRANSLATED: dict[str, tuple[str, ...]] = {
    "DataFrame": (
        "__getitem__",
        "drop",
        "rename",
        "merge",
        "join",
        "groupby",
        "sort_values",
        "set_index",
        "reset_index",
        "query",
        "eval",
        "pop",
        "assign",
        "agg",
        "aggregate",
        "melt",
        "explode",
        "dropna",
        "duplicated",
        "drop_duplicates",
        "nlargest",
        "nsmallest",
        "astype",
        "combine_first",
        "update",
    ),
    "Series": ("drop", "rename", "groupby", "sort_values", "agg", "aggregate", "astype"),
}

#: Operations that build their columns out of *values*. A dry run has no
#: values, so the schema it resolves is not the schema production gets, and
#: nothing can correct that - the run is degraded instead.
SHAPE_DRIFT: dict[str, tuple[str, ...]] = {
    "DataFrame": ("pivot", "pivot_table", "unstack", "transpose"),
    "Series": ("unstack",),
}

#: Module-level functions with the same problem. `stack` and `from_dummies`
#: are not here: both reshape by column *name*, which a rowless frame has.
SHAPE_DRIFT_FUNCTIONS: tuple[str, ...] = ("crosstab", "get_dummies")

#: Calls that take a function and never call it on zero rows, by receiver.
#: The `series` family returns one column, the `frame` family returns many.
SERIES_FUNCTIONS: tuple[str, ...] = ("apply", "map", "transform")
FRAME_FUNCTIONS: tuple[str, ...] = ("apply", "map", "transform")

#: The grouped versions. `apply` is not here: its *shape* drifts, not just its
#: dtypes, so it is handled as a degradation.
GROUP_AGGREGATIONS: tuple[str, ...] = ("agg", "aggregate", "transform")

#: NA-widening calls: on a plain numpy column the populated answer is wider
#: than the empty one.
NA_WIDENING: tuple[str, ...] = ("where", "mask")

#: True while this module is running its own bookkeeping, so a patched method
#: that the adapter itself calls is not reported as something the pipeline did.
_INSIDE: ContextVar[bool] = ContextVar("rowless_pandas_inside", default=False)

_INSTALLED = False
_CONTEXT: DryRunContext | None = None
_RUN_ID: str | None = None

#: Notes already written for this run, so a pipeline that writes in a loop says
#: so once rather than forty times.
_NOTED: set[str] = set()

#: Directories whose frames are the engine's own, not the pipeline's.
_ENGINE_ROOTS: tuple[str, ...] = ()

#: This package and core, recognised by directory as well as by module name.
_ROWLESS_ROOTS: tuple[str, ...] = ()


def install(ctx: DryRunContext) -> None:
    """Patch the pandas entry points, and bind the wrappers to this dry run.

    Called from `materialize`, which the runtime always calls before it imports
    or calls user code. Patching happens once per process; the context and the
    per-run notes are reset whenever a new run starts.
    """
    global _INSTALLED, _CONTEXT, _RUN_ID

    _CONTEXT = ctx
    if ctx.run_id != _RUN_ID:
        _RUN_ID = ctx.run_id
        _NOTED.clear()
    if _INSTALLED:
        return
    _resolve_roots()
    _install_patches()
    _INSTALLED = True


def context() -> DryRunContext | None:
    """The dry-run context the wrappers report into, if a run is open."""
    return _CONTEXT


def tracking() -> bool:
    """True while a dry run is executing user code.

    The first of the two gates on everything in this module. Outside a dry run
    the wrappers call straight through, so a process that happens to have
    imported this adapter gets unmodified pandas.
    """
    return active_log() is not None and not _INSIDE.get()


def called_from_user_code() -> bool:
    """True when the nearest non-rowless frame is not inside the engine.

    The second gate, and the one that makes patching pandas viable at all. See
    the module docstring: pandas calls its own public API constantly, and
    reporting those calls would mark every pipeline unverified before it got
    past its first operation.
    """
    frame: FrameType | None = sys._getframe(1)
    while frame is not None:
        if _is_rowless(frame):
            frame = frame.f_back
            continue
        return not _is_engine(frame)
    # Nothing but rowless frames below: this is the adapter calling itself, and
    # the pipeline did not ask for it.
    return False


def udf_return_dtype(
    function: Callable[..., Any],
    arg_schema: Schema | None = None,
    *,
    ctx: DryRunContext | None = None,
) -> DType:
    """Work out what a UDF returns, in the order `rowless.udf.infer` prefers.

    Used by `PandasAdapter.probe_udf`, and by the drift fixups below. Returns
    ANY when nothing could determine it, which lowers coverage rather than
    pretending the input dtype was the answer.
    """
    from rowless.ir.dtypes import any_

    active = ctx if ctx is not None else _CONTEXT
    found = drift.return_dtype(function, ctx=active, arg_schema=arg_schema)
    return found if found is not None else any_()


# --- Patching ----------------------------------------------------------------


def _resolve_roots() -> None:
    """Find the directories that identify engine and rowless frames."""
    global _ENGINE_ROOTS, _ROWLESS_ROOTS

    _ENGINE_ROOTS = tuple(
        root for root in (_package_root(name) for name in ("pandas", "numpy", "pyarrow")) if root
    )
    _ROWLESS_ROOTS = tuple(
        root for root in (_package_root(name) for name in ("rowless", "rowless_pandas")) if root
    )


def _package_root(name: str) -> str:
    """The directory an importable package lives in, or an empty string."""
    try:
        module = __import__(name)
    except Exception:
        return ""
    found = getattr(module, "__file__", None)
    if not found:  # pragma: no cover - a namespace package
        return ""
    return str(Path(str(found)).resolve().parent)


def _install_patches() -> None:
    """Wrap every pandas entry point this adapter needs to see."""
    import pandas as pd

    owners: dict[str, type] = {"DataFrame": pd.DataFrame, "Series": pd.Series}
    for owner_name, owner in owners.items():
        for method in ROW_DEPENDENT.get(owner_name, ()):
            _patch(owner, owner_name, method, _row_dependent_wrapper)
        for method in ROW_DEPENDENT_PROPERTIES.get(owner_name, ()):
            _patch_property(owner, owner_name, method)
        for method in EMPTY_REDUCTIONS.get(owner_name, ()):
            _patch(owner, owner_name, method, _empty_reduction_wrapper)
        for method in WRITERS.get(owner_name, ()):
            _patch(owner, owner_name, method, _writer_wrapper)
        for method in TRANSLATED.get(owner_name, ()):
            _patch(owner, owner_name, method, _translating_wrapper)
        for method in SHAPE_DRIFT.get(owner_name, ()):
            _patch(owner, owner_name, method, _shape_drift_wrapper)
        for method in NA_WIDENING:
            _patch(owner, owner_name, method, _na_widening_wrapper)
    for method in SERIES_FUNCTIONS:
        _patch(pd.Series, "Series", method, _series_function_wrapper)
    for method in FRAME_FUNCTIONS:
        _patch(pd.DataFrame, "DataFrame", method, _frame_function_wrapper)
    _patch_property(pd.DataFrame, "DataFrame", "T")
    for name, owner in _group_by_classes(pd).items():
        for method in GROUP_AGGREGATIONS:
            _patch(owner, name, method, _grouped_function_wrapper)
        _patch(owner, name, "apply", _grouped_apply_wrapper)
    for name in SHAPE_DRIFT_FUNCTIONS:
        _patch_module_function(pd, name, _shape_drift_function_wrapper)


def _group_by_classes(pd: Any) -> dict[str, type]:
    """The groupby classes to patch, by name, skipping any this pandas lacks.

    pandas has moved these around between releases, so they are looked up
    rather than imported at module scope.
    """
    found: dict[str, type] = {}
    try:
        from pandas.core.groupby.generic import DataFrameGroupBy, SeriesGroupBy

        found["DataFrameGroupBy"] = DataFrameGroupBy
        found["SeriesGroupBy"] = SeriesGroupBy
    except Exception:  # pragma: no cover - a pandas with no groupby module
        probe = pd.DataFrame({"rowless": []}).groupby("rowless")
        found[type(probe).__name__] = type(probe)
    return found


def _patch(
    owner: type,
    owner_name: str,
    method: str,
    factory: Callable[[str, Any], Any],
) -> None:
    """Replace one method with a wrapper, skipping what this pandas lacks.

    The original is found along the MRO - pandas defines `to_csv`, `where` and
    `empty` on `NDFrame` rather than on `DataFrame` - and the wrapper is set on
    the concrete class, so patching `DataFrame.to_csv` leaves `Series.to_csv`
    to its own entry rather than shadowing both from one place.
    """
    original = _resolve_method(owner, method)
    if original is None or getattr(original, "__rowless_wrapped__", False):
        return
    if not callable(original):
        return
    wrapper = factory(f"pandas.{owner_name}.{method}", original)
    wrapper.__rowless_wrapped__ = True
    with suppress(AttributeError, TypeError):
        # A C-level type would refuse the assignment. Every pandas class here
        # is a Python class, so this is belt and braces.
        setattr(owner, method, wrapper)


def _patch_property(owner: type, owner_name: str, name: str) -> None:
    """Replace a row-dependent property with one that reports the read first."""
    original = _resolve_attribute(owner, name)
    if not isinstance(original, property) or original.fget is None:
        return
    if getattr(original.fget, "__rowless_wrapped__", False):
        return
    api = f"pandas.{owner_name}.{name}"
    getter = original.fget

    @functools.wraps(getter)
    def wrapper(self: Any) -> Any:
        """Report the read, then answer with whatever pandas would have."""
        if not tracking() or not called_from_user_code():
            return getter(self)
        with own_work():
            value = getter(self)
        observe(api, returned=_summarise(value))
        return value

    wrapper.__rowless_wrapped__ = True  # type: ignore[attr-defined]
    with suppress(AttributeError, TypeError):
        setattr(owner, name, property(wrapper, original.fset, original.fdel, original.__doc__))


def _patch_module_function(pd: Any, name: str, factory: Callable[[str, Any], Any]) -> None:
    """Wrap a module-level pandas function, if this release has it."""
    original = getattr(pd, name, None)
    if original is None or getattr(original, "__rowless_wrapped__", False):
        return
    wrapper = factory(f"pandas.{name}", original)
    wrapper.__rowless_wrapped__ = True
    setattr(pd, name, wrapper)


def _resolve_method(owner: type, method: str) -> Any:
    """The function `owner.method` resolves to, from anywhere in its MRO."""
    for klass in owner.__mro__:
        found = klass.__dict__.get(method)
        if found is not None:
            return found if not isinstance(found, property) else None
    return None


def _resolve_attribute(owner: type, name: str) -> Any:
    """The raw class attribute, property objects included."""
    for klass in owner.__mro__:
        if name in klass.__dict__:
            return klass.__dict__[name]
    return None


# --- Wrappers ----------------------------------------------------------------


def _invoke(original: Any, receiver: Any, args: Any, kwargs: Any) -> Any:
    """Call a wrapped pandas method, translating its failure into a finding.

    Every wrapper in this module goes through here, which is what makes the
    translation independent of which family a method belongs to: a missing
    column reported by `groupby().agg(...)` becomes the same finding as one
    reported by `df["nope"]`, and a method that is both a drift fixup and a
    place a column can go missing does not have to pick one wrapper.

    The receiver is what supplies the near-miss suggestion, because pandas
    says which label it could not find and never says which ones it has.
    """
    try:
        with own_work():
            return original(receiver, *args, **kwargs)
    except Exception as exc:
        with own_work():
            available = _columns_of(receiver)
        errors.raise_translated(exc, available=available)


def _row_dependent_wrapper(api: str, original: Any) -> Any:
    """Report a row-count-sensitive read, then hand back what pandas returned."""

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not tracking() or not called_from_user_code():
            return original(self, *args, **kwargs)
        result = _invoke(original, self, args, kwargs)
        observe(api, returned=_summarise(result))
        return result

    return wrapper


def _empty_reduction_wrapper(api: str, original: Any) -> Any:
    """Report `idxmax` and friends, and translate the empty-sequence failure.

    pandas raises `ValueError` rather than returning something useless, which
    would otherwise be reported as a pipeline that raised - an error, failing
    CI on code that is correct. It is row dependence, so that is what it is
    reported as, and the exception rowless raises instead is still a
    `ValueError`, so a pipeline that catches one keeps working.
    """

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not tracking() or not called_from_user_code():
            return original(self, *args, **kwargs)
        observe(api, returned="nothing: there is no row to point at")
        try:
            with own_work():
                return original(self, *args, **kwargs)
        except ValueError as exc:
            # Any `ValueError` out of one of these: the reduction takes no user
            # callback, so there is no third party to blame, and which of
            # pandas and numpy raised it has changed between releases.
            drift.degraded(
                _CONTEXT,
                "empty_reduction_raises",
                f"{api} raised on a rowless frame",
            )
            raise errors.EmptyReduction(
                f"{api} has no row to point at on a rowless frame, so the dry run could not "
                "get past it. The pipeline is not necessarily wrong; rowless cannot verify "
                "what it computes from the value"
            ) from exc

    return wrapper


def _writer_wrapper(api: str, original: Any) -> Any:
    """Record a write instead of performing it, and say so in the notes."""

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        ctx = _CONTEXT
        if not tracking() or ctx is None or not called_from_user_code():
            return original(self, *args, **kwargs)
        target = _write_target(args, kwargs)
        if target is None and api.rpartition(".")[2] not in ALWAYS_WRITES:
            # `df.to_csv()` with no destination returns the CSV as a string,
            # which is a conversion and not a side effect.
            return original(self, *args, **kwargs)
        _note_once(
            ctx,
            f"{api}:{target}",
            f"{api} would write to {target or 'its configured destination'}; a dry run "
            "resolves schemas and writes nothing",
        )
        return None

    return wrapper


def _translating_wrapper(api: str, original: Any) -> Any:
    """Turn a pandas failure into a rowless finding, with the frame's columns attached.

    The columns are the point. pandas reports a missing label without saying
    what the valid ones were, so the near-miss suggestion - `amount` ->
    `amount_cents` - has to come from the receiver, which only a wrapper has
    in hand.
    """
    _ = api

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not tracking() or not called_from_user_code():
            return original(self, *args, **kwargs)
        return _invoke(original, self, args, kwargs)

    return wrapper


def _shape_drift_wrapper(api: str, original: Any) -> Any:
    """Degrade a call whose output columns come from the data, then let it run."""

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not tracking() or not called_from_user_code():
            return original(self, *args, **kwargs)
        drift.degraded(
            _CONTEXT,
            "values_become_columns",
            f"{api} builds its columns out of the values in the frame, and a rowless frame "
            "has none",
        )
        return _invoke(original, self, args, kwargs)

    return wrapper


def _shape_drift_function_wrapper(api: str, original: Any) -> Any:
    """The same, for `pd.crosstab` and `pd.get_dummies`."""

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not tracking() or not called_from_user_code():
            return original(*args, **kwargs)
        drift.degraded(
            _CONTEXT,
            "values_become_columns",
            f"{api} builds its columns out of the values it is given, and a rowless frame has none",
        )
        with own_work():
            return original(*args, **kwargs)

    return wrapper


def _na_widening_wrapper(api: str, original: Any) -> Any:
    """Report a `where` or `mask` whose populated dtype is not decidable.

    A plain numpy column has nowhere to put NA, so pandas widens it - but only
    when something is actually masked out, which depends on the data. Two
    possible answers and no way to choose, so the result is left as pandas
    computed it and the run says it could not be verified.

    The masked and arrow-backed dtypes this adapter materializes hold NA
    without changing dtype, so the common case finds nothing to report and
    stays verified. That is the whole argument for materializing them.
    """

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not tracking() or not called_from_user_code():
            return original(self, *args, **kwargs)
        result = _invoke(original, self, args, kwargs)
        if _has_other(args, kwargs) or not _is_empty(self):
            # With an `other` there is nowhere for NA to go, and with rows
            # pandas has already done whatever it does.
            return result
        with own_work():
            _report_widening(api, result)
        return result

    return wrapper


def _series_function_wrapper(api: str, original: Any) -> Any:
    """Correct `Series.apply`, `.map` and `.transform`: the function is never called."""

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not tracking() or not called_from_user_code():
            return original(self, *args, **kwargs)
        result = _invoke(original, self, args, kwargs)
        with own_work():
            if not _is_empty(self) or not _is_series(result):
                return result
            dtype = _declared_dtype(args, kwargs) or _function_dtype(args, kwargs, receiver=self)
            return _cast_series(api, result, dtype, subject=f"{self.name or 'the column'!r}")

    return wrapper


def _frame_function_wrapper(api: str, original: Any) -> Any:
    """Correct `DataFrame.apply` and `.map`, or degrade when the shape is data.

    A function returning a scalar per row or per column lands in a Series, and
    that is correctable. A function returning a Series expands into columns
    that only exist if there are rows, which is not.
    """

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not tracking() or not called_from_user_code():
            return original(self, *args, **kwargs)
        result = _invoke(original, self, args, kwargs)
        with own_work():
            if not _is_empty(self):
                return result
            dtype = _declared_dtype(args, kwargs) or _function_dtype(args, kwargs, receiver=self)
            if _is_series(result):
                return _cast_series(api, result, dtype, subject="the result")
            if _is_frame(result):
                return _cast_frame(api, result, dtype)
            return result

    return wrapper


def _grouped_function_wrapper(api: str, original: Any) -> Any:
    """Correct `GroupBy.agg` and `.transform` where the aggregation is a callable.

    A named aggregation (`"sum"`, `"mean"`) is left alone: pandas computes
    those dtypes from the input dtype rather than from the data, so they do not
    drift. Only the callables do, and only because pandas has no row to call
    them with.
    """

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not tracking() or not called_from_user_code():
            return original(self, *args, **kwargs)
        result = _invoke(original, self, args, kwargs)
        with own_work():
            if not _grouping_is_empty(self):
                return result
            if _is_series(result):
                dtype = _function_dtype(args, kwargs, receiver=self)
                return _cast_series(api, result, dtype, subject="the aggregate")
            if not _is_frame(result):
                return result
            return _cast_aggregates(api, result, args, kwargs)

    return wrapper


def _grouped_apply_wrapper(api: str, original: Any) -> Any:
    """Degrade `GroupBy.apply`: on zero rows there is no group to apply anything to."""

    @functools.wraps(original)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not tracking() or not called_from_user_code():
            return original(self, *args, **kwargs)
        drift.degraded(
            _CONTEXT,
            "groupby_apply_shape",
            f"{api} returned the ungrouped frame because a rowless frame forms no groups",
        )
        return _invoke(original, self, args, kwargs)

    return wrapper


# --- Fixup helpers -----------------------------------------------------------


def _cast_series(api: str, result: Any, dtype: DType | None, *, subject: str) -> Any:
    """Cast one corrected column, or degrade when nothing could type it."""
    if dtype is None:
        drift.degraded(
            _CONTEXT,
            _fixup_for(api),
            f"{api} kept the input dtype because pandas had no row to call the function with, "
            "and nothing could determine what the function returns",
        )
        return result
    cast = drift.cast_series(result, dtype)
    if cast is None:
        drift.degraded(
            _CONTEXT,
            _fixup_for(api),
            f"{api} returns {dtype}, which has no pandas dtype here, so the column was left "
            "at the input dtype",
        )
        return result
    drift.corrected(_CONTEXT, _fixup_for(api), f"{subject} to {dtype} after {api}")
    return cast


def _cast_frame(api: str, result: Any, dtype: DType | None) -> Any:
    """Cast every column of a corrected frame result."""
    if dtype is None:
        drift.degraded(
            _CONTEXT,
            _fixup_for(api),
            f"{api} kept the input dtypes because pandas had no row to call the function with, "
            "and nothing could determine what the function returns",
        )
        return result
    cast = drift.cast_columns(result, list(result.columns), dtype)
    if cast is None:
        return result
    drift.corrected(_CONTEXT, _fixup_for(api), f"every column to {dtype} after {api}")
    return cast


def _cast_aggregates(api: str, result: Any, args: Any, kwargs: Any) -> Any:
    """Cast the columns of an `agg` result whose aggregation was a callable."""
    corrected = result
    for column, function in _aggregated_callables(args, kwargs, result):
        dtype = drift.return_dtype(function, ctx=_CONTEXT)
        if dtype is None:
            drift.degraded(
                _CONTEXT,
                "groupby_agg_return",
                f"{api} aggregated {column!r} with a function nothing could type, so the "
                "column kept its input dtype",
            )
            continue
        cast = drift.cast_columns(corrected, [column], dtype)
        if cast is None:
            continue
        corrected = cast
        drift.corrected(_CONTEXT, "groupby_agg_return", f"{column!r} to {dtype} after {api}")
    return corrected


def _aggregated_callables(args: Any, kwargs: Any, result: Any) -> list[tuple[str, Any]]:
    """Output column to the callable that produced it, for the shapes `agg` takes.

    Four spellings, and only the ones carrying a callable matter:
    `agg(fn)` applies to every column, `agg({"amount": fn})` to one,
    `agg(total=("amount", fn))` names the output, and a string aggregation
    carries no callable at all.
    """
    columns = [str(name) for name in getattr(result, "columns", ())]
    found: list[tuple[str, Any]] = []
    for argument in args:
        if callable(argument) and not isinstance(argument, type):
            found.extend((column, argument) for column in columns)
        elif isinstance(argument, Mapping):
            found.extend(
                (str(column), function)
                for column, function in argument.items()
                if callable(function) and str(column) in columns
            )
    for name, value in kwargs.items():
        if isinstance(value, tuple) and len(value) == 2 and callable(value[1]):
            if name in columns:
                found.append((name, value[1]))
        elif callable(value) and name in columns:
            found.append((name, value))
    return found


def _function_dtype(args: Any, kwargs: Any, *, receiver: Any) -> DType | None:
    """The IR dtype the function passed to this call returns, if anything can say."""
    function = _first_callable(args, kwargs)
    if function is None:
        mapper = _first_mapping(args, kwargs)
        return drift.mapping_value_dtype(mapper) if mapper is not None else None
    return drift.return_dtype(function, ctx=_CONTEXT, arg_schema=_schema_of(receiver))


def _declared_dtype(args: Any, kwargs: Any) -> DType | None:
    """An explicit `dtype=`, which pandas 3 lets `apply` carry and rowless prefers."""
    declared = kwargs.get("dtype")
    if declared is None:
        return None
    found = dtype_map.to_ir(declared)
    return found if found.is_known else None


def _first_callable(args: Any, kwargs: Any) -> Any:
    """The function argument of an `apply`, `map`, `transform` or `agg` call."""
    for argument in args:
        if callable(argument) and not isinstance(argument, type):
            return argument
    for key in ("func", "arg", "mapper", "f"):
        value = kwargs.get(key)
        if callable(value) and not isinstance(value, type):
            return value
    return None


def _first_mapping(args: Any, kwargs: Any) -> Any:
    """The dict or Series a `Series.map` was given instead of a function."""
    for argument in args:
        if isinstance(argument, Mapping) or _is_series(argument):
            return argument
    for key in ("arg", "func", "mapper"):
        value = kwargs.get(key)
        if isinstance(value, Mapping) or _is_series(value):
            return value
    return None


def _fixup_for(api: str) -> str:
    """The drift table entry a wrapper is firing, read off the api name.

    Three receivers, three entries: a Series, a frame, and either of the
    groupby classes. The `where`/`mask` entry reports itself, since it degrades
    rather than correcting and never reaches here.
    """
    if api.startswith("pandas.Series."):
        return "series_apply_return"
    if api.startswith("pandas.DataFrame."):
        return "frame_apply_return"
    return "groupby_agg_return"


def _report_widening(api: str, result: Any) -> None:
    """Degrade a `where`/`mask` over the numpy columns whose dtype is undecidable."""
    if _is_series(result):
        at_risk = (
            [str(result.name or "the column")]
            if drift.na_upcast_dtype(result.dtype) is not None
            else []
        )
    elif _is_frame(result):
        at_risk = [
            str(column)
            for column, dtype in result.dtypes.items()
            if drift.na_upcast_dtype(dtype) is not None
        ]
    else:
        return
    if not at_risk:
        return
    drift.degraded(
        _CONTEXT,
        "na_upcast",
        f"{api} left {', '.join(sorted(at_risk))} at the input dtype because nothing was "
        "masked out of a rowless frame",
    )


# --- Frame helpers -----------------------------------------------------------


@contextmanager
def own_work() -> Iterator[None]:
    """Mark the frames below as this adapter working, not as the pipeline.

    Public, and the adapter uses it around every pandas call of its own.
    `called_from_user_code` cannot help here: the adapter is rowless, so a walk
    out of rowless frames lands on whoever called rowless - the CLI, a test -
    and reads as user code. Reading a frame's `dtypes` iterates a Series, which
    is a row-dependent accessor, so `describe` without this would report every
    pipeline it described as unverified.
    """
    token = _INSIDE.set(True)
    try:
        yield
    finally:
        _INSIDE.reset(token)


def _is_frame(obj: object) -> bool:
    """True for a pandas DataFrame."""
    import pandas as pd

    return isinstance(obj, pd.DataFrame)


def _is_series(obj: object) -> bool:
    """True for a pandas Series."""
    import pandas as pd

    return isinstance(obj, pd.Series)


def _is_empty(frame: Any) -> bool:
    """True when a frame or series has no rows, read without the patched property."""
    shape = getattr(frame, "shape", None)
    if isinstance(shape, tuple) and shape:
        return int(shape[0]) == 0
    return True


def _grouping_is_empty(grouped: Any) -> bool:
    """True when the frame behind a groupby had no rows.

    `GroupBy` does not expose its frame as public API, so the private handle is
    read defensively: an unknown pandas layout means the fixup does not fire,
    which leaves the dtype as pandas computed it rather than guessing.
    """
    for attribute in ("obj", "_obj_with_exclusions", "_selected_obj"):
        frame = getattr(grouped, attribute, None)
        if frame is not None:
            return _is_empty(frame)
    return False


def _has_other(args: Any, kwargs: Any) -> bool:
    """True when a `where`/`mask` call said what to put in place of a masked value."""
    return len(args) >= 2 or "other" in kwargs


def _schema_of(receiver: Any) -> Schema:
    """The receiver's schema, so a UDF probe gets plausible argument values."""
    try:
        if _is_series(receiver):
            name = str(receiver.name or "value")
            return Schema(fields=(Field(name=name, dtype=dtype_map.to_ir(receiver.dtype)),))
        dtypes = getattr(receiver, "dtypes", None)
        if dtypes is None:
            return Schema()
        return Schema(
            fields=tuple(
                Field(name=str(column), dtype=dtype_map.to_ir(dtype))
                for column, dtype in dtypes.items()
            )
        )
    except Exception:
        # A receiver this adapter cannot read is a receiver it will not probe
        # with invented values. The annotation path does not need the schema.
        return Schema()


def _columns_of(receiver: Any) -> tuple[str, ...]:
    """The column names behind a receiver, for the near-miss suggestion.

    A frame answers directly; a `GroupBy` answers through the frame it was
    built from, which is where a missing aggregation label was looked up; a
    Series has one name and that is the best it can do.
    """
    for candidate in (receiver, getattr(receiver, "obj", None)):
        if candidate is None:
            continue
        columns = getattr(candidate, "columns", None)
        if columns is None:
            continue
        try:
            return tuple(str(column) for column in columns)
        except Exception:  # pragma: no cover - an exotic column index
            return ()
    name = getattr(receiver, "name", None)
    return (str(name),) if name is not None else ()


def _summarise(result: object) -> object:
    """A short value for the diagnostic: what the call handed back on zero rows."""
    if isinstance(result, bool | int | float | str) or result is None:
        return result
    shape = getattr(result, "shape", None)
    if isinstance(shape, tuple) and shape and isinstance(shape[0], int):
        return f"{shape[0]} rows"
    return None


def _write_target(args: Sequence[Any], kwargs: Mapping[str, Any]) -> str | None:
    """Where a writer was pointed, or None when it was not pointed anywhere."""
    candidates: list[Any] = [*args]
    for key in ("path", "path_or_buf", "buf", "excel_writer", "fname", "name", "con", "table"):
        if key in kwargs:
            candidates.append(kwargs[key])
    for candidate in candidates:
        if isinstance(candidate, str | os.PathLike):
            return str(candidate)[:200]
        if candidate is not None and not isinstance(candidate, bool | int | float):
            return f"a {type(candidate).__qualname__}"
    return None


def _note_once(ctx: DryRunContext, key: str, message: str) -> None:
    """Record adapter commentary once per run, keyed by what happened."""
    if key in _NOTED:
        return
    _NOTED.add(key)
    ctx.note(message)


def _is_rowless(frame: FrameType) -> bool:
    """True when a stack frame belongs to rowless or to this adapter."""
    name = frame.f_globals.get("__name__", "")
    if name.startswith(("rowless", "rowless_")):
        return True
    filename = frame.f_code.co_filename
    return bool(_ROWLESS_ROOTS) and filename.startswith(_ROWLESS_ROOTS)


def _is_engine(frame: FrameType) -> bool:
    """True when a stack frame belongs to pandas, numpy or pyarrow."""
    filename = frame.f_code.co_filename
    if filename.startswith("<"):
        # A synthetic frame: `eval`, a doctest, a REPL. Not the engine, and
        # treating it as user code is the conservative direction.
        return False
    return bool(_ENGINE_ROOTS) and filename.startswith(_ENGINE_ROOTS)
