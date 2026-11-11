"""The dtype drift table: where a zero-row pandas frame is not representative.

This is the module the pandas adapter exists for. Polars resolves a plan and
answers from its own type rules, so a rowless Polars frame cannot drift. pandas
has no plan: the only way to run real pandas code is to hand it a real frame
with zero rows, and for a handful of operations the dtype pandas computes from
zero rows is **not** the dtype it would compute from data.

Every one of those is listed in `FIXUPS`, each with what pandas does, what it
would have done with rows, and what this adapter does about it. Two kinds of
entry, and the difference matters more than the list:

* `Action.CORRECT` - the adapter knows what the populated answer would be and
  casts the result to it. The correction is recorded as a note, so
  `rowless explain` shows which fixups fired on a run rather than leaving the
  user to wonder why rowless disagrees with what they see in a notebook.
* `Action.DEGRADE` - the adapter knows the empty answer is wrong and *cannot*
  know the right one, usually because the operation builds its columns out of
  values. The run is downgraded to `Confidence.UNKNOWN` with the reason. This
  is the honest half: a correction nobody could derive would be a guess, and a
  guessed schema reported as a pass is the one failure mode this project will
  not have.

Correcting silently is not an option either way. A user who sees `float64` in
their notebook and `int64` in a rowless finding has to be able to find out why
in one command, and a drift table buried in an adapter cannot be read by
anyone.

## How a correction is derived

Nearly every `CORRECT` entry has the same cause: the operation takes a
function, and pandas never calls it, because there is no row to call it with.
The output dtype is therefore whatever the *input* was, and in production it
would be whatever the function returns. So the correction is the UDF inference
in `rowless.udf.infer` - a declared `return_dtype`, then a return annotation,
then (only when `probe_udfs` is on) one synthetic call - lowered back into a
pandas dtype. When nothing can type the function, the fixup falls through to a
degradation, because the alternative is reporting the input dtype as if it were
the output.

## What is deliberately not here

`where` and `mask` without an `other` drift on plain numpy dtypes (`int64`
becomes `float64` once a NaN has to go somewhere) and do **not** drift on the
masked or arrow-backed ones, which hold NA without changing type. That is why
`rowless_pandas.dtypes` materializes masked dtypes in the first place: an
adapter that picks a dtype which cannot drift has one less thing to correct.
The entry is still here, because a pipeline that casts to plain `int64` itself
gets the numpy behaviour back.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from rowless.adapters.base import DryRunContext
from rowless.errors import UnsupportedDType
from rowless.ir.dtypes import DType
from rowless.ir.schema import Schema
from rowless.udf.infer import InferenceMethod, infer_return_dtype

from rowless_pandas import dtypes as dtype_map


class Action(StrEnum):
    """What the adapter does about one drift."""

    CORRECT = "corrected"
    """The populated dtype is derivable, so the result is cast to it."""

    DEGRADE = "degraded"
    """The populated schema is not derivable, so the run is not verified."""


@dataclass(frozen=True, slots=True)
class Fixup:
    """One place a zero-row pandas frame answers differently than a populated one."""

    id: str
    """Short stable name, quoted in notes and in `rowless explain`."""

    api: str
    """The pandas call it fires on."""

    empty: str
    """What pandas computes from zero rows."""

    populated: str
    """What pandas would have computed from data."""

    action: Action
    detail: str
    """What this adapter does, and why that is the honest option."""

    def describe(self) -> str:
        """The one-line description `dtype_drift_fixups()` reports."""
        return (
            f"{self.id} ({self.api}): on zero rows pandas gives {self.empty}, with rows "
            f"{self.populated}. rowless {self.action.value} it - {self.detail}"
        )


#: Every drift this adapter knows about. Verified against pandas 2.2 and 3.x;
#: the cases in `rowless_pandas.cases` assert the ones that are assertable
#: without data, and `tests/test_pandas_adapter.py` asserts each correction
#: against the populated answer, so an entry that stops being true fails.
FIXUPS: tuple[Fixup, ...] = (
    Fixup(
        id="series_apply_return",
        api="Series.apply, Series.map, Series.transform",
        empty="the input dtype, because the function is never called",
        populated="whatever the function returns",
        action=Action.CORRECT,
        detail=(
            "the return dtype is inferred from a return annotation, or by probing when "
            "probe_udfs is on, and the result is cast to it. With nothing to infer from, the "
            "run is degraded instead: reporting the input dtype as the output would be wrong "
            "rather than unverified. One version difference is worth knowing: a function "
            "annotated -> str lands in an object column on pandas 2.x and a str column on 3.x, "
            "and rowless reports the annotation either way, because it is a declaration and "
            "an object column is not a dtype"
        ),
    ),
    Fixup(
        id="frame_apply_return",
        api="DataFrame.apply",
        empty="float64, whatever the frame held",
        populated="the dtype of whatever the function returns, per output column",
        action=Action.CORRECT,
        detail=(
            "same inference as a Series apply when the function returns a scalar. A function "
            "that returns a Series expands into columns that exist only if there are rows, and "
            "that degrades the run"
        ),
    ),
    Fixup(
        id="groupby_agg_return",
        api="GroupBy.agg, GroupBy.aggregate, GroupBy.transform with a callable",
        empty="the input dtype, because the callable is never called",
        populated="whatever the callable returns",
        action=Action.CORRECT,
        detail=(
            "the same inference, applied per aggregated column. A named string aggregation "
            "(sum, mean) does not drift and is left alone, because pandas computes those "
            "dtypes from the input dtype rather than from the data"
        ),
    ),
    Fixup(
        id="groupby_apply_shape",
        api="GroupBy.apply",
        empty="a frame with the original columns, because no group is ever formed",
        populated="whatever the function returns per group, often a Series",
        action=Action.DEGRADE,
        detail=(
            "the output shape depends on how many groups there are, and on zero rows there are "
            "none. Nothing can be corrected here, so the run says so"
        ),
    ),
    Fixup(
        id="values_become_columns",
        api="DataFrame.pivot, pivot_table, unstack, crosstab, get_dummies, transpose",
        empty="no columns at all, or one per index level",
        populated="one column per distinct value in the data",
        action=Action.DEGRADE,
        detail=(
            "the column names are data. A dry run has none, so the schema it resolves is not "
            "the schema production gets, whatever the dtypes say"
        ),
    ),
    Fixup(
        id="na_upcast",
        api="Series.where, Series.mask, DataFrame.where, DataFrame.mask",
        empty="the input dtype, because no NA has to be stored",
        populated="the input dtype, or float64 for a numpy integer and object for a numpy "
        "bool if any row is masked out",
        action=Action.DEGRADE,
        detail=(
            "a plain numpy column has nowhere to put NA, so pandas widens it - but only if "
            "something is actually masked, which is a property of the data and not of the "
            "schema. There are two possible populated dtypes and no way to choose between "
            "them, so the run is not verified rather than corrected to a guess. The masked and "
            "arrow-backed dtypes this adapter materializes hold NA without widening and do not "
            "drift at all, which is why they are the default: the fix is a dtype, not a cast"
        ),
    ),
    Fixup(
        id="empty_reduction_raises",
        api="Series.idxmax, idxmin, argmax, argmin, DataFrame.idxmax, idxmin",
        empty="a ValueError: there is no row to point at",
        populated="a label",
        action=Action.DEGRADE,
        detail=(
            "reported as row dependence rather than as a broken pipeline. The code is fine; it "
            "is the dry run that has no rows, and failing it as an error would be a false "
            "positive in CI"
        ),
    ),
)

#: Keyed by id, for the wrappers that fire one.
BY_ID: dict[str, Fixup] = {fixup.id: fixup for fixup in FIXUPS}


def descriptions() -> tuple[str, ...]:
    """Every fixup, one line each. What `dtype_drift_fixups()` reports.

    The degradations are included deliberately. The protocol asks for the
    corrections an adapter applies, and an adapter that listed only those would
    read as though the rest of the drift did not exist - when the entries it
    left out are precisely the ones that cost a run its confidence.
    """
    return tuple(fixup.describe() for fixup in FIXUPS)


def corrected(ctx: DryRunContext | None, fixup_id: str, detail: str) -> None:
    """Record that a fixup fired, for `rowless explain`."""
    if ctx is None:
        return
    fixup = BY_ID[fixup_id]
    ctx.note(f"dtype drift fixup {fixup.id} corrected {detail} ({fixup.api})")


def degraded(ctx: DryRunContext | None, fixup_id: str, detail: str) -> None:
    """Record that a drift could not be corrected, which costs the run its confidence."""
    if ctx is None:
        return
    fixup = BY_ID[fixup_id]
    ctx.degrade(f"{fixup.id}: {detail}. {fixup.detail}")


def return_dtype(
    function: Callable[..., Any],
    *,
    ctx: DryRunContext | None,
    arg_schema: Schema | None = None,
) -> DType | None:
    """The IR dtype a UDF returns, or None when nothing could determine it.

    The order is `rowless.udf.infer`'s: a return annotation, then - only when
    `probe_udfs` is configured - one synthetic call. Probing is opt-in because
    it runs user code with invented input, and an adapter that probed by
    default would be doing the thing this project refuses to do silently.
    """
    allow_probe = bool(ctx.options.get("probe_udfs")) if ctx is not None else False
    signature = infer_return_dtype(function, arg_schema or Schema(), allow_probe=allow_probe)
    if signature.method is InferenceMethod.GAVE_UP:
        return None
    if not signature.returns.is_known or signature.returns.is_any:
        return None
    if ctx is not None:
        detail = f": {signature.note}" if signature.note else ""
        ctx.note(
            f"UDF {signature.qualname} returns {signature.returns} "
            f"({signature.method.value}){detail}"
        )
    return signature.returns


def mapping_value_dtype(mapper: Any) -> DType | None:
    """The dtype a `Series.map(dict)` would produce, unified over the values.

    `map` takes a mapping as often as it takes a function, and it drifts for
    the same reason: with no row to look up, pandas returns the input dtype
    rather than the dtype of the values it would have substituted. The values
    are right there, so this one needs no inference - only the join of what
    they are, which lands on UNKNOWN for a heterogeneous mapping rather than on
    whichever value happened to come first.
    """
    from rowless.ir.dtypes import unify
    from rowless.udf.infer import python_value_to_ir

    # A dict spells `values` as a method and a Series spells it as an array, so
    # the two are told apart by whether it is callable rather than by type.
    getter = getattr(mapper, "values", None)
    try:
        values = list(getter()) if callable(getter) else list(mapper)
    except Exception:
        return None
    dtypes = [python_value_to_ir(value) for value in values]
    if not dtypes:
        return None
    joined = dtypes[0]
    for candidate in dtypes[1:]:
        joined = unify(joined, candidate)
    return joined if joined.is_known and not joined.is_any else None


def cast_series(series: Any, dtype: DType) -> Any:
    """Cast an empty result to the dtype the populated one would have had.

    Returns the series unchanged when the dtype has no pandas equivalent: a
    refusal is reported by the caller as a degradation, and a column left at
    the wrong dtype is worse than one left at the input dtype with the run
    marked unverified.
    """
    try:
        return series.astype(dtype_map.from_ir(dtype))
    except (UnsupportedDType, TypeError, ValueError):
        return None


def cast_columns(frame: Any, columns: Sequence[str], dtype: DType) -> Any:
    """Cast named columns of an empty result, leaving the rest alone."""
    try:
        target = dtype_map.from_ir(dtype)
    except UnsupportedDType:
        return None
    present = [column for column in columns if column in frame.columns]
    if not present:
        return frame
    try:
        return frame.astype(dict.fromkeys(present, target))
    except (TypeError, ValueError):
        return None


def na_upcast_dtype(dtype: Any) -> Any:
    """The dtype a populated `where`/`mask` might widen to, or None for no drift.

    Only the plain numpy dtypes can drift: a masked or arrow-backed column
    stores NA in its mask and keeps its dtype. numpy integers widen to
    `float64` and numpy bool to `object`, because neither has a value left to
    spend on NA - but only when something is actually masked out, which is why
    the caller degrades the run instead of casting to what this returns.
    """
    import numpy as np

    if not isinstance(dtype, np.dtype):
        return None
    if dtype.kind in {"i", "u"}:
        return np.dtype("float64")
    if dtype.kind == "b":
        return np.dtype("O")
    return None
