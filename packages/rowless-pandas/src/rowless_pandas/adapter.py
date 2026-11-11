"""The pandas adapter.

The hardest of the dataframe adapters, and the most widely useful.

pandas has no lazy plan, so the only option is a real frame with zero rows.
That works, and it works better than the folklore suggests: on pandas 2.2 and
3.x, `select`, `merge`, `groupby().agg("sum")`, `concat`, arithmetic and the
`.str` and `.dt` accessors all compute the same dtypes from zero rows as they
do from data. What does not is a short, specific list, and every entry on it is
in `rowless_pandas.drift` with what pandas does, what it would have done, and
what this adapter does about it.

How the six required methods are satisfied:

* `materialize` builds `pd.DataFrame` of empty typed columns, preferring the
  dtypes that cannot drift - pyarrow-backed where pyarrow is installed, the
  masked extension dtypes otherwise.
* `describe` reads `obj.dtypes`. No execution, because there is nothing to
  execute: pandas computed the dtypes as the pipeline built the frame.
* `to_ir` and `from_ir` are in `rowless_pandas.dtypes`, where the three pandas
  dtype vocabularies and the disagreements between them are written down.
* the row dependence, the drift corrections, the error translation and the
  writer recorders are in `rowless_pandas.interception`, which explains what it
  patches and why.

## Where pandas disagrees with the other engines

The catalogue this adapter exists to write down. Each line is a place where
moving a pipeline between engines changes its output schema, and the
conformance cases in `rowless_pandas.cases` assert every one of them.

* **`groupby("status").agg(...)` puts the key in the index, not in a column.**
  The output schema is `{amount}`, where Polars and DuckDB both produce
  `{status, amount}`. This is the single biggest difference between pandas and
  everything else rowless talks to, it is invisible until something downstream
  asks for `status`, and `as_index=False` or `reset_index()` is the fix. A
  schema includes the columns, so rowless reports what pandas would hand to the
  next operation, and names the index in a note rather than quietly adding a
  column that is not one.
* **A join suffixes collisions `_x` and `_y`.** Polars suffixes `_right`,
  DuckDB keeps both columns under the same name. All three differ, and a
  declared contract notices all three.
* **`merge(left_on=, right_on=)` keeps both key columns** unless the names
  match, where Polars coalesces the right key away.
* **`amount / 2` is `float64`**, as in Polars and DuckDB. `//` keeps the
  integer, also as in both.
* **`groupby().agg("sum")` keeps the column name and the input dtype**, as
  Polars does. DuckDB renames it `sum(amount)` and widens it to 128 bits.
* **`melt` produces `variable` and `value`**, the same names Polars `unpivot`
  gives them, and `value` is an `object` column whenever the melted columns had
  different dtypes - which is a column rowless reports as UNKNOWN, because
  pandas genuinely does not know what is in it.
* **`.dt.date` is an `object` column** without pyarrow, so it has no dtype to
  report at all. With pyarrow it is `date32`. The same expression is `date` in
  Polars either way.
* **Nullability is a property of the dtype, and this adapter reports it.** A
  plain numpy `int64` or `bool` column cannot hold a missing value - pandas
  would have to widen the dtype first - while a masked, arrow-backed or
  `datetime64` column can. So `describe` reports what the dtype permits, which
  is more than Polars can say, since Polars does not model nullability at all.
  It is a statement about the column's capacity and not about its contents: a
  dry run has no contents to look at.
* **pandas keeps a named timezone.** `datetime64[ns, Europe/London]` survives
  the round trip, where DuckDB has to refuse it.

## What the empty frame hides

The honest other half, and worth knowing before trusting a pass. `astype`
coercions that would fail on data succeed on zero rows: `df["status"]
.astype("int64")` raises `ValueError: invalid literal for int()` in production
and returns an empty `int64` column here. The same goes for an aggregation that
would overflow and a `merge` whose keys would duplicate rows. rowless checks
schemas, not values, and a pipeline that type-checks can still fail on the
first row it sees. `rowless check` says what it verified; this is what it did
not.

## Inputs the IR can describe and pandas cannot

`materialize` has to produce a frame for whatever schema a source resolved. An
input column whose dtype is UNKNOWN, ANY, or one pandas genuinely lacks - a
decimal, a date, a time or binary, when pyarrow is not installed - is
materialized as `object` and the run is degraded, with the column named in the
reason. The alternative, refusing to check the pipeline at all, loses the other
fifteen columns to one unresolved source; and a degraded run cannot be reported
as a pass, so nothing is claimed that was not earned.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

from rowless.adapters.base import Capabilities, DryRunContext
from rowless.errors import DuplicateColumn, UnknownSchemaError, UnsupportedDType
from rowless.ir.dtypes import DType
from rowless.ir.schema import Field, Relation, Schema

from rowless_pandas import drift, interception
from rowless_pandas import dtypes as dtype_map

#: The oldest pandas this adapter supports. 2.0 brought the non-nanosecond
#: datetime units and `ArrowDtype`; 2.2 is the floor because it is the first
#: release where the arrow-backed dtypes carry their own `str` accessor and
#: `map` consistently, and testing a floor nobody runs is testing a guess.
MINIMUM_PANDAS: tuple[int, int] = (2, 2)


class PandasAdapter:
    """pandas, via empty typed frames plus explicit drift corrections."""

    name: ClassVar[str] = "pandas"
    requires: ClassVar[str] = "pandas>=2.2"
    capabilities: ClassVar[Capabilities] = Capabilities(
        udf_probe=True,
        nested_types=False,
    )
    """What this adapter promises.

    `nested_types` is off, and deliberately: `rowless_pandas.dtypes` maps
    lists, structs and maps exactly in both directions, but only through
    `ArrowDtype`, and pyarrow is not a dependency of this distribution. A
    capability that half the installs cannot honour is worse than one that is
    missing, because coverage scores are computed from these flags - so a
    nested column still describes correctly, and the flag stays off.

    `column_lineage` is off for a harder reason. pandas has no expression
    objects: `df.assign(total=lambda d: d.a + d.b)` hides its inputs inside a
    closure, and `df["a"] + df["b"]` produces a Series that remembers nothing
    about where it came from. Recording lineage for the name-preserving
    operations alone would be possible, and would report a computed column as
    untraced while quietly implying the rest is complete. Until there is a way
    to attribute a computed column, pandas columns are reported untraced, which
    costs `rowless diff` its blast radius for them and is the honest outcome.
    """

    def available(self) -> bool:
        """True when pandas imports at >= 2.2."""
        try:
            import pandas as pd
        except Exception:
            # An engine that fails to import is simply not available. The
            # protocol says this method may not raise, and a half-installed
            # pandas - a missing numpy ABI, say - raises on import.
            return False
        return _at_least(str(getattr(pd, "__version__", "0")), MINIMUM_PANDAS)

    def handles(self, obj: object) -> bool:
        """True for DataFrame and Series."""
        try:
            import pandas as pd
        except Exception:
            return False
        return isinstance(obj, pd.DataFrame | pd.Series)

    def materialize(self, schema: Schema, ctx: DryRunContext) -> Any:
        """Build a zero-row frame, preferring the dtypes that do not drift.

        Also the point where this adapter installs its interception: the
        runtime calls `materialize` before it imports or calls user code, so
        the wrappers are in place for whatever the pipeline does next.
        """
        import pandas as pd

        self._choose_backend(ctx)
        interception.install(ctx)
        with interception.own_work():
            columns = {
                field.name: pd.Series([], dtype=self._input_dtype(field, ctx), name=field.name)
                for field in schema.fields
            }
            if not columns:
                # A legal schema with no columns. `pd.DataFrame({})` is the
                # empty frame, which is what a zero-column relation is.
                return pd.DataFrame()
            return pd.DataFrame(columns)

    def describe(self, obj: object) -> Relation:
        """Read the dtypes pandas computed while the pipeline built the frame.

        Nothing is executed, because there is nothing left to execute: pandas
        is eager, so by the time a frame exists its dtypes are already decided.
        What this method does instead is refuse the two shapes a rowless schema
        cannot describe - duplicate column labels and a MultiIndex column axis -
        rather than silently picking one of the duplicates.
        """
        import pandas as pd

        if isinstance(obj, pd.DataFrame):
            frame = obj
        elif isinstance(obj, pd.Series):
            # A one-column relation. The name is the column name, and an
            # unnamed Series gets the same placeholder the Polars adapter uses.
            with interception.own_work():
                frame = obj.to_frame(name=obj.name if obj.name is not None else "series")
        else:
            qualified = f"{type(obj).__module__}.{type(obj).__qualname__}"
            raise UnknownSchemaError(
                f"the pandas adapter cannot describe {qualified}; a pipeline has to return a "
                "DataFrame or a Series"
            )
        # Reading a frame's dtypes means iterating a Series, which is one of
        # the accessors the interception layer reports. This is the adapter's
        # own work, not the pipeline's.
        with interception.own_work():
            _refuse_multiindex(frame)
            names = [str(column) for column in frame.columns]
            _refuse_duplicates(names)
            dtypes = list(frame.dtypes)
            nullable = _nullability(dtypes)
            notes = _index_notes(frame)
        return Relation(
            schema=Schema(
                fields=tuple(
                    Field(name=name, dtype=dtype_map.to_ir(dtype), nullable=is_nullable)
                    for name, dtype, is_nullable in zip(names, dtypes, nullable, strict=True)
                )
            ),
            engine=self.name,
            notes=notes,
        )

    def to_ir(self, native: Any) -> DType:
        """numpy, pandas extension and pyarrow dtypes to IR."""
        return dtype_map.to_ir(native)

    def from_ir(self, dtype: DType) -> Any:
        """IR to a pandas dtype, honouring the arrow_dtypes option."""
        return dtype_map.from_ir(dtype)

    def unsupported_dtypes(self) -> Sequence[DType]:
        """Corpus dtypes the installed pandas has no equivalent for.

        Empty when pyarrow is available. Without it, pandas has no decimal,
        date, time, binary or null dtype at all: all five would be `object`
        columns, which carry no dtype, so `from_ir` refuses them and the
        conformance suite checks the refusal rather than the round trip.

        Unusually for this hook, the answer depends on the environment. DuckDB
        declares a static absence because DuckDB always lacks that dtype;
        pandas lacks these five only when pyarrow is missing, and reporting
        them as absent on an install where they work would be the same kind of
        lie in the other direction.
        """
        return dtype_map.absent_dtypes()

    def dtype_drift_fixups(self) -> Sequence[str]:
        """The corrections this adapter applies, for `rowless explain`."""
        return drift.descriptions()

    def probe_udf(self, fn: Any, arg_schema: Schema, ctx: DryRunContext) -> DType:
        """Infer an `apply` return dtype from annotations, then by probing.

        The order is `rowless.udf.infer`'s: a declared dtype, then the return
        annotation, then - only when `probe_udfs` is configured - one synthetic
        call. The last is opt-in because it runs user code with invented input.
        """
        return interception.udf_return_dtype(fn, arg_schema, ctx=ctx)

    def conformance_cases(self) -> Sequence[Any]:
        """The operations battery for pandas, with the per-engine expectations."""
        from rowless_pandas.cases import conformance_cases

        return conformance_cases()

    def _choose_backend(self, ctx: DryRunContext) -> None:
        """Pick the dtype table for this run, and say so when it is the lesser one.

        The fallback is reported rather than silent: which table is in use
        decides whether five dtypes exist at all, and a user whose decimal
        column came back unresolved needs to be told that installing pyarrow
        fixes it.
        """
        option = ctx.options.get("arrow_dtypes")
        chosen = dtype_map.resolve_backend(option)
        dtype_map.use_backend(chosen)
        if chosen == "arrow":
            return
        asked = option is not None and str(option).strip().lower() not in {
            "false",
            "no",
            "off",
            "0",
            "numpy",
        }
        reason = (
            "pyarrow is not installed, so"
            if asked or option is None
            else "[engine.pandas] arrow_dtypes is off, so"
        )
        ctx.note(
            f"{reason} this run used pandas' masked extension dtypes. They hold NA without "
            "changing dtype, which is most of what the drift table exists for, but pandas has "
            "no decimal, date, time, binary or null dtype without pyarrow and columns of those "
            "types cannot be carried at all"
        )

    def _input_dtype(self, field: Field, ctx: DryRunContext) -> Any:
        """The pandas dtype for one input column, degrading when there is none."""
        import numpy as np

        if not field.dtype.is_known:
            ctx.degrade(
                f"input column {field.name!r} has an unresolved dtype "
                f"({field.dtype.reason or 'no reason given'}); it was materialized as object"
            )
            return np.dtype("O")
        if field.dtype.is_any:
            ctx.note(f"input column {field.name!r} is typed ANY; it was materialized as object")
            return np.dtype("O")
        try:
            return dtype_map.from_ir(field.dtype)
        except UnsupportedDType as exc:
            ctx.degrade(f"input column {field.name!r}: {exc}; it was materialized as object")
            return np.dtype("O")


def _nullability(dtypes: Sequence[Any]) -> list[bool]:
    """Whether each column can hold NA, which in pandas depends on the dtype.

    A masked or arrow-backed column holds NA in its mask, and a numpy
    `datetime64` has NaT, so all of those are nullable. A plain numpy integer
    or boolean column cannot hold a missing value at all: pandas would have to
    widen the dtype first, which is the drift this adapter corrects elsewhere.
    Reporting that distinction is something Polars cannot do, so it is worth
    reporting accurately rather than defaulting everything to nullable.
    """
    import numpy as np

    found: list[bool] = []
    for dtype in dtypes:
        if isinstance(dtype, np.dtype):
            found.append(dtype.kind not in {"i", "u", "b"})
        else:
            found.append(True)
    return found


def _index_notes(frame: Any) -> tuple[str, ...]:
    """Name the index levels that carry data, because the schema does not.

    The `groupby` case: `df.groupby("status").agg(...)` leaves `status` in the
    index, so it is absent from the schema rowless reports and present in
    anything that reads the frame back with its index. Saying so in a note is
    the middle road between inventing a column and leaving a user to wonder
    where their group key went.
    """
    import pandas as pd

    index = getattr(frame, "index", None)
    if index is None or isinstance(index, pd.RangeIndex):
        return ()
    named = [str(name) for name in index.names if name is not None]
    if not named:
        return ()
    return (
        f"this frame carries data in its index: {', '.join(named)}. A rowless schema describes "
        "the columns, so those are not in it - call reset_index(), or pass as_index=False to "
        "groupby, to put them back where the other engines keep them",
    )


def _refuse_duplicates(names: Sequence[str]) -> None:
    """Report two columns of the same name, which pandas allows and rowless cannot.

    pandas is happy to hold `['amount', 'amount']`: `pd.concat(..., axis=1)`
    produces it, and so does a `merge` with the suffixes turned off. A rowless
    schema is addressed by name, and so is everything downstream of it - a
    contract, a diff, a lineage edge - so the duplicate is the finding.
    """
    seen: set[str] = set()
    for name in names:
        if name in seen:
            raise DuplicateColumn(
                f"this frame has two columns named {name!r}. pandas allows duplicate column "
                "labels, but nothing downstream of a rowless schema can refer to either of "
                "them; rename one, or pass suffixes= to the merge that made them"
            )
        seen.add(name)


def _refuse_multiindex(frame: Any) -> None:
    """Refuse a MultiIndex column axis, which has no flat name to report.

    `df.groupby("k").agg(["sum", "mean"])` produces `('amount', 'sum')` as a
    column label. Flattening it to `amount_sum` would invent a name no
    downstream consumer uses, so this reports that the schema is unresolved -
    unverified rather than an error, because the pipeline may be perfectly
    correct and one `columns = [...]` away from being checkable.
    """
    import pandas as pd

    if isinstance(frame.columns, pd.MultiIndex):
        raise UnknownSchemaError(
            "this frame has a MultiIndex column axis, so its columns have no single names to "
            "report. Flatten it - df.columns = ['_'.join(c) for c in df.columns] - or use a "
            "named aggregation, agg(total=('amount', 'sum')), which produces flat names"
        )


def _at_least(version: str, minimum: tuple[int, int]) -> bool:
    """True when a version string is at least `minimum`, ignoring any suffix."""
    parts: list[int] = []
    for component in version.split(".")[:2]:
        digits = ""
        for character in component:
            if not character.isdigit():
                break
            digits += character
        parts.append(int(digits) if digits else 0)
    while len(parts) < 2:
        parts.append(0)
    return (parts[0], parts[1]) >= minimum
