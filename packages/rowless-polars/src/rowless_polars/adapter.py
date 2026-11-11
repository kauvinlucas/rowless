"""The Polars adapter. The reference implementation, and the spike target.

Polars is the easiest engine to do this well on, which is why the roadmap
starts here. `LazyFrame.collect_schema()` resolves the entire query plan
through the type rules of Polars itself and executes nothing, so the schema
rowless reports is the schema Polars would produce.

How the four required methods are satisfied:

* `materialize` builds `pl.LazyFrame(schema=...)`: a real lazy frame with the
  full schema and no data behind it. User code gets a genuine `LazyFrame`, so
  real pipelines run unmodified.
* `describe` calls `collect_schema()`, which resolves and does not execute.
* `to_ir` and `from_ir` are in `rowless_polars.dtypes`, where the dtype-level
  disagreements are written down.
* the row dependence, the plan validation, the lineage and the writer
  recorders are in `rowless_polars.interception`, which explains what it
  patches and why.

## Where Polars disagrees with the other engines

The catalogue this adapter exists to write down. Each line is a place where
moving a pipeline between engines changes its output schema, and the
conformance cases in `rowless_polars.cases` assert every one of them.

* `group_by().agg(col("amount").sum())` keeps the column name `amount` and the
  `int64` dtype. DuckDB renames it to `sum(amount)` and widens it to a 128-bit
  integer; pandas keeps the name but may widen.
* `col("amount") / 2` is `float64`. Polars divides like Python, and so does
  DuckDB: `/` is true division there too, and `//` is the integer one. The
  engines that keep an integer for `/` are the older SQL dialects.
* `col("tags").list.len()` is `uint32`, under the original column name. An
  unsigned count is a dtype change a declared contract will notice.
* `join(..., left_on=, right_on=)` coalesces the right-hand key away, where SQL
  keeps both key columns.
* a name collision in a join is suffixed `_right`. DuckDB qualifies instead,
  and pandas suffixes `_x` and `_y`.
* `unpivot` produces `variable` and `value`, both untraced. pandas `melt`
  agrees on the names, DuckDB does not.
* nullability is not modelled at all, so every described column is reported
  nullable, whatever the source declared.
* `map_elements` with no `return_dtype` makes Polars call the function once to
  find out. This adapter prefers a return annotation, and reports an unverified
  finding when there is neither.

Two of those answers also changed between Polars releases, so the adapter
normalises them rather than letting a finding depend on the installed version:
duplicate output names (2.x raises, 1.x keeps one silently) are refused by the
adapter itself, and a UDF with no determinable return dtype degrades the run on
both. `rowless-polars/README.md` has the detail, and the suite runs at the
declared floor as well as at the current release.

## Inputs the IR can describe and Polars cannot

`materialize` has to produce a frame for whatever schema a source resolved. An
input column whose dtype is UNKNOWN, ANY, or a dtype Polars genuinely lacks
(`datetime[s]`, `float16`) is materialized as `pl.Null` and the run is
degraded, with the column named in the reason. The alternative - refusing to
check the pipeline at all - loses the other fifteen columns to one unresolved
source, and a degraded run cannot be reported as a pass, so nothing is claimed
that was not earned.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

from rowless.adapters.base import Capabilities, DryRunContext
from rowless.errors import UnknownSchemaError, UnsupportedDType
from rowless.ir.dtypes import DType
from rowless.ir.schema import Field, Relation, Schema

from rowless_polars import dtypes as dtype_map
from rowless_polars import errors, interception, lineage

#: The oldest Polars this adapter supports. `collect_schema()` arrived in 1.0,
#: and everything newer that is used - 128-bit integers, `remove`, the
#: `raise_if_undetermined` flag on expression metadata - is behind a feature
#: check rather than a version check.
MINIMUM_POLARS: tuple[int, int] = (1, 0)


class PolarsAdapter:
    """Polars, via lazy plan resolution."""

    name: ClassVar[str] = "polars"
    requires: ClassVar[str] = "polars>=1.0"
    capabilities: ClassVar[Capabilities] = Capabilities(
        column_lineage=True,
        udf_probe=True,
        lazy=True,
        nested_types=True,
    )

    def available(self) -> bool:
        """True when polars imports at >= 1.0."""
        try:
            import polars as pl
        except Exception:
            # An engine that fails to import is simply not available. The
            # protocol says this method may not raise, and a half-installed
            # Polars (a missing native wheel, say) raises on import.
            return False
        return _at_least(str(getattr(pl, "__version__", "0")), MINIMUM_POLARS)

    def handles(self, obj: object) -> bool:
        """True for LazyFrame, DataFrame, Series and Expr."""
        try:
            import polars as pl
        except Exception:
            return False
        return isinstance(obj, pl.LazyFrame | pl.DataFrame | pl.Series | pl.Expr)

    def materialize(self, schema: Schema, ctx: DryRunContext) -> Any:
        """Build an empty LazyFrame with the schema declared up front.

        Also the point where this adapter installs its interception: the
        runtime calls `materialize` before it imports or calls user code, so
        the wrappers are in place for whatever the pipeline does next.
        """
        import polars as pl

        interception.install(ctx)
        columns = {field.name: self._input_dtype(field, ctx) for field in schema.fields}
        frame = pl.LazyFrame(schema=columns)
        lineage.TRACKER.seed(frame, columns)
        return frame

    def describe(self, obj: object) -> Relation:
        """Resolve the plan with `collect_schema`, translating plan errors."""
        import polars as pl

        if isinstance(obj, pl.LazyFrame):
            try:
                resolved = dict(obj.collect_schema())
            except Exception as exc:
                errors.raise_translated(exc)
        elif isinstance(obj, pl.DataFrame):
            resolved = dict(obj.schema)
        elif isinstance(obj, pl.Series):
            resolved = {obj.name or "series": obj.dtype}
        else:
            qualified = f"{type(obj).__module__}.{type(obj).__qualname__}"
            raise UnknownSchemaError(
                f"the polars adapter cannot describe {qualified}; a pipeline has to return a "
                "LazyFrame, a DataFrame or a Series"
            )
        fields = tuple(
            # Polars does not model nullability, so every column is reported
            # nullable rather than guessed at.
            Field(name=str(name), dtype=dtype_map.to_ir(dtype))
            for name, dtype in resolved.items()
        )
        return Relation(schema=Schema(fields=fields), engine=self.name)

    def to_ir(self, native: Any) -> DType:
        """Polars DataType to IR, keeping time unit, timezone and decimal scale."""
        return dtype_map.to_ir(native)

    def from_ir(self, dtype: DType) -> Any:
        """IR to Polars DataType, raising `UnsupportedDType` where none exists."""
        return dtype_map.from_ir(dtype)

    def column_lineage(self, obj: object) -> Mapping[str, tuple[str, ...]]:
        """Attribute each output column to the input columns it was computed from.

        Recorded while the plan was built, by the wrappers in
        `rowless_polars.interception`. A frame produced by an operation those
        wrappers do not cover has no entry, so its columns are reported
        untraced rather than attributed to a guess.
        """
        return lineage.TRACKER.of(obj) or {}

    def probe_udf(
        self,
        fn: Any,
        arg_schema: Schema,
        ctx: DryRunContext,
    ) -> DType:
        """Prefer `return_dtype=`, then annotations, then a single-row probe.

        The order is `rowless.udf.infer`'s, and the probe only happens when
        `probe_udfs` is configured: calling a UDF with invented values is
        opt-in, because it runs user code with made-up input.
        """
        return interception.udf_return_dtype(fn, arg_schema, ctx=ctx)

    def conformance_cases(self) -> Sequence[Any]:
        """The operations battery for Polars, with the per-engine expectations."""
        from rowless_polars.cases import conformance_cases

        return conformance_cases()

    def _input_dtype(self, field: Field, ctx: DryRunContext) -> Any:
        """The Polars dtype for one input column, degrading when there is none."""
        import polars as pl

        if not field.dtype.is_known:
            ctx.degrade(
                f"input column {field.name!r} has an unresolved dtype "
                f"({field.dtype.reason or 'no reason given'}); it was materialized as Null"
            )
            return pl.Null
        if field.dtype.is_any:
            ctx.note(f"input column {field.name!r} is typed ANY; it was materialized as Null")
            return pl.Null
        try:
            return dtype_map.from_ir(field.dtype)
        except UnsupportedDType as exc:
            ctx.degrade(f"input column {field.name!r}: {exc}; it was materialized as Null")
            return pl.Null


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
