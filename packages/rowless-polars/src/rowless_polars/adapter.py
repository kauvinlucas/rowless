"""The Polars adapter. The reference implementation, and the spike target.

Polars is the easiest engine to do this well on, which is why the roadmap starts
here. `LazyFrame.collect_schema()` resolves the entire query plan through the
type rules of Polars itself and executes nothing, so the schema rowless reports
is the schema Polars would produce.

Two details worth knowing before working on this:

* `collect_schema()` raises on an invalid plan, and the exception already names
  the missing column. Translating those into `ColumnNotFound` with a near-miss
  suggestion is most of what this adapter does.
* The resolved plan is introspectable, which is where column lineage comes from:
  walk it and attribute each output column to its inputs. That is the capability
  polars-lineage demonstrated as a proof of concept, with multiple joins to be
  handled here rather than rejected.
"""

from __future__ import annotations

from typing import Any, ClassVar

from rowless.adapters.base import Capabilities, DryRunContext
from rowless.ir.dtypes import DType
from rowless.ir.schema import Relation, Schema


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
        raise NotImplementedError("TODO(polars)")

    def handles(self, obj: object) -> bool:
        """True for LazyFrame, DataFrame and Expr."""
        raise NotImplementedError("TODO(polars)")

    def materialize(self, schema: Schema, ctx: DryRunContext) -> Any:
        """Build an empty LazyFrame with the schema declared up front."""
        raise NotImplementedError("TODO(polars)")

    def describe(self, obj: object) -> Relation:
        """Resolve the plan with `collect_schema`, translating plan errors."""
        raise NotImplementedError("TODO(polars)")

    def to_ir(self, native: Any) -> DType:
        """Polars DataType to IR, keeping time unit, timezone and decimal scale."""
        raise NotImplementedError("TODO(polars)")

    def from_ir(self, dtype: DType) -> Any:
        """IR to Polars DataType, raising `UnsupportedDType` where none exists."""
        raise NotImplementedError("TODO(polars)")

    def column_lineage(self, obj: object) -> dict[str, tuple[str, ...]]:
        """Walk the resolved plan, attributing each output column to its inputs."""
        raise NotImplementedError("TODO(polars)")

    def probe_udf(self, fn: Any, arg_schema: Schema, ctx: DryRunContext) -> DType:
        """Prefer `return_dtype=`, then annotations, then a single-row probe."""
        raise NotImplementedError("TODO(polars)")
