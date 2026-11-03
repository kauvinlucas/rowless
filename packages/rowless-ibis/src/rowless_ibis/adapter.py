"""The Ibis adapter. Structurally the cleanest fit in the project.

Ibis already separates expression building from execution: `ibis.table(schema)`
makes an unbound table, and `expr.schema()` resolves the output type without a
backend. That is precisely what rowless needs, which makes this the adapter with
the least engine-specific trickery.

The interesting part is that Ibis is also a portability check. The same
expression bound to DuckDB and to BigQuery can resolve to different schemas,
because the type systems of the backends differ. This adapter can resolve against
several backends and report where they disagree, which is a finding no other
adapter here can produce.
"""

from __future__ import annotations

from typing import Any, ClassVar

from rowless.adapters.base import Capabilities, DryRunContext
from rowless.ir.dtypes import DType
from rowless.ir.schema import Relation, Schema


class IbisAdapter:
    """Ibis, via unbound tables and expression schema resolution."""

    name: ClassVar[str] = "ibis"
    requires: ClassVar[str] = "ibis-framework>=9.0"
    capabilities: ClassVar[Capabilities] = Capabilities(
        column_lineage=True,
        sql=True,
        lazy=True,
        nested_types=True,
    )

    def available(self) -> bool:
        """True when ibis-framework imports at >= 9.0."""
        raise NotImplementedError("TODO(ibis)")

    def handles(self, obj: object) -> bool:
        """True for Table, Column and Scalar expressions."""
        raise NotImplementedError("TODO(ibis)")

    def materialize(self, schema: Schema, ctx: DryRunContext) -> Any:
        """`ibis.table()` with the IR schema: unbound, so nothing executes."""
        raise NotImplementedError("TODO(ibis)")

    def describe(self, obj: object) -> Relation:
        """Call `expr.schema()`, with no backend bound."""
        raise NotImplementedError("TODO(ibis)")

    def to_ir(self, native: Any) -> DType:
        """Ibis DataType to IR."""
        raise NotImplementedError("TODO(ibis)")

    def from_ir(self, dtype: DType) -> Any:
        """IR to an Ibis DataType."""
        raise NotImplementedError("TODO(ibis)")

    def run_sql(self, sql: str, inputs: dict[str, object], ctx: DryRunContext) -> Relation:
        """Resolve raw SQL through the parser of an unbound backend."""
        raise NotImplementedError("TODO(ibis)")

    def column_lineage(self, obj: object) -> dict[str, tuple[str, ...]]:
        """Walk the Ibis expression tree, which carries explicit column references."""
        raise NotImplementedError("TODO(ibis)")

    def backend_disagreements(self, obj: object, backends: tuple[str, ...]) -> dict[str, Schema]:
        """Resolve against several backends and report where the schemas differ.

        A portability finding no other adapter can produce: one expression is not
        guaranteed to have the same output type on DuckDB and on BigQuery.
        """
        raise NotImplementedError("TODO(ibis)")
