"""The DuckDB adapter, and the Python-to-SQL seam.

Create tables with the declared schema and no rows, then `DESCRIBE` the query.
DuckDB binds and plans without scanning, so the answer comes from its own
binder, including its coercion rules for joins and set operations.

This adapter matters more than the popularity of its engine suggests, because it
is how a SQL string becomes a node in the same graph as a Polars function: the
`SupportsSql` implementation here registers rowless relations as empty tables and
resolves arbitrary SQL against them. A dbt model and a `group_by` land in one
lineage graph, which is the gap the proposal is built around.

DuckDB can also read a Polars or pandas frame directly, so a mixed pipeline
(`duckdb.sql("select * from df")`) resolves without a copy. Those replacement
scans need registering before the query binds, and that is the fiddly part.
"""

from __future__ import annotations

from typing import Any, ClassVar

from rowless.adapters.base import Capabilities, DryRunContext
from rowless.ir.dtypes import DType
from rowless.ir.schema import Relation, Schema


class DuckDBAdapter:
    """DuckDB, via empty tables and DESCRIBE."""

    name: ClassVar[str] = "duckdb"
    requires: ClassVar[str] = "duckdb>=1.0"
    capabilities: ClassVar[Capabilities] = Capabilities(
        column_lineage=True,
        sql=True,
        lazy=True,
        nested_types=True,
    )

    def available(self) -> bool:
        """True when duckdb imports at >= 1.0."""
        raise NotImplementedError("TODO(duckdb)")

    def handles(self, obj: object) -> bool:
        """True for DuckDBPyRelation and DuckDBPyConnection."""
        raise NotImplementedError("TODO(duckdb)")

    def materialize(self, schema: Schema, ctx: DryRunContext) -> Any:
        """CREATE an empty table from the IR schema in a scratch database."""
        raise NotImplementedError("TODO(duckdb)")

    def describe(self, obj: object) -> Relation:
        """Run DESCRIBE, or read `relation.columns` and `relation.types`."""
        raise NotImplementedError("TODO(duckdb)")

    def to_ir(self, native: Any) -> DType:
        """DuckDB logical type to IR, including nested and ENUM types."""
        raise NotImplementedError("TODO(duckdb)")

    def from_ir(self, dtype: DType) -> Any:
        """IR to a DuckDB type string."""
        raise NotImplementedError("TODO(duckdb)")

    def run_sql(self, sql: str, inputs: dict[str, object], ctx: DryRunContext) -> Relation:
        """Register inputs as empty tables, then resolve the output schema."""
        raise NotImplementedError("TODO(duckdb)")

    def column_lineage(self, obj: object) -> dict[str, tuple[str, ...]]:
        """Derive lineage from the bound plan via `EXPLAIN (FORMAT JSON)`."""
        raise NotImplementedError("TODO(duckdb)")
