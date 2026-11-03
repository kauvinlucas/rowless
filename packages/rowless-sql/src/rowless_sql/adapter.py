"""The SQL adapter: schemas from SQL text, no database required.

`sqlglot` parses, qualifies and annotates types given the input schemas, so a
SQL model resolves with no engine running at all. That makes this the adapter
that works in a credential-free CI job, which is the property that makes
whatbreaks adoptable and is worth matching.

It is also the lower-confidence option, and it must say so. Type annotation in
sqlglot is dialect-aware but not complete: some functions have no annotated
return type, and the honest result is an ANY column with a note naming the
function. Where a real engine is available, the DuckDB binder is the better
answer, and `rowless check` prefers it.

Dialects are configuration, not detection: guessing wrong turns a correct query
into a parse error. The dialect comes from `[engine.sql] dialect`, or from a dbt
profile when `rowless-dbt` is installed.
"""

from __future__ import annotations

from typing import Any, ClassVar

from rowless.adapters.base import Capabilities, DryRunContext
from rowless.ir.dtypes import DType
from rowless.ir.schema import Relation, Schema


class SqlAdapter:
    """SQL strings, via sqlglot qualification and type annotation."""

    name: ClassVar[str] = "sql"
    requires: ClassVar[str] = "sqlglot>=25.0"
    capabilities: ClassVar[Capabilities] = Capabilities(
        column_lineage=True,
        sql=True,
        lazy=True,
        nested_types=True,
    )

    def available(self) -> bool:
        """True when sqlglot imports at >= 25.0."""
        raise NotImplementedError("TODO(sql)")

    def handles(self, obj: object) -> bool:
        """True for sqlglot expressions, and for str when SQL mode is on."""
        raise NotImplementedError("TODO(sql)")

    def materialize(self, schema: Schema, ctx: DryRunContext) -> Any:
        """Return a sqlglot schema mapping; there is no frame object here."""
        raise NotImplementedError("TODO(sql)")

    def describe(self, obj: object) -> Relation:
        """Qualify, annotate types, then read the projection off the select."""
        raise NotImplementedError("TODO(sql)")

    def to_ir(self, native: Any) -> DType:
        """Sqlglot DataType to IR."""
        raise NotImplementedError("TODO(sql)")

    def from_ir(self, dtype: DType) -> Any:
        """IR to a dialect-specific sqlglot DataType."""
        raise NotImplementedError("TODO(sql)")

    def run_sql(self, sql: str, inputs: dict[str, object], ctx: DryRunContext) -> Relation:
        """Resolve the output schema of a query against named input schemas."""
        raise NotImplementedError("TODO(sql)")

    def column_lineage(self, obj: object) -> dict[str, tuple[str, ...]]:
        """Use `sqlglot.lineage` to attribute each projection to source columns."""
        raise NotImplementedError("TODO(sql)")
