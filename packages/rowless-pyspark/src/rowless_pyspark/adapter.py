"""The PySpark adapter.

Last on the roadmap, because the setup cost is the hard part, not the schema
resolution.

`df.schema` triggers analysis only: Catalyst resolves and type-checks the plan
without launching a job. So the schema is exact, and it is the answer Spark
itself would give, including implicit cast rules that differ from every other
engine here in ways that matter.

Spark Connect is what makes this practical. A classic local SparkSession takes
tens of seconds to start, which breaks the linter-speed promise outright. With
Connect, analysis happens on an already-running server and a dry run costs a
round trip. The trade is that this is the only adapter needing network egress,
so the sandbox opens exactly the configured endpoint and nothing else, and
`remote=True` tells the user this adapter talks to something.

Local mode stays supported for people without a cluster, behind
`[engine.pyspark] mode = "local"`, with the session reused across every dry run
in a check so the startup cost is paid once.
"""

from __future__ import annotations

from typing import Any, ClassVar

from rowless.adapters.base import Capabilities, DryRunContext
from rowless.ir.dtypes import DType
from rowless.ir.schema import Relation, Schema


class PySparkAdapter:
    """PySpark, via the Catalyst analyzed plan over Spark Connect."""

    name: ClassVar[str] = "pyspark"
    requires: ClassVar[str] = "pyspark>=3.5"
    capabilities: ClassVar[Capabilities] = Capabilities(
        column_lineage=True,
        sql=True,
        lazy=True,
        nested_types=True,
        remote=True,
    )

    def available(self) -> bool:
        """True when pyspark imports and Connect or local mode is configured."""
        raise NotImplementedError("TODO(pyspark)")

    def handles(self, obj: object) -> bool:
        """True for pyspark.sql DataFrame and Column, Connect or classic."""
        raise NotImplementedError("TODO(pyspark)")

    def materialize(self, schema: Schema, ctx: DryRunContext) -> Any:
        """`spark.createDataFrame([], schema)` from the IR schema."""
        raise NotImplementedError("TODO(pyspark)")

    def describe(self, obj: object) -> Relation:
        """Read `df.schema`, which runs analysis but no job."""
        raise NotImplementedError("TODO(pyspark)")

    def to_ir(self, native: Any) -> DType:
        """Spark DataType to IR, including nested and decimal types."""
        raise NotImplementedError("TODO(pyspark)")

    def from_ir(self, dtype: DType) -> Any:
        """IR to a Spark DataType."""
        raise NotImplementedError("TODO(pyspark)")

    def run_sql(self, sql: str, inputs: dict[str, object], ctx: DryRunContext) -> Relation:
        """Register temp views for the inputs, then analyze the query."""
        raise NotImplementedError("TODO(pyspark)")

    def column_lineage(self, obj: object) -> dict[str, tuple[str, ...]]:
        """Walk the analyzed plan, mapping expression IDs back to source attributes."""
        raise NotImplementedError("TODO(pyspark)")
