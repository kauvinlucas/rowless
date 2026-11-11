"""The operations battery for PySpark: real pipelines, and the schemas they produce.

The conformance suite in core writes the universal half of the contract once;
this is the half that cannot be written there, because `df.select("a")` is not
the same call in Spark, Polars and pandas and inventing one abstract vocabulary
for all of them is the proxy approach the project rejects.

Every case runs through the same runtime the CLI uses, so what is asserted is
the schema a user would get. The `reason` on a case is the valuable part: it is
where an engine disagreement gets written down, and the suite prints it when
the case fails.

Nothing here calls `pyspark.sql.functions` while the cases are being
*collected*: a classic session resolves `F.col` against the active
SparkContext, so a module that built its expressions at import time would need
a JVM merely to list the tests. Every expression is built inside its pipeline,
which is where a session exists.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from rowless.errors import ColumnNotFound, DuplicateColumn
from rowless.ir.confidence import Confidence
from rowless.ir.dtypes import (
    TimeUnit,
    datetime,
    decimal,
    float_,
    int_,
    list_,
    map_,
    string,
    struct,
)
from rowless.ir.schema import Schema
from rowless.testing import fixtures
from rowless.testing.conformance import ConformanceCase

#: The timestamp dtype both fixture schemas carry, spelled once. It is UTC
#: because the adapter pins the session timezone to UTC, which is what makes a
#: zone-carrying dtype meaningful at all on an engine whose TIMESTAMP has no
#: zone of its own.
TIMESTAMP = datetime(TimeUnit.MICROSECOND, "UTC")

#: A table name no cluster has. Used by the statement-guard case: if the guard
#: ever stopped working, Spark would fail to analyze the INSERT instead of
#: quietly returning the empty frame a command returns.
MISSING_TABLE = "rowless_guard_check_missing_table"


def untyped(value: int) -> int:
    """A UDF with no `returnType` declared at the call site.

    A named function rather than a lambda so the degradation names something a
    user can find. The annotation here is deliberately *not* enough: Spark does
    not read annotations, and defaults the column to a string, which is the
    whole point of the case.
    """
    return value * 2


def conformance_cases() -> Sequence[ConformanceCase]:
    """Every operation this adapter claims to resolve, and what it resolves to."""
    orders = {"orders": fixtures.ORDERS}
    joined = {"orders": fixtures.ORDERS, "customers": fixtures.CUSTOMERS}
    events = {"events": fixtures.EVENTS}

    return (
        ConformanceCase(
            name="select",
            pipeline=lambda orders: orders.select("order_id", "amount"),
            inputs=orders,
            expect=Schema.from_mapping({"order_id": int_(64), "amount": int_(64)}),
        ),
        ConformanceCase(
            name="select_is_exact",
            pipeline=lambda orders: orders.select("order_id"),
            inputs=orders,
            expect=Schema.from_mapping({"order_id": int_(64)}),
            expect_confidence=Confidence.EXACT,
            reason=(
                "Catalyst analysis is the engine's own answer, so nothing is approximated; "
                "analysis resolves the plan and launches no job"
            ),
        ),
        ConformanceCase(
            name="alias",
            pipeline=lambda orders: orders.select(_functions().col("amount").alias("amount_cents")),
            inputs=orders,
            expect=Schema.from_mapping({"amount_cents": int_(64)}),
        ),
        ConformanceCase(
            name="rename",
            pipeline=lambda orders: orders.withColumnRenamed("amount", "amount_cents"),
            inputs=orders,
            expect=Schema.from_mapping(
                {
                    "order_id": int_(64),
                    "status": string(),
                    "amount_cents": int_(64),
                    "created_at": TIMESTAMP,
                }
            ),
            reason="withColumnRenamed keeps the column position, which a select would not",
        ),
        ConformanceCase(
            name="filter",
            pipeline=lambda orders: orders.filter(_functions().col("status") == "paid"),
            inputs=orders,
            expect=Schema.from_mapping(
                {
                    "order_id": int_(64),
                    "status": string(),
                    "amount": int_(64),
                    "created_at": TIMESTAMP,
                }
            ),
            reason="a predicate cannot change the schema, however many rows it removes",
        ),
        ConformanceCase(
            name="cast",
            pipeline=lambda orders: orders.withColumn(
                "amount", _functions().col("amount").cast("double")
            ),
            inputs=orders,
            expect=Schema.from_mapping(
                {
                    "order_id": int_(64),
                    "status": string(),
                    "amount": float_(64),
                    "created_at": TIMESTAMP,
                }
            ),
            reason="withColumn replaces the column in place rather than appending it",
        ),
        ConformanceCase(
            name="select_expr",
            pipeline=lambda orders: orders.selectExpr("amount / 2 as unit_price"),
            inputs=orders,
            expect=Schema.from_mapping({"unit_price": float_(64)}),
            reason=(
                "Spark divides like Python and returns double, which is the Polars and DuckDB "
                "answer too; the engines that keep an integer for / are the older SQL dialects"
            ),
        ),
        ConformanceCase(
            name="group_by_sum",
            pipeline=lambda orders: orders.groupBy("status").agg(_functions().sum("amount")),
            inputs=orders,
            expect=Schema.from_mapping({"status": string(), "sum(amount)": int_(64)}),
            reason=(
                "Spark names the aggregate after the expression - sum(amount) - and does not "
                "widen a bigint. DuckDB picks the same name and widens to a 128-bit integer; "
                "Polars keeps the input name amount; ibis calls it Sum(amount)"
            ),
        ),
        ConformanceCase(
            name="group_by_sum_decimal",
            pipeline=lambda orders: orders.groupBy("status").agg(
                _functions().sum(_functions().col("amount").cast("decimal(18,2)")).alias("total")
            ),
            inputs=orders,
            expect=Schema.from_mapping({"status": string(), "total": decimal(28, 2)}),
            reason=(
                "summing a decimal(18,2) gives decimal(28,2): Spark adds ten digits of headroom "
                "to a decimal sum, where Polars keeps the input precision. A contract written "
                "against one engine fails on the other for arithmetic nobody changed"
            ),
        ),
        ConformanceCase(
            name="group_by_avg_and_count",
            pipeline=lambda orders: orders.groupBy("status").agg(
                _functions().avg("amount").alias("mean_amount"),
                _functions().count("*").alias("orders"),
            ),
            inputs=orders,
            expect=Schema.from_mapping(
                {"status": string(), "mean_amount": float_(64), "orders": int_(64)}
            ),
            reason="avg of an integer is double, and count is bigint whatever it counts",
        ),
        ConformanceCase(
            name="join_separate_keys",
            pipeline=lambda orders, customers: orders.join(
                customers, orders.order_id == customers.customer_id, "inner"
            ),
            inputs=joined,
            raises=DuplicateColumn,
            reason=(
                "a join on a condition keeps both sides whole, so the colliding created_at comes "
                "back twice under the same name. Spark is content with that and a name-addressed "
                "schema cannot be: Polars coalesces the key and suffixes the collision _right, "
                "pandas suffixes _x and _y, DuckDB does what Spark does"
            ),
        ),
        ConformanceCase(
            name="join_shared_key",
            pipeline=lambda orders, customers: orders.join(customers, on="created_at", how="left"),
            inputs=joined,
            expect=Schema.from_mapping(
                {
                    "created_at": TIMESTAMP,
                    "order_id": int_(64),
                    "status": string(),
                    "amount": int_(64),
                    "customer_id": int_(64),
                    "country": string(),
                }
            ),
            reason=(
                "joining on a shared name is a USING join: the key is coalesced, appears once, "
                "and appears *first*, ahead of the left-hand columns. Polars and DuckDB keep it "
                "in its original position, and ibis does not coalesce an outer join at all"
            ),
        ),
        ConformanceCase(
            name="union_is_positional",
            pipeline=lambda orders: orders.select("order_id", "amount").union(
                orders.select("amount", "order_id")
            ),
            inputs=orders,
            expect=Schema.from_mapping({"order_id": int_(64), "amount": int_(64)}),
            reason=(
                "the most expensive disagreement in this adapter. union pairs columns by "
                "position, so two frames whose columns are in different orders are unioned into "
                "the left-hand names with the right-hand values swapped - and because both are "
                "bigint here, nothing errors. pandas concat and Polars concat align on names"
            ),
        ),
        ConformanceCase(
            name="union_by_name",
            pipeline=lambda orders: orders.select("order_id", "amount").unionByName(
                orders.select("amount", "order_id")
            ),
            inputs=orders,
            expect=Schema.from_mapping({"order_id": int_(64), "amount": int_(64)}),
            reason=(
                "unionByName is the one that aligns on names, which is what the other engines do"
            ),
        ),
        ConformanceCase(
            name="nested_struct_field",
            pipeline=lambda events: events.select("payload.source"),
            inputs=events,
            expect=Schema.from_mapping({"source": string()}),
            reason="pulling a struct field out names the column after the field",
        ),
        ConformanceCase(
            name="nested_round_trip",
            pipeline=lambda events: events.select("payload", "tags", "attrs"),
            inputs=events,
            expect=Schema.from_mapping(
                {
                    "payload": struct({"source": string(), "retries": int_(32)}),
                    "tags": list_(string()),
                    "attrs": map_(string(), string()),
                }
            ),
            reason=(
                "Spark has a real MapType, so a map round trips with nothing lost. Polars has to "
                "encode one as list[struct{key, value}]"
            ),
        ),
        ConformanceCase(
            name="list_length_is_signed",
            pipeline=lambda events: events.select(_functions().size(_functions().col("tags"))),
            inputs=events,
            expect=Schema.from_mapping({"size(tags)": int_(32)}),
            reason=(
                "a list length is int32 and the column is named size(tags). Polars reports uint32 "
                "under the original column name: a dtype change and a name change in one call"
            ),
        ),
        ConformanceCase(
            name="explode",
            pipeline=lambda events: events.select(
                _functions().explode(_functions().col("tags")).alias("tag")
            ),
            inputs=events,
            expect=Schema.from_mapping({"tag": string()}),
            reason="explode changes the row count and the element dtype is what the schema shows",
        ),
        ConformanceCase(
            name="statement_that_is_not_a_query_is_not_sent",
            pipeline=lambda orders: orders.sparkSession.sql(
                f"INSERT INTO {MISSING_TABLE} SELECT 1 AS a"
            ),
            inputs=orders,
            expect=Schema.empty(),
            reason=(
                "the case that matters most on this engine. A dry run holds a socket to a live "
                "cluster, so an INSERT in a pipeline is recorded and not sent, and the empty "
                "frame Spark returns for a command is returned in its place. The table does not "
                "exist anywhere, so a guard that stopped working would fail to analyze instead "
                "of resolving to no columns"
            ),
        ),
        ConformanceCase(
            name="missing_column",
            pipeline=lambda orders: orders.select("amount_cents"),
            inputs=orders,
            raises=ColumnNotFound,
            reason=(
                "the finding from the proposal. Spark reports UNRESOLVED_COLUMN.WITH_SUGGESTION "
                "with the column in objectName and the alternatives it computed in proposal, so "
                "the near miss is the engine's own and not something rowless guessed"
            ),
        ),
        ConformanceCase(
            name="duplicate_column",
            pipeline=lambda orders: orders.select(
                _functions().col("amount"), _functions().col("amount")
            ),
            inputs=orders,
            raises=DuplicateColumn,
            reason=(
                "Spark is the one engine here that *allows* two output columns of the same name. "
                "Polars refuses it, DuckDB produces it only through a join, and rowless has to "
                "report it either way: nothing downstream of a name-addressed schema could refer "
                "to either column"
            ),
        ),
        ConformanceCase(
            name="udf_without_return_type",
            pipeline=lambda orders: orders.withColumn(
                "doubled", _functions().udf(untyped)(_functions().col("amount"))
            ),
            inputs=orders,
            expect_confidence=Confidence.UNKNOWN,
            reason=(
                "Spark does not infer a UDF return type: with no returnType it types the column "
                "StringType, silently. That is a default dressed as an answer, so the adapter "
                "degrades the run and names the function. Passing returnType= makes it go away"
            ),
        ),
        ConformanceCase(
            name="row_dependent_branch",
            pipeline=lambda orders: orders.select("order_id") if orders.count() == 0 else orders,
            inputs=orders,
            expect_confidence=Confidence.UNKNOWN,
            reason=(
                "count() is 0 on a rowless frame, so the dry run took the branch production "
                "never would"
            ),
        ),
        ConformanceCase(
            name="row_dependent_collect",
            pipeline=lambda orders: orders.withColumn(
                "ceiling",
                _functions().lit(
                    max([row[0] for row in orders.select("amount").collect()], default=0)
                ),
            ),
            inputs=orders,
            expect_confidence=Confidence.UNKNOWN,
            reason=(
                "collect() pulls rows into the driver, so the literal column it builds would be "
                "a different number in production. This is where the adapter parts company with "
                "the Polars one: a Polars pipeline ends in collect() and a Spark pipeline does "
                "not, so the same word is treated differently"
            ),
        ),
    )


def _functions() -> Any:
    """`pyspark.sql.functions`, imported at call time rather than at import time.

    A classic session resolves `F.col` against the active SparkContext, so
    importing this module must not need one: the import happens inside the
    pipeline, where a session exists. Under Connect the same module dispatches
    to the Connect implementations, so one spelling covers both modes - which
    is also how a real pipeline is written.
    """
    from pyspark.sql import functions

    return functions
