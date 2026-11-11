"""The operations battery for Polars: real pipelines, and the schemas they produce.

The conformance suite in core writes the universal half of the contract once;
this is the half that cannot be written there, because `df.select("a")` is not
the same call in Polars, pandas and DuckDB and inventing one abstract
vocabulary for all of them is the proxy approach the project rejects.

Every case runs through the same runtime the CLI uses, so what is asserted is
the schema a user would get. The `reason` on a case is the valuable part: it is
where an engine disagreement gets written down, and the suite prints it when
the case fails.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from rowless.errors import ColumnNotFound, DuplicateColumn
from rowless.ir.confidence import Confidence
from rowless.ir.dtypes import (
    TimeUnit,
    date,
    datetime,
    float_,
    int_,
    list_,
    map_,
    string,
    struct,
    uint,
)
from rowless.ir.schema import Schema
from rowless.testing import fixtures
from rowless.testing.conformance import ConformanceCase

#: The timestamp dtype both fixture schemas carry, spelled once.
TIMESTAMP = datetime(TimeUnit.MICROSECOND, "UTC")


def annotated_cents(amount: int) -> float:
    """A UDF with a return annotation, which is what the adapter prefers.

    Deliberately a named function and not a lambda: the point of the case is
    that the annotation is read, and a lambda cannot carry one.
    """
    return amount / 100


def unknowable(amount: int) -> Any:
    """A UDF nothing can type: no annotation worth having, and it raises when probed."""
    raise RuntimeError("this UDF cannot be called with an invented value")


def conformance_cases() -> Sequence[ConformanceCase]:
    """Every operation this adapter claims to resolve, and what it resolves to."""
    import polars as pl

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
            reason="a lazy plan resolution is the engine own answer, so nothing is approximated",
        ),
        ConformanceCase(
            name="alias",
            pipeline=lambda orders: orders.select(pl.col("amount").alias("amount_cents")),
            inputs=orders,
            expect=Schema.from_mapping({"amount_cents": int_(64)}),
        ),
        ConformanceCase(
            name="rename",
            pipeline=lambda orders: orders.rename({"amount": "amount_cents"}),
            inputs=orders,
            expect=Schema.from_mapping(
                {
                    "order_id": int_(64),
                    "status": string(),
                    "amount_cents": int_(64),
                    "created_at": TIMESTAMP,
                }
            ),
            reason="rename keeps the column position, which a select-based rename would not",
        ),
        ConformanceCase(
            name="filter",
            pipeline=lambda orders: orders.filter(pl.col("status") == "paid"),
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
            pipeline=lambda orders: orders.with_columns(pl.col("amount").cast(pl.Float64)),
            inputs=orders,
            expect=Schema.from_mapping(
                {
                    "order_id": int_(64),
                    "status": string(),
                    "amount": float_(64),
                    "created_at": TIMESTAMP,
                }
            ),
            reason="with_columns replaces the column in place rather than appending it",
        ),
        ConformanceCase(
            name="integer_division_is_float",
            pipeline=lambda orders: orders.select((pl.col("amount") / 2).alias("unit_price")),
            inputs=orders,
            expect=Schema.from_mapping({"unit_price": float_(64)}),
            reason="Polars divides like Python and returns float64, which is the DuckDB answer too",
        ),
        ConformanceCase(
            name="group_by_sum",
            pipeline=lambda orders: orders.group_by("status").agg(pl.col("amount").sum()),
            inputs=orders,
            expect=Schema.from_mapping({"status": string(), "amount": int_(64)}),
            reason=(
                "Polars keeps the aggregated column name and its int64 dtype; DuckDB renames it "
                "sum(amount) and widens to a 128-bit integer"
            ),
        ),
        ConformanceCase(
            name="group_by_computed_key",
            pipeline=lambda orders: (
                orders.filter(pl.col("status") == "paid")
                .group_by(pl.col("created_at").dt.date().alias("day"))
                .agg(pl.col("amount").sum())
            ),
            inputs=orders,
            expect=Schema.from_mapping({"day": date(), "amount": int_(64)}),
            reason="the example from the proposal, with the column that still exists",
        ),
        ConformanceCase(
            name="join_separate_keys",
            pipeline=lambda orders, customers: orders.join(
                customers, left_on="order_id", right_on="customer_id", how="inner"
            ),
            inputs=joined,
            expect=Schema.from_mapping(
                {
                    "order_id": int_(64),
                    "status": string(),
                    "amount": int_(64),
                    "created_at": TIMESTAMP,
                    "country": string(),
                    "created_at_right": TIMESTAMP,
                }
            ),
            reason=(
                "Polars coalesces the right key away and suffixes the colliding created_at with "
                "_right; SQL keeps both keys, pandas suffixes _x and _y"
            ),
        ),
        ConformanceCase(
            name="join_shared_key",
            pipeline=lambda orders, customers: orders.join(customers, on="created_at", how="left"),
            inputs=joined,
            expect=Schema.from_mapping(
                {
                    "order_id": int_(64),
                    "status": string(),
                    "amount": int_(64),
                    "created_at": TIMESTAMP,
                    "customer_id": int_(64),
                    "country": string(),
                }
            ),
            reason="joining on a shared name produces one column, not two",
        ),
        ConformanceCase(
            name="union",
            pipeline=lambda orders: pl.concat([orders, orders.filter(pl.col("status") == "paid")]),
            inputs=orders,
            expect=Schema.from_mapping(
                {
                    "order_id": int_(64),
                    "status": string(),
                    "amount": int_(64),
                    "created_at": TIMESTAMP,
                }
            ),
            reason="pl.concat is the lazy union, and it requires the schemas to match exactly",
        ),
        ConformanceCase(
            name="nested_struct_field",
            pipeline=lambda events: events.select(pl.col("payload").struct.field("source")),
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
                "Polars has no Map type, so a map travels as the Arrow encoding "
                "list[struct{key, value}] and comes back as a map"
            ),
        ),
        ConformanceCase(
            name="list_length_is_unsigned",
            pipeline=lambda events: events.select(pl.col("tags").list.len()),
            inputs=events,
            expect=Schema.from_mapping({"tags": uint(32)}),
            reason="a list length is uint32 and keeps the column name, which surprises everybody",
        ),
        ConformanceCase(
            name="udf_with_annotation",
            pipeline=lambda orders: orders.select(pl.col("amount").map_elements(annotated_cents)),
            inputs=orders,
            expect=Schema.from_mapping({"amount": float_(64)}),
            reason=(
                "the adapter reads the return annotation and passes it to Polars as return_dtype, "
                "so the function is never called with an invented value"
            ),
        ),
        ConformanceCase(
            name="udf_with_no_inferable_type",
            pipeline=lambda orders: orders.select(pl.col("amount").map_elements(unknowable)),
            inputs=orders,
            expect_confidence=Confidence.UNKNOWN,
            reason=(
                "nothing can type this UDF: no usable annotation, and the Polars inference call "
                "raises. What the engine does about it is version-dependent - 1.x types the "
                "column Unknown, 2.x raises - so the adapter asserts the one thing that must "
                "hold either way: the result is not verified, and it says why"
            ),
        ),
        ConformanceCase(
            name="missing_column",
            pipeline=lambda orders: orders.select("amount_cents"),
            inputs=orders,
            raises=ColumnNotFound,
            reason=(
                "the finding from the proposal. Polars names the column and lists the valid ones, "
                "which is what makes the near-miss suggestion possible"
            ),
        ),
        ConformanceCase(
            name="duplicate_column",
            pipeline=lambda orders: orders.select(pl.col("amount"), pl.col("amount")),
            inputs=orders,
            raises=DuplicateColumn,
            reason="Polars refuses a duplicate output name rather than keeping the last one",
        ),
        ConformanceCase(
            name="row_dependent_branch",
            pipeline=lambda orders: (
                orders.select("order_id") if orders.collect().is_empty() else orders
            ),
            inputs=orders,
            expect_confidence=Confidence.UNKNOWN,
            reason=(
                "collect() executes and is_empty() is True on a rowless frame, so the dry run "
                "took the branch production never would"
            ),
        ),
        ConformanceCase(
            name="row_dependent_scalar",
            pipeline=lambda orders: orders.with_columns(
                pl.lit(orders.select("amount").collect()["amount"].max()).alias("ceiling")
            ),
            inputs=orders,
            expect_confidence=Confidence.UNKNOWN,
            reason=(
                "Series.max() is None on zero rows, so the literal column it builds would be a "
                "number in production. Expression-level max() is not row-dependent and is not "
                "intercepted"
            ),
        ),
    )
