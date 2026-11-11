"""The operations battery for Ibis: real pipelines, and the schemas they produce.

The conformance suite in core writes the universal half of the contract once;
this is the half that cannot be written there, because `df.select("a")` is not
the same call in Polars, pandas and ibis and inventing one abstract vocabulary
for all of them is the proxy approach the project rejects.

Every case runs through the same runtime the CLI uses, so what is asserted is
the schema a user would get. The `reason` on a case is the valuable part: it is
where an engine disagreement gets written down, and the suite prints it when
the case fails.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from rowless.errors import ColumnNotFound, DTypeMismatch, DuplicateColumn
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
)
from rowless.ir.schema import Schema
from rowless.testing import fixtures
from rowless.testing.conformance import ConformanceCase

#: The timestamp dtype both fixture schemas carry, spelled once.
TIMESTAMP = datetime(TimeUnit.MICROSECOND, "UTC")

#: The SQL a pipeline hands to `Table.sql`, which the adapter resolves through
#: ibis's own parser rather than through a backend.
INLINE_SQL = "SELECT status, sum(amount) AS total FROM orders GROUP BY status"


def paid_by_day(orders: Any) -> Any:
    """The example from the proposal: daily revenue for paid orders.

    A named function rather than a lambda because it takes two steps, and the
    second one has to refer to the first: ibis dereferences a column of the
    original table onto the filtered one, but spelling it out is what a user
    would write.
    """
    paid = orders.filter(orders.status == "paid")
    return paid.group_by(day=paid.created_at.date()).agg(total=paid.amount.sum())


def positional_aggregate(orders: Any) -> Any:
    """Aggregate without naming the output, which is where the engines diverge."""
    return orders.group_by("status").agg(orders.amount.sum())


def inline_sql(orders: Any) -> Any:
    """A pipeline that drops into raw SQL part way through.

    `Table.alias` is how ibis names the table a SQL string refers to, and it is
    what makes this work with no backend: the adapter resolves the query
    against the alias rather than asking a warehouse what it returns.
    """
    return orders.alias("orders").sql(INLINE_SQL)


def row_dependent_branch(orders: Any) -> Any:
    """Branch on how many rows there are, which a dry run cannot answer."""
    if orders.count().execute() == 0:
        return orders.select("order_id")
    return orders


def row_dependent_literal(orders: Any) -> Any:
    """Build a literal column out of values read from the frame."""
    import ibis

    ceiling = max(orders.amount.to_list(), default=0)
    return orders.mutate(ceiling=ibis.literal(ceiling))


def conformance_cases() -> Sequence[ConformanceCase]:
    """Every operation this adapter claims to resolve, and what it resolves to."""
    import ibis

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
                "an unbound expression's schema is the engine's own answer, resolved by ibis's "
                "type system with no backend and nothing approximated"
            ),
        ),
        ConformanceCase(
            name="column_access",
            pipeline=lambda orders: orders.select(orders.amount),
            inputs=orders,
            expect=Schema.from_mapping({"amount": int_(64)}),
            reason="attribute access is how an ibis pipeline reads a column",
        ),
        ConformanceCase(
            name="alias",
            pipeline=lambda orders: orders.select(amount_cents=orders.amount),
            inputs=orders,
            expect=Schema.from_mapping({"amount_cents": int_(64)}),
            reason="ibis names an output column with a keyword, where Polars uses .alias()",
        ),
        ConformanceCase(
            name="rename",
            pipeline=lambda orders: orders.rename(amount_cents="amount"),
            inputs=orders,
            expect=Schema.from_mapping(
                {
                    "order_id": int_(64),
                    "status": string(),
                    "amount_cents": int_(64),
                    "created_at": TIMESTAMP,
                }
            ),
            reason=(
                "rename keeps the column position, and takes new=old - the opposite order to "
                "the mapping Polars takes"
            ),
        ),
        ConformanceCase(
            name="filter",
            pipeline=lambda orders: orders.filter(orders.status == "paid"),
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
            pipeline=lambda orders: orders.mutate(amount=orders.amount.cast("float64")),
            inputs=orders,
            expect=Schema.from_mapping(
                {
                    "order_id": int_(64),
                    "status": string(),
                    "amount": float_(64),
                    "created_at": TIMESTAMP,
                }
            ),
            reason="mutate replaces a column of the same name in place rather than appending it",
        ),
        ConformanceCase(
            name="integer_division_is_float",
            pipeline=lambda orders: orders.select(unit_price=orders.amount / 2),
            inputs=orders,
            expect=Schema.from_mapping({"unit_price": float_(64)}),
            reason="ibis divides like Python and returns float64, which is the DuckDB answer too",
        ),
        ConformanceCase(
            name="group_by_sum",
            pipeline=lambda orders: orders.group_by("status").agg(total=orders.amount.sum()),
            inputs=orders,
            expect=Schema.from_mapping({"status": string(), "total": int_(64)}),
            reason=(
                "ibis keeps the int64 of the summed column, as Polars does; DuckDB widens the "
                "same aggregate to a 128-bit integer"
            ),
        ),
        ConformanceCase(
            name="group_by_unnamed_aggregate",
            pipeline=positional_aggregate,
            inputs=orders,
            expect=Schema.from_mapping({"status": string(), "Sum(amount)": int_(64)}),
            reason=(
                "an unnamed aggregate is named after the operation: ibis says Sum(amount), "
                "DuckDB says sum(amount) and Polars keeps the input name amount. Three engines, "
                "three names, one call - which is why ibis takes a keyword"
            ),
        ),
        ConformanceCase(
            name="group_by_computed_key",
            pipeline=paid_by_day,
            inputs=orders,
            expect=Schema.from_mapping({"day": date(), "total": int_(64)}),
            reason="the example from the proposal, with the column that still exists",
        ),
        ConformanceCase(
            name="join_separate_keys",
            pipeline=lambda orders, customers: orders.join(
                customers, orders.order_id == customers.customer_id
            ),
            inputs=joined,
            expect=Schema.from_mapping(
                {
                    "order_id": int_(64),
                    "status": string(),
                    "amount": int_(64),
                    "created_at": TIMESTAMP,
                    "customer_id": int_(64),
                    "country": string(),
                    "created_at_right": TIMESTAMP,
                }
            ),
            reason=(
                "ibis keeps both key columns and suffixes the colliding created_at with _right; "
                "Polars coalesces the right key away, and DuckDB keeps both columns under the "
                "same name"
            ),
        ),
        ConformanceCase(
            name="join_shared_key",
            pipeline=lambda orders, customers: orders.join(customers, "created_at"),
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
            reason=(
                "an inner join on a shared name coalesces the key, which is the one join shape "
                "where ibis, Polars and DuckDB all agree"
            ),
        ),
        ConformanceCase(
            name="outer_join_shared_key_keeps_both",
            pipeline=lambda orders, customers: orders.join(customers, "created_at", how="left"),
            inputs=joined,
            expect=Schema.from_mapping(
                {
                    "order_id": int_(64),
                    "status": string(),
                    "amount": int_(64),
                    "created_at": TIMESTAMP,
                    "customer_id": int_(64),
                    "country": string(),
                    "created_at_right": TIMESTAMP,
                }
            ),
            reason=(
                "the same join as above, outer rather than inner, produces an extra column: "
                "ibis coalesces a shared key only when the join is inner, because an outer join "
                "can leave the right key null where the left one is not, and the two columns "
                "therefore hold different values. Polars coalesces a left join anyway, so a "
                "pipeline moved between the two engines gains or loses a column here"
            ),
        ),
        ConformanceCase(
            name="union",
            pipeline=lambda orders: orders.union(orders.filter(orders.status == "paid")),
            inputs=orders,
            expect=Schema.from_mapping(
                {
                    "order_id": int_(64),
                    "status": string(),
                    "amount": int_(64),
                    "created_at": TIMESTAMP,
                }
            ),
            reason="ibis requires the schemas to be equal for a set operation, as Polars does",
        ),
        ConformanceCase(
            name="nested_struct_field",
            pipeline=lambda events: events.select(events.payload.source),
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
                "ibis has a real Map type, so a map survives the round trip as a map rather "
                "than as the Arrow list[struct{key, value}] encoding Polars has to use"
            ),
        ),
        ConformanceCase(
            name="list_length_is_signed",
            pipeline=lambda events: events.select(events.tags.length()),
            inputs=events,
            expect=Schema.from_mapping({"ArrayLength(tags)": int_(64)}),
            reason=(
                "a list length is int64 under the operation's name; Polars reports uint32 under "
                "the original column name, so this one call differs in both dtype and name"
            ),
        ),
        ConformanceCase(
            name="inline_sql",
            pipeline=inline_sql,
            inputs=orders,
            expect=Schema.from_mapping({"status": string(), "total": int_(64)}),
            reason=(
                "Table.sql() normally asks the backend what a query returns. There is no "
                "backend here, so the adapter resolves it through ibis's own SQL parser against "
                "the schema it already has - which is the same seam run_sql uses"
            ),
        ),
        ConformanceCase(
            name="missing_column",
            pipeline=lambda orders: orders.select("amount_cents"),
            inputs=orders,
            raises=ColumnNotFound,
            reason=(
                "the finding from the proposal. Ibis names the column and lists the existing "
                "ones, which is what makes the near-miss suggestion possible"
            ),
        ),
        ConformanceCase(
            name="missing_column_through_attribute",
            pipeline=lambda orders: orders.select(cents=orders.amount_cents),
            inputs=orders,
            raises=ColumnNotFound,
            reason=(
                "attribute access fails with AttributeError in ibis, with no column list "
                "attached; the adapter reports it as the flagship finding while keeping it an "
                "AttributeError, so nothing that probes an object with getattr breaks"
            ),
        ),
        ConformanceCase(
            name="duplicate_column",
            pipeline=lambda orders: orders.select(orders.amount, orders.amount),
            inputs=orders,
            raises=DuplicateColumn,
            reason="ibis refuses a duplicate output name rather than keeping the last one",
        ),
        ConformanceCase(
            name="dtype_mismatch_in_arithmetic",
            pipeline=lambda orders: orders.select(mixed=orders.amount + orders.status),
            inputs=orders,
            raises=DTypeMismatch,
            reason=(
                "ibis refuses to add an integer to a string while the *argument* to select is "
                "being built, before select is called, so the translation has to sit on the "
                "operator rather than on the table method"
            ),
        ),
        ConformanceCase(
            name="dtype_mismatch_in_predicate",
            pipeline=lambda orders: orders.filter(orders.amount),
            inputs=orders,
            raises=DTypeMismatch,
            reason="a predicate has to be boolean; ibis refuses an integer rather than coercing",
        ),
        ConformanceCase(
            name="row_dependent_branch",
            pipeline=row_dependent_branch,
            inputs=orders,
            expect_confidence=Confidence.UNKNOWN,
            reason=(
                "execute() reads data, and a rowless frame has none. Ibis cannot execute an "
                "unbound expression at all, so rather than let it fail as a broken pipeline the "
                "adapter reports the branch as row-dependent and the result as unverified"
            ),
        ),
        ConformanceCase(
            name="row_dependent_literal",
            pipeline=row_dependent_literal,
            inputs=orders,
            expect_confidence=Confidence.UNKNOWN,
            reason=(
                "to_list() on a rowless column is empty here and populated in production, so "
                "the literal column built from it is not the one production would build. The "
                "expression-level aggregate amount.max() is *not* row-dependent in ibis - it "
                "builds a query - which is why far fewer ibis pipelines come out unverified"
            ),
        ),
        ConformanceCase(
            name="literal_column_is_untraced",
            pipeline=lambda orders: orders.select("order_id").mutate(source=ibis.literal("web")),
            inputs=orders,
            expect=Schema.from_mapping({"order_id": int_(64), "source": string()}),
            reason=(
                "a literal column has no input column behind it, so it is reported untraced "
                "rather than attributed to whatever the pipeline happened to read"
            ),
        ),
    )
