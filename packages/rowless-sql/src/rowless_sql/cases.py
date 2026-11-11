"""The operations battery for SQL: real queries, and the schemas they produce.

The conformance suite in core writes the universal half of the contract once;
this is the half that cannot be written there, because `df.select("a")` is not
the same call in Polars, pandas and SQL and inventing one abstract vocabulary
for all of them is the proxy approach the project rejects.

A pipeline on this engine is written one of two ways, and both are exercised
here. A SQL string is the common one - that is what a dbt model is, and
`str()` of a rowless frame renders the table name, so an f-string refers to the
input it was given. A `sqlglot` expression built with the library's own builder
API is the other, and it resolves through exactly the same path.

Every case runs through the same runtime the CLI uses, so what is asserted is
the schema a user would get. The `reason` on a case is the valuable part: it is
where an engine disagreement gets written down, and the suite prints it when
the case fails.

Two kinds of case here are worth reading on their own. The ones that assert a
**refusal** - an unaliased aggregate, a table nothing registered, a write
statement - are findings this adapter produces that its engine does not, and
each exists because reporting sqlglot's own answer would be reporting something
no warehouse would agree with. And `no_type_checking` asserts the **gap**: a
query DuckDB refuses outright resolves here without complaint, which is the
trade a user makes by checking SQL with no database behind it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from rowless.errors import ColumnNotFound, DuplicateColumn
from rowless.ir.confidence import Confidence
from rowless.ir.dtypes import (
    TimeUnit,
    any_,
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

from rowless_sql.errors import (
    AmbiguousColumn,
    SqlNotParsable,
    StatementNotAQuery,
    TableNotRegistered,
    UnnamedProjection,
)

#: The timestamp dtype both fixture schemas carry, spelled once.
TIMESTAMP = datetime(TimeUnit.MICROSECOND, "UTC")

#: The schema `SELECT *` over the orders fixture resolves to.
ALL_ORDERS = Schema.from_mapping(
    {
        "order_id": int_(64),
        "status": string(),
        "amount": int_(64),
        "created_at": TIMESTAMP,
    }
)


def paid_by_day(orders: Any) -> str:
    """The example from the proposal: daily revenue for paid orders.

    A named function rather than a lambda because it is the case the whole
    project is built around, and because it is worth seeing what a pipeline on
    this engine looks like: a SQL string that names its input by interpolating
    the frame rowless handed it.
    """
    return f"""
        SELECT CAST(created_at AS DATE) AS day, sum(amount) AS total
        FROM {orders}
        WHERE status = 'paid'
        GROUP BY 1
    """


def through_a_cte(orders: Any) -> str:
    """The same aggregate, behind a CTE.

    Here because a CTE is where the DuckDB adapter stops being able to trace
    columns and this one does not: sqlglot resolves the projection through the
    scope, so `total` is still attributed to `amount`.
    """
    return f"""
        WITH paid AS (SELECT * FROM {orders} WHERE status = 'paid')
        SELECT status, sum(amount) AS total
        FROM paid
        GROUP BY status
    """


def built_with_the_builder(orders: Any) -> Any:
    """A pipeline that builds its query with sqlglot rather than with a string."""
    from sqlglot import exp

    return exp.select("status", "amount").from_(orders).where("status = 'paid'")


def dbt_shaped_reference(orders: Any) -> str:
    """A query that names its input the way dbt compiles `ref()`.

    dbt turns `ref('stg_orders')` into a fully qualified, quoted relation. The
    catalogue is keyed on the final identifier, so this resolves against the
    same entry a bare name would - which is what makes a compiled model
    resolvable without rewriting its SQL.
    """
    return f'SELECT amount FROM "analytics"."staging"."{orders}"'


def union_widens(orders: Any) -> str:
    """A set operation whose sides have different integer widths."""
    return f"""
        SELECT CAST(order_id AS INT) AS id FROM {orders}
        UNION ALL
        SELECT order_id AS id FROM {orders}
    """


def union_of_incompatible(orders: Any) -> str:
    """A set operation with no common dtype, which SQL resolves anyway."""
    return f"""
        SELECT status AS value FROM {orders}
        UNION ALL
        SELECT amount AS value FROM {orders}
    """


def conformance_cases() -> Sequence[ConformanceCase]:
    """Every operation this adapter claims to resolve, and what it resolves to."""
    orders = {"orders": fixtures.ORDERS}
    joined = {"orders": fixtures.ORDERS, "customers": fixtures.CUSTOMERS}
    events = {"events": fixtures.EVENTS}

    return (
        ConformanceCase(
            name="select",
            pipeline=lambda orders: f"SELECT order_id, amount FROM {orders}",
            inputs=orders,
            expect=Schema.from_mapping({"order_id": int_(64), "amount": int_(64)}),
        ),
        ConformanceCase(
            name="select_is_exact",
            pipeline=lambda orders: f"SELECT order_id FROM {orders}",
            inputs=orders,
            expect=Schema.from_mapping({"order_id": int_(64)}),
            expect_confidence=Confidence.EXACT,
            reason=(
                "a projection's types come from sqlglot's own annotator against the registered "
                "schema, with nothing approximated - so a query made only of columns it knows "
                "is exact, with no database anywhere"
            ),
        ),
        ConformanceCase(
            name="select_star",
            pipeline=lambda orders: f"SELECT * FROM {orders}",
            inputs=orders,
            expect=ALL_ORDERS,
            reason=(
                "qualification expands the star against the registered schema, so a star query "
                "resolves to real columns rather than to an unknown width"
            ),
        ),
        ConformanceCase(
            name="builder_expression",
            pipeline=built_with_the_builder,
            inputs=orders,
            expect=Schema.from_mapping({"status": string(), "amount": int_(64)}),
            reason=(
                "sqlglot's builder API is the other way to write a pipeline on this engine, and "
                "it resolves through the same path as a string: the frame is a real sqlglot "
                "table, so `.from_(orders)` is the ordinary spelling"
            ),
        ),
        ConformanceCase(
            name="filter",
            pipeline=lambda orders: f"SELECT * FROM {orders} WHERE status = 'paid'",
            inputs=orders,
            expect=ALL_ORDERS,
            reason="a predicate cannot change the schema, however many rows it removes",
        ),
        ConformanceCase(
            name="rename",
            pipeline=lambda orders: f"SELECT amount AS amount_cents FROM {orders}",
            inputs=orders,
            expect=Schema.from_mapping({"amount_cents": int_(64)}),
            reason="an alias is how SQL renames, and the new name is what rowless reports",
        ),
        ConformanceCase(
            name="cast",
            pipeline=lambda orders: f"SELECT CAST(amount AS DOUBLE) AS amount FROM {orders}",
            inputs=orders,
            expect=Schema.from_mapping({"amount": float_(64)}),
            reason="a cast is the one place SQL states a type outright, so it is always exact",
        ),
        ConformanceCase(
            name="integer_division_is_float",
            pipeline=lambda orders: f"SELECT amount / 2 AS unit_price FROM {orders}",
            inputs=orders,
            expect=Schema.from_mapping({"unit_price": float_(64)}),
            reason=(
                "SQL engines are often said to do integer division; sqlglot's annotator does "
                "not, and neither does DuckDB. Polars and Ibis agree, so all four produce a "
                "float here and the folklore is wrong"
            ),
        ),
        ConformanceCase(
            name="group_by_sum",
            pipeline=lambda orders: (
                f"SELECT status, sum(amount) AS total FROM {orders} GROUP BY status"
            ),
            inputs=orders,
            expect=Schema.from_mapping({"status": string(), "total": int_(64)}),
            reason=(
                "an aggregate keeps its operand's type: int64 in, int64 out, which agrees with "
                "Polars and Ibis. DuckDB widens the same aggregate to a 128-bit integer, so a "
                "contract written against DuckDB's answer fails here and vice versa"
            ),
        ),
        ConformanceCase(
            name="count_star_is_int64",
            pipeline=lambda orders: f"SELECT count(*) AS n FROM {orders}",
            inputs=orders,
            expect=Schema.from_mapping({"n": int_(64)}),
            reason="count is int64 in every engine rowless targets - the one easy agreement",
        ),
        ConformanceCase(
            name="average_is_float",
            pipeline=lambda orders: f"SELECT avg(amount) AS average FROM {orders}",
            inputs=orders,
            expect=Schema.from_mapping({"average": float_(64)}),
            reason="avg of an integer is a double, where sum of one is not",
        ),
        ConformanceCase(
            name="group_by_computed_key",
            pipeline=paid_by_day,
            inputs=orders,
            expect=Schema.from_mapping({"day": date(), "total": int_(64)}),
            reason=(
                "the example from the proposal, with the column that still exists. The "
                "positional GROUP BY resolves too: qualification rewrites `GROUP BY 1` against "
                "the projection it refers to"
            ),
        ),
        ConformanceCase(
            name="group_by_through_a_cte",
            pipeline=through_a_cte,
            inputs=orders,
            expect=Schema.from_mapping({"status": string(), "total": int_(64)}),
            reason=(
                "a CTE is a scope, and sqlglot resolves through it - which is also why this is "
                "the adapter whose lineage survives one. The DuckDB adapter refuses to "
                "attribute a projection behind a CTE rather than guess across it"
            ),
        ),
        ConformanceCase(
            name="join_explicit_projection",
            pipeline=lambda orders, customers: (
                f"SELECT o.order_id, c.country FROM {orders} o "
                f"JOIN {customers} c ON o.order_id = c.customer_id"
            ),
            inputs=joined,
            expect=Schema.from_mapping({"order_id": int_(64), "country": string()}),
            reason="a qualified projection over a join is the shape that resolves cleanly",
        ),
        ConformanceCase(
            name="join_star_duplicates_the_collision",
            pipeline=lambda orders, customers: (
                f"SELECT * FROM {orders} JOIN {customers} ON order_id = customer_id"
            ),
            inputs=joined,
            raises=DuplicateColumn,
            reason=(
                "SQL keeps both key columns and does not rename a collision, so `SELECT *` over "
                "two tables that both have created_at produces that name twice. DuckDB agrees; "
                "Polars coalesces the key and suffixes the collision _right; pandas suffixes _x "
                "and _y. rowless reports the duplicate, because a name-addressed schema cannot "
                "describe it"
            ),
        ),
        ConformanceCase(
            name="union",
            pipeline=lambda orders, customers: (
                f"SELECT order_id FROM {orders} UNION ALL SELECT customer_id FROM {customers}"
            ),
            inputs=joined,
            expect=Schema.from_mapping({"order_id": int_(64)}),
            reason=(
                "the names of a set operation come from the left-hand side, which is what every "
                "SQL engine does - and unlike Polars and Ibis, SQL does not require the two "
                "sides to have matching names at all"
            ),
        ),
        ConformanceCase(
            name="union_widens_through_the_ir",
            pipeline=union_widens,
            inputs=orders,
            expect=Schema.from_mapping({"id": int_(64)}),
            reason=(
                "sqlglot reports a union's types as the left-hand side's, so this query is int32 "
                "to it. rowless combines the two sides through its own dtype lattice instead: "
                "int32 unioned with int64 is int64, which is what the warehouse will store"
            ),
        ),
        ConformanceCase(
            name="union_of_incompatible_is_unresolved",
            pipeline=union_of_incompatible,
            inputs=orders,
            expect_confidence=Confidence.UNKNOWN,
            reason=(
                "a string unioned with an integer has no common dtype. SQL's own rules would "
                "coerce one side and sqlglot reports the left side's type, which is how a tool "
                "ends up claiming a column is a string when the warehouse will refuse the query "
                "or store something else. rowless reports it unresolved and says why"
            ),
        ),
        ConformanceCase(
            name="nested_struct_field",
            pipeline=lambda events: f"SELECT payload.source AS source FROM {events}",
            inputs=events,
            expect=Schema.from_mapping({"source": string()}),
            reason="sqlglot types a struct field access from the struct's own declaration",
        ),
        ConformanceCase(
            name="nested_round_trip",
            pipeline=lambda events: f"SELECT payload, tags, attrs FROM {events}",
            inputs=events,
            expect=Schema.from_mapping(
                {
                    "payload": struct({"source": string(), "retries": int_(32)}),
                    "tags": list_(string()),
                    "attrs": map_(string(), string()),
                }
            ),
            reason=(
                "SQL has a real MAP type, so a map survives as a map rather than as the Arrow "
                "list[struct{key, value}] encoding Polars has to use"
            ),
        ),
        ConformanceCase(
            name="dbt_shaped_reference",
            pipeline=dbt_shaped_reference,
            inputs=orders,
            expect=Schema.from_mapping({"amount": int_(64)}),
            reason=(
                "dbt compiles ref() into a three-part quoted relation. The catalogue is keyed on "
                "the final identifier, so a compiled model resolves without rewriting its SQL - "
                "which is the whole point of the dbt seam"
            ),
        ),
        ConformanceCase(
            name="unannotated_function_is_any",
            pipeline=lambda orders: (
                f"SELECT order_id, some_warehouse_udf(amount) AS score FROM {orders}"
            ),
            inputs=orders,
            expect=Schema.from_mapping({"order_id": int_(64), "score": any_()}),
            expect_confidence=Confidence.PARTIAL,
            reason=(
                "sqlglot has no return type for a function it does not know, and this is the "
                "headline limit of the adapter: the column comes back ANY with a note naming "
                "the function, the typed-coverage score drops, and nothing is claimed. A real "
                "binder - rowless-duckdb - types the same query completely"
            ),
        ),
        ConformanceCase(
            name="no_type_checking",
            pipeline=lambda orders: f"SELECT amount + status AS mixed FROM {orders}",
            inputs=orders,
            expect=Schema.from_mapping({"mixed": int_(64)}),
            reason=(
                "the trade this adapter makes, asserted rather than hidden. Adding an integer "
                "to a string is refused outright by DuckDB and by Ibis; sqlglot has no binder, "
                "annotates the expression as BIGINT and reports nothing. This adapter finds "
                "renames, missing columns, ambiguity and duplicates - not type errors"
            ),
        ),
        ConformanceCase(
            name="missing_column",
            pipeline=lambda orders: f"SELECT amount_cents FROM {orders}",
            inputs=orders,
            raises=ColumnNotFound,
            reason=(
                "the finding from the proposal. sqlglot names the column it could not resolve "
                "and does not list the alternatives, so the near-miss suggestion comes from the "
                "registered schemas instead"
            ),
        ),
        ConformanceCase(
            name="ambiguous_column",
            pipeline=lambda orders, customers: (
                f"SELECT created_at FROM {orders} JOIN {customers} ON order_id = customer_id"
            ),
            inputs=joined,
            raises=AmbiguousColumn,
            reason=(
                "both inputs have created_at, and sqlglot reports that with the same message it "
                "uses for a column that exists nowhere. The catalogue is asked which tables "
                "have the name, so the finding says 'qualify this' rather than sending the "
                "reader to look for a rename"
            ),
        ),
        ConformanceCase(
            name="unaliased_aggregate_is_refused",
            pipeline=lambda orders: f"SELECT status, sum(amount) FROM {orders} GROUP BY status",
            inputs=orders,
            raises=UnnamedProjection,
            reason=(
                "sqlglot calls an unaliased projection _col_0, which is its own placeholder: "
                "DuckDB would call that column sum(amount), Spark _col0 and Postgres sum. "
                "Reporting a name no engine uses would put a column in a contract that cannot "
                "hold anywhere, so the query is unverified until the expression is aliased"
            ),
        ),
        ConformanceCase(
            name="unregistered_table_is_unverified",
            pipeline=lambda orders: f"SELECT amount FROM not_a_registered_table, {orders}",
            inputs=orders,
            raises=TableNotRegistered,
            reason=(
                "a query that reads a table rowless has no schema for is reported, not guessed "
                "at: the table very likely exists in the warehouse, and rowless resolves "
                "against the schemas it was given. sqlglot would resolve a *qualified* column "
                "against an unknown table without complaint, which is why the catalogue is "
                "checked before qualification"
            ),
        ),
        ConformanceCase(
            name="write_statement_is_refused",
            pipeline=lambda orders: f"INSERT INTO totals SELECT * FROM {orders}",
            inputs=orders,
            raises=StatementNotAQuery,
            reason=(
                "a dry run resolves what a query returns. A statement whose job is to write has "
                "no output schema, and resolving the select buried inside it would report a "
                "schema for something rowless did not analyse"
            ),
        ),
        ConformanceCase(
            name="unreadable_sql_is_unverified",
            pipeline=lambda orders: f"SELECT FROM WHERE {orders}",
            inputs=orders,
            raises=SqlNotParsable,
            reason=(
                "a parse failure is not proof that a model is broken: valid Snowflake or T-SQL "
                "fails exactly this way when the dialect was not named, and the message says "
                "which setting fixes it. Reporting valid SQL as wrong is how a checker gets "
                "switched off"
            ),
        ),
    )
