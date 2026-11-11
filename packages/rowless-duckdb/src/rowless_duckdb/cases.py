"""The operations battery for DuckDB: real pipelines, and the schemas they produce.

The conformance suite in core writes the universal half of the contract once;
this is the half that cannot be written there, because `df.select("a")` is not
the same call in Polars, pandas and DuckDB and inventing one abstract
vocabulary for all of them is the proxy approach the project rejects.

Every case runs through the same runtime the CLI uses, so what is asserted is
the schema a user would get. The `reason` on a case is the valuable part: it is
where an engine disagreement gets written down, and the suite prints it when
the case fails.

Two deliberate constraints on how the pipelines are written:

* **nothing newer than DuckDB 1.0.** `with_columns` and `rename` arrived later,
  so a cast here restates the projection the way a SQL user would, and the
  battery runs unchanged at the declared floor. The adapter still intercepts
  those methods where the installed release has them.
* **aliases on the relations being joined**, rather than the table names
  `materialize` chose. A join condition that spelled out `stg_orders.order_id`
  would be asserting the adapter's private naming rather than DuckDB's join
  semantics.
"""

from __future__ import annotations

from collections.abc import Sequence

from rowless.errors import ColumnNotFound, DuplicateColumn
from rowless.ir.confidence import Confidence
from rowless.ir.dtypes import (
    TimeUnit,
    boolean,
    date,
    datetime,
    decimal,
    enum,
    float_,
    int_,
    list_,
    map_,
    null,
    string,
    struct,
)
from rowless.ir.schema import Schema
from rowless.testing import fixtures
from rowless.testing.conformance import ConformanceCase

#: The timestamp dtype both fixture schemas carry, spelled once. DuckDB stores
#: it as `TIMESTAMP WITH TIME ZONE`, which is a UTC instant.
TIMESTAMP = datetime(TimeUnit.MICROSECOND, "UTC")

#: What `sum` of a 64-bit integer comes back as. DuckDB widens to 128 bits so
#: the sum cannot overflow; Polars keeps `int64` and lets it.
SUMMED = int_(128)


def conformance_cases() -> Sequence[ConformanceCase]:
    """Every operation this adapter claims to resolve, and what it resolves to."""
    orders = {"orders": fixtures.ORDERS}
    joined = {"orders": fixtures.ORDERS, "customers": fixtures.CUSTOMERS}
    events = {"events": fixtures.EVENTS}
    drift = {"drift": fixtures.DRIFT_PRONE}

    return (
        ConformanceCase(
            name="project",
            pipeline=lambda orders: orders.project("order_id, amount"),
            inputs=orders,
            expect=Schema.from_mapping({"order_id": int_(64), "amount": int_(64)}),
        ),
        ConformanceCase(
            name="project_is_exact",
            pipeline=lambda orders: orders.project("order_id"),
            inputs=orders,
            expect=Schema.from_mapping({"order_id": int_(64)}),
            expect_confidence=Confidence.EXACT,
            reason="the binder resolved this itself against a real table, so nothing is guessed",
        ),
        ConformanceCase(
            name="alias",
            pipeline=lambda orders: orders.project("amount AS amount_cents"),
            inputs=orders,
            expect=Schema.from_mapping({"amount_cents": int_(64)}),
        ),
        ConformanceCase(
            name="filter",
            pipeline=lambda orders: orders.filter("status = 'paid'"),
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
            pipeline=lambda orders: orders.project(
                "order_id, status, CAST(amount AS DOUBLE) AS amount, created_at"
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
            reason=(
                "DuckDB has no with_columns at the declared floor, so a cast restates the "
                "projection; the column keeps its position because the select list says so"
            ),
        ),
        ConformanceCase(
            name="division_is_double",
            pipeline=lambda orders: orders.project("amount / 2 AS unit_price"),
            inputs=orders,
            expect=Schema.from_mapping({"unit_price": float_(64)}),
            reason=(
                "SQL engines are said to do integer division, and DuckDB does not: / is true "
                "division and returns DOUBLE, which is the Polars answer too"
            ),
        ),
        ConformanceCase(
            name="integer_division_stays_integer",
            pipeline=lambda orders: orders.project("amount // 2 AS unit_price"),
            inputs=orders,
            expect=Schema.from_mapping({"unit_price": int_(64)}),
            reason=(
                "// is the integer division DuckDB keeps BIGINT for; Polars has no such operator"
            ),
        ),
        ConformanceCase(
            name="aggregate_renames_and_widens",
            pipeline=lambda orders: orders.aggregate("status, sum(amount)"),
            inputs=orders,
            expect=Schema.from_mapping({"status": string(), "sum(amount)": SUMMED}),
            reason=(
                "two differences from Polars in one call: DuckDB names the column after the "
                "expression that produced it, and widens the sum to HUGEINT so it cannot "
                "overflow. Polars keeps the name amount and the int64"
            ),
        ),
        ConformanceCase(
            name="aggregate_with_alias",
            pipeline=lambda orders: orders.aggregate("status, sum(amount) AS total"),
            inputs=orders,
            expect=Schema.from_mapping({"status": string(), "total": SUMMED}),
            reason="an alias is how a SQL user avoids the sum(amount) name; the width still widens",
        ),
        ConformanceCase(
            name="count_is_bigint",
            pipeline=lambda orders: orders.aggregate("status, count(*) AS orders"),
            inputs=orders,
            expect=Schema.from_mapping({"status": string(), "orders": int_(64)}),
            reason=(
                "count does not widen the way sum does, so an aggregate output's width depends "
                "on which aggregate it was"
            ),
        ),
        ConformanceCase(
            name="group_by_computed_key",
            pipeline=lambda orders: orders.filter("status = 'paid'").aggregate(
                "CAST(created_at AS DATE) AS day, sum(amount) AS amount"
            ),
            inputs=orders,
            expect=Schema.from_mapping({"day": date(), "amount": SUMMED}),
            reason=(
                "the example from the proposal, with the column that still exists. The "
                "relational aggregate groups by every non-aggregated expression, so the "
                "computed key needs no second argument"
            ),
        ),
        ConformanceCase(
            name="join_shared_name_coalesces",
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
                "joining on a shared name coalesces the key, which is the one join shape where "
                "DuckDB and Polars agree"
            ),
        ),
        ConformanceCase(
            name="join_on_condition_duplicates_the_collision",
            pipeline=lambda orders, customers: orders.set_alias("o").join(
                customers.set_alias("c"), "o.order_id = c.customer_id"
            ),
            inputs=joined,
            raises=DuplicateColumn,
            reason=(
                "DuckDB keeps both columns of a colliding name and renames neither, so this "
                "join produces two columns called created_at. Polars coalesces the key and "
                "suffixes the collision _right; pandas suffixes _x and _y. A name-addressed "
                "schema cannot describe the DuckDB answer, so rowless reports it"
            ),
        ),
        ConformanceCase(
            name="union",
            pipeline=lambda orders: orders.union(orders.filter("status = 'paid'")),
            inputs=orders,
            expect=Schema.from_mapping(
                {
                    "order_id": int_(64),
                    "status": string(),
                    "amount": int_(64),
                    "created_at": TIMESTAMP,
                }
            ),
            reason="union is UNION ALL on a relation, and it requires the schemas to line up",
        ),
        ConformanceCase(
            name="sql_through_a_relation",
            pipeline=lambda orders: orders.query(
                "orders", "SELECT status, sum(amount) AS total FROM orders GROUP BY status"
            ),
            inputs=orders,
            expect=Schema.from_mapping({"status": string(), "total": SUMMED}),
            reason=(
                "the seam, in one line: a SQL string resolved against a rowless relation, with "
                "the same binder and the same answer as the relational spelling above"
            ),
        ),
        ConformanceCase(
            name="nested_round_trip",
            pipeline=lambda events: events.project("payload, tags, attrs"),
            inputs=events,
            expect=Schema.from_mapping(
                {
                    "payload": struct({"source": string(), "retries": int_(32)}),
                    "tags": list_(string()),
                    "attrs": map_(string(), string()),
                }
            ),
            reason=(
                "DuckDB has a real MAP type, so a map stays a map; Polars has to carry one as "
                "the Arrow list[struct{key, value}] encoding"
            ),
        ),
        ConformanceCase(
            name="nested_struct_field",
            pipeline=lambda events: events.project("payload.source AS source"),
            inputs=events,
            expect=Schema.from_mapping({"source": string()}),
            reason="dot notation reaches into a struct, and the alias is what names the column",
        ),
        ConformanceCase(
            name="list_length_is_bigint",
            pipeline=lambda events: events.project("len(tags) AS tag_count"),
            inputs=events,
            expect=Schema.from_mapping({"tag_count": int_(64)}),
            reason=(
                "a list length is BIGINT here and uint32 in Polars, which a declared contract "
                "will notice"
            ),
        ),
        ConformanceCase(
            name="drift_prone_schema_survives",
            pipeline=lambda drift: drift.project("nullable_bool, category, money"),
            inputs=drift,
            expect=Schema.from_mapping(
                {
                    "nullable_bool": boolean(),
                    "category": enum(("a", "b", "c")),
                    "money": decimal(18, 2),
                }
            ),
            reason=(
                "the dtypes that drift to object or float64 on an empty pandas frame. None of "
                "these drift here, because a DuckDB table carries its declared types whether or "
                "not it has rows - which is why this adapter needs no fixup table. The one "
                "exception is the all-null column, left out of this projection because what it "
                "resolves to depends on the DuckDB release; see the package README"
            ),
        ),
        ConformanceCase(
            name="null_column_survives_being_read",
            pipeline=lambda drift: drift.filter("nullable_bool"),
            inputs=drift,
            expect=Schema.from_mapping(
                {
                    "all_null": null(),
                    "nullable_bool": boolean(),
                    "category": enum(("a", "b", "c")),
                    "naive_ts": datetime(TimeUnit.MICROSECOND),
                    "money": decimal(18, 2),
                }
            ),
            reason=(
                "an all-null column keeps the null type as long as nothing projects it: a table "
                "read is the declared schema on every release. Projecting it is where 1.0 and "
                "2.0 part company"
            ),
        ),
        ConformanceCase(
            name="decimal_sum_widens_to_the_maximum",
            pipeline=lambda drift: drift.aggregate("sum(money) AS total"),
            inputs=drift,
            expect=Schema.from_mapping({"total": decimal(38, 2)}),
            reason=(
                "summing a DECIMAL(18,2) widens the precision to 38 and keeps the scale, so a "
                "contract that declares the sum as decimal(18,2) fails on DuckDB and passes on "
                "an engine that does not widen"
            ),
        ),
        ConformanceCase(
            name="missing_column",
            pipeline=lambda orders: orders.project("amount_cents"),
            inputs=orders,
            raises=ColumnNotFound,
            reason=(
                "the finding from the proposal. DuckDB binds eagerly, so this raises at the "
                "line that wrote it, and the binder lists the candidate bindings - which is "
                "what makes the near-miss suggestion possible"
            ),
        ),
        ConformanceCase(
            name="missing_column_shadowed_by_an_alias",
            pipeline=lambda orders: orders.aggregate("status, sum(amount_cents) AS amount_cents"),
            inputs=orders,
            raises=ColumnNotFound,
            reason=(
                "the rename case as it actually arrives. When the column a query reads is "
                "missing *and* something in the same select list is aliased to that name, "
                "DuckDB stops talking about the missing column and complains that an alias "
                "cannot be referenced before it is defined. It is still a missing column, and "
                "reporting it as a lateral-reference puzzle would send the user to the wrong "
                "line - this is the shape `sum(amount) AS amount` takes after a rename, which "
                "is to say the common one"
            ),
        ),
        ConformanceCase(
            name="row_dependent_fetch",
            pipeline=lambda orders: orders.project("order_id") if not orders.fetchall() else orders,
            inputs=orders,
            expect_confidence=Confidence.UNKNOWN,
            reason=(
                "fetchall() returns nothing on a rowless relation, so the dry run took the "
                "branch production never would"
            ),
        ),
        ConformanceCase(
            name="row_dependent_length",
            pipeline=lambda orders: orders.limit(1) if len(orders) == 0 else orders,
            inputs=orders,
            expect_confidence=Confidence.UNKNOWN,
            reason=(
                "len() on a relation runs a count, and the count is zero. The aggregate "
                "methods are not row-dependent: rel.max('amount') returns a relation, so it "
                "resolves to a dtype without reading anything"
            ),
        ),
    )
