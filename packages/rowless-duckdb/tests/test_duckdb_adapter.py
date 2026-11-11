"""Unit tests for the parts of the adapter the conformance suite cannot reach.

The suite in core is the specification, and it covers the contract. What it
cannot cover is the behaviour that is specific to this engine: which DuckDB
exceptions become which findings, which dtypes it refuses rather than
approximates, whether the SQL seam resolves a dbt-style relation name, and
whether the lineage read out of DuckDB's parser is the lineage a user would
expect. That is what is here.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import duckdb
import pytest
from rowless.adapters.base import DryRunContext, SupportsColumnLineage, SupportsSql
from rowless.diagnostics.codes import Code
from rowless.errors import (
    ColumnNotFound,
    DuplicateColumn,
    SchemaNotFound,
    SideEffectBlocked,
    UnknownSchemaError,
    UnsupportedDType,
)
from rowless.ir.confidence import Confidence
from rowless.ir.dtypes import (
    Kind,
    TimeUnit,
    any_,
    binary,
    datetime,
    decimal,
    duration,
    enum,
    float_,
    int_,
    list_,
    map_,
    string,
    struct,
    time,
    unknown,
)
from rowless.ir.schema import Field, Schema
from rowless.runtime.dryrun import dry_run_callable
from rowless.runtime.rowdep import RowDependencyLog
from rowless.testing import fixtures
from rowless_duckdb import database, errors, interception, lineage, session
from rowless_duckdb.adapter import DuckDBAdapter, _at_least
from rowless_duckdb.dtypes import from_ir, to_ir, type_spec

pytestmark = pytest.mark.requires_engine

ADAPTER = DuckDBAdapter()

TIMESTAMP = datetime(TimeUnit.MICROSECOND, "UTC")


def _engine_reports_nullability() -> bool:
    """Whether this DuckDB reports `NOT NULL` at all. It started doing so in 1.2.

    Asked of DuckDB directly rather than of the adapter, so the tests below
    still check that the adapter reads the answer correctly wherever the answer
    exists - and skip, rather than fail, at the declared floor, which has no
    answer to read.
    """
    connection = duckdb.connect(":memory:")
    connection.execute("CREATE TABLE t (a BIGINT NOT NULL)")
    described = connection.sql("DESCRIBE SELECT * FROM t").fetchall()
    return bool(described) and str(described[0][2]).upper() == "NO"


REPORTS_NULLABILITY = _engine_reports_nullability()

needs_nullability = pytest.mark.skipif(
    not REPORTS_NULLABILITY,
    reason="this DuckDB does not report NOT NULL; it arrived in 1.2 and the floor is 1.0",
)


@pytest.fixture
def ctx(tmp_path: Path) -> DryRunContext:
    """A fresh dry-run context, which also resets the scratch database."""
    context = DryRunContext(run_id=f"test-{tmp_path.name}", workdir=str(tmp_path))
    session.begin(context)
    return context


@pytest.fixture
def tracked(ctx: DryRunContext) -> Iterator[RowDependencyLog]:
    """An open dry run, so the wrappers in `interception` are live.

    They are gated on one deliberately: outside a dry run DuckDB behaves
    normally, which is what lets a process import this adapter without its
    own DuckDB usage changing. Anything testing translation or lineage has to
    be inside a run, the same way a real check is.
    """
    from rowless.runtime import rowdep

    with rowdep.recording() as log:
        yield log


@pytest.fixture
def orders(ctx: DryRunContext, tracked: RowDependencyLog) -> Any:
    """The proposal's orders relation, materialized as an empty DuckDB table."""
    return ADAPTER.materialize(fixtures.ORDERS, ctx)


# --- Availability and routing ------------------------------------------------


def test_available_with_duckdb_installed() -> None:
    assert ADAPTER.available() is True


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("0.10.3", False),
        ("1.0.0", True),
        ("1.3.2", True),
        ("2.0.0", True),
        ("2.1.0-dev123", True),
        ("", False),
    ],
)
def test_version_floor(version: str, expected: bool) -> None:
    assert _at_least(version, (1, 0)) is expected


def test_handles_relations_and_connections(orders: Any) -> None:
    assert ADAPTER.handles(orders) is True
    assert ADAPTER.handles(duckdb.connect(":memory:")) is True
    assert ADAPTER.handles(object()) is False
    assert ADAPTER.handles("SELECT 1") is False


def test_declares_the_capabilities_it_implements() -> None:
    assert isinstance(ADAPTER, SupportsSql)
    assert isinstance(ADAPTER, SupportsColumnLineage)
    assert ADAPTER.capabilities.sql is True
    assert ADAPTER.capabilities.column_lineage is True
    # DuckDB cannot probe a Python UDF's return dtype: a function registered
    # with create_function has to declare its type up front.
    assert ADAPTER.capabilities.udf_probe is False


# --- Dtypes ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("spelling", "expected"),
    [
        ("BIGINT", int_(64)),
        ("HUGEINT", int_(128)),
        ("UHUGEINT", Kind.UINT),
        ("FLOAT", float_(32)),
        ("TIMESTAMP_S", datetime(TimeUnit.SECOND)),
        ("TIMESTAMP WITH TIME ZONE", TIMESTAMP),
        ("INTERVAL", duration(TimeUnit.MICROSECOND)),
        ("INTEGER[3]", list_(int_(32))),
        ("MAP(VARCHAR, BIGINT)", map_(string(), int_(64))),
        ("ENUM('a','b','c')", enum(("a", "b", "c"))),
        ('"NULL"', Kind.NULL),
        ("BLOB", binary()),
    ],
)
def test_to_ir_reads_the_sql_spelling(spelling: str, expected: object) -> None:
    """`DESCRIBE` reports types as strings, so `to_ir` accepts both spellings."""
    from_string = to_ir(spelling)
    from_type = to_ir(duckdb.sqltype(spelling))
    assert from_string == from_type
    if isinstance(expected, Kind):
        assert from_string.kind is expected
    else:
        assert from_string == expected


@pytest.mark.parametrize(
    "spelling",
    ["UUID", "BIT", "VARINT", "UNION(a INTEGER)", "TIME WITH TIME ZONE"],
)
def test_to_ir_refuses_types_with_no_ir_equivalent(spelling: str) -> None:
    """A type the IR cannot describe poisons comparisons rather than passing them."""
    found = to_ir(spelling)
    assert found.kind is Kind.UNKNOWN
    assert found.reason


def test_to_ir_of_an_unparseable_string_is_unknown() -> None:
    assert to_ir("NOT A TYPE").kind is Kind.UNKNOWN
    assert to_ir(object()).kind is Kind.UNKNOWN


def test_array_loses_its_width() -> None:
    """The IR has no fixed-size list, and the element type is what matters."""
    assert to_ir("INTEGER[3]") == to_ir("INTEGER[]")


@pytest.mark.parametrize(
    ("dtype", "spelling"),
    [
        (int_(8), "TINYINT"),
        (int_(128), "HUGEINT"),
        (float_(64), "DOUBLE"),
        (decimal(38, 9), "DECIMAL(38,9)"),
        (string(), "VARCHAR"),
        (time(TimeUnit.MICROSECOND), "TIME"),
        (datetime(TimeUnit.NANOSECOND), "TIMESTAMP_NS"),
        (TIMESTAMP, "TIMESTAMP WITH TIME ZONE"),
        (datetime(TimeUnit.MICROSECOND, "utc"), "TIMESTAMP WITH TIME ZONE"),
        (list_(string()), "VARCHAR[]"),
        (map_(string(), string()), "MAP(VARCHAR, VARCHAR)"),
        (enum(("a", "b")), "ENUM('a', 'b')"),
    ],
)
def test_from_ir_spells_the_type(dtype: object, spelling: str) -> None:
    assert type_spec(dtype) == spelling  # type: ignore[arg-type]


def test_struct_field_names_are_quoted() -> None:
    """A field called `select` is legal in the IR and a keyword in SQL."""
    spelled = type_spec(struct({"select": string(), "from": int_(32)}))
    assert spelled == 'STRUCT("select" VARCHAR, "from" INTEGER)'
    assert to_ir(duckdb.sqltype(spelled)) == struct({"select": string(), "from": int_(32)})


def test_enum_variants_are_escaped() -> None:
    quoted = type_spec(enum(("it's", "fine")))
    assert quoted == "ENUM('it''s', 'fine')"
    assert to_ir(duckdb.sqltype(quoted)) == enum(("it's", "fine"))


@pytest.mark.parametrize(
    ("dtype", "fragment"),
    [
        (datetime(TimeUnit.MICROSECOND, "Europe/London"), "named zone"),
        (datetime(TimeUnit.NANOSECOND, "UTC"), "microsecond resolution"),
        (time(TimeUnit.MILLISECOND), "microsecond type"),
        (duration(TimeUnit.NANOSECOND), "microsecond type"),
        (decimal(39, 2), "at most 38 digits"),
        (float_(16), "16-bit float"),
        (unknown("nothing resolved this"), "no dtype for"),
    ],
)
def test_from_ir_refuses_rather_than_approximating(dtype: object, fragment: str) -> None:
    """Every refusal names the reason, because the user has to fix the contract."""
    with pytest.raises(UnsupportedDType, match=fragment):
        from_ir(dtype)  # type: ignore[arg-type]


def test_utc_is_the_only_zone_that_round_trips() -> None:
    """`TIMESTAMPTZ` stores instants, so UTC is the only honest reading of it."""
    assert to_ir(from_ir(TIMESTAMP)) == TIMESTAMP


def test_unsupported_dtypes_are_declared_and_refused() -> None:
    """The conformance suite reads this list; it must not be aspirational."""
    declared = list(ADAPTER.unsupported_dtypes())
    assert declared
    for dtype in declared:
        with pytest.raises(UnsupportedDType):
            ADAPTER.from_ir(dtype)


# --- Materialize and describe ------------------------------------------------


@needs_nullability
def test_materialize_declares_not_null(orders: Any) -> None:
    """DuckDB is the one engine here that tracks nullability, so it is declared."""
    described = ADAPTER.describe(orders).schema
    assert described["order_id"].nullable is False
    assert described["status"].nullable is True


def test_nullability_is_reported_nullable_when_the_engine_will_not_say(orders: Any) -> None:
    """Not knowing is reported as nullable, which is the safe direction."""
    described = ADAPTER.describe(orders).schema
    if not REPORTS_NULLABILITY:
        assert all(field.nullable for field in described.fields)


@needs_nullability
def test_not_null_survives_a_projection(orders: Any) -> None:
    """Which is what makes a non-nullable contract checkable at all."""
    projected = ADAPTER.describe(orders.project("order_id, amount")).schema
    assert projected["order_id"].nullable is False
    assert projected["amount"].nullable is True


def test_materialize_degrades_an_unresolved_input_dtype(ctx: DryRunContext) -> None:
    schema = Schema(
        fields=(
            Field(name="order_id", dtype=int_(64)),
            Field(name="mystery", dtype=unknown("the source would not say")),
        ),
        name="partly_known",
    )
    relation = ADAPTER.materialize(schema, ctx)
    assert ADAPTER.describe(relation).schema["mystery"].dtype.kind is Kind.NULL
    assert any("mystery" in reason for reason in ctx.degradations)


def test_materialize_notes_an_any_input_without_degrading(ctx: DryRunContext) -> None:
    """ANY is a deliberate escape hatch, so it is reported but does not degrade."""
    schema = Schema(fields=(Field(name="whatever", dtype=any_()),), name="gradual")
    ADAPTER.materialize(schema, ctx)
    assert ctx.degradations == []
    assert any("whatever" in note for note in ctx.notes)


def test_materialize_degrades_a_dtype_duckdb_lacks(ctx: DryRunContext) -> None:
    schema = Schema(
        fields=(Field(name="when", dtype=datetime(TimeUnit.NANOSECOND, "Europe/London")),),
        name="zoned",
    )
    ADAPTER.materialize(schema, ctx)
    assert any("when" in reason for reason in ctx.degradations)


def test_two_inputs_with_the_same_name_get_separate_tables(
    ctx: DryRunContext, tracked: RowDependencyLog
) -> None:
    """A run with two relations called `orders` must not resolve against one table."""
    first = ADAPTER.materialize(fixtures.ORDERS, ctx)
    second = ADAPTER.materialize(fixtures.ORDERS.relabel("stg_orders"), ctx)
    assert first.sql_query() != second.sql_query()


def test_describe_refuses_a_duplicate_output_name(
    ctx: DryRunContext, tracked: RowDependencyLog
) -> None:
    """DuckDB keeps both colliding columns; a name-addressed schema cannot."""
    orders = ADAPTER.materialize(fixtures.ORDERS, ctx)
    customers = ADAPTER.materialize(fixtures.CUSTOMERS, ctx)
    collided = orders.set_alias("o").join(customers.set_alias("c"), "o.order_id = c.customer_id")
    with pytest.raises(DuplicateColumn, match="created_at"):
        ADAPTER.describe(collided)


def test_describe_a_connection_reads_its_last_result(ctx: DryRunContext) -> None:
    """A pipeline written against the DBAPI surface returns a connection."""
    connection = duckdb.connect(":memory:")
    connection.execute("SELECT 1 AS one, 'x' AS two")
    described = ADAPTER.describe(connection).schema
    assert tuple(described) == ("one", "two")
    assert described["two"].dtype == string()


def test_describe_a_connection_with_no_result_is_unresolved() -> None:
    with pytest.raises(UnknownSchemaError, match="no result to describe"):
        ADAPTER.describe(duckdb.connect(":memory:"))


def test_describe_a_foreign_object_is_unresolved() -> None:
    with pytest.raises(UnknownSchemaError, match="cannot describe"):
        ADAPTER.describe({"not": "a relation"})


def test_an_empty_schema_cannot_be_materialized(ctx: DryRunContext) -> None:
    """DuckDB has no zero-column relation, and saying so beats inventing one."""
    with pytest.raises(UnknownSchemaError, match="zero-column"):
        ADAPTER.materialize(Schema.empty(name="nothing"), ctx)


# --- Error translation -------------------------------------------------------


def test_missing_column_names_the_column_and_suggests_the_near_miss(orders: Any) -> None:
    with pytest.raises(ColumnNotFound) as raised:
        orders.project("amount_cents")
    assert raised.value.column == "amount_cents"
    assert raised.value.suggestion == "amount"


@pytest.mark.parametrize(
    "message",
    [
        # 2.x lists the candidates unqualified, 1.x qualifies them.
        'Referenced column "amount_cents" not found in FROM clause!\n'
        'Candidate bindings: "amount", "created_at"',
        'Referenced column "amount_cents" not found in FROM clause!\n'
        'Candidate bindings: "orders.amount", "orders.created_at"',
    ],
)
def test_candidate_bindings_are_read_whichever_way_they_are_spelled(message: str) -> None:
    translated = errors.translate(duckdb.BinderException(message))
    assert isinstance(translated, ColumnNotFound)
    assert translated.suggestion == "amount"


def test_an_ambiguous_reference_is_a_duplicate_column() -> None:
    translated = errors.translate(
        duckdb.BinderException('Ambiguous reference to column name "created_at" (use: ...)')
    )
    assert isinstance(translated, DuplicateColumn)
    assert translated.code is Code.DUPLICATE_COLUMN


def test_a_column_shadowed_by_an_alias_is_still_a_missing_column(orders: Any) -> None:
    """DuckDB talks about the alias; the column is what the user has to fix.

    `sum(amount) AS amount` over a table where `amount` was renamed is the
    shape the rename case actually arrives in, and the binder's message for it
    mentions only the lateral reference.
    """
    with pytest.raises(ColumnNotFound) as raised:
        orders.aggregate("status, sum(amount_cents) AS amount_cents")
    assert raised.value.column == "amount_cents"
    assert raised.value.suggestion == "amount"


def test_an_unknown_table_is_a_missing_schema() -> None:
    translated = errors.translate(duckdb.CatalogException("Table with name nope does not exist!"))
    assert isinstance(translated, SchemaNotFound)


@pytest.mark.parametrize(
    "message",
    [
        "Scalar Function with name no_such_function does not exist!",
        "Type with name no_such_type does not exist!",
    ],
)
def test_a_missing_function_is_not_a_missing_schema(message: str) -> None:
    """It is a mistake in the query, and `SCHEMA_NOT_FOUND` reads as unchecked.

    Core reports an untranslated error under `PIPELINE_RAISED` at error level,
    which is what a wrong query deserves; `SCHEMA_NOT_FOUND` is an unverified
    code, and would make a broken pipeline look like one rowless skipped.
    """
    assert errors.translate(duckdb.CatalogException(message)) is None


def test_an_untranslated_error_still_points_at_the_pipeline() -> None:
    """DuckDB 2.x raises through a Python layer, which must not take the blame.

    Core picks the line for a finding by walking the traceback for the deepest
    frame outside rowless. Left alone, every unclassified DuckDB error would be
    reported against `site-packages/duckdb/_relation.py`, which no user can
    act on.
    """

    def pipeline(orders: Any) -> Any:
        return orders.project("no_such_function(status) AS oops")

    result = dry_run_callable(pipeline, {"orders": fixtures.ORDERS}, engine="duckdb")
    assert result.output is None
    (finding,) = [d for d in result.diagnostics if d.code is Code.PIPELINE_RAISED]
    assert finding.origin is not None
    assert finding.origin.file == __file__
    assert "no_such_function" in (finding.detail or "")


def test_a_blocked_write_says_rowless_blocked_it() -> None:
    translated = errors.translate(
        duckdb.InvalidInputException(
            'Cannot execute statement of type "CREATE" on database "warehouse" which is '
            "attached in read-only mode!"
        )
    )
    assert isinstance(translated, SideEffectBlocked)


def test_an_unclassifiable_duckdb_error_is_left_alone() -> None:
    """Core reports it under PIPELINE_RAISED, which beats a guessed code."""
    assert errors.translate(duckdb.InternalException("something went very wrong")) is None


def test_a_user_exception_is_not_mistaken_for_an_engine_failure() -> None:
    class BinderException(Exception):
        """Same name, nothing to do with DuckDB."""

    assert errors.translate(BinderException("not ours")) is None


def test_the_headline_drops_the_query_dump() -> None:
    exception = duckdb.BinderException(
        'Referenced column "x" not found in FROM clause!\nLINE 1: SELECT x FROM t\n               ^'
    )
    assert "LINE 1" not in errors.headline(exception)


# --- The SQL seam ------------------------------------------------------------


def test_run_sql_resolves_against_registered_relations(ctx: DryRunContext) -> None:
    orders = ADAPTER.materialize(fixtures.ORDERS, ctx)
    resolved = ADAPTER.run_sql(
        "SELECT status, sum(amount) AS total FROM stg_orders GROUP BY status",
        {"stg_orders": orders},
        ctx,
    )
    assert tuple(resolved.schema) == ("status", "total")
    assert resolved.schema["total"].dtype == int_(128)
    assert resolved.engine == "duckdb"


def test_run_sql_accepts_a_schema_without_a_relation(ctx: DryRunContext) -> None:
    """A caller holding a resolved schema has no reason to build a relation first."""
    resolved = ADAPTER.run_sql(
        "SELECT order_id FROM stg_orders",
        {"stg_orders": fixtures.ORDERS},
        ctx,
    )
    assert resolved.schema["order_id"].dtype == int_(64)


def test_run_sql_resolves_a_dbt_style_relation_name(ctx: DryRunContext) -> None:
    """dbt compiles `ref()` into a three-part quoted name, in a catalog we lack."""
    resolved = ADAPTER.run_sql(
        'SELECT country FROM "warehouse"."analytics"."stg_customers"',
        {'"warehouse"."analytics"."stg_customers"': fixtures.CUSTOMERS},
        ctx,
    )
    assert tuple(resolved.schema) == ("country",)


def test_run_sql_attaches_column_lineage(ctx: DryRunContext) -> None:
    resolved = ADAPTER.run_sql(
        "SELECT status, sum(amount) AS total FROM stg_orders GROUP BY status",
        {"stg_orders": fixtures.ORDERS},
        ctx,
    )
    assert resolved.schema["status"].derived_from == ("status",)
    assert resolved.schema["total"].derived_from == ("amount",)


def test_run_sql_reports_a_subquery_as_untraced(ctx: DryRunContext) -> None:
    """Attributing across a scope rowless cannot see is how a tool invents a rename."""
    resolved = ADAPTER.run_sql(
        "SELECT total FROM (SELECT sum(amount) AS total FROM stg_orders)",
        {"stg_orders": fixtures.ORDERS},
        ctx,
    )
    assert resolved.schema["total"].dtype == int_(128)
    assert resolved.schema["total"].derived_from == ()
    assert any("untraced" in note for note in ctx.notes)


def test_run_sql_reports_a_missing_column_as_a_finding(ctx: DryRunContext) -> None:
    with pytest.raises(ColumnNotFound):
        ADAPTER.run_sql("SELECT amount_cents FROM stg_orders", {"stg_orders": fixtures.ORDERS}, ctx)


def test_run_sql_refuses_a_statement_that_is_not_a_query(ctx: DryRunContext) -> None:
    """`DESCRIBE` will not take a DDL statement, which is the point of using it."""
    with pytest.raises(Exception):  # noqa: B017 - the kind depends on the release
        ADAPTER.run_sql(
            "CREATE TABLE output AS SELECT * FROM stg_orders",
            {"stg_orders": fixtures.ORDERS},
            ctx,
        )


def test_run_sql_refuses_an_input_it_cannot_read(ctx: DryRunContext) -> None:
    with pytest.raises(UnknownSchemaError, match="cannot register"):
        ADAPTER.run_sql("SELECT 1", {"orders": "a string is not a relation"}, ctx)


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("stg_orders", ("stg_orders",)),
        ("analytics.stg_orders", ("analytics", "stg_orders")),
        ('"warehouse"."analytics"."stg_orders"', ("warehouse", "analytics", "stg_orders")),
        ("`warehouse`.`stg_orders`", ("warehouse", "stg_orders")),
        ("[warehouse].[stg_orders]", ("warehouse", "stg_orders")),
        ("", ()),
    ],
)
def test_relation_names_are_split_whatever_quotes_them(
    path: str, expected: tuple[str, ...]
) -> None:
    assert database.split_identifier(path) == expected


# --- Column lineage ----------------------------------------------------------


def test_lineage_composes_through_a_chain(orders: Any) -> None:
    chained = orders.filter("status = 'paid'").project("amount AS amount_cents, order_id")
    assert ADAPTER.column_lineage(chained) == {
        "amount_cents": ("amount",),
        "order_id": ("order_id",),
    }


def test_lineage_of_an_expression_reads_every_column_it_touches(orders: Any) -> None:
    computed = orders.project("amount * order_id AS weighted")
    assert ADAPTER.column_lineage(computed) == {"weighted": ("amount", "order_id")}


def test_lineage_survives_an_aggregate(orders: Any) -> None:
    aggregated = orders.aggregate("status, sum(amount) AS total")
    assert ADAPTER.column_lineage(aggregated) == {"status": ("status",), "total": ("amount",)}


def test_a_predicate_is_not_lineage(orders: Any) -> None:
    """`WHERE` decides which rows survive, not what a column holds."""
    filtered = orders.filter("status = 'paid'").project("amount")
    assert ADAPTER.column_lineage(filtered) == {"amount": ("amount",)}


def test_lineage_of_a_literal_column_is_empty(orders: Any) -> None:
    """A column computed from no input column has no parent, and says so."""
    constant = orders.project("1 AS one, amount")
    assert ADAPTER.column_lineage(constant) == {"amount": ("amount",)}


def test_lineage_of_a_join_takes_each_column_from_its_own_side(
    ctx: DryRunContext, tracked: RowDependencyLog
) -> None:
    orders = ADAPTER.materialize(fixtures.ORDERS, ctx)
    customers = ADAPTER.materialize(fixtures.CUSTOMERS, ctx)
    joined = orders.join(customers, "created_at")
    traced = ADAPTER.column_lineage(joined)
    assert traced["amount"] == ("amount",)
    assert traced["country"] == ("country",)


def test_lineage_of_a_union_merges_both_sides(orders: Any) -> None:
    merged = orders.union(orders.filter("status = 'paid'"))
    assert ADAPTER.column_lineage(merged)["amount"] == ("amount",)


def test_a_star_projection_carries_rather_than_guessing(orders: Any) -> None:
    """`*` expands to a count the text does not give, so names are carried."""
    starred = orders.project("*")
    assert ADAPTER.column_lineage(starred)["amount"] == ("amount",)


def test_lineage_of_an_untracked_relation_is_read_off_its_sql(ctx: DryRunContext) -> None:
    """A relation from `conn.sql` was never watched being built."""
    ADAPTER.materialize(fixtures.ORDERS, ctx)
    connection = database.connection()
    produced = connection.sql("SELECT status, sum(amount) AS total FROM stg_orders GROUP BY status")
    lineage.TRACKER.clear()
    assert ADAPTER.column_lineage(produced) == {"status": ("status",), "total": ("amount",)}


def test_lineage_of_a_foreign_object_is_empty() -> None:
    assert ADAPTER.column_lineage(object()) == {}


def test_lineage_is_dropped_when_the_relation_is_collected(orders: Any) -> None:
    before = len(lineage.TRACKER)
    projected = orders.project("amount")
    assert len(lineage.TRACKER) == before + 1
    # Dropping the relation drops its entry, rather than leaving it to be
    # inherited by whatever reuses the id().
    del projected
    assert len(lineage.TRACKER) == before


def test_parse_query_refuses_a_set_operation() -> None:
    """A union's columns come from several scopes; `merged` handles those."""
    parsed = lineage.parse_query("SELECT a FROM t UNION ALL SELECT b FROM u")
    assert parsed.resolvable is False


def test_parse_query_refuses_a_cte() -> None:
    parsed = lineage.parse_query("WITH x AS (SELECT a FROM t) SELECT a FROM x")
    assert parsed.resolvable is False


def test_parse_query_survives_a_syntax_error() -> None:
    assert lineage.parse_query("SELECT FROM").resolvable is False


def test_a_table_function_is_not_an_input_relation() -> None:
    """`read_parquet(...)` columns are not input columns, so nothing is claimed.

    Core attributes an unqualified parent to the single input when a pipeline
    has one, so claiming these names would invent an edge to a column that
    relation does not have.
    """
    assert lineage.of_query("SELECT i FROM range(3) t(i)", ["i"]) == {}
    assert lineage.of_query("SELECT a AS b FROM t", ["b"]) == {"b": ("a",)}


# --- Interception ------------------------------------------------------------


def test_a_writer_is_recorded_rather_than_performed(tmp_path: Path) -> None:
    target = tmp_path / "revenue.parquet"

    def pipeline(orders: Any) -> Any:
        orders.write_parquet(str(target))
        return orders.project("order_id")

    result = dry_run_callable(pipeline, {"orders": fixtures.ORDERS}, engine="duckdb")
    assert result.output is not None
    assert not target.exists()
    assert any("revenue.parquet" in note for note in result.output.notes)


def test_a_copy_statement_is_recognised_and_skipped(tmp_path: Path) -> None:
    """DuckDB writes through its own filesystem, so the audit hook never sees this."""
    target = tmp_path / "exported.parquet"

    def pipeline(orders: Any) -> Any:
        database.connection().execute(f"COPY (SELECT 1) TO '{target.as_posix()}'")
        return orders.project("order_id")

    result = dry_run_callable(pipeline, {"orders": fixtures.ORDERS}, engine="duckdb")
    assert result.output is not None
    assert not target.exists()
    assert any("COPY" in note for note in result.output.notes)


def test_the_scratch_database_refuses_file_access(ctx: DryRunContext) -> None:
    """The other half of the same hole: no COPY, no read_csv, no httpfs."""
    with pytest.raises(duckdb.Error, match="disabled"):
        database.connection().execute("COPY (SELECT 1) TO 'rowless_must_not_exist.parquet'")
    assert not Path("rowless_must_not_exist.parquet").exists()


def test_connecting_to_a_missing_database_is_blocked() -> None:
    def pipeline(orders: Any) -> Any:
        duckdb.connect("rowless_must_not_be_created.db")
        return orders

    result = dry_run_callable(pipeline, {"orders": fixtures.ORDERS}, engine="duckdb")
    assert result.output is None
    assert Code.SIDE_EFFECT_BLOCKED in [d.code for d in result.diagnostics]
    assert not Path("rowless_must_not_be_created.db").exists()


def test_connecting_to_a_real_database_is_read_only(tmp_path: Path) -> None:
    existing = tmp_path / "warehouse.db"
    writer = duckdb.connect(str(existing))
    writer.execute("CREATE TABLE orders (order_id BIGINT)")
    writer.close()

    def pipeline(orders: Any) -> Any:
        connection = duckdb.connect(str(existing))
        return connection.table("orders")

    result = dry_run_callable(pipeline, {"orders": fixtures.ORDERS}, engine="duckdb")
    assert result.output is not None
    assert tuple(result.output.schema) == ("order_id",)
    assert any("read-only" in note for note in result.output.notes)


def test_an_in_memory_connection_is_left_alone() -> None:
    def pipeline(orders: Any) -> Any:
        connection = duckdb.connect(":memory:")
        connection.execute("CREATE TABLE scratch (a BIGINT)")
        return connection.table("scratch")

    result = dry_run_callable(pipeline, {"orders": fixtures.ORDERS}, engine="duckdb")
    assert result.output is not None
    assert tuple(result.output.schema) == ("a",)


@pytest.mark.parametrize(
    "read",
    [
        # Only accessors that need nothing but duckdb: `df()`, `pl()` and
        # `to_arrow_table()` are intercepted the same way, and this package
        # depends on none of pandas, Polars or pyarrow.
        lambda relation: relation.fetchall(),
        lambda relation: len(relation),
        lambda relation: relation.shape,
    ],
)
def test_reading_values_costs_the_run_its_confidence(read: Callable[[Any], object]) -> None:
    def pipeline(orders: Any) -> Any:
        read(orders)
        return orders.project("order_id")

    result = dry_run_callable(pipeline, {"orders": fixtures.ORDERS}, engine="duckdb")
    assert result.coverage.confidence is Confidence.UNKNOWN
    assert result.coverage.reasons


def test_reading_values_off_a_connection_costs_the_same() -> None:
    """A pipeline on the DBAPI surface loses confidence the same way."""

    def pipeline(orders: Any) -> Any:
        connection = duckdb.connect(":memory:")
        connection.execute("SELECT 1")
        connection.fetchall()
        return orders.project("order_id")

    result = dry_run_callable(pipeline, {"orders": fixtures.ORDERS}, engine="duckdb")
    assert result.coverage.confidence is Confidence.UNKNOWN


def test_an_aggregate_method_is_not_row_dependent() -> None:
    """The difference from Polars: `rel.max(...)` builds a relation, not a value."""

    def pipeline(orders: Any) -> Any:
        return orders.max("amount")

    result = dry_run_callable(pipeline, {"orders": fixtures.ORDERS}, engine="duckdb")
    assert result.coverage.confidence is not Confidence.UNKNOWN


def test_duckdb_behaves_normally_outside_a_dry_run(tmp_path: Path) -> None:
    """The wrappers are installed once per process and gated per run."""
    interception.install(DryRunContext(run_id="gating", workdir=str(tmp_path)))
    connection = duckdb.connect(":memory:")
    connection.execute("CREATE TABLE t (a BIGINT)")
    connection.execute("INSERT INTO t VALUES (1), (2)")
    assert connection.table("t").fetchall() == [(1,), (2,)]
    target = tmp_path / "written.parquet"
    connection.table("t").write_parquet(str(target))
    assert target.exists()
