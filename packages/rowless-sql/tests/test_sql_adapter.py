"""Unit tests for the parts of the adapter the conformance suite cannot reach.

The suite in core is the specification, and it covers the contract. What it
cannot cover is the behaviour specific to this engine: which sqlglot failures
become which findings, which dtypes it refuses rather than approximates,
whether a dbt-shaped relation name resolves against the catalogue, whether the
lineage sqlglot reports survives a CTE, what happens to an identifier the
dialect folds, and where the dialect itself comes from. That is what is here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import sqlglot
from rowless.adapters.base import DryRunContext, SupportsColumnLineage, SupportsSql
from rowless.diagnostics.codes import Code
from rowless.errors import (
    ColumnNotFound,
    ConfigError,
    DuplicateColumn,
    RowlessError,
    UnknownSchemaError,
    UnsupportedDType,
)
from rowless.ir.confidence import Confidence
from rowless.ir.dtypes import (
    DType,
    Kind,
    TimeUnit,
    any_,
    binary,
    boolean,
    date,
    datetime,
    decimal,
    duration,
    enum,
    float_,
    int_,
    list_,
    map_,
    null,
    string,
    struct,
    time,
    uint,
    unknown,
)
from rowless.ir.schema import Field, Schema
from rowless.runtime.dryrun import dry_run_callable
from rowless.testing import fixtures
from rowless_sql import catalog as catalogue
from rowless_sql import dialects, errors, lineage, resolve, session
from rowless_sql.adapter import MINIMUM_SQLGLOT, SqlAdapter, _at_least
from rowless_sql.dtypes import from_ir, to_ir

pytestmark = pytest.mark.requires_engine

ADAPTER = SqlAdapter()

if not ADAPTER.available():
    # Below the declared floor these tests would fail rather than say
    # anything: the adapter supports sqlglot >= 25.7, and CI runs this suite
    # against exactly that release as well as the current one.
    pytest.skip(f"rowless-sql needs {SqlAdapter.requires}", allow_module_level=True)

TIMESTAMP = datetime(TimeUnit.MICROSECOND, "UTC")

#: A schema whose columns are spelled the way a hand-written warehouse table
#: often spells them, for the identifier-folding tests.
MIXED_CASE = Schema(
    fields=(
        Field(name="CustomerID", dtype=int_(64), nullable=False),
        Field(name="Country", dtype=string()),
    ),
    name="Dim_Customer",
)


@pytest.fixture
def ctx(tmp_path: Path) -> DryRunContext:
    """A fresh dry-run context, which also resets this adapter's per-run state."""
    context = DryRunContext(run_id=f"test-{tmp_path.name}", workdir=str(tmp_path))
    session.begin(context)
    dialects.forget_dbt()
    return context


@pytest.fixture
def orders(ctx: DryRunContext) -> Any:
    """The orders fixture, materialized and registered."""
    return ADAPTER.materialize(fixtures.ORDERS, ctx)


@pytest.fixture
def customers(ctx: DryRunContext) -> Any:
    """The customers fixture, which collides with orders on `created_at`."""
    return ADAPTER.materialize(fixtures.CUSTOMERS, ctx)


@pytest.fixture
def events(ctx: DryRunContext) -> Any:
    """The nested-types fixture."""
    return ADAPTER.materialize(fixtures.EVENTS, ctx)


def columns_of(relation: Any) -> list[tuple[str, str]]:
    """Name and dtype spelling per column, which is what most tests assert on."""
    return [(field.name, str(field.dtype)) for field in relation.schema.fields]


# --- Availability and routing ------------------------------------------------


def test_available_with_sqlglot_installed() -> None:
    assert ADAPTER.available() is True
    assert ADAPTER.name == "sql"
    assert ADAPTER.capabilities.sql is True
    assert isinstance(ADAPTER, SupportsSql)
    assert isinstance(ADAPTER, SupportsColumnLineage)


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("25.7.0", True),
        ("25.6.1", False),
        ("25.18.0", True),
        ("26.0.0", True),
        ("30.27.0", True),
        ("25.7", True),
        ("1.0", False),
        ("not-a-version", False),
    ],
)
def test_version_floor(version: str, expected: bool) -> None:
    assert _at_least(version, MINIMUM_SQLGLOT) is expected


def test_handles_sqlglot_expressions(orders: Any) -> None:
    assert ADAPTER.handles(orders) is True
    assert ADAPTER.handles(sqlglot.parse_one("SELECT 1")) is True
    assert ADAPTER.handles(object()) is False


def test_a_string_is_sql_only_inside_a_run(ctx: DryRunContext) -> None:
    """Outside a run the registry must not route every string to this adapter."""
    assert session.sql_mode() is True
    assert ADAPTER.handles("SELECT 1") is True

    # Reaching into the module is the only way to simulate "no run open":
    # `begin` is the only public way in, and there is no public way out.
    session._CONTEXT = None
    try:
        assert session.sql_mode() is False
        assert ADAPTER.handles("SELECT 1") is False
    finally:
        session.begin(ctx)


# --- Dtypes ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ir_dtype", "spelling"),
    [
        (boolean(), "BOOLEAN"),
        (int_(8), "TINYINT"),
        (int_(16), "SMALLINT"),
        (int_(32), "INT"),
        (int_(64), "BIGINT"),
        (int_(128), "INT128"),
        (uint(8), "UTINYINT"),
        (uint(32), "UINT"),
        (float_(32), "FLOAT"),
        (float_(64), "DOUBLE"),
        (decimal(18, 2), "DECIMAL(18, 2)"),
        (string(), "VARCHAR"),
        (binary(), "VARBINARY"),
        (date(), "DATE"),
        (time(TimeUnit.MILLISECOND), "TIME(3)"),
        (datetime(TimeUnit.MICROSECOND), "TIMESTAMP(6)"),
        (datetime(TimeUnit.MICROSECOND, "UTC"), "TIMESTAMPTZ(6)"),
        (duration(TimeUnit.MICROSECOND), "INTERVAL"),
        (null(), "NULL"),
        (list_(string()), "ARRAY<VARCHAR>"),
        (map_(string(), int_(64)), "MAP<VARCHAR, BIGINT>"),
        (struct({"source": string(), "retries": int_(32)}), "STRUCT<source VARCHAR, retries INT>"),
        (enum(("a", "b", "c")), "ENUM('a', 'b', 'c')"),
    ],
)
def test_dtype_lowering_is_exact(ir_dtype: DType, spelling: str) -> None:
    """The SQL spelling is asserted, not just the round trip.

    A round trip can agree with itself while being wrong: `uint8` lowered to
    `BIGINT` would round-trip through a mapping that read BIGINT back as
    `uint8`. Writing the spelling down is what catches that.
    """
    assert from_ir(ir_dtype).sql() == spelling
    assert to_ir(from_ir(ir_dtype)) == ir_dtype


def test_unsigned_types_lower_without_a_dialect() -> None:
    """Built node by node, because `UTINYINT` is a parse error in generic SQL.

    The reason `from_ir` does not go through `exp.DataType.build`: an unsigned
    column would otherwise only lower in a dialect that happens to have the
    type, and the catalogue in the adapter docstring would stop being true for
    every other one.
    """
    with pytest.raises(sqlglot.errors.ParseError):
        sqlglot.exp.DataType.build("UTINYINT")
    assert to_ir(from_ir(uint(8))) == uint(8)


@pytest.mark.parametrize(
    ("spelling", "expected"),
    [
        ("varchar(100)", string()),
        ("numeric(18,2)", decimal(18, 2)),
        ("timestamp", datetime(TimeUnit.MICROSECOND)),
        ("timestamp with time zone", datetime(TimeUnit.MICROSECOND, "UTC")),
        ("timestamp(3)", datetime(TimeUnit.MILLISECOND)),
        ("timestamp(0)", datetime(TimeUnit.SECOND)),
        ("timestamp(9)", datetime(TimeUnit.NANOSECOND)),
        ("time(3)", time(TimeUnit.MILLISECOND)),
        ("interval", duration(TimeUnit.MICROSECOND)),
        ("array<int64>", list_(int_(64))),
        ("map<string, int>", map_(string(), int_(32))),
        ("enum('paid','refunded')", enum(("paid", "refunded"))),
        ("enum", string()),
    ],
)
def test_type_strings_map_onto_the_ir(spelling: str, expected: DType) -> None:
    from rowless_sql.dtypes import parse

    parsed = parse(spelling, "bigquery" if "int64" in spelling else "")
    assert parsed is not None
    assert to_ir(parsed) == expected


def test_a_named_zone_is_refused_rather_than_moved_to_utc() -> None:
    """The one declared absence, and the reason it is one."""
    assert ADAPTER.unsupported_dtypes() == (datetime(TimeUnit.NANOSECOND, "Europe/London"),)
    with pytest.raises(UnsupportedDType, match="Europe/London"):
        from_ir(datetime(TimeUnit.NANOSECOND, "Europe/London"))
    # UTC is an instant, which is exactly what SQL's aware timestamp stores.
    assert to_ir(from_ir(datetime(TimeUnit.NANOSECOND, "UTC"))) == datetime(
        TimeUnit.NANOSECOND, "UTC"
    )


@pytest.mark.parametrize(
    "refused",
    [
        duration(TimeUnit.MILLISECOND),
        float_(16),
        enum(()),
        unknown("nothing could resolve this"),
        any_(),
    ],
)
def test_dtypes_with_no_faithful_sql_type_are_refused(refused: DType) -> None:
    """Refused rather than approximated: a dtype that came back changed would
    be reported downstream as a change nobody made."""
    with pytest.raises(UnsupportedDType):
        from_ir(refused)


@pytest.mark.parametrize("spelling", ["json", "variant", "uuid", "geography", "timetz"])
def test_types_with_no_ir_equivalent_are_any(spelling: str) -> None:
    from rowless_sql.dtypes import parse

    parsed = parse(spelling, "snowflake")
    assert parsed is not None
    dtype = to_ir(parsed, spelling=spelling)

    assert dtype.is_any
    assert dtype.reason is not None and spelling in dtype.reason


@pytest.mark.parametrize(
    ("spelling", "fragment"),
    [("decimal", "declares no precision"), ("array", "does not say what it holds")],
)
def test_types_rowless_cannot_read_are_unknown(spelling: str, fragment: str) -> None:
    from rowless_sql.dtypes import parse

    parsed = parse(spelling)
    assert parsed is not None
    dtype = to_ir(parsed, spelling=spelling)

    assert dtype.kind is Kind.UNKNOWN
    assert dtype.reason is not None and fragment in dtype.reason


def test_to_ir_of_a_foreign_object_is_unknown() -> None:
    assert to_ir(object()).kind is Kind.UNKNOWN
    assert to_ir(None).kind is Kind.UNKNOWN


# --- The catalogue -----------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("stg_orders", "stg_orders"),
        ('"analytics"."staging"."stg_orders"', "stg_orders"),
        ("analytics.staging.stg_orders", "stg_orders"),
        ("`project`.`dataset`.`table`", "table"),
        ("[db].[schema].[table]", "table"),
    ],
)
def test_a_qualified_name_is_keyed_on_its_final_identifier(name: str, expected: str) -> None:
    assert catalogue.bare_name(name) == expected


def test_a_trailing_semicolon_is_dropped() -> None:
    assert catalogue.strip_terminator("SELECT 1;\n") == "SELECT 1"
    assert catalogue.strip_terminator("  SELECT 1  ") == "SELECT 1"


def test_two_inputs_of_the_same_name_stay_two_tables(ctx: DryRunContext) -> None:
    """Otherwise a join between them would silently become a self-join."""
    first = ADAPTER.materialize(fixtures.ORDERS, ctx)
    second = ADAPTER.materialize(fixtures.ORDERS, ctx)

    assert str(first) == "stg_orders"
    assert str(second) == "stg_orders_1"
    assert len(session.catalog()) == 2


def test_an_unnamed_schema_still_gets_a_usable_table(ctx: DryRunContext) -> None:
    table = ADAPTER.materialize(Schema.from_mapping({"a": int_(64)}), ctx)

    assert str(table) == catalogue.DEFAULT_TABLE
    assert columns_of(ADAPTER.describe(table)) == [("a", "int64")]


def test_a_colliding_run_sql_input_is_reported_not_resolved(ctx: DryRunContext) -> None:
    cat = catalogue.Catalog()
    columns = {"a": from_ir(int_(64))}
    assert cat.add("orders", Schema.from_mapping({"a": int_(64)}), columns) is None
    # The same schema under the same name is not a collision.
    assert cat.add("orders", Schema.from_mapping({"a": int_(64)}), columns) is None

    collision = cat.add("other.orders", Schema.from_mapping({"b": string()}), columns)
    assert collision is not None and "orders" in collision

    kept = cat.get("orders")
    assert kept is not None
    assert list(kept.schema) == ["a"]  # the first one is kept


def test_tables_read_leaves_ctes_out() -> None:
    """A CTE reference is an `exp.Table` too, and reporting one as a missing
    input would report a correct model as broken."""
    query = sqlglot.parse_one(
        "WITH paid AS (SELECT * FROM orders) SELECT * FROM paid JOIN db.sch.customers c ON TRUE"
    )
    assert catalogue.tables_read(query) == ("orders", "customers")


# --- Materialize -------------------------------------------------------------


def test_the_frame_is_a_real_table_that_renders_its_name(orders: Any) -> None:
    """What makes an f-string SQL model work without the pipeline knowing anything."""
    assert isinstance(orders, sqlglot.exp.Table)
    assert f"SELECT * FROM {orders}" == "SELECT * FROM stg_orders"


def test_an_unresolved_input_dtype_degrades_the_run(ctx: DryRunContext) -> None:
    schema = Schema(
        fields=(
            Field(name="amount", dtype=int_(64)),
            Field(name="mystery", dtype=unknown("the parquet footer said nothing")),
        )
    )
    relation = ADAPTER.describe(ADAPTER.materialize(schema, ctx))

    assert ctx.degradations and "mystery" in ctx.degradations[0]
    assert columns_of(relation) == [("amount", "int64"), ("mystery", "null")]


def test_an_any_input_is_a_note_not_a_degradation(ctx: DryRunContext) -> None:
    """ANY is the escape hatch: it lowers coverage and warns, never degrades."""
    schema = Schema(fields=(Field(name="payload", dtype=any_()),))
    ADAPTER.materialize(schema, ctx)

    assert ctx.degradations == []
    assert any("payload" in note for note in ctx.notes)


def test_an_input_dtype_sql_lacks_degrades_and_names_the_column(ctx: DryRunContext) -> None:
    schema = Schema(fields=(Field(name="when", dtype=datetime(TimeUnit.NANOSECOND, "Asia/Tokyo")),))
    ADAPTER.materialize(schema, ctx)

    assert ctx.degradations and "when" in ctx.degradations[0]
    assert "Asia/Tokyo" in ctx.degradations[0]


# --- Describe ----------------------------------------------------------------


def test_describing_an_input_reports_its_declared_nullability(orders: Any) -> None:
    """The input is the one thing this adapter can report nullability for."""
    relation = ADAPTER.describe(orders)

    assert relation.engine == "sql"
    assert relation.schema["order_id"].nullable is False
    assert relation.schema["status"].nullable is True
    assert relation.schema["order_id"].derived_from == ("order_id",)


def test_describing_a_query_reports_every_column_nullable(orders: Any, ctx: DryRunContext) -> None:
    relation = ADAPTER.describe(f"SELECT order_id FROM {orders}")

    assert relation.schema["order_id"].nullable is True
    assert any("nullability" in note for note in ctx.notes)


def test_describe_accepts_a_string_and_an_expression(orders: Any) -> None:
    from sqlglot import exp

    text = ADAPTER.describe(f"SELECT status, amount FROM {orders}")
    built = ADAPTER.describe(exp.select("status", "amount").from_(orders))

    assert columns_of(text) == columns_of(built) == [("status", "string"), ("amount", "int64")]


def test_describe_refuses_an_object_it_cannot_resolve(orders: Any) -> None:
    with pytest.raises(UnknownSchemaError, match="cannot describe"):
        ADAPTER.describe(object())


def test_describe_refuses_a_table_this_run_never_registered(ctx: DryRunContext) -> None:
    from sqlglot import exp

    with pytest.raises(UnknownSchemaError, match="no schema for the table"):
        ADAPTER.describe(exp.table_(exp.to_identifier("somebody_elses_table")))


def test_describe_does_not_rewrite_the_pipeline_s_expression(orders: Any) -> None:
    """Qualification rewrites in place, and the expression may be the user's."""
    from sqlglot import exp

    query = exp.select("status").from_(orders)
    before = query.sql()
    ADAPTER.describe(query)

    assert query.sql() == before


def test_nested_types_resolve_through_a_projection(events: Any) -> None:
    relation = ADAPTER.describe(f"SELECT payload.source AS source, tags FROM {events}")

    assert columns_of(relation) == [("source", "string"), ("tags", "list[string]")]


# --- Resolution --------------------------------------------------------------


def test_a_set_operation_unifies_both_sides(orders: Any) -> None:
    relation = ADAPTER.describe(
        f"SELECT CAST(order_id AS INT) AS id FROM {orders} "
        f"UNION ALL SELECT order_id AS id FROM {orders}"
    )

    assert columns_of(relation) == [("id", "int64")]


def test_a_set_operation_with_no_common_dtype_is_unresolved(
    orders: Any,
    ctx: DryRunContext,
) -> None:
    relation = ADAPTER.describe(
        f"SELECT status AS value FROM {orders} UNION ALL SELECT amount AS value FROM {orders}"
    )

    assert relation.schema["value"].dtype.kind is Kind.UNKNOWN
    assert ctx.degradations and "no common dtype" in ctx.degradations[0]


def test_a_set_operation_of_different_widths_is_reported(orders: Any, customers: Any) -> None:
    with pytest.raises(errors.SetOperationMismatch) as raised:
        ADAPTER.describe(
            f"SELECT order_id FROM {orders} UNION ALL SELECT customer_id, country FROM {customers}"
        )

    assert raised.value.code is Code.DTYPE_MISMATCH
    assert "1 column(s) on the left and 2 on the right" in str(raised.value)


def test_a_duplicate_output_name_is_refused(orders: Any) -> None:
    with pytest.raises(DuplicateColumn, match="two columns named 'amount'"):
        ADAPTER.describe(f"SELECT amount, amount FROM {orders}")


def test_an_unaliased_projection_is_refused(orders: Any) -> None:
    with pytest.raises(errors.UnnamedProjection) as raised:
        ADAPTER.describe(f"SELECT sum(amount) FROM {orders}")

    assert raised.value.code is Code.UNRESOLVED_SOURCE
    assert "Alias them" in str(raised.value)


def test_a_literal_keeps_the_name_the_query_gave_it(orders: Any) -> None:
    relation = ADAPTER.describe(f"SELECT order_id, 'web' AS source FROM {orders}")

    assert columns_of(relation) == [("order_id", "int64"), ("source", "string")]


def test_an_unknown_function_is_any_with_the_function_named(
    orders: Any,
    ctx: DryRunContext,
) -> None:
    relation = ADAPTER.describe(f"SELECT some_warehouse_udf(amount) AS score FROM {orders}")

    assert relation.schema["score"].dtype.is_any
    assert any("some_warehouse_udf" in note.lower() for note in ctx.notes)


def test_a_write_statement_is_refused(orders: Any) -> None:
    with pytest.raises(errors.StatementNotAQuery, match="INSERT"):
        ADAPTER.describe(f"INSERT INTO totals SELECT * FROM {orders}")
    with pytest.raises(errors.StatementNotAQuery, match="CREATE"):
        ADAPTER.describe(f"CREATE TABLE totals AS SELECT * FROM {orders}")


def test_an_empty_string_is_not_a_query(ctx: DryRunContext) -> None:
    with pytest.raises(errors.SqlNotParsable, match="empty"):
        resolve.parse("   ")


def test_a_table_nothing_registered_names_what_is_registered(orders: Any) -> None:
    with pytest.raises(errors.TableNotRegistered) as raised:
        ADAPTER.describe(f"SELECT nope.amount FROM nope, {orders}")

    assert "stg_orders" in str(raised.value)
    assert raised.value.code is Code.UNRESOLVED_SOURCE


# --- Error translation -------------------------------------------------------


def test_a_missing_column_carries_the_near_miss(orders: Any) -> None:
    with pytest.raises(ColumnNotFound) as raised:
        ADAPTER.describe(f"SELECT amount_cents FROM {orders}")

    assert raised.value.column == "amount_cents"
    assert raised.value.suggestion == "amount"
    assert raised.value.code is Code.COLUMN_NOT_FOUND


def test_an_ambiguous_column_names_the_tables_to_choose_between(
    orders: Any,
    customers: Any,
) -> None:
    with pytest.raises(errors.AmbiguousColumn) as raised:
        ADAPTER.describe(
            f"SELECT created_at FROM {orders} JOIN {customers} ON order_id = customer_id"
        )

    message = str(raised.value)
    assert raised.value.code is Code.AMBIGUOUS_COLUMN
    assert "stg_orders" in message and "stg_customers" in message
    assert "Qualify" in message


@pytest.mark.parametrize(
    "message",
    [
        "Column 'created_at' could not be resolved. Line: 1, Col: 8",
        "Column '\"created_at\"' could not be resolved",
        "Ambiguous column 'created_at' (Line: 1, Col: 8)",
    ],
)
def test_both_message_shapes_are_read_as_ambiguity(message: str) -> None:
    """The oldest supported sqlglot reports ambiguity as an unresolved column."""
    translated = errors.translate(
        sqlglot.errors.OptimizeError(message),
        ambiguous={"created_at": ("orders", "customers")},
    )

    assert isinstance(translated, errors.AmbiguousColumn)


def test_an_unresolved_column_is_not_ambiguity_when_only_one_table_has_it() -> None:
    translated = errors.translate(
        sqlglot.errors.OptimizeError("Column 'amount_cents' could not be resolved"),
        available=("amount", "status"),
    )

    assert isinstance(translated, ColumnNotFound)
    assert translated.suggestion == "amount"


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Unknown table: orders", errors.TableNotRegistered),
        ("Unknown column: amount_cents", ColumnNotFound),
        ("Unknown output column: total", ColumnNotFound),
        ("Cannot resolve 'amount' due to missing source schemas", errors.TableNotRegistered),
        ("Invalid set operation due to column mismatch: SELECT 1", errors.SetOperationMismatch),
        ("Cannot automatically join: created_at", errors.SqlNotParsable),
    ],
)
def test_optimizer_messages_are_classified(message: str, expected: type[RowlessError]) -> None:
    assert isinstance(errors.translate(sqlglot.errors.OptimizeError(message)), expected)


def test_a_parse_failure_names_the_dialect_to_set() -> None:
    """A dialect's own syntax fails this way, so the message has to say so."""
    generic = errors.translate(sqlglot.errors.ParseError("Invalid expression"))
    named = errors.translate(sqlglot.errors.ParseError("Invalid expression"), dialect="snowflake")

    assert isinstance(generic, errors.SqlNotParsable)
    assert "[engine.sql] dialect" in str(generic)
    assert isinstance(named, errors.SqlNotParsable)
    assert "snowflake" in str(named)


def test_terminal_highlighting_is_stripped_from_a_finding() -> None:
    translated = errors.translate(
        sqlglot.errors.ParseError("Invalid expression.\n  SELECT \x1b[4mFROM\x1b[0m")
    )

    assert translated is not None
    assert "\x1b" not in str(translated)


def test_translate_leaves_what_it_cannot_classify_alone() -> None:
    assert errors.translate(ValueError("something else entirely")) is None
    assert errors.translate(sqlglot.errors.OptimizeError("a message nobody has seen")) is None
    assert errors.translate(ColumnNotFound("amount")) is None


def test_is_sqlglot_error_goes_by_hierarchy_and_module() -> None:
    assert errors.is_sqlglot_error(sqlglot.errors.ParseError("x")) is True
    assert errors.is_sqlglot_error(ValueError("x")) is False


def test_raise_translated_passes_through_what_it_cannot_classify() -> None:
    with pytest.raises(ValueError):
        errors.raise_translated(ValueError("not mine"))


# --- Dialects ----------------------------------------------------------------


def test_an_explicit_dialect_wins() -> None:
    assert dialects.for_options({"dialect": "snowflake"}) == "snowflake"


def test_an_unknown_dialect_is_a_configuration_error() -> None:
    """Falling back to generic SQL after a typo would turn one wrong setting
    into a pile of parse failures a hundred lines away."""
    with pytest.raises(ConfigError) as raised:
        dialects.for_options({"dialect": "snowflak"})

    assert raised.value.code is Code.CONFIG_INVALID
    assert "snowflake" in str(raised.value)


def test_no_dialect_configured_reads_as_generic_sql(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    dialects.forget_dbt()

    assert dialects.for_options({}) == ""


def test_the_dialect_changes_how_a_query_is_read(ctx: DryRunContext) -> None:
    """A dialect is configuration, and this is what it buys.

    Snowflake's `::` cast and its folding of unquoted identifiers to upper case
    are both dialect behaviour, so the same query read as generic SQL and as
    Snowflake produces differently named columns.
    """
    snowflake = DryRunContext(
        run_id="dialect", workdir=ctx.workdir, options={"dialect": "snowflake"}
    )
    session.begin(snowflake)
    table = ADAPTER.materialize(fixtures.ORDERS, snowflake)
    relation = ADAPTER.describe(f"SELECT amount::DOUBLE AS amt FROM {table}")

    assert columns_of(relation) == [("AMT", "float64")]


# --- Identifier folding ------------------------------------------------------


def test_a_folded_input_column_keeps_its_declared_spelling(ctx: DryRunContext) -> None:
    """Otherwise a diff would report a rename that nobody made."""
    table = ADAPTER.materialize(MIXED_CASE, ctx)
    relation = ADAPTER.describe(f"SELECT * FROM {table}")

    assert [field.name for field in relation.schema.fields] == ["CustomerID", "Country"]
    assert any("folds unquoted identifiers" in note for note in ctx.notes)


def test_an_alias_the_query_wrote_keeps_the_fold(ctx: DryRunContext) -> None:
    """That really is what the warehouse will call the column."""
    table = ADAPTER.materialize(MIXED_CASE, ctx)
    relation = ADAPTER.describe(f"SELECT CustomerID AS CustomerKey FROM {table}")

    assert [field.name for field in relation.schema.fields] == ["customerkey"]


def test_lineage_parents_use_the_declared_spelling(ctx: DryRunContext) -> None:
    table = ADAPTER.materialize(MIXED_CASE, ctx)
    traced = ADAPTER.column_lineage(f"SELECT CustomerID AS CustomerKey FROM {table}")

    assert traced == {"customerkey": ("CustomerID",)}


# --- Lineage -----------------------------------------------------------------


def test_lineage_survives_a_cte(orders: Any) -> None:
    """The case the DuckDB adapter refuses on purpose rather than guess at."""
    traced = ADAPTER.column_lineage(
        f"WITH paid AS (SELECT * FROM {orders} WHERE status = 'paid') "
        "SELECT status, sum(amount) AS total FROM paid GROUP BY status"
    )

    assert traced == {"status": ("status",), "total": ("amount",)}


def test_lineage_survives_a_subquery(orders: Any) -> None:
    traced = ADAPTER.column_lineage(
        f"SELECT inner_q.amount AS amount FROM (SELECT amount FROM {orders}) AS inner_q"
    )

    assert traced == {"amount": ("amount",)}


def test_a_literal_column_is_untraced(orders: Any) -> None:
    """Untraced rather than attributed to whatever the query happened to read."""
    traced = ADAPTER.column_lineage(f"SELECT order_id, 'web' AS source FROM {orders}")

    assert traced == {"order_id": ("order_id",)}


def test_a_predicate_is_not_a_parent(orders: Any) -> None:
    """`WHERE status = 'paid'` decides which rows survive, not what a column holds."""
    traced = ADAPTER.column_lineage(f"SELECT amount FROM {orders} WHERE status = 'paid'")

    assert traced == {"amount": ("amount",)}


def test_a_union_column_comes_from_both_sides(orders: Any, customers: Any) -> None:
    traced = ADAPTER.column_lineage(
        f"SELECT order_id FROM {orders} UNION ALL SELECT customer_id FROM {customers}"
    )

    assert traced == {"order_id": ("order_id", "customer_id")}


def test_an_input_table_is_its_own_lineage(orders: Any) -> None:
    traced = ADAPTER.column_lineage(orders)

    assert traced == {name: (name,) for name in fixtures.ORDERS}


def test_lineage_of_an_unregistered_table_is_empty(ctx: DryRunContext) -> None:
    from sqlglot import exp

    assert ADAPTER.column_lineage(exp.table_(exp.to_identifier("elsewhere"))) == {}
    assert ADAPTER.column_lineage(object()) == {}


def test_lineage_reuses_the_resolution_describe_already_did(orders: Any) -> None:
    """One qualification per pipeline, not two: the runtime calls both."""
    query = f"SELECT status FROM {orders}"
    relation = ADAPTER.describe(query)
    traced = ADAPTER.column_lineage(query)

    assert columns_of(relation) == [("status", "string")]
    assert traced == {"status": ("status",)}


def test_lineage_of_a_query_that_cannot_be_scoped_is_empty(orders: Any) -> None:
    assert lineage.of_resolved(object(), ["a"], schema=session.catalog().mapping()) == {}


# --- The SQL seam ------------------------------------------------------------


def test_run_sql_resolves_a_dbt_shaped_relation(orders: Any, ctx: DryRunContext) -> None:
    """dbt offers every spelling of a node's name; any one of them resolves."""
    relation = ADAPTER.run_sql(
        'SELECT amount FROM "analytics"."staging"."stg_orders";',
        {'"analytics"."staging"."stg_orders"': orders},
        ctx,
    )

    assert columns_of(relation) == [("amount", "int64")]
    assert relation.schema["amount"].derived_from == ("amount",)


def test_run_sql_accepts_a_schema_without_a_frame(ctx: DryRunContext) -> None:
    """`rowless-dbt` has a schema per upstream model and no reason to build a table."""
    relation = ADAPTER.run_sql(
        "SELECT status, count(*) AS n FROM stg_orders GROUP BY status",
        {"stg_orders": fixtures.ORDERS},
        ctx,
    )

    assert columns_of(relation) == [("status", "string"), ("n", "int64")]


def test_run_sql_traces_columns_through_the_model(ctx: DryRunContext) -> None:
    relation = ADAPTER.run_sql(
        "WITH paid AS (SELECT * FROM stg_orders WHERE status = 'paid') "
        "SELECT sum(amount) AS total FROM paid",
        {"stg_orders": fixtures.ORDERS},
        ctx,
    )

    assert relation.schema["total"].derived_from == ("amount",)


def test_run_sql_reports_a_name_collision_rather_than_resolving_it(ctx: DryRunContext) -> None:
    other = Schema.from_mapping({"something_else": string()}, name="stg_orders")
    with pytest.raises(RowlessError):
        ADAPTER.run_sql(
            "SELECT something_else FROM stg_orders",
            {"stg_orders": fixtures.ORDERS, "other.stg_orders": other},
            ctx,
        )

    assert ctx.degradations and "stg_orders" in ctx.degradations[0]


def test_run_sql_refuses_an_input_it_cannot_read(ctx: DryRunContext) -> None:
    with pytest.raises(UnknownSchemaError, match="cannot register"):
        ADAPTER.run_sql("SELECT 1 AS one FROM t", {"t": object()}, ctx)


def test_run_sql_refuses_a_table_from_another_run(ctx: DryRunContext) -> None:
    from sqlglot import exp

    with pytest.raises(UnknownSchemaError, match="no schema for the table"):
        ADAPTER.run_sql(
            "SELECT 1 AS one FROM t",
            {"t": exp.table_(exp.to_identifier("t"))},
            ctx,
        )


def test_run_sql_notes_an_untraceable_query(ctx: DryRunContext) -> None:
    relation = ADAPTER.run_sql(
        "SELECT 'fixed' AS source FROM stg_orders",
        {"stg_orders": fixtures.ORDERS},
        ctx,
    )

    assert relation.schema["source"].derived_from == ()
    assert any("untraced" in note for note in ctx.notes)


# --- End to end --------------------------------------------------------------


def test_a_sql_pipeline_resolves_through_the_runtime() -> None:
    """The path `rowless check` takes, with no database and no credentials."""

    def daily_revenue(orders: Any) -> str:
        return f"""
            SELECT CAST(created_at AS DATE) AS day, sum(amount) AS total
            FROM {orders}
            WHERE status = 'paid'
            GROUP BY 1
        """

    result = dry_run_callable(daily_revenue, {"orders": fixtures.ORDERS}, engine="sql")

    assert result.output is not None
    assert columns_of(result.output) == [("day", "date"), ("total", "int64")]
    assert result.coverage.confidence is Confidence.EXACT
    assert result.coverage.traced_ratio == 1.0
    assert result.ok is True


def test_the_renamed_column_is_the_finding() -> None:
    """The proposal's example, on SQL instead of Polars."""

    def daily_revenue(orders: Any) -> str:
        return f"SELECT sum(amount) AS total FROM {orders}"

    renamed = Schema(
        fields=tuple(
            Field(name="amount_cents" if field.name == "amount" else field.name, dtype=field.dtype)
            for field in fixtures.ORDERS.fields
        ),
        name="stg_orders",
    )
    result = dry_run_callable(daily_revenue, {"orders": renamed}, engine="sql")

    assert result.output is None
    assert [diagnostic.code for diagnostic in result.diagnostics] == [Code.COLUMN_NOT_FOUND]
    assert "amount_cents" in (result.diagnostics[0].suggestion or "")


def test_nothing_in_a_dry_run_touches_the_filesystem_or_network() -> None:
    """The property that makes this adapter usable in a credential-free CI job.

    The conformance suite asserts it for `describe`; this asserts it for the
    whole run, including parsing and lineage, because "no database" is the
    claim the adapter is chosen for. The temp directory is allowed because the
    runtime creates the run's own scratch directory inside the sandbox; what
    is being asserted is that nothing opens a socket and nothing writes
    anywhere else.
    """
    import tempfile

    from rowless.runtime.sandbox import Policy, sandboxed

    with sandboxed(Policy.strict().allowing_writes_to(tempfile.gettempdir())) as log:
        result = dry_run_callable(
            lambda orders: f"SELECT status FROM {orders}",
            {"orders": fixtures.ORDERS},
            engine="sql",
        )

    assert result.output is not None
    assert log.network == [] and log.blocked == []


# --- The one mapping --------------------------------------------------------


def test_rowless_dbt_delegates_its_type_mapping_here() -> None:
    """One sqlglot-to-IR mapping in the workspace, not two.

    `rowless_dbt.dtypes` parses a manifest's type strings and hands the parsed
    type to this adapter's `to_ir`. Asserted rather than assumed, because the
    duplicate it replaced is exactly the kind of thing that drifts.
    """
    dbt_dtypes = pytest.importorskip(
        "rowless_dbt.dtypes", reason="the delegation only exists when rowless-dbt is installed"
    )
    from rowless_sql.dtypes import parse

    for spelling, dialect in [
        ("numeric(18,2)", "postgres"),
        ("timestamp with time zone", "postgres"),
        ("number", "snowflake"),
        ("array<int64>", "bigquery"),
        ("json", ""),
    ]:
        parsed = parse(spelling, dialect)
        assert parsed is not None
        assert dbt_dtypes.dtype_from_sql(spelling, dialect=dialect) == to_ir(
            parsed, spelling=spelling
        )
