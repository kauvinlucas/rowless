"""Unit tests for the parts of the adapter the conformance suite cannot reach.

The suite in core is the specification, and it covers the contract. What it
cannot cover is the behaviour that is specific to this engine: which ibis
exceptions become which findings, which dtypes it refuses rather than
approximates, whether the SQL seam resolves a dbt-style relation name with no
database behind it, whether the lineage read off the expression tree is the
lineage a user would expect, and whether the portability findings - the thing
no other adapter here can produce - actually name the backends that disagree.
That is what is here.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import ibis
import ibis.expr.datatypes as dt
import pytest
from rowless.adapters.base import (
    DryRunContext,
    SupportsColumnLineage,
    SupportsSql,
    SupportsUdfProbe,
)
from rowless.diagnostics.codes import Code
from rowless.errors import (
    ColumnNotFound,
    DTypeMismatch,
    DuplicateColumn,
    SchemaNotFound,
    UnknownSchemaError,
    UnsupportedDType,
)
from rowless.ir.confidence import Confidence
from rowless.ir.dtypes import (
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
    string,
    struct,
    time,
    uint,
    unknown,
)
from rowless.ir.schema import Field, Schema
from rowless.runtime import rowdep
from rowless.runtime.dryrun import dry_run_callable
from rowless.runtime.rowdep import RowDependencyLog
from rowless.testing import fixtures
from rowless_ibis import errors, lineage, portability, session
from rowless_ibis import sql as sql_seam
from rowless_ibis.adapter import MINIMUM_IBIS, IbisAdapter, _at_least
from rowless_ibis.dtypes import from_ir, is_nullable, to_ir, with_nullability

pytestmark = pytest.mark.requires_engine

ADAPTER = IbisAdapter()

if not ADAPTER.available():
    # Below the declared floor these tests would fail rather than say anything:
    # the adapter supports ibis >= 10.0, and CI runs this suite against exactly
    # that release as well as the current one. A skip is reported as a skip.
    pytest.skip(
        f"rowless-ibis needs {IbisAdapter.requires}",
        allow_module_level=True,
    )

TIMESTAMP = datetime(TimeUnit.MICROSECOND, "UTC")


@pytest.fixture
def ctx(tmp_path: Path) -> DryRunContext:
    """A fresh dry-run context, which also resets this adapter's per-run state."""
    context = DryRunContext(run_id=f"test-{tmp_path.name}", workdir=str(tmp_path))
    session.begin(context)
    return context


@pytest.fixture
def tracked(ctx: DryRunContext) -> Iterator[RowDependencyLog]:
    """An open dry run, so the wrappers in `interception` are live.

    They are gated on one deliberately: outside a dry run ibis behaves
    normally, which is what lets a process import this adapter without its own
    ibis usage changing. Anything testing translation, row dependence or the
    inline SQL seam has to be inside a run, the same way a real check is.
    """
    with rowdep.recording() as log:
        yield log


@pytest.fixture
def orders(ctx: DryRunContext, tracked: RowDependencyLog) -> Any:
    """The proposal's orders relation, materialized as an unbound ibis table."""
    return ADAPTER.materialize(fixtures.ORDERS, ctx)


@pytest.fixture
def customers(ctx: DryRunContext, tracked: RowDependencyLog) -> Any:
    """The join partner, with its deliberate `created_at` collision."""
    return ADAPTER.materialize(fixtures.CUSTOMERS, ctx)


@pytest.fixture
def events(ctx: DryRunContext, tracked: RowDependencyLog) -> Any:
    """The nested-types relation."""
    return ADAPTER.materialize(fixtures.EVENTS, ctx)


# --- Availability and routing ------------------------------------------------


def test_available_with_ibis_installed() -> None:
    assert ADAPTER.available() is True


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("9.0.0", False),
        ("9.5.0", False),
        ("10.0.0", True),
        ("11.0.0", True),
        ("12.0.0", True),
        ("13.0.0.dev7", True),
        ("", False),
    ],
)
def test_version_floor(version: str, expected: bool) -> None:
    """The floor is 10.0, and CI runs this suite against it."""
    assert _at_least(version, MINIMUM_IBIS) is expected


def test_handles_tables_columns_and_scalars(orders: Any) -> None:
    assert ADAPTER.handles(orders) is True
    assert ADAPTER.handles(orders.amount) is True
    assert ADAPTER.handles(orders.amount.sum()) is True
    assert ADAPTER.handles(object()) is False
    assert ADAPTER.handles(fixtures.ORDERS) is False


# --- Dtypes ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ir_dtype", "native"),
    [
        (boolean(), dt.Boolean()),
        (int_(8), dt.Int8()),
        (int_(64), dt.Int64()),
        (uint(32), dt.UInt32()),
        (float_(16), dt.Float16()),
        (float_(64), dt.Float64()),
        (decimal(18, 2), dt.Decimal(18, 2)),
        (string(), dt.String()),
        (binary(), dt.Binary()),
        (date(), dt.Date()),
        (time(), dt.Time()),
        (datetime(TimeUnit.MILLISECOND), dt.Timestamp(scale=3)),
        (datetime(TimeUnit.MICROSECOND, "UTC"), dt.Timestamp(timezone="UTC", scale=6)),
        (
            datetime(TimeUnit.NANOSECOND, "Europe/London"),
            dt.Timestamp(timezone="Europe/London", scale=9),
        ),
        (duration(TimeUnit.MICROSECOND), dt.Interval(unit="us")),
        (list_(string()), dt.Array(dt.String())),
        (map_(string(), int_(64)), dt.Map(dt.String(), dt.Int64())),
        (struct({"source": string()}), dt.Struct({"source": dt.String()})),
    ],
)
def test_dtype_lowering_is_exact(ir_dtype: Any, native: Any) -> None:
    """`from_ir` produces the ibis dtype, not merely something compatible."""
    assert from_ir(ir_dtype) == native
    assert to_ir(native) == ir_dtype


def test_float16_exists_here_and_not_in_polars() -> None:
    """Half precision is a real ibis dtype, which Polars has no equivalent for."""
    assert to_ir(from_ir(float_(16))) == float_(16)


def test_enum_is_refused_because_ibis_enum_is_a_string() -> None:
    """`dt.Enum` is an alias for `String`, so the variant set would be lost."""
    assert dt.Enum is dt.String
    with pytest.raises(UnsupportedDType, match="alias for String"):
        from_ir(enum(("a", "b", "c")))
    assert enum(("a", "b", "c")) in ADAPTER.unsupported_dtypes()


def test_a_string_column_does_not_read_back_as_an_enum() -> None:
    """The other half of the enum story: nothing is promoted to a variant set."""
    assert to_ir(dt.String()) == string()


@pytest.mark.parametrize("unit", [TimeUnit.SECOND, TimeUnit.MILLISECOND, TimeUnit.NANOSECOND])
def test_time_is_microseconds_only(unit: TimeUnit) -> None:
    """Ibis has one unparameterised `Time`, so every other resolution is refused.

    Lowering `time[ms]` would read back as `time[us]` and be reported
    downstream as a dtype change nobody made.
    """
    with pytest.raises(UnsupportedDType, match="unparameterised Time"):
        from_ir(time(unit))


@pytest.mark.parametrize(
    ("scale", "expected"),
    [
        (0, datetime(TimeUnit.SECOND)),
        (3, datetime(TimeUnit.MILLISECOND)),
        (6, datetime(TimeUnit.MICROSECOND)),
        (9, datetime(TimeUnit.NANOSECOND)),
    ],
)
def test_timestamp_scale_is_read_as_a_time_unit(scale: int, expected: Any) -> None:
    assert to_ir(dt.Timestamp(scale=scale)) == expected


def test_timestamp_without_a_scale_is_read_as_microseconds() -> None:
    """`scale=None` means "whatever the backend does", and they do microseconds."""
    assert to_ir(dt.Timestamp()) == datetime(TimeUnit.MICROSECOND)
    assert to_ir(dt.Timestamp(timezone="UTC")) == datetime(TimeUnit.MICROSECOND, "UTC")


def test_a_scale_the_ir_cannot_spell_is_unknown() -> None:
    """Ibis allows `timestamp(4)`; the IR has no unit for it, so it is not guessed."""
    resolved = to_ir(dt.Timestamp(scale=4))
    assert resolved.kind is Kind.UNKNOWN
    assert resolved.reason is not None and "scale 4" in resolved.reason


def test_a_naive_timestamp_is_not_utc() -> None:
    """The distinction the IR exists to keep: naive and zoned are different values."""
    assert to_ir(dt.Timestamp(scale=6)).tz is None
    assert to_ir(dt.Timestamp(timezone="UTC", scale=6)).tz == "UTC"


@pytest.mark.parametrize("unit", ["Y", "Q", "M", "W", "D", "h", "m"])
def test_calendar_intervals_are_not_durations(unit: str) -> None:
    """A month is not a fixed length of time, so it is reported unresolved."""
    resolved = to_ir(dt.Interval(unit=unit))
    assert resolved.kind is Kind.UNKNOWN
    assert resolved.reason is not None and "calendar arithmetic" in resolved.reason


def test_128_bit_integers_are_refused() -> None:
    """Ibis tops out at 64 bits, and narrowing would overflow in production only."""
    with pytest.raises(UnsupportedDType, match="128-bit integer"):
        from_ir(int_(128))
    with pytest.raises(UnsupportedDType, match="128-bit integer"):
        from_ir(uint(128))


def test_decimal_needs_both_parameters() -> None:
    with pytest.raises(UnsupportedDType, match="precision and a scale"):
        from_ir(decimal(10, 2).with_(precision=None))
    resolved = to_ir(dt.Decimal())
    assert resolved.kind is Kind.UNKNOWN
    assert resolved.reason is not None and "no precision" in resolved.reason


@pytest.mark.parametrize(
    "native",
    [dt.JSON(), dt.UUID(), dt.INET(), dt.MACADDR(), dt.Point(), dt.Geometry()],
)
def test_types_with_no_ir_equivalent_are_unknown_not_strings(native: Any) -> None:
    """Reporting a UUID as a string would make a contract comparing them pass."""
    assert to_ir(native).kind is Kind.UNKNOWN


def test_unknown_is_refused_and_foreign_objects_are_unknown() -> None:
    with pytest.raises(UnsupportedDType):
        from_ir(unknown("nothing resolved this"))
    assert to_ir(object()).kind is Kind.UNKNOWN
    assert to_ir(dt.Unknown()).kind is Kind.UNKNOWN


def test_any_lowers_to_the_one_dtype_that_casts_into_everything() -> None:
    """The gradual-typing top type; an ANY column is already counted against coverage."""
    assert from_ir(any_()) == dt.Null()


def test_dtype_strings_are_accepted_as_well_as_instances() -> None:
    """A pipeline may have declared its table with the string shorthand."""
    assert to_ir("int32") == int_(32)
    assert to_ir("array<string>") == list_(string())
    assert to_ir("not a dtype at all").kind is Kind.UNKNOWN


def test_nullability_helpers_round_trip() -> None:
    assert is_nullable(dt.Int64()) is True
    assert is_nullable(with_nullability(dt.Int64(), nullable=False)) is False
    # A dtype that cannot carry nullability is reported nullable rather than
    # failing the materialize.
    assert is_nullable(with_nullability(object(), nullable=False)) is True


# --- Materialize -------------------------------------------------------------


def test_materialize_carries_nullability(ctx: DryRunContext) -> None:
    """Ibis keeps nullability on the dtype, so the declared `NOT NULL` survives."""
    table = ADAPTER.materialize(fixtures.ORDERS, ctx)
    assert table.schema()["order_id"].nullable is False
    assert table.schema()["status"].nullable is True


def test_materialize_is_unbound(ctx: DryRunContext) -> None:
    """The whole premise: a real ibis table with a schema and no backend."""
    import ibis.expr.operations as ops

    table = ADAPTER.materialize(fixtures.ORDERS, ctx)
    assert isinstance(table.op(), ops.UnboundTable)
    assert table.op().name == "stg_orders"


def test_two_inputs_with_the_same_name_get_separate_tables(ctx: DryRunContext) -> None:
    """Ibis interns its nodes, so identical names would collapse into one table.

    A join between two inputs that resolved from identically named sources
    would then silently be a self-join.
    """
    first = ADAPTER.materialize(fixtures.ORDERS, ctx)
    second = ADAPTER.materialize(fixtures.ORDERS, ctx)
    assert first.op() != second.op()
    assert second.op().name == "stg_orders_1"


def test_an_unresolved_input_dtype_degrades_the_run(ctx: DryRunContext) -> None:
    """One unresolved source must not cost the other fifteen columns."""
    schema = Schema(
        fields=(
            Field(name="good", dtype=int_(64)),
            Field(name="bad", dtype=unknown("the source could not be resolved")),
        ),
        name="partly_unresolved",
    )
    table = ADAPTER.materialize(schema, ctx)
    assert table.schema()["good"] == dt.Int64()
    assert table.schema()["bad"] == dt.Null()
    assert any("'bad'" in reason for reason in ctx.degradations)
    assert not any("'good'" in reason for reason in ctx.degradations)


def test_an_any_input_is_a_note_not_a_degradation(ctx: DryRunContext) -> None:
    """ANY is a deliberate escape hatch, so it lowers coverage without failing."""
    schema = Schema(fields=(Field(name="loose", dtype=any_()),), name="loose")
    ADAPTER.materialize(schema, ctx)
    assert any("'loose'" in note for note in ctx.notes)
    assert ctx.degradations == []


def test_an_input_dtype_ibis_lacks_degrades_and_names_the_column(ctx: DryRunContext) -> None:
    """The enum case, which is the one the fixtures actually contain."""
    ADAPTER.materialize(fixtures.DRIFT_PRONE, ctx)
    assert any("'category'" in reason for reason in ctx.degradations)


# --- Describe ----------------------------------------------------------------


def test_describe_reports_nullability_from_the_engine(orders: Any) -> None:
    """Of the engines rowless targets, only ibis propagates this through a filter."""
    relation = ADAPTER.describe(orders.filter(orders.amount > 0))
    assert relation.engine == "ibis"
    assert relation.schema["order_id"].nullable is False
    assert relation.schema["amount"].nullable is True


def test_describe_a_column_expression(orders: Any) -> None:
    relation = ADAPTER.describe(orders.amount)
    assert tuple(relation.schema) == ("amount",)
    assert relation.schema["amount"].dtype == int_(64)


def test_describe_a_scalar_names_it_after_the_operation(orders: Any) -> None:
    """`Sum(amount)` is the name a user would see in the result."""
    relation = ADAPTER.describe(orders.amount.sum())
    assert tuple(relation.schema) == ("Sum(amount)",)


def test_describe_refuses_a_foreign_object(orders: Any) -> None:
    with pytest.raises(UnknownSchemaError, match="cannot describe"):
        ADAPTER.describe(object())


def test_describe_nested_types(events: Any) -> None:
    relation = ADAPTER.describe(events)
    assert relation.schema["payload"].dtype == struct({"source": string(), "retries": int_(32)})
    assert relation.schema["tags"].dtype == list_(string())
    # Unlike Polars, ibis has a real Map, so this is not the Arrow encoding.
    assert relation.schema["attrs"].dtype == map_(string(), string())


# --- Error translation -------------------------------------------------------


def test_missing_column_carries_the_near_miss(orders: Any) -> None:
    """The flagship finding, from the proposal's rename case."""
    with pytest.raises(ColumnNotFound) as caught:
        orders.select("amount_cents")
    assert caught.value.column == "amount_cents"
    assert caught.value.suggestion == "amount"
    assert "amount_cents" in str(caught.value)


def test_missing_column_through_attribute_access_is_both_errors(orders: Any) -> None:
    """`orders.amount_cents` is the idiomatic spelling, and it must stay an AttributeError."""
    with pytest.raises(ColumnNotFound) as caught:
        _ = orders.amount_cents
    assert isinstance(caught.value, AttributeError)
    assert caught.value.suggestion == "amount"


def test_attribute_probes_still_behave(orders: Any) -> None:
    """Keeping the `AttributeError` half is what makes this safe to do at all."""
    assert hasattr(orders, "nope") is False
    assert getattr(orders, "nope", "fallback") == "fallback"
    assert hasattr(orders, "amount") is True


def test_a_private_attribute_is_not_a_column(orders: Any) -> None:
    """A library probing the object is not a pipeline reading a column."""
    with pytest.raises(AttributeError) as caught:
        _ = orders._not_a_column
    assert not isinstance(caught.value, ColumnNotFound)


def test_a_mistyped_method_keeps_ibis_own_hint(orders: Any) -> None:
    """Ibis suggests `order_by` for `sort`, which beats anything a column list can say."""
    with pytest.raises(AttributeError, match="order_by") as caught:
        _ = orders.sort
    assert not isinstance(caught.value, ColumnNotFound)


def test_duplicate_output_name_is_reported(orders: Any) -> None:
    with pytest.raises(DuplicateColumn, match="amount"):
        orders.select(orders.amount, orders.amount)


def test_a_mixed_type_expression_is_a_dtype_mismatch(orders: Any) -> None:
    """The operator itself is translated: this raises before `select` is called."""
    with pytest.raises(DTypeMismatch) as caught:
        orders.select(mixed=orders.amount + orders.status)
    message = str(caught.value)
    assert "StringConcat" in message
    # The plan dump ibis embeds in the message is stripped; the complaint is not.
    assert "UnboundTable" not in message
    assert len(message.splitlines()) == 1


def test_a_non_boolean_predicate_is_a_dtype_mismatch(orders: Any) -> None:
    with pytest.raises(DTypeMismatch):
        orders.filter(orders.amount)


def test_a_mismatched_set_operation_is_reported(orders: Any, customers: Any) -> None:
    """Ibis refuses to union two different schemas, where DuckDB coerces them.

    How much detail the message carries is version-dependent - the releases at
    the top of the supported range list the columns missing from each side, the
    floor says only that the schemas differ - so the assertion is on the
    finding rather than on the prose.
    """
    with pytest.raises(DTypeMismatch, match="set operations"):
        orders.union(customers)


def test_translate_leaves_what_it_cannot_classify_alone() -> None:
    """Mislabelling an unrecognised failure is worse than letting core say so."""
    assert errors.translate(ValueError("something else entirely")) is None
    assert errors.translate(ColumnNotFound("already translated")) is None


def test_translate_recognises_an_unbound_execution() -> None:
    import ibis.common.exceptions as ibis_errors

    translated = errors.translate(
        ibis_errors.IbisError("Expression contains unbound tables and cannot be executed")
    )
    assert isinstance(translated, errors.ExecutionAttempted)
    assert translated.code is Code.ROW_DEPENDENT_CONTROL_FLOW


def test_translate_a_table_not_found() -> None:
    import ibis.common.exceptions as ibis_errors

    assert isinstance(
        errors.translate(ibis_errors.TableNotFound("orders")),
        SchemaNotFound,
    )


def test_translate_a_sqlglot_unresolved_column() -> None:
    """The SQL seam fails with sqlglot's exceptions, not with ibis's."""
    from sqlglot.errors import OptimizeError

    translated = errors.translate_sql_error(
        OptimizeError("Column 'amount' could not be resolved. Line: 1, Col: 8"),
        available=("amount_cents", "order_id"),
    )
    assert isinstance(translated, ColumnNotFound)
    assert translated.suggestion == "amount_cents"


def test_a_sql_syntax_error_is_not_blamed_on_the_pipeline() -> None:
    """A dialect's own syntax fails to parse the same way valid SQL does."""
    from sqlglot.errors import ParseError

    assert errors.translate_sql_error(ParseError("no idea")) is None


def test_fallback_available_columns_are_used_when_ibis_lists_none() -> None:
    import ibis.common.exceptions as ibis_errors

    translated = errors.translate(
        ibis_errors.IbisTypeError("Column 'amount' is not found in table."),
        relation="stg_orders",
        available=("amount_cents",),
    )
    assert isinstance(translated, ColumnNotFound)
    assert translated.suggestion == "amount_cents"
    assert translated.relation == "stg_orders"


# --- Interception ------------------------------------------------------------


def test_reading_values_is_reported_as_row_dependence(
    orders: Any, tracked: RowDependencyLog
) -> None:
    """There is no data to hand back, so the call is refused rather than faked."""
    with pytest.raises(errors.ExecutionAttempted):
        orders.to_pandas()
    assert [entry.api for entry in tracked.observed] == ["ibis.Table.to_pandas"]
    assert tracked.taints_result is True


@pytest.mark.parametrize(
    "read",
    [
        lambda table: table.execute(),
        lambda table: table.to_pyarrow(),
        lambda table: table.cache(),
        lambda table: table.amount.to_list(),
        lambda table: table.amount.sum().to_pyarrow(),
    ],
)
def test_every_executor_is_covered(orders: Any, read: Any) -> None:
    with pytest.raises(errors.ExecutionAttempted):
        read(orders)


def test_expression_level_aggregates_cost_nothing(orders: Any, tracked: RowDependencyLog) -> None:
    """The ibis difference: `amount.max()` builds a query rather than reading a value.

    On Polars and pandas the same spelling reads data and costs all of the
    confidence, which is why far fewer ibis pipelines come out unverified.
    """
    assert ADAPTER.describe(orders.amount.max()).schema["Max(amount)"].dtype == int_(64)
    assert tracked.observed == []


def test_a_writer_is_recorded_rather_than_performed(
    orders: Any, ctx: DryRunContext, tmp_path: Path
) -> None:
    target = tmp_path / "out.parquet"
    assert orders.to_parquet(target) is None
    assert not target.exists()
    assert any("out.parquet" in note for note in ctx.notes)


def test_a_writer_in_a_loop_says_so_once(orders: Any, ctx: DryRunContext) -> None:
    for _ in range(5):
        orders.to_parquet("out.parquet")
    assert len([note for note in ctx.notes if "out.parquet" in note]) == 1


def test_the_wrappers_do_nothing_outside_a_dry_run(ctx: DryRunContext) -> None:
    """Importing this adapter must not change how ibis behaves for anybody else."""
    import ibis.common.exceptions as ibis_errors

    table = ibis.table({"amount": "int64"}, name="plain")
    with pytest.raises(ibis_errors.IbisTypeError):
        table.select("amount_cents")
    with pytest.raises(AttributeError) as caught:
        _ = table.amount_cents
    assert not isinstance(caught.value, ColumnNotFound)


def test_inline_sql_is_resolved_without_a_backend(orders: Any, ctx: DryRunContext) -> None:
    """`Table.sql` normally asks the backend; there is no backend in a dry run."""
    resolved = orders.alias("orders").sql(
        "SELECT status, sum(amount) AS total FROM orders GROUP BY status"
    )
    relation = ADAPTER.describe(resolved)
    assert tuple(relation.schema) == ("status", "total")
    assert relation.schema["total"].dtype == int_(64)
    assert any("SQL parser" in note for note in ctx.notes)


def test_inline_sql_columns_are_traced(orders: Any) -> None:
    """An opaque SQL view would have no lineage; an ibis expression does."""
    resolved = orders.alias("orders").sql("SELECT amount AS cents FROM orders")
    assert ADAPTER.column_lineage(resolved) == {"cents": ("amount",)}


def test_inline_sql_reads_the_tables_own_name(orders: Any) -> None:
    """A query that names the table rather than an alias resolves too."""
    resolved = orders.sql("SELECT order_id FROM stg_orders")
    assert tuple(ADAPTER.describe(resolved).schema) == ("order_id",)


def test_inline_sql_missing_column_is_the_flagship_finding(orders: Any) -> None:
    with pytest.raises(ColumnNotFound) as caught:
        orders.alias("orders").sql("SELECT amount_cents FROM orders")
    assert caught.value.suggestion == "amount"


def test_inline_sql_that_cannot_be_read_is_unverified(orders: Any) -> None:
    """Unverified, not broken: the SQL may be perfectly valid for its warehouse."""
    with pytest.raises(errors.SqlNotResolvable, match="dialect"):
        orders.alias("orders").sql("SELECT * FROM a_table_nobody_registered")


# --- Lineage -----------------------------------------------------------------


def test_lineage_follows_a_chain(orders: Any) -> None:
    filtered = orders.filter(orders.status == "paid")
    traced = ADAPTER.column_lineage(
        filtered.select(ident=filtered.order_id, cents=filtered.amount * 100)
    )
    assert traced == {"ident": ("order_id",), "cents": ("amount",)}


def test_a_predicate_is_not_a_parent(orders: Any) -> None:
    """`WHERE status = 'paid'` decides which rows survive, not what a column holds.

    This is the case a naive graph search gets wrong: `ops.Field` holds its
    relation as a child, so searching from one walks into the filter below and
    comes back with every column the predicate mentions.
    """
    filtered = orders.filter(orders.status == "paid")
    traced = ADAPTER.column_lineage(filtered.select(ident=filtered.order_id))
    assert traced["ident"] == ("order_id",)


def test_an_aggregate_is_attributed_to_what_it_summed(orders: Any) -> None:
    traced = ADAPTER.column_lineage(orders.group_by("status").agg(total=orders.amount.sum()))
    assert traced == {"status": ("status",), "total": ("amount",)}


def test_a_computed_column_names_every_input(orders: Any) -> None:
    traced = ADAPTER.column_lineage(orders.select(combo=orders.amount + orders.order_id))
    assert set(traced["combo"]) == {"amount", "order_id"}


def test_a_literal_column_is_untraced(orders: Any) -> None:
    """No input column behind it, so no edge is invented."""
    traced = ADAPTER.column_lineage(orders.select("order_id").mutate(src=ibis.literal("web")))
    assert "src" not in traced
    assert traced["order_id"] == ("order_id",)


def test_a_union_is_attributed_to_both_sides(orders: Any) -> None:
    traced = ADAPTER.column_lineage(orders.union(orders.filter(orders.amount > 0)))
    assert traced["amount"] == ("amount",)


def test_a_join_keeps_each_side_separate(orders: Any, customers: Any) -> None:
    traced = ADAPTER.column_lineage(
        orders.join(customers, orders.order_id == customers.customer_id)
    )
    assert traced["country"] == ("country",)
    assert traced["created_at_right"] == ("created_at",)


def test_a_struct_field_is_attributed_to_its_column(events: Any) -> None:
    traced = ADAPTER.column_lineage(events.select(events.payload.source))
    assert traced == {"source": ("payload",)}


def test_lineage_of_a_column_expression(orders: Any) -> None:
    assert ADAPTER.column_lineage(orders.amount * 2) == {"Multiply(amount, 2)": ("amount",)}


def test_lineage_of_a_foreign_object_is_empty() -> None:
    assert lineage.of_expression(object()) == {}


def test_lineage_depth_is_bounded(orders: Any) -> None:
    """A pathological chain must not hang the dry run."""
    expr = orders
    for _ in range(50):
        expr = expr.filter(expr.amount > 0)
    assert ADAPTER.column_lineage(expr)["amount"] == ("amount",)


# --- The SQL seam ------------------------------------------------------------


def test_run_sql_resolves_with_no_database(ctx: DryRunContext) -> None:
    relation = ADAPTER.run_sql(
        "SELECT status, sum(amount) AS total FROM stg_orders GROUP BY status",
        {"stg_orders": fixtures.ORDERS},
        ctx,
    )
    assert tuple(relation.schema) == ("status", "total")
    assert relation.schema["total"].dtype == int_(64)
    assert relation.engine == "ibis"


def test_run_sql_traces_columns(ctx: DryRunContext) -> None:
    """A SQL model's columns join the same graph as a Python pipeline's."""
    relation = ADAPTER.run_sql(
        "SELECT order_id, amount / 2 AS half FROM stg_orders",
        {"stg_orders": fixtures.ORDERS},
        ctx,
    )
    assert relation.schema["half"].derived_from == ("amount",)
    assert relation.schema["half"].dtype == float_(64)


def test_run_sql_resolves_a_dbt_relation_name(ctx: DryRunContext) -> None:
    """dbt compiles `ref()` into a quoted three-part relation name."""
    relation = ADAPTER.run_sql(
        'SELECT order_id FROM "warehouse"."analytics"."stg_orders";',
        {'"warehouse"."analytics"."stg_orders"': fixtures.ORDERS},
        ctx,
    )
    assert tuple(relation.schema) == ("order_id",)


def test_run_sql_reads_a_cte(ctx: DryRunContext) -> None:
    relation = ADAPTER.run_sql(
        "WITH paid AS (SELECT * FROM stg_orders WHERE status = 'paid') "
        "SELECT status, sum(amount) AS total FROM paid GROUP BY status",
        {"stg_orders": fixtures.ORDERS},
        ctx,
    )
    assert tuple(relation.schema) == ("status", "total")


def test_run_sql_accepts_a_materialized_table(ctx: DryRunContext) -> None:
    """A caller that already has a frame has no reason to build a schema for it."""
    table = ADAPTER.materialize(fixtures.ORDERS, ctx)
    relation = ADAPTER.run_sql("SELECT status FROM stg_orders", {"stg_orders": table}, ctx)
    assert tuple(relation.schema) == ("status",)


def test_run_sql_refuses_an_input_it_cannot_read(ctx: DryRunContext) -> None:
    with pytest.raises(UnknownSchemaError, match="cannot register"):
        ADAPTER.run_sql("SELECT 1 AS one FROM stg_orders", {"stg_orders": object()}, ctx)


def test_run_sql_reports_a_missing_column(ctx: DryRunContext) -> None:
    with pytest.raises(ColumnNotFound) as caught:
        ADAPTER.run_sql("SELECT amount_cents FROM stg_orders", {"stg_orders": fixtures.ORDERS}, ctx)
    assert caught.value.suggestion == "amount"


def test_run_sql_degrades_on_a_name_collision(ctx: DryRunContext) -> None:
    """Two inputs cannot both be `orders`, and picking one would resolve it wrongly."""
    other = Schema.from_mapping({"different": int_(64)}, name="stg_orders")
    ADAPTER.run_sql(
        "SELECT order_id FROM stg_orders",
        {"first.stg_orders": fixtures.ORDERS, "second.stg_orders": other},
        ctx,
    )
    assert any("stg_orders" in reason for reason in ctx.degradations)


def test_run_sql_honours_the_configured_dialect(tmp_path: Path) -> None:
    """dbt compiles for a warehouse, and a Snowflake model is not portable SQL."""
    ctx = DryRunContext(run_id="dialect", workdir=str(tmp_path), options={"dialect": "snowflake"})
    relation = ADAPTER.run_sql(
        "SELECT amount::float AS amt FROM stg_orders", {"stg_orders": fixtures.ORDERS}, ctx
    )
    # A float, and the engine's own answer about its width: Snowflake's FLOAT is
    # double precision, which ibis 12 reports and earlier releases in the
    # supported range read as float32. The width is ibis's to get right, so the
    # assertion is that the dialect was read at all.
    assert relation.schema["amt"].dtype.kind is Kind.FLOAT


def test_run_sql_notes_a_query_it_cannot_trace(ctx: DryRunContext) -> None:
    relation = ADAPTER.run_sql(
        "SELECT count(*) AS n FROM stg_orders", {"stg_orders": fixtures.ORDERS}, ctx
    )
    assert relation.schema["n"].derived_from == ()
    assert any("untraced" in note for note in ctx.notes)


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("orders", "orders"),
        ("analytics.orders", "orders"),
        ('"warehouse"."analytics"."orders"', "orders"),
        ("`warehouse`.`orders`", "orders"),
        ("[dbo].[orders]", "orders"),
    ],
)
def test_names_are_matched_on_the_final_identifier(given: str, expected: str) -> None:
    assert sql_seam.bare_name(given) == expected


def test_a_trailing_terminator_is_dropped() -> None:
    assert sql_seam.strip_terminator("SELECT 1;\n") == "SELECT 1"


# --- Portability -------------------------------------------------------------


def test_backend_disagreements_names_the_backends_that_differ(ctx: DryRunContext) -> None:
    """The finding no other adapter here can produce, and nothing is installed."""
    schema = Schema(
        fields=(
            Field(name="narrow", dtype=int_(32)),
            Field(name="money", dtype=decimal(18, 2)),
        ),
        name="wide",
    )
    table = ADAPTER.materialize(schema, ctx)
    found = ADAPTER.backend_disagreements(table, ("duckdb", "bigquery"))
    # DuckDB has both types exactly; BigQuery has one integer type, so the
    # narrow one widens, and its NUMERIC is fixed at (38, 9), so the decimal
    # cannot be held as declared. What the widened decimal reads back as is
    # version-dependent - ibis 12 says decimal(38, 9), earlier releases in the
    # range refuse to read NUMERIC back at all - and either answer is the same
    # finding: a contract declaring decimal(18, 2) cannot hold on BigQuery.
    assert set(found) == {"bigquery"}
    assert found["bigquery"]["narrow"].dtype == int_(64)
    assert not found["bigquery"]["money"].dtype.matches(decimal(18, 2))


def test_a_named_zone_is_lost_on_the_backends_that_store_instants(
    ctx: DryRunContext,
) -> None:
    """The silent-hour-shift case, reported before the migration rather than after."""
    schema = Schema(
        fields=(Field(name="when", dtype=datetime(TimeUnit.NANOSECOND, "Europe/London")),),
        name="zoned",
    )
    table = ADAPTER.materialize(schema, ctx)
    found = ADAPTER.backend_disagreements(table, ("duckdb",))
    assert found["duckdb"]["when"].dtype == datetime(TimeUnit.MICROSECOND, "UTC")


def test_a_type_a_backend_lacks_is_unresolved_rather_than_approximated(
    ctx: DryRunContext,
) -> None:
    """SQLite has no array type, so the column cannot be stored at all."""
    schema = Schema(fields=(Field(name="tags", dtype=list_(string())),), name="nested")
    table = ADAPTER.materialize(schema, ctx)
    found = ADAPTER.backend_disagreements(table, ("sqlite",))
    assert found["sqlite"]["tags"].dtype.kind is Kind.UNKNOWN


def test_a_portable_expression_reports_nothing(orders: Any) -> None:
    assert ADAPTER.backend_disagreements(orders.select("status"), ("duckdb", "bigquery")) == {}


def test_an_unknown_backend_is_a_typo_not_a_finding(orders: Any) -> None:
    with pytest.raises(portability.UnknownBackend, match="mongodb"):
        ADAPTER.backend_disagreements(orders, ("mongodb",))


def test_a_dtype_ibis_cannot_express_is_reported_per_backend() -> None:
    mapper = portability.type_mapper("duckdb")
    resolved = portability.round_trip(enum(("a", "b")), mapper, backend="duckdb")
    assert resolved.kind is Kind.UNKNOWN
    assert resolved.reason is not None and "ibis cannot express" in resolved.reason


def test_the_default_backends_cover_the_four_failure_shapes() -> None:
    """Chosen for how far apart their type systems are, not for popularity."""
    assert portability.DEFAULT_BACKENDS == ("duckdb", "bigquery", "snowflake", "sqlite")


# --- Capabilities ------------------------------------------------------------


def test_declared_capabilities_match_what_is_implemented() -> None:
    """A capability that is advertised but missing would inflate a coverage score."""
    assert isinstance(ADAPTER, SupportsColumnLineage)
    assert isinstance(ADAPTER, SupportsSql)
    assert not isinstance(ADAPTER, SupportsUdfProbe)
    assert ADAPTER.capabilities.udf_probe is False
    assert ADAPTER.capabilities.remote is False


def test_the_operations_battery_covers_the_checklist() -> None:
    """docs/writing-an-adapter.md names what a battery has to exercise."""
    names = {case.name for case in ADAPTER.conformance_cases()}
    for required in (
        "select",
        "rename",
        "filter",
        "cast",
        "group_by_sum",
        "join_separate_keys",
        "join_shared_key",
        "union",
        "row_dependent_branch",
    ):
        assert required in names


# --- Session -----------------------------------------------------------------


def test_a_new_run_resets_the_table_namer(ctx: DryRunContext, tmp_path: Path) -> None:
    first = ADAPTER.materialize(fixtures.ORDERS, ctx)
    session.begin(DryRunContext(run_id="a-different-run", workdir=str(tmp_path)))
    second = ADAPTER.materialize(fixtures.ORDERS, ctx)
    assert first.op().name == second.op().name == "stg_orders"


def test_notes_and_degradations_need_no_open_context(tmp_path: Path) -> None:
    """A note with nowhere to go is dropped rather than raising."""
    session.begin(DryRunContext(run_id="open", workdir=str(tmp_path)))
    session.note("recorded once", once_for="key")
    session.note("recorded once", once_for="key")
    context = session.context()
    assert context is not None
    assert context.notes == ["recorded once"]
    session.degrade("lost it")
    assert context.degradations == ["lost it"]


def test_own_work_is_not_user_code(orders: Any) -> None:
    """A wrapper resolving a schema must not be reported as the pipeline reading data."""
    assert session.tracking() is True
    with session.own_work():
        assert session.tracking() is False


# --- End to end --------------------------------------------------------------


def test_dry_run_reports_the_renamed_column_at_the_users_line() -> None:
    """The whole point, through the same runtime the CLI uses."""
    result = dry_run_callable(
        lambda orders: orders.select(cents=orders.amount_cents),
        {"orders": fixtures.ORDERS},
        engine="ibis",
    )
    assert result.output is None
    (diagnostic,) = [d for d in result.diagnostics if d.code is Code.COLUMN_NOT_FOUND]
    assert diagnostic.suggestion is not None and "amount" in diagnostic.suggestion
    assert diagnostic.origin is not None
    assert diagnostic.origin.file is not None and diagnostic.origin.file.endswith(
        "test_ibis_adapter.py"
    )


def test_dry_run_of_a_working_pipeline_is_exact_and_traced() -> None:
    result = dry_run_callable(
        lambda orders: orders.group_by("status").agg(total=orders.amount.sum()),
        {"orders": fixtures.ORDERS},
        engine="ibis",
    )
    assert result.ok is True
    assert result.coverage.confidence is Confidence.EXACT
    assert result.coverage.traced_ratio == 1.0
    assert result.output is not None
    assert result.output.schema["total"].derived_from == ("amount",)


def test_dry_run_of_a_row_dependent_pipeline_is_unverified() -> None:
    def pipeline(orders: Any) -> Any:
        return orders.limit(1) if orders.count().execute() else orders

    result = dry_run_callable(pipeline, {"orders": fixtures.ORDERS}, engine="ibis")
    assert result.ok is False
    assert result.coverage.confidence is Confidence.UNKNOWN
    assert result.coverage.reasons
    assert any(d.code is Code.ROW_DEPENDENT_CONTROL_FLOW for d in result.diagnostics)
