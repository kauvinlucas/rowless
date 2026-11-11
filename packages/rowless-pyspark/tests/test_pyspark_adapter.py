"""Unit tests for the parts of the adapter the conformance suite cannot reach.

The suite in core is the specification, and it covers the contract - but every
test in it needs a Spark to ask, which means a Connect endpoint or a JVM. These
tests deliberately need **neither**, so the half of this adapter that is pure
Python is checked everywhere, including on a laptop with the wrong Java and in
the lint job:

* the dtype table, including the refusals that exist so a value is never
  silently moved by an hour or truncated to a narrower width;
* the error translation, built from real `AnalysisException` objects carrying
  the conditions a server sends;
* the lineage rules, and the expression reader that walks a Spark Connect
  expression tree without a session;
* the statement guard, which is the one piece of this adapter whose failure
  mode is data loss rather than a wrong schema;
* session selection: which mode is chosen, what egress it asks the sandbox
  for, and how the Java version is read.

What is *not* here is anything that needs analysis: materialize, describe, and
the operations battery are the conformance suite's job, and it runs in CI
against a local session and against the floor release.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pyspark.sql.types as t
import pytest
from pyspark.errors import AnalysisException
from pyspark.errors.exceptions.connect import SparkConnectGrpcException
from rowless.adapters.base import (
    DryRunContext,
    NeedsEgress,
    SupportsColumnLineage,
    SupportsSql,
    SupportsUdfProbe,
)
from rowless.errors import (
    ColumnNotFound,
    DTypeMismatch,
    DuplicateColumn,
    UnknownSchemaError,
    UnsupportedDType,
)
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
    string,
    struct,
    time,
    uint,
    unknown,
)
from rowless.ir.schema import Field, Schema
from rowless.testing import fixtures
from rowless_pyspark import adapter as adapter_module
from rowless_pyspark import dtypes as dtype_map
from rowless_pyspark import errors, interception, lineage
from rowless_pyspark import session as run
from rowless_pyspark import spark as cluster
from rowless_pyspark.adapter import PySparkAdapter

ADAPTER = PySparkAdapter()


@pytest.fixture(autouse=True)
def _clean_run_state(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Every test starts with no run open, no session cached and no endpoint set."""
    for variable in cluster.REMOTE_ENV:
        monkeypatch.delenv(variable, raising=False)
    cluster.forget()
    yield
    cluster.forget()


def context(**options: Any) -> DryRunContext:
    """A dry-run context with adapter options, and a scratch dir that exists."""
    import tempfile

    return DryRunContext(run_id="test", workdir=tempfile.gettempdir(), options=options)


def _connect_is_importable() -> bool:
    """Whether a Connect `Column` can be built in this environment at all.

    It cannot on the 3.5 floor under Python 3.12: that client imports pandas
    through `distutils`, which 3.12 removed. The tests that read a Connect
    expression tree skip there, and everything that does not need one - the
    lineage rules, the dtype table, the error translation, the statement guard -
    runs on both.
    """
    try:
        import pyspark.sql.connect.functions as functions

        functions.col("probe")
    except Exception:
        return False
    return True


CONNECT_READY = _connect_is_importable()

requires_connect = pytest.mark.skipif(
    not CONNECT_READY,
    reason="this PySpark cannot build a Connect expression without a session",
)


def connect_column(name: str) -> Any:
    """A Spark Connect `Column`, which needs no session to build.

    The whole reason the lineage reader prefers the Connect expression tree:
    `pyspark.sql.functions.col` resolves against an active SparkContext, and
    the Connect spelling of the same call is pure Python.
    """
    import pyspark.sql.connect.functions as functions

    return functions.col(name)


def classic_column() -> Any:
    """A classic `Column` with no JVM behind it.

    Enough for the checks that only ask what an object *is* - `handles`, and
    `describe` refusing one - and buildable with no session, which is what
    makes those two testable on every install.
    """
    from pyspark.sql.column import Column

    return Column(None)


# --- Dtypes ------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype",
    [
        boolean(),
        int_(8),
        int_(16),
        int_(32),
        int_(64),
        float_(32),
        float_(64),
        decimal(18, 2),
        decimal(38, 9),
        string(),
        binary(),
        date(),
        datetime(TimeUnit.MICROSECOND),
        datetime(TimeUnit.MICROSECOND, "UTC"),
        duration(TimeUnit.MICROSECOND),
        list_(int_(64)),
        map_(string(), int_(64)),
        struct({"source": string(), "retries": int_(32)}),
    ],
    ids=str,
)
def test_dtype_round_trips(dtype: DType) -> None:
    assert dtype_map.to_ir(dtype_map.from_ir(dtype)) == dtype


def test_time_round_trips_when_spark_has_the_type() -> None:
    if not dtype_map.time_type_available():
        pytest.skip("this PySpark has no TIME type; it arrived in Spark 4.1")
    assert dtype_map.to_ir(dtype_map.from_ir(time(TimeUnit.MICROSECOND))) == time(
        TimeUnit.MICROSECOND
    )
    assert dtype_map.from_ir(time(TimeUnit.MILLISECOND)).precision == 3


@pytest.mark.parametrize(
    "dtype",
    [
        uint(8),
        uint(32),
        uint(64),
        int_(128),
        float_(16),
        enum(("a", "b")),
        datetime(TimeUnit.MILLISECOND),
        datetime(TimeUnit.NANOSECOND),
        datetime(TimeUnit.MICROSECOND, "Europe/London"),
        duration(TimeUnit.MILLISECOND),
        time(TimeUnit.NANOSECOND),
        decimal(40, 2),
        unknown("nothing resolved this"),
    ],
    ids=str,
)
def test_from_ir_refuses_rather_than_approximating(dtype: DType) -> None:
    with pytest.raises(UnsupportedDType):
        dtype_map.from_ir(dtype)


def test_unsigned_refusal_says_why() -> None:
    with pytest.raises(UnsupportedDType, match="unsigned"):
        dtype_map.from_ir(uint(32))


def test_named_zone_refusal_names_the_session_timezone() -> None:
    with pytest.raises(UnsupportedDType, match="'UTC'"):
        dtype_map.from_ir(datetime(TimeUnit.MICROSECOND, "Europe/London"))


def test_timestamp_dtype_follows_the_session_timezone(monkeypatch: pytest.MonkeyPatch) -> None:
    """`TimestampType` has no zone of its own, so the IR dtype tracks the session's.

    Reporting UTC for a session Spark will render in another zone would be a
    dtype rowless invented, and the round trip has to follow the same rule.
    """
    monkeypatch.setattr(cluster, "_TIMEZONE", "America/Denver")
    assert dtype_map.to_ir(t.TimestampType()) == datetime(TimeUnit.MICROSECOND, "America/Denver")
    assert isinstance(
        dtype_map.from_ir(datetime(TimeUnit.MICROSECOND, "America/Denver")), t.TimestampType
    )
    with pytest.raises(UnsupportedDType):
        dtype_map.from_ir(datetime(TimeUnit.MICROSECOND, "UTC"))


def test_naive_and_aware_timestamps_stay_apart() -> None:
    assert dtype_map.to_ir(t.TimestampNTZType()) == datetime(TimeUnit.MICROSECOND)
    assert dtype_map.to_ir(t.TimestampType()).tz == "UTC"


def test_year_month_interval_is_unknown_with_a_reason() -> None:
    found = dtype_map.to_ir(t.YearMonthIntervalType())
    assert found.kind is Kind.UNKNOWN
    assert "month" in (found.reason or "")


def test_char_and_varchar_are_strings() -> None:
    assert dtype_map.to_ir(t.CharType(10)) == string()
    assert dtype_map.to_ir(t.VarcharType(10)) == string()


def test_variant_is_refused_rather_than_guessed() -> None:
    variant = getattr(t, "VariantType", None)
    if variant is None:  # pragma: no cover - only on an older PySpark
        pytest.skip("this PySpark has no VariantType")
    assert dtype_map.to_ir(variant()).kind is Kind.UNKNOWN


def test_to_ir_of_a_foreign_object_is_unknown() -> None:
    assert dtype_map.to_ir(object()).kind is Kind.UNKNOWN
    assert dtype_map.to_ir(t.StringType).kind is Kind.STRING


def test_any_becomes_the_one_type_spark_coerces_from() -> None:
    assert isinstance(dtype_map.from_ir(any_()), t.NullType)


def test_struct_type_carries_nullability() -> None:
    native = dtype_map.struct_type(fixtures.ORDERS.fields)
    by_name = {field.name: field for field in native.fields}
    assert by_name["order_id"].nullable is False
    assert by_name["status"].nullable is True


def test_to_schema_reads_nullability_back() -> None:
    native = t.StructType(
        [
            t.StructField("order_id", t.LongType(), nullable=False),
            t.StructField("status", t.StringType(), nullable=True),
        ]
    )
    schema = dtype_map.to_schema(native)
    assert tuple(schema) == ("order_id", "status")
    assert schema["order_id"].nullable is False
    assert schema["status"].dtype == string()


def test_declared_absences_are_the_ones_the_suite_checks() -> None:
    absent = set(dtype_map.unsupported_dtypes())
    assert {uint(8), uint(32), enum(("a", "b", "c")), datetime(TimeUnit.MILLISECOND)} <= absent
    for dtype in absent:
        with pytest.raises(UnsupportedDType):
            dtype_map.from_ir(dtype)


def test_every_corpus_dtype_either_round_trips_or_is_declared_absent() -> None:
    """The contract the conformance suite enforces, asserted without a session."""
    absent = set(dtype_map.unsupported_dtypes())
    for dtype in fixtures.round_trip_corpus():
        if dtype in absent:
            continue
        assert dtype_map.to_ir(dtype_map.from_ir(dtype)) == dtype, dtype


# --- Errors ------------------------------------------------------------------


def analysis_error(
    condition: str = "",
    params: dict[str, str] | None = None,
    message: str = "",
) -> AnalysisException:
    """An `AnalysisException` carrying a server-side condition and its parameters.

    Built through whichever constructor the installed PySpark offers - the
    keyword names changed between 3.5 and 4 - and through a subclass when
    neither takes them. The subclass is faithful rather than a convenience: a
    classic session raises a *captured* exception, whose condition and
    parameters are exactly these two methods, so this is the shape the adapter
    reads in both modes.
    """
    parameters = dict(params or {})
    text = message or f"[{condition}] something went wrong. SQLSTATE: 42703"
    spellings: tuple[dict[str, Any], ...] = (
        {"errorClass": condition, "messageParameters": parameters},
        {"error_class": condition, "message_parameters": parameters},
    )
    build: Any = AnalysisException
    for keywords in spellings:
        if not condition:
            break
        try:
            return cast("AnalysisException", build(message=text, **keywords))
        except Exception:
            # 3.5 takes the other spelling *and* refuses a message alongside
            # it, so neither keyword set works there and the subclass below is
            # what that version is tested through.
            continue

    class Captured(AnalysisException):
        """What a classic session raises: a condition plus a parameter map."""

        def getErrorClass(self) -> str:
            return condition

        def getMessageParameters(self) -> dict[str, str]:
            return parameters

    return Captured(text)


def test_unresolved_column_becomes_the_flagship_finding() -> None:
    exc = analysis_error(
        "UNRESOLVED_COLUMN.WITH_SUGGESTION",
        {"objectName": "`amount_cents`", "proposal": "`amount`, `status`"},
    )
    found = errors.translate(exc)
    assert isinstance(found, ColumnNotFound)
    assert found.column == "amount_cents"
    assert found.suggestion == "amount"


def test_near_miss_prefers_sparks_own_proposal_over_the_column_list() -> None:
    exc = analysis_error(
        "UNRESOLVED_COLUMN.WITH_SUGGESTION",
        {"objectName": "`amount_cents`", "proposal": "`amount`"},
    )
    found = errors.translate(exc, available=("completely", "different"))
    assert isinstance(found, ColumnNotFound)
    assert found.available == ("amount",)


def test_unresolved_column_without_a_proposal_uses_the_frames_columns() -> None:
    exc = analysis_error("UNRESOLVED_COLUMN.WITHOUT_SUGGESTION", {"objectName": "`amount_cents`"})
    found = errors.translate(exc, available=("order_id", "amount"))
    assert isinstance(found, ColumnNotFound)
    assert found.suggestion == "amount"


def test_unresolved_field_says_where_it_looked() -> None:
    exc = analysis_error(
        "UNRESOLVED_FIELD.WITH_SUGGESTION",
        {"fieldName": "`sources`", "columnPath": "`payload`", "proposal": "`source`"},
    )
    found = errors.translate(exc)
    assert isinstance(found, ColumnNotFound)
    assert found.column == "sources"
    assert found.relation == "payload"


def test_legacy_prose_message_is_still_classified() -> None:
    """A classic session on an older Spark reports this as a sentence."""
    exc = AnalysisException(
        message=(
            "cannot resolve 'amount_cents' given input columns: [order_id, amount, status];\n"
            "'Project [unresolvedalias('amount_cents)]\n+- LocalRelation <empty>"
        )
    )
    found = errors.translate(exc)
    assert isinstance(found, ColumnNotFound)
    assert found.column == "amount_cents"
    assert found.suggestion == "amount"


@pytest.mark.parametrize(
    ("condition", "expected"),
    [
        ("AMBIGUOUS_REFERENCE", DuplicateColumn),
        ("COLUMN_ALREADY_EXISTS", DuplicateColumn),
        ("DATATYPE_MISMATCH.BINARY_OP_DIFF_TYPES", DTypeMismatch),
        ("NUM_COLUMNS_MISMATCH", DTypeMismatch),
        ("INCOMPATIBLE_COLUMN_TYPE", DTypeMismatch),
        ("TABLE_OR_VIEW_NOT_FOUND", UnknownSchemaError),
        ("PARSE_SYNTAX_ERROR", UnknownSchemaError),
    ],
)
def test_conditions_map_to_findings(condition: str, expected: Any) -> None:
    found = errors.translate(analysis_error(condition, {}))
    assert isinstance(found, expected)


def test_subconditions_land_on_the_same_finding() -> None:
    """A condition rowless has not catalogued still dispatches on its family."""
    found = errors.translate(analysis_error("UNRESOLVED_COLUMN.SOMETHING_NEW_IN_SPARK_5", {}))
    assert isinstance(found, ColumnNotFound)


def test_transport_failure_is_not_blamed_on_the_pipeline() -> None:
    found = errors.translate(SparkConnectGrpcException(message="channel closed"))
    assert isinstance(found, UnknownSchemaError)
    assert "endpoint" in str(found)


def test_an_unclassified_spark_error_is_left_alone() -> None:
    assert errors.translate(analysis_error("SOME_CONDITION_NOBODY_CATALOGUED", {})) is None


def test_a_user_exception_is_not_mistaken_for_an_engine_failure() -> None:
    class AnalysisException(Exception):  # deliberately the same name as Spark's
        pass

    assert errors.translate(AnalysisException("user code raised this")) is None


def test_the_plan_dump_and_sqlstate_are_stripped() -> None:
    exc = analysis_error(
        "TABLE_OR_VIEW_NOT_FOUND",
        {},
        message="[TABLE_OR_VIEW_NOT_FOUND] `orders` cannot be found. SQLSTATE: 42P01;\n"
        "'Project [*]\n+- 'UnresolvedRelation [orders]",
    )
    found = errors.translate(exc)
    assert found is not None
    assert "Project" not in str(found)
    assert "SQLSTATE" not in str(found)


def test_raise_translated_passes_an_unknown_error_through() -> None:
    original = analysis_error("SOMETHING_ELSE", {})
    with pytest.raises(AnalysisException):
        errors.raise_translated(original)


# --- Lineage -----------------------------------------------------------------


def test_carry_keeps_the_surviving_columns() -> None:
    parent = {"a": ("a",), "b": ("b",)}
    assert lineage.carry(parent, ["b"]) == {"b": ("b",)}


@requires_connect
def test_positional_attribution_matches_expressions_to_output_columns() -> None:
    parent = {"amount": ("amount",), "status": ("status",)}
    terms = [connect_column("amount"), connect_column("status")]
    assert lineage.positional(parent, ["sum(amount)", "status"], terms) == {
        "sum(amount)": ("amount",),
        "status": ("status",),
    }


@requires_connect
def test_positional_attribution_reads_a_compound_expression() -> None:
    parent = {"a": ("a",), "b": ("b",)}
    total = (connect_column("a") + connect_column("b")).alias("total")
    assert lineage.positional(parent, ["total"], [total]) == {"total": ("a", "b")}


def test_positional_falls_back_to_names_when_a_star_expanded() -> None:
    parent = {"a": ("a",), "b": ("b",)}
    assert lineage.positional(parent, ["a", "b"], ["*"]) == parent


@requires_connect
def test_a_literal_column_is_untraced_rather_than_attributed() -> None:
    import pyspark.sql.connect.functions as functions

    parent = {"amount": ("amount",)}
    assert lineage.positional(parent, ["ceiling"], [functions.lit(1).alias("ceiling")]) == {}


@requires_connect
def test_a_qualified_reference_points_at_the_column_it_reads() -> None:
    assert lineage.root_names("payload.source") == ("payload",)
    assert lineage.root_names(connect_column("payload.source")) == ("payload",)


def test_renamed_moves_the_lineage_with_the_name() -> None:
    parent = {"amount": ("amount",), "status": ("status",)}
    found = lineage.renamed(parent, ["amount_cents", "status"], {"amount": "amount_cents"})
    assert found == {"amount_cents": ("amount",), "status": ("status",)}


@requires_connect
def test_added_attributes_the_named_column_and_keeps_the_rest() -> None:
    parent = {"amount": ("amount",)}
    found = lineage.added(parent, ["amount", "doubled"], {"doubled": connect_column("amount")})
    assert found == {"amount": ("amount",), "doubled": ("amount",)}


def test_joined_takes_each_column_from_the_side_that_has_it() -> None:
    left = {"order_id": ("order_id",)}
    right = {"country": ("country",)}
    found = lineage.joined(left, right, ["order_id", "country", "invented"])
    assert found == {"order_id": ("order_id",), "country": ("country",)}


def test_union_pairs_by_position_when_the_names_do_not_match() -> None:
    """`union` is positional in Spark, and the lineage has to say so."""
    left = {"order_id": ("order_id",)}
    right = {"amount": ("amount",)}
    assert lineage.merged([left, right], ["order_id"]) == {"order_id": ("order_id", "amount")}


def test_union_by_name_pairs_the_same_names() -> None:
    left = {"amount": ("amount",)}
    right = {"amount": ("amount_cents",)}
    assert lineage.merged([left, right], ["amount"]) == {"amount": ("amount", "amount_cents")}


@requires_connect
def test_terms_flattens_the_shapes_a_builder_accepts() -> None:
    found = lineage.terms((["a", connect_column("b")], "c"), {"how": "inner"})
    assert len(found) == 3
    assert lineage.root_names(found[0]) == ("a",)
    assert lineage.root_names(found[1]) == ("b",)


@requires_connect
def test_output_name_is_only_read_from_an_alias() -> None:
    assert lineage.output_name(connect_column("amount").alias("cents")) == "cents"
    assert lineage.output_name(connect_column("amount")) is None


@requires_connect
def test_star_is_recognised_in_both_spellings() -> None:
    import pyspark.sql.connect.functions as functions

    assert lineage.is_star("*")
    assert lineage.is_star(functions.col("*"))
    assert not lineage.is_star(connect_column("amount"))


def test_the_tracker_distinguishes_untracked_from_empty() -> None:
    tracker = lineage.PlanLineage()
    frame = SimpleNamespace()
    assert tracker.of(frame) is None
    tracker.record(frame, {})
    assert tracker.of(frame) == {}


def test_the_tracker_seeds_identity_lineage_for_an_input() -> None:
    tracker = lineage.PlanLineage()
    frame = SimpleNamespace()
    tracker.seed(frame, ["a", "b"])
    assert tracker.of(frame) == {"a": ("a",), "b": ("b",)}


def test_the_tracker_forgets_a_collected_frame() -> None:
    tracker = lineage.PlanLineage()

    class Frame:
        pass

    frame: Any = Frame()
    tracker.record(frame, {"a": ("a",)})
    assert len(tracker) == 1
    del frame
    import gc

    gc.collect()
    assert len(tracker) == 0


# --- The statement guard -----------------------------------------------------


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        ("SELECT 1", True),
        ("  with a as (select 1) select * from a", True),
        ("-- a comment\nSELECT 1", True),
        ("/* block */ SELECT 1", True),
        ("(SELECT 1)", True),
        ("VALUES (1)", True),
        ("DESCRIBE TABLE orders", True),
        ("INSERT INTO orders SELECT * FROM staging", False),
        ("insert overwrite table orders select 1", False),
        ("CREATE TABLE orders (id BIGINT)", False),
        ("DROP TABLE orders", False),
        ("MERGE INTO orders USING staging ON a = b", False),
        ("TRUNCATE TABLE orders", False),
        ("UPDATE orders SET amount = 0", False),
        ("DELETE FROM orders", False),
        ("GRANT SELECT ON orders TO alice", False),
        ("-- insert into orders\nSELECT 1", True),
        ("", False),
    ],
)
def test_only_a_query_is_allowed_through(sql: str, expected: Any) -> None:
    """The one piece of this adapter whose failure mode is data loss."""
    assert interception.is_query(sql) is expected


def test_statement_kind_names_what_was_refused() -> None:
    assert interception.statement_kind("/* x */ insert into t values (1)") == "INSERT"
    assert interception.statement_kind("") == ""


def test_write_targets_are_read_from_the_call_and_from_the_writer() -> None:
    assert interception._write_target(("s3://bucket/orders",), {}) == "s3://bucket/orders"
    assert interception._write_target((), {"path": "/tmp/out"}) == "/tmp/out"
    writer = SimpleNamespace(_table_name="warehouse.orders")
    assert interception._writer_target(writer) == "warehouse.orders"
    options = SimpleNamespace(_options={"path": "abfss://lake/orders"})
    assert interception._writer_target(options) == "abfss://lake/orders"


@requires_connect
def test_the_builder_argument_shapes_are_understood() -> None:
    column = connect_column("amount")
    assert interception._additions(("doubled", column), {}) == {"doubled": column}
    assert interception._additions(({"doubled": column},), {}) == {"doubled": column}
    assert interception._renames(("amount", "cents"), {}) == {"amount": "cents"}
    assert interception._renames(({"amount": "cents"},), {}) == {"amount": "cents"}


def test_installing_the_patches_is_idempotent() -> None:
    interception.install(context())
    interception.install(context())
    import pyspark.sql.dataframe as dataframe

    assert getattr(dataframe.DataFrame.select, "__rowless_wrapped__", False)
    assert getattr(dataframe.DataFrame.collect, "__rowless_wrapped__", False)
    assert interception.installed()


def test_the_wrappers_do_nothing_outside_a_dry_run() -> None:
    """Importing this adapter must not change how PySpark behaves for anybody else."""
    interception.install(context())
    assert run.tracking() is False


def test_validation_can_be_turned_off_per_repository() -> None:
    run.begin(context(validate_each_call=False))
    assert interception.validating() is False
    run.begin(context())
    assert interception.validating() is True


# --- Session selection -------------------------------------------------------


def test_an_endpoint_in_the_options_selects_connect_mode() -> None:
    run.begin(context(remote="sc://cluster.internal:15002"))
    assert cluster.mode() == "connect"
    assert cluster.endpoint() == "sc://cluster.internal:15002"
    assert cluster.egress_hosts() == ("cluster.internal",)


def test_an_endpoint_in_the_environment_is_honoured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPARK_REMOTE", "sc://from-the-shell:15002")
    run.begin(context())
    assert cluster.mode() == "connect"
    assert cluster.egress_hosts() == ("from-the-shell",)


def test_configured_local_mode_beats_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """An endpoint exported by a shell profile is not consent to send a dry run to it."""
    monkeypatch.setenv("SPARK_REMOTE", "sc://production:15002")
    run.begin(context(mode="local"))
    assert cluster.mode() == "local"
    assert "production" not in cluster.egress_hosts()


def test_local_mode_asks_for_loopback_and_a_subprocess() -> None:
    needs = ADAPTER.egress({"mode": "local"})
    assert needs.subprocesses is True
    assert "127.0.0.1" in needs.hosts
    assert needs.reason


def test_connect_mode_asks_for_exactly_one_host() -> None:
    needs = ADAPTER.egress({"remote": "sc://warehouse.example:15002/;token=secret"})
    assert needs.hosts == ("warehouse.example",)
    assert needs.subprocesses is False


def test_no_endpoint_asks_for_nothing() -> None:
    assert ADAPTER.egress({"mode": "connect"}).hosts == ()


@pytest.mark.parametrize(
    ("url", "host"),
    [
        ("sc://host:15002", "host"),
        ("sc://host", "host"),
        ("host:15002", "host"),
        ("sc://host:15002/;token=abc", "host"),
    ],
)
def test_the_endpoint_host_is_what_gets_allowlisted(url: str, host: str) -> None:
    assert cluster.endpoint_host(url) == host


@pytest.mark.parametrize(
    ("release", "expected"),
    [
        ('JAVA_VERSION="17.0.11"\n', 17),
        ('JAVA_VERSION="1.8.0_451"\n', 8),
        ('JAVA_VERSION="21"\n', 21),
        ("NOTHING_USEFUL=1\n", None),
    ],
)
def test_the_java_version_is_read_from_the_jdk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    release: str,
    expected: Any,
) -> None:
    """Read from the `release` file, because a dry run may not spawn `java -version`."""
    (tmp_path / "release").write_text(release, encoding="utf-8")
    monkeypatch.setenv("JAVA_HOME", str(tmp_path))
    assert cluster.java_version() == expected


def test_an_old_java_makes_local_mode_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Java 8 is enough for Spark 3.5 and not for Spark 4, so the Spark is pinned."""
    monkeypatch.setattr(cluster, "pyspark_version", lambda: (4, 0))
    (tmp_path / "release").write_text('JAVA_VERSION="1.8.0_451"\n', encoding="utf-8")
    monkeypatch.setenv("JAVA_HOME", str(tmp_path))
    run.begin(context(mode="local"))
    assert cluster.java_is_new_enough() is False
    assert ADAPTER.available() is False
    reason = ADAPTER.unavailable_reason()
    assert "Java" in reason and "Connect" in reason


def test_an_unreadable_java_is_reported_rather_than_assumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("JAVA_HOME", str(tmp_path))
    run.begin(context(mode="local"))
    assert cluster.java_version() is None
    assert ADAPTER.available() is False
    assert "JAVA_HOME" in ADAPTER.unavailable_reason()


def test_a_connect_client_that_will_not_import_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 3.5 client needs Python 3.11, so an endpoint is not enough on its own."""
    monkeypatch.setattr(cluster, "connect_client_error", lambda: "ModuleNotFoundError: distutils")
    run.begin(context(remote="sc://cluster:15002"))

    assert ADAPTER.available() is False
    reason = ADAPTER.unavailable_reason()
    assert "distutils" in reason and "pyspark>=4" in reason


def test_the_connect_client_is_imported_rather_than_counted() -> None:
    """Counting dependencies says yes where the client still fails to import."""
    assert cluster.connect_client_available() == (cluster.connect_client_error() == "")


def test_a_configured_endpoint_makes_the_adapter_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An endpoint plus a client that imports is the whole requirement.

    The client is stubbed as importable rather than assumed: on the 3.5 floor
    under a modern Python it genuinely is not, which the test above covers.
    """
    monkeypatch.setattr(cluster, "connect_client_error", lambda: "")
    run.begin(context(remote="sc://cluster:15002"))

    assert ADAPTER.available() is True
    assert ADAPTER.unavailable_reason() == ""


def test_missing_endpoint_in_connect_mode_says_what_to_configure() -> None:
    run.begin(context(mode="connect"))
    assert ADAPTER.available() is False
    assert "endpoint" in ADAPTER.unavailable_reason()


@pytest.mark.parametrize(
    ("version", "expected"),
    [("4.0.0", (4, 0)), ("3.5.3", (3, 5)), ("4.1.0.dev0", (4, 1)), ("nonsense", None)],
)
def test_the_pyspark_version_is_parsed_for_the_floor(version: str, expected: Any) -> None:
    assert cluster.parse_version(version) == expected


def test_the_java_floor_follows_the_installed_spark(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cluster, "pyspark_version", lambda: (3, 5))
    assert cluster.required_java() == 8
    monkeypatch.setattr(cluster, "pyspark_version", lambda: (4, 0))
    assert cluster.required_java() == 17


def test_an_adopted_session_keeps_its_own_timezone(monkeypatch: pytest.MonkeyPatch) -> None:
    """Somebody else's session is not this adapter's to reconfigure."""
    adopted = SimpleNamespace(
        conf=SimpleNamespace(get=lambda key: "America/Denver"), is_stopped=False
    )
    monkeypatch.setattr(cluster, "_active_session", lambda: adopted)
    assert cluster.session() is adopted
    assert cluster.active_mode() == "adopted"
    assert cluster.session_timezone() == "America/Denver"


def test_a_stopped_session_is_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pipeline is entitled to call `spark.stop()`, and a check runs many."""
    stopped = SimpleNamespace(conf=SimpleNamespace(get=lambda key: "UTC"), is_stopped=False)
    monkeypatch.setattr(cluster, "_active_session", lambda: stopped)
    assert cluster.session() is stopped
    stopped.is_stopped = True
    cluster.forget_if_stopped()
    assert cluster.active_mode() is None


# --- The adapter surface -----------------------------------------------------


def test_capabilities_match_the_protocols_implemented() -> None:
    assert ADAPTER.capabilities.remote is True
    assert ADAPTER.capabilities.lazy is True
    assert isinstance(ADAPTER, SupportsColumnLineage)
    assert isinstance(ADAPTER, SupportsSql)
    assert isinstance(ADAPTER, NeedsEgress)
    assert not isinstance(ADAPTER, SupportsUdfProbe)
    assert ADAPTER.capabilities.udf_probe is False


def test_handles_claims_frames_and_columns_in_both_modes() -> None:
    import pyspark.sql.dataframe as classic_frames

    owners: list[type] = [classic_frames.DataFrame]
    if CONNECT_READY:
        import pyspark.sql.connect.dataframe as connect_frames

        owners.append(connect_frames.DataFrame)
    for owner in owners:
        assert ADAPTER.handles(object.__new__(owner))
    assert ADAPTER.handles(classic_column())
    assert not ADAPTER.handles(object())
    assert not ADAPTER.handles("a string")


def test_describe_refuses_a_column_and_names_the_fix() -> None:
    with pytest.raises(UnknownSchemaError, match=r"df\.select"):
        ADAPTER.describe(classic_column())


def test_describe_refuses_something_that_is_not_a_frame() -> None:
    with pytest.raises(UnknownSchemaError, match="DataFrame"):
        ADAPTER.describe(object())


def test_duplicate_output_columns_are_reported() -> None:
    duplicated = Schema(
        fields=(
            Field(name="created_at", dtype=string()),
            Field(name="created_at", dtype=string()),
        )
    )
    with pytest.raises(DuplicateColumn, match="created_at"):
        adapter_module._refuse_duplicates(duplicated)


def test_an_unresolved_input_dtype_degrades_the_run() -> None:
    ctx = context()
    found = ADAPTER._input_dtype(Field(name="mystery", dtype=unknown("source unreachable")), ctx)
    assert found.kind is Kind.NULL
    assert ctx.degradations and "mystery" in ctx.degradations[0]


def test_a_dtype_spark_lacks_degrades_the_run_rather_than_being_approximated() -> None:
    ctx = context()
    found = ADAPTER._input_dtype(Field(name="counter", dtype=uint(32)), ctx)
    assert found.kind is Kind.NULL
    assert ctx.degradations and "counter" in ctx.degradations[0]


def test_an_any_typed_input_is_noted_but_does_not_degrade() -> None:
    ctx = context()
    found = ADAPTER._input_dtype(Field(name="whatever", dtype=any_()), ctx)
    assert found.kind is Kind.NULL
    assert not ctx.degradations
    assert ctx.notes


def test_a_supported_input_dtype_is_left_alone() -> None:
    ctx = context()
    assert ADAPTER._input_dtype(Field(name="amount", dtype=int_(64)), ctx) == int_(64)
    assert not ctx.degradations


def test_run_sql_refuses_a_statement_that_could_change_something() -> None:
    with pytest.raises(UnknownSchemaError, match="INSERT"):
        ADAPTER.run_sql("INSERT INTO orders SELECT 1", {}, context())


def test_a_qualified_relation_name_registers_under_its_leaf() -> None:
    assert adapter_module._leaf_name('"analytics"."marts"."orders"') == "orders"
    assert adapter_module._leaf_name("orders") == "orders"


def test_a_trailing_semicolon_is_stripped() -> None:
    assert adapter_module._strip_terminator("SELECT 1;  ") == "SELECT 1"


def test_the_adapter_declares_its_conformance_cases() -> None:
    cases = ADAPTER.conformance_cases()
    assert len(cases) > 10
    named = {case.name for case in cases}
    assert {
        "select",
        "rename",
        "filter",
        "cast",
        "group_by_sum",
        "join_shared_key",
        "union_is_positional",
        "missing_column",
        "row_dependent_branch",
        "statement_that_is_not_a_query_is_not_sent",
    } <= named


def test_every_case_either_expects_a_schema_or_a_failure() -> None:
    for case in ADAPTER.conformance_cases():
        asserted = case.expect is not None or case.raises is not None
        asserted = asserted or case.expect_confidence is not None
        assert asserted, f"{case.name} asserts nothing"


def test_the_disagreements_are_written_down() -> None:
    """`reason` is the catalogue, and the suite prints it when a case fails."""
    cases = {case.name: case for case in ADAPTER.conformance_cases()}
    assert "by position" in (cases["union_is_positional"].reason or "")
    assert "decimal" in (cases["group_by_sum_decimal"].reason or "")
    assert "collect" in (cases["row_dependent_collect"].reason or "")
