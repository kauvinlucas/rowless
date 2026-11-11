"""Unit tests for the parts of the adapter the conformance suite cannot reach.

The suite in core is the specification, and it covers the contract. What it
cannot cover is the behaviour that is specific to this engine: which Polars
exceptions become which findings, what happens to an input dtype Polars does
not have, and whether the lineage recorded while a plan is built is the lineage
a user would expect. That is what is here.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import polars as pl
import pytest
from rowless.adapters.base import DryRunContext
from rowless.diagnostics.codes import Code
from rowless.errors import (
    ColumnNotFound,
    DTypeMismatch,
    DuplicateColumn,
    UnknownSchemaError,
    UnsupportedDType,
)
from rowless.ir.confidence import Confidence
from rowless.ir.dtypes import (
    TimeUnit,
    any_,
    datetime,
    float_,
    int_,
    string,
    unknown,
)
from rowless.ir.schema import Field, Schema
from rowless.runtime.dryrun import dry_run_callable
from rowless.testing import fixtures
from rowless_polars import errors, lineage
from rowless_polars.adapter import PolarsAdapter, _at_least

pytestmark = pytest.mark.requires_engine

ADAPTER = PolarsAdapter()


@pytest.fixture
def ctx(tmp_path: Path) -> DryRunContext:
    return DryRunContext(run_id="test", workdir=str(tmp_path))


# --- Availability and routing ------------------------------------------------


def test_available_with_polars_installed() -> None:
    assert ADAPTER.available() is True


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("1.0.0", True),
        ("1.0.0-rc.1", True),
        ("2.1.0", True),
        ("0.20.31", False),
        ("nonsense", False),
    ],
)
def test_version_floor(version: str, expected: bool) -> None:
    assert _at_least(version, (1, 0)) is expected


def test_handles_the_polars_types_and_nothing_else() -> None:
    assert ADAPTER.handles(pl.LazyFrame())
    assert ADAPTER.handles(pl.DataFrame())
    assert ADAPTER.handles(pl.Series("a", [], dtype=pl.Int64))
    assert ADAPTER.handles(pl.col("a"))
    assert not ADAPTER.handles(object())
    assert not ADAPTER.handles({"a": "int64"})


# --- Dtypes ------------------------------------------------------------------


def test_from_ir_refuses_rather_than_approximating() -> None:
    with pytest.raises(UnsupportedDType):
        ADAPTER.from_ir(unknown("the source could not say"))
    with pytest.raises(UnsupportedDType, match="ms, us and ns"):
        ADAPTER.from_ir(datetime(TimeUnit.SECOND))
    with pytest.raises(UnsupportedDType, match="Float32 and Float64"):
        ADAPTER.from_ir(float_(16))


def test_to_ir_of_an_untyped_polars_dtype_is_unknown() -> None:
    for native in (pl.Object, pl.Unknown):
        dtype = ADAPTER.to_ir(native)
        assert not dtype.is_known
        assert dtype.reason


def test_categorical_is_a_string_and_enum_is_not() -> None:
    assert ADAPTER.to_ir(pl.Categorical) == string()
    assert ADAPTER.to_ir(pl.Enum(["a", "b"])).names == ("a", "b")


def test_fixed_size_array_degrades_to_a_list() -> None:
    assert ADAPTER.to_ir(pl.Array(pl.Int64, 3)) == ADAPTER.to_ir(pl.List(pl.Int64))


def test_any_lowers_to_null_so_a_gradual_schema_still_materializes(ctx: DryRunContext) -> None:
    schema = Schema(fields=(Field(name="mystery", dtype=any_()),))
    frame = ADAPTER.materialize(schema, ctx)
    assert frame.collect_schema() == pl.Schema({"mystery": pl.Null})
    assert not ctx.degradations
    assert any("ANY" in note for note in ctx.notes)


def test_unresolved_input_dtype_degrades_the_run_instead_of_failing_it(ctx: DryRunContext) -> None:
    schema = Schema(
        fields=(
            Field(name="order_id", dtype=int_(64)),
            Field(name="mystery", dtype=unknown("source has no type for this")),
        )
    )
    frame = ADAPTER.materialize(schema, ctx)
    assert tuple(frame.collect_schema()) == ("order_id", "mystery")
    assert any("mystery" in reason for reason in ctx.degradations)


def test_unsupported_input_dtype_degrades_the_run(ctx: DryRunContext) -> None:
    schema = Schema(fields=(Field(name="when", dtype=datetime(TimeUnit.SECOND)),))
    ADAPTER.materialize(schema, ctx)
    assert any("when" in reason for reason in ctx.degradations)


# --- describe ----------------------------------------------------------------


def test_describe_reads_a_lazy_frame_without_executing(ctx: DryRunContext) -> None:
    frame = ADAPTER.materialize(fixtures.ORDERS, ctx)
    relation = ADAPTER.describe(frame)
    assert tuple(relation.schema) == tuple(fixtures.ORDERS)
    assert relation.engine == "polars"


def test_describe_reads_an_eager_frame_and_a_series(ctx: DryRunContext) -> None:
    frame = ADAPTER.materialize(fixtures.ORDERS, ctx).collect()
    assert tuple(ADAPTER.describe(frame).schema) == tuple(fixtures.ORDERS)
    series = pl.Series("amount", [], dtype=pl.Int64)
    assert ADAPTER.describe(series).schema["amount"].dtype == int_(64)


def test_describe_refuses_an_expression() -> None:
    with pytest.raises(UnknownSchemaError, match="cannot describe"):
        ADAPTER.describe(pl.col("amount"))


def test_describe_reports_every_column_nullable(ctx: DryRunContext) -> None:
    # Polars does not model nullability. The IR says so rather than inventing
    # the non-null constraint the fixture declares.
    relation = ADAPTER.describe(ADAPTER.materialize(fixtures.ORDERS, ctx))
    assert fixtures.ORDERS["order_id"].nullable is False
    assert relation.schema["order_id"].nullable is True


# --- Error translation -------------------------------------------------------


def test_column_not_found_carries_the_name_and_the_near_miss() -> None:
    raised = pl.exceptions.ColumnNotFoundError(
        'unable to find column "amount"; valid columns: ["order_id", "amount_cents"]\n\n'
        "Resolved plan until failure:\n\n\t---> FAILED HERE RESOLVING 'select' <---"
    )
    translated = errors.translate(raised)
    assert isinstance(translated, ColumnNotFound)
    assert translated.column == "amount"
    assert translated.available == ("order_id", "amount_cents")
    assert translated.suggestion == "amount_cents"
    assert "Resolved plan" not in str(translated)


@pytest.mark.parametrize(
    ("raised", "expected"),
    [
        (pl.exceptions.DuplicateError("duplicate output name 'amount'"), DuplicateColumn),
        (pl.exceptions.SchemaError("expected String, got Int64"), DTypeMismatch),
        (pl.exceptions.InvalidOperationError("cannot sum a string"), DTypeMismatch),
        (pl.exceptions.StructFieldNotFoundError("source"), ColumnNotFound),
    ],
)
def test_classified_polars_errors(raised: Exception, expected: type[Exception]) -> None:
    assert isinstance(errors.translate(raised), expected)


def test_udf_inference_failure_is_unverified_not_an_error() -> None:
    raised = pl.exceptions.InvalidOperationError(
        "UDF called without return type, but was not able to infer the output type"
    )
    translated = errors.translate(raised)
    assert isinstance(translated, errors.UnknownUdfReturn)
    assert translated.code is Code.UNKNOWN_UDF_RETURN
    assert "return_dtype" in str(translated)


def test_an_unclassifiable_polars_error_is_left_alone() -> None:
    # A ComputeError says something went wrong without saying what, and core
    # reports it under PIPELINE_RAISED with the engine message attached.
    assert errors.translate(pl.exceptions.ComputeError("something went wrong")) is None
    assert errors.translate(ValueError("user code raised this")) is None


def test_polars_errors_are_recognised_by_hierarchy() -> None:
    assert errors.is_polars_error(pl.exceptions.ComputeError("x"))
    assert not errors.is_polars_error(ValueError("x"))


# --- Plan validation at the user line ----------------------------------------


def broken_pipeline(orders: pl.LazyFrame) -> pl.LazyFrame:
    return orders.select("order_id", "amount_cents")


def test_a_plan_error_points_at_the_line_that_built_it() -> None:
    result = dry_run_callable(broken_pipeline, {"orders": fixtures.ORDERS}, engine="polars")
    expected_line = inspect.getsourcelines(broken_pipeline)[1] + 1
    assert result.output is None
    (finding,) = result.diagnostics
    assert finding.code is Code.COLUMN_NOT_FOUND
    assert finding.origin is not None
    assert finding.origin.file == __file__
    assert finding.origin.line == expected_line
    assert finding.suggestion is not None


# --- Lineage rules -----------------------------------------------------------


def test_carry_keeps_only_the_columns_that_survived() -> None:
    parent = {"a": ("a",), "b": ("b",)}
    assert lineage.carry(parent, ["a"]) == {"a": ("a",)}


def test_expressions_attribute_an_alias_to_its_roots() -> None:
    parent = {"amount": ("amount",), "order_id": ("order_id",)}
    resolved = lineage.from_expressions(
        parent,
        ["total"],
        [(pl.col("amount") + pl.col("order_id")).alias("total")],
        carry_rest=False,
    )
    assert resolved == {"total": ("amount", "order_id")}


def test_a_column_from_no_input_column_is_untraced() -> None:
    resolved = lineage.from_expressions(
        {"amount": ("amount",)},
        ["n"],
        [pl.len().alias("n")],
        carry_rest=True,
    )
    assert resolved == {}


def test_wildcards_fall_back_to_the_names_that_did_not_change() -> None:
    resolved = lineage.from_expressions(
        {"amount": ("amount",)},
        ["amount"],
        [pl.all()],
        carry_rest=True,
    )
    assert resolved == {"amount": ("amount",)}


def test_rename_moves_lineage_to_the_new_name() -> None:
    resolved = lineage.renamed(
        {"amount": ("amount",)},
        ["amount_cents"],
        {"amount": "amount_cents"},
    )
    assert resolved == {"amount_cents": ("amount",)}


def test_join_takes_each_column_from_the_side_it_came_from() -> None:
    resolved = lineage.joined(
        {"order_id": ("order_id",)},
        {"country": ("country",), "created_at": ("created_at",)},
        ["order_id", "country", "created_at_right"],
        suffix="_right",
    )
    assert resolved == {
        "order_id": ("order_id",),
        "country": ("country",),
        "created_at_right": ("created_at",),
    }


def test_concat_unions_the_sources_of_each_column() -> None:
    resolved = lineage.merged(
        [{"amount": ("amount",)}, {"amount": ("order_id",)}, None],
        ["amount"],
    )
    assert resolved == {"amount": ("amount", "order_id")}


# --- Lineage end to end ------------------------------------------------------


def traced_pipeline(orders: pl.LazyFrame, customers: pl.LazyFrame) -> pl.LazyFrame:
    joined = orders.join(customers, left_on="order_id", right_on="customer_id", how="inner")
    return (
        joined.filter(pl.col("status") == "paid")
        .rename({"amount": "amount_cents"})
        .group_by(pl.col("created_at").dt.date().alias("day"), "country")
        .agg(pl.col("amount_cents").sum().alias("revenue"), pl.len().alias("orders"))
    )


def test_lineage_survives_a_join_a_rename_and_an_aggregation() -> None:
    result = dry_run_callable(
        traced_pipeline,
        {"orders": fixtures.ORDERS, "customers": fixtures.CUSTOMERS},
        engine="polars",
    )
    assert result.coverage.confidence is Confidence.EXACT
    assert result.output is not None
    traced = {column.name: column.derived_from for column in result.output.schema.fields}
    assert traced == {
        "day": ("created_at",),
        "country": ("country",),
        "revenue": ("amount",),
        # `pl.len()` counts rows; no input column produced it, and the coverage
        # score is what reports that honestly.
        "orders": (),
    }
    assert result.coverage.traced_ratio == 0.75


def test_an_untracked_frame_reports_no_lineage() -> None:
    assert ADAPTER.column_lineage(pl.LazyFrame()) == {}


# --- Interception ------------------------------------------------------------


def collecting_pipeline(orders: pl.LazyFrame) -> pl.DataFrame:
    return orders.select("order_id", "amount").collect()


def test_collecting_is_noted_but_does_not_cost_confidence() -> None:
    result = dry_run_callable(collecting_pipeline, {"orders": fixtures.ORDERS}, engine="polars")
    assert result.coverage.confidence is Confidence.EXACT
    assert result.output is not None
    assert any("executed the plan" in note for note in result.output.notes)
    # Lineage survives the hop from LazyFrame to DataFrame.
    assert result.output.schema["amount"].derived_from == ("amount",)


def row_dependent_pipeline(orders: pl.LazyFrame) -> pl.LazyFrame:
    frame = orders.collect()
    return orders.select("order_id") if frame.is_empty() else orders


def test_a_row_dependent_branch_is_reported_unverified() -> None:
    result = dry_run_callable(row_dependent_pipeline, {"orders": fixtures.ORDERS}, engine="polars")
    assert result.coverage.confidence is Confidence.UNKNOWN
    assert any("is_empty" in reason for reason in result.coverage.reasons)
    codes = [diagnostic.code for diagnostic in result.diagnostics]
    assert Code.ROW_DEPENDENT_CONTROL_FLOW in codes


def expression_reduction_pipeline(orders: pl.LazyFrame) -> pl.LazyFrame:
    # The counterpart to the test above: `sum()` on an expression is resolved
    # by the type rules, so it must not be reported as row-dependent.
    return orders.group_by("status").agg(
        pl.col("amount").sum().alias("total"),
        pl.col("amount").max().alias("largest"),
    )


def test_expression_reductions_are_not_row_dependent() -> None:
    result = dry_run_callable(
        expression_reduction_pipeline, {"orders": fixtures.ORDERS}, engine="polars"
    )
    assert result.coverage.confidence is Confidence.EXACT
    assert not result.coverage.reasons


def writing_pipeline(orders: pl.LazyFrame) -> pl.DataFrame:
    frame = orders.select("amount").collect()
    frame.write_parquet("warehouse/revenue.parquet")
    return frame


def test_a_write_is_recorded_rather_than_performed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = dry_run_callable(writing_pipeline, {"orders": fixtures.ORDERS}, engine="polars")
    assert result.output is not None
    assert any("would write to" in note for note in result.output.notes)
    assert not list(tmp_path.glob("**/*.parquet"))


def converting_pipeline(orders: pl.LazyFrame) -> pl.DataFrame:
    # `write_csv` with no destination is a conversion, not a side effect, and
    # has to keep working.
    frame = orders.select("amount").collect()
    assert isinstance(frame.write_csv(), str)
    return frame


def test_a_writer_with_no_destination_is_passed_through() -> None:
    result = dry_run_callable(converting_pipeline, {"orders": fixtures.ORDERS}, engine="polars")
    assert result.output is not None


def annotated(amount: int) -> float:
    return amount / 100


def test_a_udf_return_annotation_is_used_without_calling_the_function() -> None:
    schema = Schema(fields=(Field(name="amount", dtype=int_(64)),))
    ctx = DryRunContext(run_id="udf", workdir=".")
    assert ADAPTER.probe_udf(annotated, schema, ctx) == float_(64)
    assert any("annotation" in note for note in ctx.notes)


def test_a_udf_with_nothing_to_go_on_is_any_until_probing_is_enabled() -> None:
    schema = Schema(fields=(Field(name="status", dtype=string()),))
    plain = DryRunContext(run_id="udf", workdir=".")
    assert ADAPTER.probe_udf(lambda value: value.upper(), schema, plain).is_any

    probing = DryRunContext(run_id="udf", workdir=".", options={"probe_udfs": True})
    assert ADAPTER.probe_udf(lambda value: value.upper(), schema, probing) == string()


def test_polars_is_unmodified_outside_a_dry_run() -> None:
    # Every wrapper is gated on an open dry run, so a process that imported the
    # adapter still gets lazy Polars: a bad plan resolves, and fails later.
    frame = pl.LazyFrame(schema={"amount": pl.Int64}).select("amount_cents")
    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        frame.collect_schema()
