"""Unit tests for the parts of the adapter the conformance suite cannot reach.

The suite in core is the specification, and it covers the contract. What it
cannot cover is the behaviour specific to this engine: which pandas exceptions
become which findings, what happens to an input dtype pandas does not have,
whether the interception reports the pipeline's own calls and not the engine's,
and - the part this adapter exists for - whether each entry in the drift table
corrects an empty frame to the dtype a *populated* frame would have produced.

That last battery is the important one. Every `Action.CORRECT` entry in
`rowless_pandas.drift` is run twice: once through rowless on a zero-row frame,
and once for real on three rows. An entry whose correction stops matching what
pandas does fails here, which is what keeps the table honest as pandas changes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from rowless.adapters.base import DryRunContext
from rowless.diagnostics.codes import Code
from rowless.errors import (
    DTypeMismatch,
    DuplicateColumn,
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
    enum,
    float_,
    int_,
    null,
    string,
    uint,
    unknown,
)
from rowless.ir.schema import Field, Schema
from rowless.runtime.dryrun import DryRunResult, dry_run_callable
from rowless.testing import fixtures
from rowless_pandas import drift, errors, interception
from rowless_pandas import dtypes as dtype_map
from rowless_pandas.adapter import PandasAdapter, _at_least

pytestmark = pytest.mark.requires_engine

ADAPTER = PandasAdapter()

#: A small schema with the three dtypes every case below needs, and nothing
#: that depends on pyarrow, so the whole file runs on either dtype table.
ORDERS = Schema(
    fields=(
        Field(name="order_id", dtype=int_(64), nullable=False),
        Field(name="status", dtype=string()),
        Field(name="amount", dtype=int_(64)),
    ),
    name="orders",
)

BACKENDS = ["arrow", "numpy"] if dtype_map.arrow_available() else ["numpy"]


@pytest.fixture
def ctx(tmp_path: Path) -> DryRunContext:
    return DryRunContext(run_id="test", workdir=str(tmp_path))


@pytest.fixture(autouse=True)
def _reset_backend() -> Any:
    """Leave the dtype table as the environment found it after every test."""
    yield
    dtype_map.use_backend(None)


def run(pipeline: Any, inputs: dict[str, Schema] | None = None, **options: Any) -> DryRunResult:
    """Dry-run one pipeline the way the CLI does, through the real runtime."""
    return dry_run_callable(
        pipeline, inputs or {"orders": ORDERS}, engine="pandas", options=options
    )


def populated() -> pd.DataFrame:
    """The same schema as `ORDERS`, with three rows in it.

    Built through `from_ir`, so a correction is compared against pandas running
    on the dtypes this adapter actually materializes rather than on whatever
    the constructor would have guessed.
    """
    return pd.DataFrame(
        {
            "order_id": pd.Series([1, 2, 3], dtype=dtype_map.from_ir(int_(64))),
            "status": pd.Series(["paid", "void", "paid"], dtype=dtype_map.from_ir(string())),
            "amount": pd.Series([10, 20, 30], dtype=dtype_map.from_ir(int_(64))),
        }
    )


# --- Availability and routing ------------------------------------------------


def test_available_with_pandas_installed() -> None:
    assert ADAPTER.available() is True


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("2.2.0", True),
        ("2.2.3", True),
        ("3.1.0", True),
        ("2.1.4", False),
        ("1.5.3", False),
        ("nonsense", False),
    ],
)
def test_version_floor(version: str, expected: bool) -> None:
    assert _at_least(version, (2, 2)) is expected


def test_handles_the_pandas_types_and_nothing_else() -> None:
    assert ADAPTER.handles(pd.DataFrame())
    assert ADAPTER.handles(pd.Series([], dtype="int64"))
    assert not ADAPTER.handles(object())
    assert not ADAPTER.handles({"amount": "int64"})
    assert not ADAPTER.handles(np.dtype("int64"))


def test_capabilities_are_what_the_adapter_implements() -> None:
    # The conformance suite checks the protocol-backed flags; these two are
    # the deliberate absences, and a future port should have to change this
    # test on purpose rather than discover the flags drifted.
    assert ADAPTER.capabilities.column_lineage is False
    assert ADAPTER.capabilities.nested_types is False
    assert ADAPTER.capabilities.udf_probe is True
    assert not hasattr(ADAPTER, "column_lineage")


# --- Dtypes ------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_from_ir_refuses_rather_than_approximating(backend: str) -> None:
    dtype_map.use_backend(backend)  # type: ignore[arg-type]
    with pytest.raises(UnsupportedDType):
        ADAPTER.from_ir(unknown("the source could not say"))
    with pytest.raises(UnsupportedDType, match="8, 16, 32 and 64"):
        ADAPTER.from_ir(int_(128))


def test_the_numpy_table_refuses_the_five_dtypes_pandas_lacks() -> None:
    dtype_map.use_backend("numpy")
    for dtype in (decimal(18, 2), binary(), date(), null()):
        with pytest.raises(UnsupportedDType, match="without pyarrow"):
            ADAPTER.from_ir(dtype)
    assert dtype_map.absent_dtypes(prefer="numpy") != ()
    assert dtype_map.absent_dtypes(prefer="arrow") == ()


def test_to_ir_of_an_object_column_is_unknown_with_a_reason() -> None:
    found = ADAPTER.to_ir(np.dtype("O"))
    assert found.kind is Kind.UNKNOWN
    assert found.reason is not None and "arbitrary Python values" in found.reason


def test_to_ir_reads_every_pandas_spelling() -> None:
    assert ADAPTER.to_ir(np.dtype("int32")) == int_(32)
    assert ADAPTER.to_ir(pd.Int32Dtype()) == int_(32)
    assert ADAPTER.to_ir("Int32") == int_(32)
    assert ADAPTER.to_ir(pd.BooleanDtype()) == boolean()
    assert ADAPTER.to_ir(pd.StringDtype()) == string()
    assert ADAPTER.to_ir(np.dtype("uint8")) == uint(8)
    assert ADAPTER.to_ir("datetime64[ms]") == datetime(TimeUnit.MILLISECOND)
    assert ADAPTER.to_ir("timedelta64[us]").kind is Kind.DURATION


def test_a_named_timezone_survives_the_round_trip() -> None:
    aware = datetime(TimeUnit.NANOSECOND, "Europe/London")
    for backend in BACKENDS:
        dtype_map.use_backend(backend)  # type: ignore[arg-type]
        assert ADAPTER.to_ir(ADAPTER.from_ir(aware)) == aware


def test_a_tz_aware_dtype_is_not_read_as_a_naive_one() -> None:
    # `DatetimeTZDtype.numpy_dtype` is the naive dtype, so reading it through
    # the generic numpy path would drop the zone. The order of the checks in
    # `to_ir` is what prevents that, and this is the test that pins it.
    found = ADAPTER.to_ir(pd.DatetimeTZDtype(unit="us", tz="UTC"))
    assert found.tz == "UTC"


def test_a_string_categorical_is_an_enum_and_a_numeric_one_is_its_values() -> None:
    assert ADAPTER.to_ir(pd.CategoricalDtype(["a", "b", "c"])) == enum(("a", "b", "c"))
    # The IR has no numeric enum, so the closed variant set is what is lost -
    # here, where it can be explained, rather than silently downstream.
    assert ADAPTER.to_ir(pd.CategoricalDtype([1, 2, 3])).kind is Kind.INT


def test_pandas_dtypes_with_no_ir_equivalent_are_unknown() -> None:
    assert ADAPTER.to_ir(pd.PeriodDtype("D")).kind is Kind.UNKNOWN
    assert ADAPTER.to_ir(pd.IntervalDtype("int64")).kind is Kind.UNKNOWN


def test_sparse_reads_through_to_its_subtype() -> None:
    assert ADAPTER.to_ir(pd.SparseDtype("int32")) == int_(32)


def test_any_lowers_to_object_so_a_gradual_schema_still_materializes(
    ctx: DryRunContext,
) -> None:
    dtype_map.use_backend("numpy")
    schema = Schema(fields=(Field(name="whatever", dtype=any_()),))
    frame = ADAPTER.materialize(schema, ctx)
    assert list(frame.columns) == ["whatever"]
    assert any("typed ANY" in note for note in ctx.notes)
    assert ctx.degradations == []


def test_unresolved_input_dtype_degrades_the_run_instead_of_failing_it(
    ctx: DryRunContext,
) -> None:
    schema = Schema(
        fields=(
            Field(name="order_id", dtype=int_(64)),
            Field(name="mystery", dtype=unknown("the source would not say")),
        )
    )
    frame = ADAPTER.materialize(schema, ctx)
    assert list(frame.columns) == ["order_id", "mystery"]
    assert any("mystery" in reason for reason in ctx.degradations)


def test_an_input_dtype_pandas_lacks_degrades_the_run(ctx: DryRunContext) -> None:
    numpy_only = DryRunContext(run_id="numpy", workdir=ctx.workdir, options={"arrow_dtypes": False})
    schema = Schema(fields=(Field(name="money", dtype=decimal(18, 2)),))
    frame = ADAPTER.materialize(schema, numpy_only)
    assert frame.dtypes["money"] == np.dtype("O")
    assert any("money" in reason for reason in numpy_only.degradations)


def test_the_numpy_fallback_is_reported_rather_than_silent(ctx: DryRunContext) -> None:
    context = DryRunContext(run_id="r", workdir=ctx.workdir, options={"arrow_dtypes": False})
    ADAPTER.materialize(ORDERS, context)
    assert any("masked extension dtypes" in note for note in context.notes)


# --- Describe ----------------------------------------------------------------


def test_describe_reads_a_frame_and_a_series(ctx: DryRunContext) -> None:
    frame = ADAPTER.materialize(ORDERS, ctx)
    relation = ADAPTER.describe(frame)
    assert tuple(relation.schema) == ("order_id", "status", "amount")
    assert relation.engine == "pandas"
    series = ADAPTER.describe(frame["amount"])
    assert tuple(series.schema) == ("amount",)


def test_describe_names_an_unnamed_series(ctx: DryRunContext) -> None:
    relation = ADAPTER.describe(pd.Series([], dtype="int64"))
    assert tuple(relation.schema) == ("series",)


def test_describe_refuses_something_that_is_not_a_frame() -> None:
    with pytest.raises(UnknownSchemaError, match="DataFrame or a Series"):
        ADAPTER.describe({"amount": [1, 2]})


def test_describe_refuses_duplicate_column_labels(ctx: DryRunContext) -> None:
    frame = ADAPTER.materialize(ORDERS, ctx)
    doubled = pd.concat([frame[["amount"]], frame[["amount"]]], axis=1)
    with pytest.raises(DuplicateColumn, match="two columns named 'amount'"):
        ADAPTER.describe(doubled)


def test_describe_refuses_a_multiindex_column_axis(ctx: DryRunContext) -> None:
    frame = ADAPTER.materialize(ORDERS, ctx)
    stacked = frame.groupby("status", observed=True).agg(["sum", "mean"])
    with pytest.raises(UnknownSchemaError, match="MultiIndex"):
        ADAPTER.describe(stacked)


def test_describe_reports_nullability_from_the_dtype(ctx: DryRunContext) -> None:
    frame = pd.DataFrame(
        {
            "plain": pd.Series([], dtype="int64"),
            "masked": pd.Series([], dtype="Int64"),
            "stamp": pd.Series([], dtype="datetime64[us]"),
        }
    )
    relation = ADAPTER.describe(frame)
    assert relation.schema["plain"].nullable is False
    assert relation.schema["masked"].nullable is True
    assert relation.schema["stamp"].nullable is True


def test_a_group_key_in_the_index_is_named_in_a_note(ctx: DryRunContext) -> None:
    frame = ADAPTER.materialize(ORDERS, ctx)
    grouped = frame.groupby("status", observed=True).agg({"amount": "sum"})
    relation = ADAPTER.describe(grouped)
    assert tuple(relation.schema) == ("amount",)
    assert any("status" in note for note in relation.notes)


def test_describe_does_not_report_its_own_reads_as_row_dependence() -> None:
    # Reading `frame.dtypes` iterates a Series, which is a patched accessor.
    # Without the own-work guard every described pipeline would come back
    # unverified, which is the bug this test exists to catch.
    result = run(lambda orders: orders[["order_id"]])
    assert result.coverage.confidence is Confidence.EXACT
    assert result.coverage.reasons == ()


# --- Error translation -------------------------------------------------------


def test_column_not_found_carries_the_name_and_the_near_miss() -> None:
    schema = Schema(
        fields=(
            Field(name="order_id", dtype=int_(64)),
            Field(name="amount_cents", dtype=int_(64)),
        )
    )
    result = run(lambda orders: orders[["amount"]], {"orders": schema})
    assert result.output is None
    (finding,) = [d for d in result.diagnostics if d.code is Code.COLUMN_NOT_FOUND]
    assert "amount" in finding.message
    assert finding.suggestion is not None and "amount_cents" in finding.suggestion


@pytest.mark.parametrize(
    "pipeline",
    [
        pytest.param(lambda orders: orders["nope"], id="getitem"),
        pytest.param(lambda orders: orders[["nope"]], id="projection"),
        pytest.param(lambda orders: orders.drop(columns=["nope"]), id="drop"),
        pytest.param(lambda orders: orders.set_index("nope"), id="set_index"),
        pytest.param(lambda orders: orders.sort_values("nope"), id="sort_values"),
        pytest.param(lambda orders: orders.groupby("nope"), id="groupby"),
        pytest.param(lambda orders: orders.query("nope > 1"), id="query"),
        pytest.param(lambda orders: orders.agg({"nope": "sum"}), id="agg"),
    ],
)
def test_every_spelling_of_a_missing_label_is_one_finding(pipeline: Any) -> None:
    result = run(pipeline)
    reported = [d.code for d in result.diagnostics]
    assert Code.COLUMN_NOT_FOUND in reported, reported


def test_a_user_key_error_is_not_mistaken_for_a_missing_column() -> None:
    # The whole reason `raised_in_pandas` exists: a KeyError from the
    # pipeline's own dictionary must not be reported as a column that does not
    # exist, and it must not be reported as nothing either.
    def pipeline(orders: pd.DataFrame) -> pd.DataFrame:
        config: dict[str, str] = {}
        return orders[[config["region"]]]

    result = run(pipeline)
    reported = [d.code for d in result.diagnostics]
    assert Code.COLUMN_NOT_FOUND not in reported
    assert Code.PIPELINE_RAISED in reported


def test_a_merge_with_no_common_column_is_a_join_finding() -> None:
    other = Schema(fields=(Field(name="country", dtype=string()),))
    result = run(
        lambda orders, customers: orders.merge(customers),
        {"orders": ORDERS, "customers": other},
    )
    assert Code.JOIN_KEY_MISMATCH in [d.code for d in result.diagnostics]


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda: pd.Series([1]).str.upper(), id="str_accessor_on_an_integer"),
        pytest.param(
            lambda: pd.Series(["a"], dtype="string") + pd.Series([1]), id="string_plus_integer"
        ),
        pytest.param(
            lambda: pd.Series(["a"], dtype="string").astype("int64"), id="uncastable_string"
        ),
    ],
)
def test_a_dtype_pandas_refuses_is_a_dtype_finding(call: Any) -> None:
    # pandas raises a bare TypeError or ValueError for all three and rewords
    # the message between releases, so these are matched by what the message
    # says rather than by the exception type. The parameters are the shapes
    # that have to keep working across the supported pandas range.
    assert isinstance(errors.translate(_raised_inside_pandas(call)), DTypeMismatch)


def test_an_unclassifiable_pandas_error_is_left_alone() -> None:
    raised = _raised_inside_pandas(lambda: pd.DataFrame({"a": [1]}).rename(index=0))
    assert errors.translate(raised) is None


def test_errors_raised_outside_pandas_are_left_alone() -> None:
    assert errors.translate(KeyError("region")) is None
    assert errors.translate(ValueError("nope")) is None


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("'amount'", ("amount",)),
        ("\"['amount'] not found in axis\"", ("amount",)),
        ("None of [Index(['amount'], dtype='str')] are in the [columns]", ("amount",)),
        ("None of ['amount'] are in the columns", ("amount",)),
        ("Label(s) ['amount'] do not exist", ("amount",)),
        ("['amount', 'status'] not found in axis", ("amount", "status")),
    ],
)
def test_every_message_shape_yields_the_label(message: str, expected: tuple[str, ...]) -> None:
    assert errors._labels(message) == expected


def _raised_inside_pandas(call: Any) -> BaseException:
    """Run `call`, expecting it to raise, and hand back the exception."""
    try:
        call()
    except Exception as exc:
        return exc
    raise AssertionError("the call was supposed to raise")


# --- Row dependence ----------------------------------------------------------


@pytest.mark.parametrize(
    "pipeline",
    [
        pytest.param(lambda orders: orders[["order_id"]] if orders.empty else orders, id="empty"),
        pytest.param(lambda orders: orders.assign(n=len(orders)), id="len"),
        pytest.param(lambda orders: orders.assign(top=orders["amount"].max()), id="series_max"),
        pytest.param(lambda orders: orders.assign(n=orders["status"].nunique()), id="nunique"),
        pytest.param(
            lambda orders: orders.assign(total=orders[["amount"]].sum()["amount"]), id="frame_sum"
        ),
        pytest.param(
            lambda orders: orders.assign(first=next(iter(orders["status"]), "none")), id="iter"
        ),
    ],
)
def test_a_value_read_costs_the_run_its_confidence(pipeline: Any) -> None:
    result = run(pipeline)
    assert result.coverage.confidence is Confidence.UNKNOWN
    assert result.coverage.reasons
    assert Code.ROW_DEPENDENT_CONTROL_FLOW in [d.code for d in result.diagnostics]


def test_the_engines_own_calls_are_not_reported_as_the_pipelines() -> None:
    # pandas reads `len(self)`, `.values` and `empty` constantly while it
    # works. Without the caller rule this pipeline would be unverified several
    # times over, which would make the adapter useless in CI.
    def pipeline(orders: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
        merged = orders.merge(customers, left_on="order_id", right_on="customer_id", how="left")
        merged = merged.sort_values("amount").drop_duplicates(subset=["order_id"])
        return merged.assign(label=merged["status"].str.upper()).reset_index(drop=True)

    result = run(pipeline, {"orders": ORDERS, "customers": fixtures.CUSTOMERS})
    assert result.coverage.confidence is Confidence.EXACT, result.coverage.reasons


def test_a_reduction_that_raises_on_zero_rows_is_unverified_not_an_error() -> None:
    result = run(lambda orders: orders.assign(top=orders["amount"].idxmax()))
    assert result.coverage.confidence is Confidence.UNKNOWN
    codes = [d.code for d in result.diagnostics]
    assert Code.ROW_DEPENDENT_CONTROL_FLOW in codes
    assert Code.PIPELINE_RAISED not in codes


def test_the_empty_reduction_error_is_still_a_value_error() -> None:
    # A pipeline that already guards `idxmax` with `except ValueError` keeps
    # working, because that is what pandas raises in production.
    def pipeline(orders: pd.DataFrame) -> pd.DataFrame:
        try:
            top = orders["amount"].idxmax()
        except ValueError:
            top = 0
        return orders.assign(top=top)

    result = run(pipeline)
    assert result.output is not None
    assert result.coverage.confidence is Confidence.UNKNOWN


# --- Writers -----------------------------------------------------------------


def test_a_write_is_recorded_rather_than_performed(tmp_path: Path) -> None:
    target = tmp_path / "orders.parquet"

    def pipeline(orders: pd.DataFrame) -> pd.DataFrame:
        orders.to_parquet(target)
        return orders

    result = run(pipeline)
    assert not target.exists()
    assert result.output is not None
    assert any("would write to" in note for note in result.output.notes)


def test_a_writer_with_no_destination_is_passed_through() -> None:
    def pipeline(orders: pd.DataFrame) -> pd.DataFrame:
        assert isinstance(orders.to_csv(), str)
        return orders

    assert run(pipeline).output is not None


def test_a_database_write_is_recorded_without_a_path() -> None:
    def pipeline(orders: pd.DataFrame) -> pd.DataFrame:
        orders.to_sql("warehouse.orders", con=object())
        return orders

    result = run(pipeline)
    assert result.output is not None
    assert any("to_sql" in note for note in result.output.notes)


# --- The drift table ---------------------------------------------------------


def annotated_cents(amount: int) -> float:
    """A UDF the adapter can type from its annotation."""
    return amount / 100


def annotated_label(amount: int) -> str:
    """The same, returning a string, so a wrong correction is visible."""
    return f"#{amount}"


def labelled_row(row: Any) -> str:
    """A row-wise UDF, for `DataFrame.apply(axis=1)`."""
    return f"#{row['amount']}"


def scaled_total(values: Any) -> float:
    """An aggregation pandas cannot type without calling it."""
    return float(values.sum()) / 100


#: Each correction, as a pipeline that runs on both an empty frame and a real
#: one. The assertion is that rowless and pandas agree, which is the only
#: definition of a correct fixup there is.
CORRECTIONS: dict[str, Any] = {
    "series_apply": lambda df: df.assign(x=df["amount"].apply(annotated_cents))[["x"]],
    "series_map_mapping": lambda df: df.assign(x=df["status"].map({"paid": 1.5, "void": 2.5}))[
        ["x"]
    ],
    "series_transform": lambda df: df.assign(x=df["amount"].transform(annotated_cents))[["x"]],
    "frame_map": lambda df: df[["amount"]].map(annotated_cents),
    "groupby_agg_dict": lambda df: df.groupby("status", as_index=False, observed=True).agg(
        {"amount": scaled_total}
    ),
    "groupby_agg_named": lambda df: df.groupby("status", as_index=False, observed=True).agg(
        total=("amount", scaled_total)
    ),
    "series_groupby_agg": lambda df: (
        df.groupby("status", observed=True)["amount"].agg(scaled_total).reset_index()
    ),
}


@pytest.mark.parametrize("name", sorted(CORRECTIONS))
@pytest.mark.parametrize("backend", BACKENDS)
def test_a_corrected_dtype_is_the_populated_dtype(name: str, backend: str) -> None:
    dtype_map.use_backend(backend)  # type: ignore[arg-type]
    pipeline = CORRECTIONS[name]
    result = run(pipeline, {"df": ORDERS}, arrow_dtypes=backend == "arrow")
    assert result.output is not None, result.coverage.reasons
    corrected = {field.name: field.dtype for field in result.output.schema.fields}

    real = pipeline(populated())
    frame = real.to_frame() if isinstance(real, pd.Series) else real
    truth = {str(column): ADAPTER.to_ir(dtype) for column, dtype in frame.dtypes.items()}

    assert corrected == truth
    assert result.coverage.confidence is Confidence.EXACT, result.coverage.reasons


#: The same battery for a UDF annotated `-> str`, which is the one place the
#: correction and the engine disagree. Kept separate so the disagreement is
#: asserted rather than hidden behind a looser comparison above.
STRING_CORRECTIONS: dict[str, Any] = {
    "series_map_function": lambda df: df.assign(x=df["amount"].map(annotated_label))[["x"]],
    "series_apply_function": lambda df: df.assign(x=df["amount"].apply(annotated_label))[["x"]],
    "frame_apply_rowwise": lambda df: df.apply(labelled_row, axis=1).to_frame("x"),
}


@pytest.mark.parametrize("name", sorted(STRING_CORRECTIONS))
def test_a_string_returning_udf_is_corrected_to_the_annotated_dtype(name: str) -> None:
    # The version difference, pinned in both directions. A function annotated
    # `-> str` lands in an `object` column on pandas 2.x and a `str` column on
    # 3.x. rowless reports `string` either way: the annotation is a
    # declaration, and an object column is not a dtype but the absence of one.
    pipeline = STRING_CORRECTIONS[name]
    result = run(pipeline, {"df": ORDERS})
    assert result.output is not None, result.coverage.reasons
    assert result.output.schema["x"].dtype == string()

    real = pipeline(populated())
    frame = real.to_frame() if isinstance(real, pd.Series) else real
    engine_answer = ADAPTER.to_ir(frame.dtypes["x"])
    assert engine_answer in (string(), unknown()), engine_answer
    if engine_answer.kind is Kind.UNKNOWN:
        assert not _at_least(pd.__version__, (3, 0)), (
            "pandas 3 types a str-returning UDF as str; only 2.x leaves it object"
        )


def test_a_correction_that_fired_is_in_the_notes() -> None:
    result = run(lambda orders: orders.assign(x=orders["amount"].apply(annotated_cents)))
    assert result.output is not None
    assert any("series_apply_return" in note for note in result.output.notes)


def test_an_untypeable_udf_degrades_rather_than_reporting_the_input_dtype() -> None:
    result = run(lambda orders: orders.assign(x=orders["amount"].apply(lambda v: v / 100)))
    assert result.coverage.confidence is Confidence.UNKNOWN
    assert any("series_apply_return" in reason for reason in result.coverage.reasons)


def test_probing_types_a_udf_that_nothing_else_can() -> None:
    result = run(
        lambda orders: orders.assign(x=orders["amount"].apply(lambda v: v / 100))[["x"]],
        probe_udfs=True,
    )
    assert result.output is not None
    assert result.output.schema["x"].dtype == float_(64)


@pytest.mark.parametrize(
    ("name", "pipeline"),
    [
        pytest.param(
            "groupby_apply_shape",
            lambda orders: orders.groupby("status", observed=True).apply(
                lambda group: group["amount"].sum()
            ),
            id="groupby_apply",
        ),
        pytest.param(
            "values_become_columns",
            lambda orders: orders.set_index(["status", "order_id"])["amount"].unstack(),
            id="unstack",
        ),
        pytest.param(
            "values_become_columns",
            lambda orders: orders.pivot_table(
                index="status", columns="order_id", values="amount", aggfunc="sum"
            ),
            id="pivot_table",
        ),
        pytest.param(
            "na_upcast",
            lambda orders: orders.assign(
                x=orders["amount"].astype("int64").where(orders["amount"] > 1)
            ),
            id="where_on_numpy",
        ),
    ],
)
def test_an_uncorrectable_drift_degrades_the_run(name: str, pipeline: Any) -> None:
    result = run(pipeline)
    assert result.coverage.confidence is Confidence.UNKNOWN
    assert any(name in reason for reason in result.coverage.reasons), result.coverage.reasons


def test_a_masked_dtype_does_not_drift_through_a_mask() -> None:
    # The argument for materializing masked dtypes: the same `where` that
    # degrades the run above costs nothing here.
    result = run(lambda orders: orders.assign(x=orders["amount"].where(orders["amount"] > 1)))
    assert result.coverage.confidence is Confidence.EXACT, result.coverage.reasons


def test_the_fixup_table_is_reported_in_full() -> None:
    described = ADAPTER.dtype_drift_fixups()
    assert len(described) == len(drift.FIXUPS)
    assert all(fixup.id in " ".join(described) for fixup in drift.FIXUPS)
    # Both kinds are reported: an adapter that listed only its corrections
    # would read as though the rest of the drift did not exist.
    assert any(drift.Action.CORRECT.value in line for line in described)
    assert any(drift.Action.DEGRADE.value in line for line in described)


# --- Interception hygiene ----------------------------------------------------


def test_pandas_is_unmodified_outside_a_dry_run(ctx: DryRunContext) -> None:
    ADAPTER.materialize(ORDERS, ctx)
    frame = pd.DataFrame({"amount": [1, 2, 3]})
    # Every wrapper is installed by now, and every one of them has to be a
    # no-op: this process is not running a dry run.
    assert frame.empty is False
    assert len(frame) == 3
    assert frame["amount"].max() == 3
    assert frame["amount"].apply(lambda v: v / 2).tolist() == [0.5, 1.0, 1.5]
    assert interception.tracking() is False


def test_installing_twice_does_not_double_wrap(ctx: DryRunContext) -> None:
    ADAPTER.materialize(ORDERS, ctx)
    once = pd.DataFrame.__getitem__
    ADAPTER.materialize(ORDERS, DryRunContext(run_id="other", workdir=ctx.workdir))
    assert pd.DataFrame.__getitem__ is once


def test_the_caller_rule_recognises_an_engine_frame() -> None:
    # The mechanism behind `called_from_user_code`: the frame above a callback
    # pandas invoked is pandas' own, and a wrapper reached from there is the
    # engine working rather than the pipeline.
    import sys

    seen: list[Any] = []
    pd.DataFrame({"amount": [1]}).apply(lambda row: seen.append(sys._getframe(1)), axis=1)
    assert seen and interception._is_engine(seen[0])
    assert not interception._is_rowless(seen[0])
