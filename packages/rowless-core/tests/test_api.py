"""The public API: `source`, `pipeline`, `schema_of` and `assert_schema`."""

from __future__ import annotations

import pytest
import rowless
from rowless.adapters.base import DryRunContext
from rowless.api import dry_run_session, marker_of
from rowless.errors import RowlessError, UnknownSchemaError
from rowless.ir.dtypes import any_, int_, string
from rowless.ir.schema import Schema
from rowless.testing import fixtures
from rowless_fake import FakeAdapter, FakeFrame

ORDERS = {"orders": fixtures.ORDERS}


def test_source_outside_a_dry_run_refuses_to_read_anything() -> None:
    """Production code reads its inputs the way it always did."""
    with pytest.raises(RowlessError, match="only works inside a dry run"):
        rowless.source("lake://stg_orders")


def test_source_inside_a_dry_run_returns_a_real_frame(
    fake_engine: FakeAdapter,
    scratch: object,
) -> None:
    """A module-level declaration resolves to the schema the parent fetched."""
    ctx = DryRunContext(run_id="test", workdir=str(scratch))
    with dry_run_session(fake_engine, ctx, {"lake://stg_orders": fixtures.ORDERS}):
        frame = rowless.source("lake://stg_orders")

        assert isinstance(frame, FakeFrame)
        assert tuple(frame.schema) == tuple(fixtures.ORDERS)

        with pytest.raises(UnknownSchemaError, match="resolved inputs"):
            rowless.source("lake://unknown")


def test_pipeline_marker_leaves_the_function_alone() -> None:
    """Rowless is a checker, not a runtime: the decorator wraps nothing."""

    @rowless.pipeline(inputs={"orders": "lake://stg_orders"}, engine="fake")
    def daily_revenue(orders: FakeFrame) -> FakeFrame:
        return orders

    frame = FakeFrame(fixtures.ORDERS)
    assert daily_revenue(frame) is frame

    marker = marker_of(daily_revenue)
    assert marker is not None
    assert marker.inputs == {"orders": "lake://stg_orders"}
    assert marker.engine == "fake"


def test_pipeline_works_bare_as_well() -> None:
    """`@rowless.pipeline` with no arguments is still a marker."""

    @rowless.pipeline
    def totals(orders: FakeFrame) -> FakeFrame:
        return orders

    assert marker_of(totals) is not None


def test_schema_of_routes_through_the_adapter(fake_engine: FakeAdapter) -> None:
    """The same call the runtime makes, exposed for tests and notebooks."""
    assert rowless.schema_of(FakeFrame(fixtures.ORDERS)) == fixtures.ORDERS


def test_assert_schema_passes_on_a_matching_contract(fake_engine: FakeAdapter) -> None:
    """Contract tests in milliseconds: the pytest plugin's whole premise."""

    def daily_revenue(orders: FakeFrame) -> FakeFrame:
        return orders.select("order_id", "status")

    rowless.assert_schema(
        daily_revenue,
        {"order_id": "int64", "status": "string"},
        inputs=ORDERS,
    )


def test_assert_schema_reports_the_difference(fake_engine: FakeAdapter) -> None:
    """An AssertionError has to say which column and which dtype."""

    def daily_revenue(orders: FakeFrame) -> FakeFrame:
        return orders.select("order_id", "status")

    with pytest.raises(AssertionError) as caught:
        rowless.assert_schema(
            daily_revenue,
            {"order_id": "int32", "missing": "date"},
            inputs=ORDERS,
            exact=False,
        )

    message = str(caught.value)
    assert "order_id: expected int32, got int64" in message
    assert "missing column 'missing'" in message


def test_assert_schema_exactness_is_the_default(fake_engine: FakeAdapter) -> None:
    """Extra columns are a contract violation unless the test opted out."""

    def daily_revenue(orders: FakeFrame) -> FakeFrame:
        return orders.select("order_id", "status")

    with pytest.raises(AssertionError, match="unexpected column"):
        rowless.assert_schema(daily_revenue, {"order_id": "int64"}, inputs=ORDERS)

    rowless.assert_schema(daily_revenue, {"order_id": "int64"}, inputs=ORDERS, exact=False)


def test_assert_schema_never_passes_on_an_unverified_result(fake_engine: FakeAdapter) -> None:
    """A contract test that silently stops testing is worse than no test."""

    def row_dependent(orders: FakeFrame) -> FakeFrame:
        if orders.is_empty():
            return orders
        return orders.select("order_id")

    with pytest.raises(AssertionError, match="proves nothing"):
        rowless.assert_schema(row_dependent, {"order_id": "int64"}, inputs=ORDERS)


def test_assert_schema_refuses_an_any_column(fake_engine: FakeAdapter) -> None:
    """ANY matches anything in inference; in a test it is not evidence."""

    def guessing(orders: FakeFrame) -> FakeFrame:
        return orders.select("order_id").derive("amount", any_())

    with pytest.raises(AssertionError, match="not evidence"):
        rowless.assert_schema(
            guessing,
            {"order_id": "int64", "amount": "int64"},
            inputs=ORDERS,
        )


def test_assert_schema_takes_a_schema_object_too(fake_engine: FakeAdapter) -> None:
    """The shorthand is a convenience, not the only way in."""

    def daily_revenue(orders: FakeFrame) -> FakeFrame:
        return orders.select("order_id", "status")

    rowless.assert_schema(
        daily_revenue,
        Schema.from_mapping({"order_id": int_(64), "status": string()}),
        inputs=ORDERS,
    )


def test_assert_schema_uses_declared_inputs_when_none_are_passed(
    fake_engine: FakeAdapter,
    tmp_path: object,
) -> None:
    """A user declares their inputs once, on the pipeline."""
    schema_file = f"{tmp_path}/orders.json".replace("\\", "/")
    with open(schema_file, "w", encoding="utf-8") as handle:
        handle.write('{"order_id": "int64"}')

    @rowless.pipeline(inputs={"orders": f"json://{schema_file}"}, engine="fake")
    def totals(orders: FakeFrame) -> FakeFrame:
        return orders.select("order_id")

    rowless.assert_schema(totals, {"order_id": "int64"})


def test_assert_schema_says_what_to_do_without_inputs(fake_engine: FakeAdapter) -> None:
    """The error names the fix rather than failing deep inside the runtime."""

    def totals(orders: FakeFrame) -> FakeFrame:
        return orders

    with pytest.raises(AssertionError, match=r"@rowless\.pipeline"):
        rowless.assert_schema(totals, {"order_id": "int64"})
