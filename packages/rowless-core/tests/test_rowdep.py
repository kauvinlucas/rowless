"""Row-dependent control flow detection. The silent-wrong-answer guard."""

from __future__ import annotations

from rowless.diagnostics.codes import Code
from rowless.ir.confidence import Confidence
from rowless.runtime.dryrun import dry_run_callable
from rowless.runtime.rowdep import ROW_DEPENDENT_APIS, observe, recording
from rowless.testing import fixtures
from rowless_fake import FakeAdapter, FakeFrame

INPUTS = {"orders": fixtures.ORDERS}


def test_is_empty_downgrades_to_unknown(fake_engine: FakeAdapter) -> None:
    """A branch on `is_empty()` makes the whole result UNKNOWN."""

    def pipeline(orders: FakeFrame) -> FakeFrame:
        if orders.is_empty():
            return orders.select("order_id")
        return orders.select("order_id", "amount")

    result = dry_run_callable(pipeline, INPUTS, engine=fake_engine.name)

    assert result.output is not None, "the schema is still reported; it is just not trusted"
    assert result.coverage.confidence is Confidence.UNKNOWN
    assert not result.ok
    assert any(
        diagnostic.code is Code.ROW_DEPENDENT_CONTROL_FLOW for diagnostic in result.diagnostics
    )


def test_item_access_downgrades_to_unknown(fake_engine: FakeAdapter) -> None:
    """Reading a scalar out of an empty frame is not a usable result."""

    def pipeline(orders: FakeFrame) -> FakeFrame:
        threshold = orders.item()
        _ = threshold
        return orders.select("order_id")

    result = dry_run_callable(pipeline, INPUTS, engine=fake_engine.name)

    assert result.coverage.confidence is Confidence.UNKNOWN
    assert "item" in " ".join(result.coverage.reasons)


def test_reason_names_the_user_line() -> None:
    """The recorded location is the user frame, not the adapter frame."""
    with recording() as log:
        frame = FakeFrame(fixtures.ORDERS)
        frame.is_empty()

    assert len(log.observed) == 1
    entry = log.observed[0]
    assert entry.file is not None
    assert entry.file.endswith("test_rowdep.py"), f"pointed at {entry.file} instead of the test"
    assert entry.api == "fake.FakeFrame.is_empty"
    assert "production would take a different path" in entry.explanation


def test_clean_pipeline_stays_exact(fake_engine: FakeAdapter) -> None:
    """A pipeline with no row dependence keeps Confidence.EXACT."""

    def pipeline(orders: FakeFrame) -> FakeFrame:
        return orders.rename({"amount": "amount_cents"})

    result = dry_run_callable(pipeline, INPUTS, engine=fake_engine.name)

    assert result.coverage.confidence is Confidence.EXACT
    assert result.ok
    assert result.diagnostics == ()


def test_observe_outside_a_dry_run_is_a_no_op() -> None:
    """An adapter calls `observe` unconditionally; nothing is driving here."""
    observe("fake.FakeFrame.is_empty", returned=False)


def test_the_api_list_is_the_documented_one() -> None:
    """One canonical list, so a diagnostic can be looked up."""
    assert "is_empty" in ROW_DEPENDENT_APIS
    assert "iter_rows" in ROW_DEPENDENT_APIS
    assert "collect" in ROW_DEPENDENT_APIS
