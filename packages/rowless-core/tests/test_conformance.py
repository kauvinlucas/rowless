"""The conformance suite, run against the fake adapter.

Two things are checked here. The obvious one is that the suite works: an
adapter author's first move is to run it, and a harness that errors on
collection teaches them nothing. The less obvious one is that the suite
actually *fails* a dishonest adapter, because a conformance suite that passes
everything is worse than none.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from rowless.adapters.base import Capabilities
from rowless.ir.dtypes import Kind, int_, string, unknown
from rowless.ir.schema import Schema
from rowless.testing.conformance import (
    ConformanceCase,
    adapter_conformance,
    dtype_corpus,
    pipeline_battery,
)
from rowless_fake import FakeAdapter, install

#: The generated suite for the fake adapter. Collected and run by pytest like
#: any other test class, which is exactly how an adapter package uses it.
TestFakeConformance = adapter_conformance(FakeAdapter())


@pytest.fixture(autouse=True)
def _registered() -> None:
    """Keep the fake adapter in the registry for every test in this module.

    Registered per test rather than at import: other test modules clear the
    registry when they finish, and the suite looks its adapter up by name at
    the moment it runs.
    """
    install()


def test_the_corpus_follows_declared_capabilities() -> None:
    """An adapter that cannot do nested types skips them rather than failing."""

    class Flat(FakeAdapter):
        capabilities = Capabilities(lazy=True, nested_types=False)

    nested = {Kind.LIST, Kind.STRUCT, Kind.MAP}

    assert any(dtype.kind in nested for dtype in dtype_corpus(FakeAdapter()))
    assert not any(dtype.kind in nested for dtype in dtype_corpus(Flat()))


def test_an_adapter_with_no_cases_is_reported_as_incomplete() -> None:
    """Operations are per-engine data; an adapter that supplies none is not done."""

    class Bare:
        name = "bare"
        requires = ""
        capabilities = Capabilities()

        def available(self) -> bool:
            return True

    assert pipeline_battery(Bare()) == ()  # type: ignore[arg-type]

    suite = adapter_conformance(Bare())()  # type: ignore[arg-type]
    with pytest.raises(AssertionError, match="conformance_cases"):
        suite.test_declares_conformance_cases()


def test_a_lying_capability_flag_fails() -> None:
    """A capability advertised but not implemented skews every coverage score."""

    class Liar(FakeAdapter):
        capabilities = Capabilities(column_lineage=True, sql=True, lazy=True)

    suite = adapter_conformance(Liar())()

    with pytest.raises(AssertionError, match="sql"):
        suite.test_capabilities_are_truthful()


def test_a_guessed_dtype_fails_the_round_trip() -> None:
    """Substituting a near-enough dtype is the bug this suite exists to catch."""

    class Sloppy(FakeAdapter):
        def from_ir(self, dtype: object) -> object:
            # Pretends every integer is 64-bit, which is what a careless
            # lowering does and what a mysterious production diff looks like.
            if getattr(dtype, "kind", None) is Kind.INT:
                return int_(64)
            return super().from_ir(dtype)  # type: ignore[arg-type]

    suite = adapter_conformance(Sloppy())()

    with pytest.raises(AssertionError, match="round trip lost information"):
        suite.test_dtype_round_trips(int_(32))


def test_an_adapter_that_guesses_instead_of_refusing_fails() -> None:
    """`from_ir(UNKNOWN)` must raise; returning something is a false pass."""

    class Guesser(FakeAdapter):
        def from_ir(self, dtype: object) -> object:
            return string()

    suite = adapter_conformance(Guesser())()

    # pytest's own failure is a BaseException, not an Exception: the suite
    # failing is the expected outcome here.
    with pytest.raises(BaseException, match="DID NOT RAISE"):
        suite.test_from_ir_refuses_unknown()


def test_a_confident_answer_for_row_dependent_code_fails() -> None:
    """The silent-wrong-answer case, caught by the suite rather than in CI."""

    class Overconfident(FakeAdapter):
        def conformance_cases(self) -> Sequence[ConformanceCase]:
            from rowless.ir.confidence import Confidence

            return (
                ConformanceCase(
                    name="claims_certainty",
                    pipeline=lambda orders: orders.select("order_id"),
                    inputs={"orders": Schema.from_mapping({"order_id": int_(64)})},
                    expect_confidence=Confidence.UNKNOWN,
                    reason="declared unverified, but nothing row-dependent happens",
                ),
            )

    adapter = Overconfident()
    install_name = adapter.name
    suite = adapter_conformance(adapter)()
    cases = pipeline_battery(adapter)

    assert install_name == "fake"
    with pytest.raises(AssertionError, match="expected confidence UNKNOWN"):
        suite.test_operations_resolve(cases[0])


def test_unknown_dtype_mapping_is_unknown_not_a_guess() -> None:
    """`to_ir` of a foreign type must come back UNKNOWN."""
    assert FakeAdapter().to_ir(object()).kind is Kind.UNKNOWN
    assert unknown().kind is Kind.UNKNOWN
