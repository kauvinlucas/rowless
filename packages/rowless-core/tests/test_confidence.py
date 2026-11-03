"""Coverage scoring. The honesty tests: rowless must never overstate itself."""

from __future__ import annotations

from rowless.ir.confidence import Confidence, Coverage, degrade
from rowless.ir.dtypes import any_, int_, string, unknown
from rowless.ir.schema import Field, Schema


def test_merge_takes_weaker_confidence() -> None:
    """EXACT merged with PARTIAL is PARTIAL."""
    exact = Coverage(columns_total=2, columns_typed=2, confidence=Confidence.EXACT)
    partial = Coverage(columns_total=2, columns_typed=1, confidence=Confidence.PARTIAL)

    merged = exact.merge(partial)

    assert merged.confidence is Confidence.PARTIAL
    assert merged.columns_total == 4
    assert merged.columns_typed == 3
    assert merged.merge(degrade("worker died")).confidence is Confidence.UNKNOWN


def test_any_column_lowers_typed_ratio() -> None:
    """An ANY column counts against typed coverage."""
    schema = Schema(
        fields=(
            Field(name="day", dtype=string()),
            Field(name="amount", dtype=int_(64)),
            Field(name="whatever", dtype=any_()),
            Field(name="mystery", dtype=unknown("the adapter gave up")),
        )
    )

    coverage = Coverage.score(schema)

    assert coverage.columns_total == 4
    assert coverage.columns_typed == 2
    assert coverage.typed_ratio == 0.5
    assert coverage.confidence is Confidence.PARTIAL


def test_untraced_column_lowers_traced_ratio() -> None:
    """A column with no derived_from counts against lineage coverage."""
    schema = Schema(
        fields=(
            Field(name="day", dtype=string(), derived_from=("created_at",)),
            Field(name="amount", dtype=int_(64)),
        )
    )

    coverage = Coverage.score(schema)

    assert coverage.traced_ratio == 0.5
    assert coverage.typed_ratio == 1.0
    assert coverage.confidence is Confidence.EXACT


def test_nothing_resolved_is_unknown_not_partial() -> None:
    """A schema where no column resolved is not a partial result."""
    schema = Schema(fields=(Field(name="x", dtype=unknown("no idea")),))

    assert Coverage.score(schema).confidence is Confidence.UNKNOWN


def test_degrade_carries_reason() -> None:
    """Every confidence loss has a reason attached for `rowless explain`."""
    assert degrade("row-dependent branch").reasons == ("row-dependent branch",)

    scored = Coverage.score(Schema(fields=(Field(name="x", dtype=int_(64)),)))
    degraded = scored.degraded("if df.is_empty() at revenue.py:9")

    assert degraded.confidence is Confidence.UNKNOWN
    assert degraded.reasons == ("if df.is_empty() at revenue.py:9",)
    assert degraded.columns_total == 1, "the counts survive, so explain can show the scale"


def test_reasons_are_deduplicated() -> None:
    """Two pipelines failing the same way say so once."""
    first = degrade("unresolvable source")
    second = degrade("unresolvable source")

    assert first.merge(second).reasons == ("unresolvable source",)
