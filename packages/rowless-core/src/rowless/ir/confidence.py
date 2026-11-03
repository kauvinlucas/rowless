"""Gradual schema typing: how much of a result rowless actually verified.

Hard problem 03 in the proposal. The failure mode to design against is a tool
that reports "safe" when it has simply lost track of a column. So every result
carries a `Confidence`, every ANY-typed or untraced column lowers a coverage
score, and the CLI refuses to print a pass when coverage is below the
configured floor.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum

from rowless.ir.schema import Schema


class Confidence(IntEnum):
    """How much to trust a schema result. Ordered: take the minimum when combining."""

    UNKNOWN = 0
    """Nothing was resolved: a dry run that failed, or control flow that depends
    on row count (`if df.is_empty()`, `.item()`, `.to_list()`)."""

    PARTIAL = 1
    """Some columns resolved, others are ANY or untraced. Reported as partial,
    never as a pass."""

    EXACT = 2
    """The engine resolved every column itself, with no rowless guesswork."""


@dataclass(frozen=True, slots=True)
class Coverage:
    """The coverage score reported per pipeline, and in aggregate by the CLI."""

    columns_total: int = 0

    columns_typed: int = 0
    """Columns with a known, non-ANY dtype."""

    columns_traced: int = 0
    """Columns with non-empty `derived_from`, so usable for lineage."""

    confidence: Confidence = Confidence.UNKNOWN

    reasons: tuple[str, ...] = ()
    """Human-readable causes of lost confidence, shown by `rowless explain`."""

    @property
    def typed_ratio(self) -> float:
        """Fraction of columns with a resolved dtype. The headline CI number."""
        return self._ratio(self.columns_typed)

    @property
    def traced_ratio(self) -> float:
        """Fraction of columns with lineage, which gates diff blast radius."""
        return self._ratio(self.columns_traced)

    def _ratio(self, part: int) -> float:
        """Share of the total, with no columns counting as fully covered.

        A result with no columns at all is not a coverage problem; it is either
        a legitimately empty schema or a failed run, and the failed run is
        already reported through `confidence`.
        """
        if self.columns_total == 0:
            return 1.0
        return part / self.columns_total

    def merge(self, other: Coverage) -> Coverage:
        """Combine two results: sum the counts, take the weaker confidence."""
        return Coverage(
            columns_total=self.columns_total + other.columns_total,
            columns_typed=self.columns_typed + other.columns_typed,
            columns_traced=self.columns_traced + other.columns_traced,
            confidence=min(self.confidence, other.confidence),
            reasons=_dedupe(self.reasons + other.reasons),
        )

    def degraded(self, reason: str) -> Coverage:
        """This result, downgraded to UNKNOWN, with `reason` recorded.

        The counts are kept: knowing that rowless lost confidence about a
        14-column output is more useful than pretending the output was empty.
        """
        return replace(
            self,
            confidence=Confidence.UNKNOWN,
            reasons=_dedupe((*self.reasons, reason)),
        )

    @classmethod
    def score(
        cls,
        schema: Schema,
        *,
        confidence: Confidence | None = None,
        reasons: tuple[str, ...] = (),
    ) -> Coverage:
        """Score a resolved schema.

        Confidence is derived from the schema itself unless a caller overrides
        it: every column known and concrete is EXACT, a mix is PARTIAL, and a
        schema where nothing resolved is UNKNOWN. Callers override when they
        know something the schema does not show, which in practice means row
        dependence.
        """
        typed = sum(1 for f in schema.fields if f.dtype.is_known and not f.dtype.is_any)
        traced = sum(1 for f in schema.fields if f.derived_from)
        derived = _confidence_for(len(schema.fields), typed)
        return cls(
            columns_total=len(schema.fields),
            columns_typed=typed,
            columns_traced=traced,
            confidence=confidence if confidence is not None else derived,
            reasons=_dedupe(reasons),
        )


def _confidence_for(total: int, typed: int) -> Confidence:
    """Confidence implied by how many of `total` columns came out typed."""
    if total == 0:
        return Confidence.EXACT
    if typed == total:
        return Confidence.EXACT
    if typed == 0:
        return Confidence.UNKNOWN
    return Confidence.PARTIAL


def degrade(reason: str) -> Coverage:
    """Coverage for a result rowless gave up on, carrying the reason forward."""
    return Coverage(confidence=Confidence.UNKNOWN, reasons=(reason,))


def _dedupe(reasons: tuple[str, ...]) -> tuple[str, ...]:
    """Drop repeats while keeping the order they were reported in.

    Two pipelines degrading for the same reason should say so once; a user
    reading `rowless explain` is looking for distinct causes.
    """
    return tuple(dict.fromkeys(reasons))
