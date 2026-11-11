"""Keeping a squiggle under the right line while the file is being edited.

A dry run's findings are about the file as it was when the check ran. The
moment a line is inserted above one of them, its line number is wrong, and a
diagnostic pointing at the wrong line is worse than no diagnostic: it accuses
code that is fine.

The fix is to remember what the flagged line *said*, not only where it was.
On every republish the text is looked for again: unchanged in place, the
finding stays; moved a little, it follows; gone, it is dropped and waits for
the next save to tell the truth.

Dropping is the important half. The alternative - leaving a finding on a line
whose text no longer matches - is how a checker ends up blamed for a bug it
reported correctly three edits ago.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

from rowless.diagnostics.models import Diagnostic

#: How far from its old position a line is still recognised as the same line.
#: Wide enough to survive an import block or a new function being added above,
#: narrow enough that an identical line elsewhere in the file is not mistaken
#: for this one.
WINDOW = 40


@dataclass(frozen=True, slots=True)
class Anchored:
    """A finding plus the text of the line it was reported against."""

    diagnostic: Diagnostic
    text: str
    """The flagged line, stripped. Empty when the finding has no line, or the
    line was blank, in which case the finding cannot be re-anchored and is
    kept where it is."""

    @property
    def movable(self) -> bool:
        """True when there is enough text to recognise the line again."""
        return bool(self.text)


def anchor(diagnostics: Sequence[Diagnostic], lines: Sequence[str]) -> tuple[Anchored, ...]:
    """Record what each finding's line said, at the moment it was produced."""
    return tuple(Anchored(diagnostic=d, text=_line(lines, d)) for d in diagnostics)


def reanchor(
    anchored: Sequence[Anchored],
    lines: Sequence[str],
    *,
    window: int = WINDOW,
) -> tuple[Diagnostic, ...]:
    """Re-point findings at the buffer as it is now, dropping the lost ones.

    A finding with no location, or one whose line was blank, passes through
    untouched: there is nothing to match on, and a report-level finding like
    coverage has no line to be wrong about in the first place.
    """
    kept: list[Diagnostic] = []
    for item in anchored:
        if not item.movable:
            kept.append(item.diagnostic)
            continue
        origin = item.diagnostic.origin
        if origin is None or origin.line is None:
            kept.append(item.diagnostic)
            continue
        moved = _find(item.text, lines, origin.line, window)
        if moved is None:
            continue
        if moved == origin.line:
            kept.append(item.diagnostic)
            continue
        kept.append(replace(item.diagnostic, origin=replace(origin, line=moved)))
    return tuple(kept)


def _find(text: str, lines: Sequence[str], was: int, window: int) -> int | None:
    """The 1-based line now holding `text`, searched outwards from `was`.

    Outwards rather than top-down, so that duplicated lines - two pipelines
    ending in the same `return orders.select(...)` - resolve to the one
    nearest where the finding already was.
    """
    if _at(lines, was) == text:
        return was
    for distance in range(1, window + 1):
        for candidate in (was - distance, was + distance):
            if _at(lines, candidate) == text:
                return candidate
    return None


def _at(lines: Sequence[str], line: int) -> str | None:
    """One 1-based line, stripped, or None when it is out of range."""
    if 1 <= line <= len(lines):
        return lines[line - 1].strip()
    return None


def _line(lines: Sequence[str], diagnostic: Diagnostic) -> str:
    """The stripped text of a finding's line, or an empty string."""
    origin = diagnostic.origin
    if origin is None or origin.line is None:
        return ""
    return _at(lines, origin.line) or ""
