"""The diagnostic record. One shape, rendered many ways."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from rowless.diagnostics.codes import Code
from rowless.ir.confidence import Coverage
from rowless.ir.diff import Change, ChangeKind, Severity
from rowless.ir.dtypes import Kind
from rowless.ir.schema import Origin


class Level(StrEnum):
    """Severity as the user sees it."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    UNVERIFIED = "unverified"
    """Distinct from warning on purpose: it means rowless does not know, which
    is different from knowing something is suspect."""


#: Levels that change the exit code. UNVERIFIED is in here because a result
#: rowless could not verify must not read as a pass; `fail_on_unverified` is
#: applied earlier, when a code is turned into a level.
FAILING_LEVELS: frozenset[Level] = frozenset({Level.ERROR, Level.UNVERIFIED})

#: Diagnostic code prefix for rowless's own failures, as opposed to the user's.
#: Exit code 2 exists so CI can tell those apart.
INTERNAL_PREFIX = "RWL5"

#: Which code a classified change is reported under.
_CHANGE_CODES: dict[ChangeKind, Code] = {
    ChangeKind.COLUMN_REMOVED: Code.BREAKING_COLUMN_REMOVED,
    ChangeKind.COLUMN_RENAMED: Code.BREAKING_COLUMN_RENAMED,
    ChangeKind.DTYPE_CHANGED: Code.BREAKING_DTYPE_CHANGED,
    ChangeKind.NULLABILITY_CHANGED: Code.RISKY_NULLABILITY_WIDENED,
    ChangeKind.SCHEMA_UNRESOLVED: Code.UNVERIFIED_CHANGE,
}


def _code_for(change: Change) -> Code:
    """The code a change is reported under.

    A timezone change is spelled out separately from any other retype, because
    it is the one dtype change that keeps every query working and silently
    moves the values.
    """
    before, after = change.before, change.after
    if (
        change.kind is ChangeKind.DTYPE_CHANGED
        and before is not None
        and after is not None
        and before.kind is Kind.DATETIME
        and after.kind is Kind.DATETIME
        and before.tz != after.tz
    ):
        return Code.RISKY_TIMEZONE_CHANGED
    return _CHANGE_CODES.get(change.kind, Code.UNVERIFIED_CHANGE)


@dataclass(frozen=True, slots=True)
class Diagnostic:
    """One finding.

    Carries enough for every surface: terminal text, a GitHub annotation, an
    LSP diagnostic and a JUnit case. Adding a surface must never require
    adding a field here.
    """

    code: Code
    level: Level
    message: str
    """One line, no trailing period, naming the thing: `ColumnNotFound 'amount'`."""

    origin: Origin | None = None
    """Primary location: the file and line the squiggle goes under."""

    related: tuple[Origin, ...] = ()
    """Secondary locations, which is where the cross-language payoff shows up:
    the Python error points at the SQL model that renamed the column."""

    detail: str | None = None
    """The indented explanation under the message."""

    suggestion: str | None = None
    """A concrete fix: the nearest column name, the extra to install."""

    @property
    def fails_run(self) -> bool:
        """True when this diagnostic should change the exit code."""
        return self.level in FAILING_LEVELS

    @property
    def is_internal(self) -> bool:
        """True when rowless itself failed, rather than the pipeline."""
        return self.code.value.startswith(INTERNAL_PREFIX)

    @classmethod
    def from_change(
        cls,
        change: Change,
        *,
        relation: str,
        origin: Origin | None = None,
    ) -> Diagnostic:
        """Turn a classified schema change into a reportable finding.

        A RISKY change is a warning, an UNVERIFIED one gets its own level, and
        everything breaking is an error. The blast radius travels in `detail`,
        which is what renders as the tree in the proposal's terminal mock.
        """
        level = {
            Severity.BREAKING: Level.ERROR,
            Severity.RISKY: Level.WARNING,
            Severity.UNVERIFIED: Level.UNVERIFIED,
            Severity.COMPATIBLE: Level.INFO,
        }[change.severity]
        detail = None
        if change.affects:
            detail = "\n".join(f"|- {affected}" for affected in change.affects)
        return cls(
            code=_code_for(change),
            level=level,
            message=f"{change.severity.value.upper()}  {relation}.{change.describe()}",
            origin=origin,
            detail=detail,
        )


@dataclass(frozen=True, slots=True)
class Report:
    """Everything one `rowless check` or `rowless diff` produced."""

    diagnostics: tuple[Diagnostic, ...] = ()
    pipelines_checked: int = 0
    pipelines_skipped: int = 0

    coverage: Coverage = field(default_factory=Coverage)
    """Aggregate coverage across every pipeline checked. Rendered next to the
    findings, because a finding count without a coverage figure beside it
    overstates what was actually verified."""

    duration_ms: float = 0.0

    @property
    def exit_code(self) -> int:
        """0 clean, 1 findings, 2 rowless itself failed.

        Three codes, because CI needs to distinguish "your pipeline is broken"
        from "rowless is broken", and a tool that conflates them gets disabled.
        """
        if any(diagnostic.is_internal for diagnostic in self.diagnostics):
            return 2
        if any(diagnostic.fails_run for diagnostic in self.diagnostics):
            return 1
        return 0

    @property
    def failures(self) -> tuple[Diagnostic, ...]:
        """Just the diagnostics that changed the exit code."""
        return tuple(diagnostic for diagnostic in self.diagnostics if diagnostic.fails_run)

    def merge(self, other: Report) -> Report:
        """Combine two reports, as the CLI does across pipelines."""
        return Report(
            diagnostics=self.diagnostics + other.diagnostics,
            pipelines_checked=self.pipelines_checked + other.pipelines_checked,
            pipelines_skipped=self.pipelines_skipped + other.pipelines_skipped,
            coverage=self.coverage.merge(other.coverage),
            duration_ms=self.duration_ms + other.duration_ms,
        )
