"""Schema diffing: what `rowless diff` and the GitHub Action report."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum

from rowless.ir.dtypes import DType, Kind
from rowless.ir.schema import Field, Schema


class Severity(StrEnum):
    """How bad a change is for downstream consumers."""

    BREAKING = "breaking"
    """Removes or retypes something consumers read. Fails the CI gate."""

    RISKY = "risky"
    """Compatible in principle, but behaviour-changing: nullability widened, a
    timezone added, int32 to int64 where a downstream cast may overflow."""

    COMPATIBLE = "compatible"
    """Additive. Reported, never gated."""

    UNVERIFIED = "unverified"
    """Coverage was too low to classify. Treated as breaking by the gate,
    because the alternative is a false all-clear."""


#: Worst-first ordering for `SchemaDiff.severity`. UNVERIFIED sorts just under
#: BREAKING: it gates like one, but it is a different thing to tell a user.
SEVERITY_ORDER: Mapping[Severity, int] = {
    Severity.COMPATIBLE: 0,
    Severity.RISKY: 1,
    Severity.UNVERIFIED: 2,
    Severity.BREAKING: 3,
}

#: Severities that fail the gate by default. `fail_on_unverified` is what makes
#: the second entry configurable, and it defaults to on.
GATING_SEVERITIES: frozenset[Severity] = frozenset({Severity.BREAKING, Severity.UNVERIFIED})


class ChangeKind(StrEnum):
    """The shape of a single schema change."""

    COLUMN_ADDED = "column_added"
    COLUMN_REMOVED = "column_removed"
    COLUMN_RENAMED = "column_renamed"
    DTYPE_CHANGED = "dtype_changed"
    NULLABILITY_CHANGED = "nullability_changed"
    ORDER_CHANGED = "order_changed"
    SCHEMA_UNRESOLVED = "schema_unresolved"


@dataclass(frozen=True, slots=True)
class Change:
    """One schema change, with its blast radius attached."""

    kind: ChangeKind
    severity: Severity
    column: str
    before: DType | None = None
    after: DType | None = None
    renamed_to: str | None = None

    #: Downstream nodes that read this column, rendered as the tree in the
    #: proposal's terminal mock. Filled in from the lineage graph.
    affects: tuple[str, ...] = ()

    def describe(self) -> str:
        """One line, the way the terminal and the PR comment both want it."""
        match self.kind:
            case ChangeKind.COLUMN_RENAMED:
                return f"{self.column} -> {self.renamed_to}"
            case ChangeKind.COLUMN_REMOVED:
                return f"{self.column} removed"
            case ChangeKind.COLUMN_ADDED:
                return f"{self.column} added ({self.after})"
            case ChangeKind.DTYPE_CHANGED:
                return f"{self.column} {self.before} -> {self.after}"
            case ChangeKind.NULLABILITY_CHANGED:
                return f"{self.column} nullability changed"
            case ChangeKind.ORDER_CHANGED:
                return f"{self.column} moved"
            case ChangeKind.SCHEMA_UNRESOLVED:
                return f"{self.column} could not be resolved"


@dataclass(frozen=True, slots=True)
class SchemaDiff:
    """All changes between two versions of one relation."""

    relation: str
    changes: tuple[Change, ...] = ()

    @property
    def severity(self) -> Severity:
        """The worst severity present. The CLI exit code derives from this."""
        return worst(change.severity for change in self.changes)

    @property
    def is_breaking(self) -> bool:
        """True when the gate should fail."""
        return any(change.severity in GATING_SEVERITIES for change in self.changes)


def worst(severities: Iterable[Severity]) -> Severity:
    """The most severe of `severities`, COMPATIBLE when there are none."""
    return max(
        severities,
        key=lambda severity: SEVERITY_ORDER[severity],
        default=Severity.COMPATIBLE,
    )


def diff_schemas(before: Schema, after: Schema, *, detect_renames: bool = True) -> SchemaDiff:
    """Compare two schemas.

    Rename detection is heuristic (position, dtype, and lineage evidence when
    available) and is what collapses a removal plus an addition into the single
    "amount to amount_cents" line the proposal shows. It is opt-out, because a
    wrong guess here is worse than reporting two separate changes.
    """
    relation = after.name or before.name or "<anonymous>"
    before_names = tuple(before)
    after_names = tuple(after)
    removed = [name for name in before_names if name not in after]
    added = [name for name in after_names if name not in before]

    renames = _detect_renames(before, after, removed, added) if detect_renames else {}
    changes: list[Change] = []

    for name in removed:
        if name in renames:
            target = renames[name]
            changes.append(
                Change(
                    kind=ChangeKind.COLUMN_RENAMED,
                    severity=Severity.BREAKING,
                    column=name,
                    before=before[name].dtype,
                    after=after[target].dtype,
                    renamed_to=target,
                )
            )
        else:
            changes.append(
                Change(
                    kind=ChangeKind.COLUMN_REMOVED,
                    severity=Severity.BREAKING,
                    column=name,
                    before=before[name].dtype,
                )
            )

    renamed_targets = set(renames.values())
    for name in added:
        if name in renamed_targets:
            continue
        changes.append(
            Change(
                kind=ChangeKind.COLUMN_ADDED,
                severity=Severity.COMPATIBLE,
                column=name,
                after=after[name].dtype,
            )
        )

    for name in before_names:
        if name not in after:
            continue
        changes.extend(_compare_field(before[name], after[name]))

    changes.extend(_order_changes(before_names, after_names))
    return SchemaDiff(relation=relation, changes=tuple(changes))


def _compare_field(before: Field, after: Field) -> list[Change]:
    """Classify the change to one column present on both sides."""
    changes: list[Change] = []
    if not before.dtype.is_known or not after.dtype.is_known:
        # One side never resolved, so rowless cannot say whether this is safe.
        # Saying so is the whole point; guessing "compatible" here is the
        # failure mode the project exists to avoid.
        return [
            Change(
                kind=ChangeKind.SCHEMA_UNRESOLVED,
                severity=Severity.UNVERIFIED,
                column=before.name,
                before=before.dtype,
                after=after.dtype,
            )
        ]

    if not before.dtype.matches(after.dtype):
        changes.append(
            Change(
                kind=ChangeKind.DTYPE_CHANGED,
                severity=_dtype_severity(before.dtype, after.dtype),
                column=before.name,
                before=before.dtype,
                after=after.dtype,
            )
        )

    if after.nullable and not before.nullable:
        # Widening starts producing nulls in a column consumers never had to
        # handle. Narrowing only strengthens a promise, changes nothing
        # downstream, and is left unreported rather than adding noise.
        changes.append(
            Change(
                kind=ChangeKind.NULLABILITY_CHANGED,
                severity=Severity.RISKY,
                column=before.name,
                before=before.dtype,
                after=after.dtype,
            )
        )
    return changes


def _dtype_severity(before: DType, after: DType) -> Severity:
    """Severity of a dtype change, from the consumer's point of view."""
    if before.kind is Kind.DATETIME and after.kind is Kind.DATETIME and before.tz != after.tz:
        # Adding or dropping a timezone keeps the column readable and shifts
        # every value in it, which is exactly what RISKY is for.
        return Severity.RISKY
    if after.accepts(before):
        # A widened type still holds the old values, but a downstream cast,
        # an Arrow schema or a warehouse DDL can still trip over it.
        return Severity.RISKY
    return Severity.BREAKING


def _order_changes(before_names: tuple[str, ...], after_names: tuple[str, ...]) -> list[Change]:
    """Flag reordering, but only when nothing else explains it.

    Positional consumers (`df.to_numpy()`, `INSERT INTO t SELECT *`) break on a
    reorder, so it is worth a line. Adds and removals shuffle positions on
    their own, and reporting that as well would be noise.
    """
    if set(before_names) != set(after_names) or before_names == after_names:
        return []
    moved = [name for index, name in enumerate(after_names) if before_names[index] != name]
    return [
        Change(kind=ChangeKind.ORDER_CHANGED, severity=Severity.RISKY, column=name)
        for name in moved[:1]
    ]


def _detect_renames(
    before: Schema,
    after: Schema,
    removed: list[str],
    added: list[str],
) -> dict[str, str]:
    """Pair removed columns with added ones, best evidence first.

    Three kinds of evidence, in descending order of trust: the new column's
    lineage names the old one, the new column sits at the old one's position
    with the same dtype, or there is exactly one candidate on each side. The
    last rule is the weakest, and is why `detect_renames` can be turned off.
    """
    positions_before = {name: index for index, name in enumerate(before)}
    positions_after = {name: index for index, name in enumerate(after)}
    pairs: dict[str, str] = {}
    unclaimed = list(added)

    for name in removed:
        for candidate in unclaimed:
            if name in after[candidate].derived_from:
                pairs[name] = candidate
                unclaimed.remove(candidate)
                break

    for name in removed:
        if name in pairs:
            continue
        for candidate in unclaimed:
            same_position = positions_before[name] == positions_after[candidate]
            if same_position and before[name].dtype.matches(after[candidate].dtype):
                pairs[name] = candidate
                unclaimed.remove(candidate)
                break

    leftover = [name for name in removed if name not in pairs]
    if len(leftover) == 1 and len(unclaimed) == 1:
        name, candidate = leftover[0], unclaimed[0]
        if before[name].dtype.matches(after[candidate].dtype):
            pairs[name] = candidate

    return pairs
