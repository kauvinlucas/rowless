"""Schema diffing and severity classification."""

from __future__ import annotations

from rowless.ir.diff import ChangeKind, Severity, diff_schemas
from rowless.ir.dtypes import TimeUnit, datetime, int_, string, unknown
from rowless.ir.schema import Field, Schema

ORDERS = Schema(
    fields=(
        Field(name="order_id", dtype=int_(64), nullable=False),
        Field(name="amount", dtype=int_(64)),
    ),
    name="stg_orders",
)


def _kinds(diff: object) -> list[ChangeKind]:
    """The change kinds a diff reported, for compact assertions."""
    return [change.kind for change in diff.changes]  # type: ignore[attr-defined]


def test_removed_column_is_breaking() -> None:
    """Dropping a column downstream reads is BREAKING."""
    after = ORDERS.drop(("amount",))

    diff = diff_schemas(ORDERS, after)

    assert _kinds(diff) == [ChangeKind.COLUMN_REMOVED]
    assert diff.severity is Severity.BREAKING
    assert diff.is_breaking


def test_added_column_is_compatible() -> None:
    """A new column is COMPATIBLE and never gates."""
    after = ORDERS.with_fields(Field(name="currency", dtype=string()))

    diff = diff_schemas(ORDERS, after)

    assert _kinds(diff) == [ChangeKind.COLUMN_ADDED]
    assert diff.severity is Severity.COMPATIBLE
    assert not diff.is_breaking


def test_rename_detected_as_single_change() -> None:
    """A drop plus an add at the same position with the same dtype is a rename."""
    after = ORDERS.rename({"amount": "amount_cents"})

    diff = diff_schemas(ORDERS, after)

    assert _kinds(diff) == [ChangeKind.COLUMN_RENAMED]
    change = diff.changes[0]
    assert change.column == "amount"
    assert change.renamed_to == "amount_cents"
    assert change.severity is Severity.BREAKING
    assert change.describe() == "amount -> amount_cents"


def test_rename_detection_can_be_disabled() -> None:
    """With detect_renames off, the same case reports two changes."""
    after = ORDERS.rename({"amount": "amount_cents"})

    diff = diff_schemas(ORDERS, after, detect_renames=False)

    assert sorted(_kinds(diff)) == sorted([ChangeKind.COLUMN_REMOVED, ChangeKind.COLUMN_ADDED])


def test_rename_follows_lineage_when_it_is_available() -> None:
    """Lineage evidence beats position: it is the only non-heuristic signal."""
    after = Schema(
        fields=(
            Field(name="amount_cents", dtype=int_(64), derived_from=("amount",)),
            Field(name="order_id", dtype=int_(64), nullable=False),
        ),
        name="stg_orders",
    )

    diff = diff_schemas(ORDERS, after)

    renames = [change for change in diff.changes if change.kind is ChangeKind.COLUMN_RENAMED]
    assert [(change.column, change.renamed_to) for change in renames] == [
        ("amount", "amount_cents")
    ]


def test_unresolved_schema_is_unverified_not_safe() -> None:
    """An unresolvable schema yields UNVERIFIED, which fails the gate."""
    after = Schema(
        fields=(
            Field(name="order_id", dtype=int_(64), nullable=False),
            Field(name="amount", dtype=unknown("adapter could not resolve this")),
        ),
        name="stg_orders",
    )

    diff = diff_schemas(ORDERS, after)

    assert _kinds(diff) == [ChangeKind.SCHEMA_UNRESOLVED]
    assert diff.severity is Severity.UNVERIFIED
    assert diff.is_breaking, "unverified must fail the gate; the alternative is a false all-clear"


def test_nullability_widening_is_risky() -> None:
    """Non-null to nullable is RISKY, not COMPATIBLE."""
    after = Schema(
        fields=(
            Field(name="order_id", dtype=int_(64), nullable=True),
            Field(name="amount", dtype=int_(64)),
        ),
        name="stg_orders",
    )

    diff = diff_schemas(ORDERS, after)

    assert _kinds(diff) == [ChangeKind.NULLABILITY_CHANGED]
    assert diff.severity is Severity.RISKY


def test_nullability_narrowing_is_not_reported() -> None:
    """A strengthened promise breaks nothing downstream, so it is not noise."""
    before = Schema(fields=(Field(name="amount", dtype=int_(64), nullable=True),))
    after = Schema(fields=(Field(name="amount", dtype=int_(64), nullable=False),))

    assert diff_schemas(before, after).changes == ()


def test_timezone_change_is_risky() -> None:
    """Naive to UTC is RISKY: it silently shifts values in downstream code."""
    before = Schema(fields=(Field(name="created_at", dtype=datetime(TimeUnit.MICROSECOND)),))
    after = Schema(fields=(Field(name="created_at", dtype=datetime(TimeUnit.MICROSECOND, "UTC")),))

    diff = diff_schemas(before, after)

    assert _kinds(diff) == [ChangeKind.DTYPE_CHANGED]
    assert diff.severity is Severity.RISKY


def test_incompatible_retype_is_breaking() -> None:
    """int64 to string is not a widening, whatever the engine does with it."""
    after = Schema(
        fields=(
            Field(name="order_id", dtype=int_(64), nullable=False),
            Field(name="amount", dtype=string()),
        ),
        name="stg_orders",
    )

    assert diff_schemas(ORDERS, after).severity is Severity.BREAKING


def test_widening_a_dtype_is_risky_not_silent() -> None:
    """int32 to int64 keeps the data and can still break a downstream cast."""
    before = Schema(fields=(Field(name="amount", dtype=int_(32)),))
    after = Schema(fields=(Field(name="amount", dtype=int_(64)),))

    assert diff_schemas(before, after).severity is Severity.RISKY


def test_reorder_is_reported_once() -> None:
    """Positional consumers break on a reorder, and one line says so."""
    after = ORDERS.select(("amount", "order_id"))

    diff = diff_schemas(ORDERS, after)

    assert _kinds(diff) == [ChangeKind.ORDER_CHANGED]
    assert diff.severity is Severity.RISKY
