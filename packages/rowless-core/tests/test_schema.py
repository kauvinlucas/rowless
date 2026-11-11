"""Schemas: lookup, derivation and the errors they raise."""

from __future__ import annotations

import pytest
from rowless.errors import ColumnNotFound
from rowless.ir.dtypes import int_, string
from rowless.ir.schema import Field, Origin, Schema

ORDERS = Schema.from_mapping(
    {"order_id": int_(64), "status": string(), "amount": int_(64)},
    name="stg_orders",
)


def test_lookup_suggests_the_near_miss() -> None:
    """The rename case is the one in the proposal; a bare name is half an answer."""
    schema = ORDERS.rename({"amount": "amount_cents"})

    with pytest.raises(ColumnNotFound) as caught:
        _ = schema["amount"]

    assert caught.value.suggestion == "amount_cents"
    assert "amount_cents" in str(caught.value)
    assert caught.value.relation == "stg_orders"


def test_membership_works_without_raising() -> None:
    """`Schema` is a Mapping, so `in` and `get` have to behave like one."""
    assert "status" in ORDERS
    assert "nope" not in ORDERS
    assert ORDERS.get("nope") is None
    assert tuple(ORDERS) == ("order_id", "status", "amount")
    assert len(ORDERS) == 3


def test_select_preserves_the_order_given() -> None:
    """Projection order is the caller's, not the schema's."""
    assert tuple(ORDERS.select(("amount", "order_id"))) == ("amount", "order_id")


def test_derivations_do_not_mutate() -> None:
    """A dry run keeps its input schema to diff against the output."""
    dropped = ORDERS.drop(("status",))

    assert tuple(dropped) == ("order_id", "amount")
    assert tuple(ORDERS) == ("order_id", "status", "amount")


def test_derivations_error_on_a_missing_column() -> None:
    """Dropping or renaming something absent is a bug, not a no-op."""
    with pytest.raises(ColumnNotFound):
        ORDERS.drop(("nope",))
    with pytest.raises(ColumnNotFound):
        ORDERS.rename({"nope": "whatever"})


def test_rename_keeps_position() -> None:
    """A rename is not a drop and an append; downstream order survives."""
    renamed = ORDERS.rename({"status": "state"})

    assert tuple(renamed) == ("order_id", "state", "amount")


def test_with_fields_replaces_by_name() -> None:
    """Appending a column that exists replaces it, keeping its position."""
    updated = ORDERS.with_fields(Field(name="status", dtype=int_(32)))

    assert tuple(updated) == ("order_id", "status", "amount")
    assert updated["status"].dtype == int_(32)

    appended = ORDERS.with_fields(Field(name="day", dtype=string()))
    assert tuple(appended)[-1] == "day"


def test_origin_renders_as_a_clickable_location() -> None:
    """Diagnostics quote `file:line`, which is what makes them actionable."""
    assert str(Origin(kind="python", ref="x", file="revenue.py", line=9)) == "revenue.py:9"
    assert str(Origin(kind="source", ref="lake://orders")) == "lake://orders"
