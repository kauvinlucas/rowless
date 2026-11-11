"""Schema history and field-ID identity: the part only Iceberg can do.

Everything here is about one claim: that a rename reported from Iceberg
metadata is a recorded fact rather than a pairing rowless guessed. The tests
that matter are the ones where a name-based comparison would get it wrong -
a column dropped while another is added, a column renamed twice, a rename
alongside an unrelated addition.
"""

from __future__ import annotations

from pathlib import Path

import iceberg_tables
import pytest
from rowless.errors import SchemaNotFound
from rowless.ir.dtypes import TimeUnit, datetime, int_
from rowless.sources.base import SourceRef
from rowless_iceberg import catalog, history

FIRST, MIDDLE, LATEST = iceberg_tables.ORDERS_SNAPSHOTS


def _table(tmp_path: Path, fixture: str = iceberg_tables.ORDERS) -> object:
    """The fixture table, loaded the way the source loads it."""
    path = iceberg_tables.metadata_path(tmp_path, fixture)
    return catalog.load_table(SourceRef.parse(f"iceberg://sales.orders?metadata={path.as_posix()}"))


def test_retains_every_schema_the_table_kept(tmp_path: Path) -> None:
    """Three schemas, oldest first, which is the order a history is read in."""
    table = _table(tmp_path)

    assert history.retained_schema_ids(table) == (0, 1, 2)
    assert history.current_schema_id(table) == 2


def test_field_ids_survive_the_mapping(tmp_path: Path) -> None:
    """A column's identity is its field ID, and it is not the column's name."""
    table = _table(tmp_path)

    assert history.field_ids(table, 0) == {
        "order_id": 1,
        "amount": 2,
        "status": 3,
        "customer_email": 4,
    }
    # Same IDs 1 to 3, two of them under different names.
    assert history.field_ids(table, 2) == {
        "order_id": 1,
        "amount_cents": 2,
        "order_status": 3,
        "created_at": 5,
        "tags": 6,
    }


def test_name_history_collects_every_name_a_field_id_held(tmp_path: Path) -> None:
    """Oldest first, repeats collapsed."""
    table = _table(tmp_path)

    assert history.name_history(table, through=2) == {
        1: ("order_id",),
        2: ("amount", "amount_cents"),
        3: ("status", "order_status"),
        4: ("customer_email",),
        5: ("created_at",),
        6: ("tags",),
    }


def test_name_history_is_cut_off_at_the_requested_schema(tmp_path: Path) -> None:
    """A schema resolved at an old snapshot must not see its own future.

    Field 2 is called `amount` in schema 0. Reading the whole history there
    would hand it `amount_cents` as a former name, which is backwards and
    would make `rowless diff` report a rename that has not happened yet.
    """
    table = _table(tmp_path)

    assert history.name_history(table, through=0) == {
        1: ("order_id",),
        2: ("amount",),
        3: ("status",),
        4: ("customer_email",),
    }


def test_renames_are_exact_across_a_drop_and_an_add(tmp_path: Path) -> None:
    """The case a name-based comparison gets wrong.

    Between schema 0 and schema 2: field 2 and field 3 are renamed, field 4 is
    dropped, fields 5 and 6 are added. A heuristic looking at removed and added
    names sees four removals and four additions and has to guess. Field IDs
    make it two renames and nothing else.
    """
    table = _table(tmp_path)

    assert history.renames_between(table, 0, 2) == {
        "amount": "amount_cents",
        "status": "order_status",
    }


def test_renames_between_adjacent_schemas_report_only_that_step(tmp_path: Path) -> None:
    """Field 2 was already renamed by schema 1, so only field 3 moved after it."""
    table = _table(tmp_path)

    assert history.renames_between(table, 1, 2) == {"status": "order_status"}
    assert history.renames_between(table, 0, 1) == {"amount": "amount_cents"}


def test_renames_between_a_schema_and_itself_is_empty(tmp_path: Path) -> None:
    """No change is no change, not an empty guess."""
    assert history.renames_between(_table(tmp_path), 2, 2) == {}


def test_a_snapshot_resolves_to_the_schema_it_ran_against(tmp_path: Path) -> None:
    """Which is the whole point: a diff needs no git checkout."""
    table = _table(tmp_path)

    assert history.schema_id_at_snapshot(table, FIRST) == 0
    assert history.schema_id_at_snapshot(table, MIDDLE) == 1
    assert history.schema_id_at_snapshot(table, LATEST) == 2


def test_a_branch_or_tag_resolves_through_its_snapshot(tmp_path: Path) -> None:
    """Named refs are how a team points at a release rather than at an id."""
    table = _table(tmp_path)

    assert history.schema_id_at_ref(table, "main") == 2
    assert history.schema_id_at_ref(table, "release-2024-11") == 1
    assert history.schema_id_at_ref(table, "audit") == 0


def test_an_unknown_branch_lists_the_ones_there_are(tmp_path: Path) -> None:
    """A typo in a ref name has to be fixable from the message."""
    with pytest.raises(SchemaNotFound) as caught:
        history.schema_id_at_ref(_table(tmp_path), "relase-2024-11")

    message = str(caught.value)
    assert "relase-2024-11" in message
    assert "release-2024-11" in message


def test_an_expired_schema_lists_the_retained_ones(tmp_path: Path) -> None:
    """Iceberg expires metadata, so this is ordinary rather than a bug."""
    with pytest.raises(SchemaNotFound) as caught:
        history.iceberg_schema(_table(tmp_path), 97)

    assert "retained schema ids: 0, 1, 2" in str(caught.value)


def test_an_expired_snapshot_lists_recent_ones(tmp_path: Path) -> None:
    """Most recent first: a stale `?snapshot=` is usually one release behind."""
    with pytest.raises(SchemaNotFound) as caught:
        history.schema_id_at_snapshot(_table(tmp_path), 1)

    message = str(caught.value)
    assert "snapshot 1 is not in the table metadata" in message
    assert str(LATEST) in message


def test_a_snapshot_with_no_schema_id_is_refused(tmp_path: Path) -> None:
    """The one historical comparison this source cannot make.

    `schema-id` is optional on a snapshot, and an old writer left it out.
    Answering with the current schema would be a guess presented as a recorded
    fact, which is the one thing this package exists not to do.
    """
    table = _table(tmp_path, iceberg_tables.LEGACY)

    with pytest.raises(SchemaNotFound) as caught:
        history.schema_id_at_snapshot(table, 5566778899001122)

    assert "records no schema id" in str(caught.value)


def test_to_ir_carries_dtypes_docs_nullability_and_origin(tmp_path: Path) -> None:
    """One assertion per thing the IR can hold, so a regression names itself."""
    table = _table(tmp_path)

    schema = history.to_ir(table, 2, uri="iceberg://sales.orders", name="orders")

    assert schema.name == "orders"
    assert tuple(schema) == ("order_id", "amount_cents", "order_status", "created_at", "tags")
    assert schema["order_id"].dtype == int_(64)
    assert schema["order_id"].nullable is False
    # `amount` was an int32 of dollars and became an int64 of cents: a rename
    # and an Iceberg-legal int-to-long promotion in one step.
    assert schema["amount_cents"].dtype == int_(64)
    assert schema["created_at"].dtype == datetime(TimeUnit.MICROSECOND, "UTC")
    assert schema["order_status"].description == "placed, shipped or cancelled"
    origin = schema["order_id"].origin
    assert origin is not None
    assert origin.kind == "source"
    assert origin.ref == "iceberg://sales.orders"
    assert origin.file is not None and origin.file.endswith(iceberg_tables.ORDERS)


def test_to_ir_fills_in_derived_from_with_former_names(tmp_path: Path) -> None:
    """The evidence `rowless.ir.diff` trusts above its own heuristics.

    Most recent former name first, nothing for a column that never moved, and
    nothing for one that was only ever added.
    """
    table = _table(tmp_path)

    schema = history.to_ir(table, 2, uri="iceberg://sales.orders")

    assert schema["amount_cents"].derived_from == ("amount",)
    assert schema["order_status"].derived_from == ("status",)
    assert schema["order_id"].derived_from == ()
    assert schema["tags"].derived_from == ()


def test_to_ir_at_an_old_schema_claims_no_future_renames(tmp_path: Path) -> None:
    """Schema 0 is the start of the history, so nothing is derived from anything."""
    schema = history.to_ir(_table(tmp_path), 0, uri="iceberg://sales.orders")

    assert [field.derived_from for field in schema.fields] == [(), (), (), ()]
    assert tuple(schema) == ("order_id", "amount", "status", "customer_email")
