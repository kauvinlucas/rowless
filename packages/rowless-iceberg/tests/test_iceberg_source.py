"""The `iceberg://` schema source, end to end through `rowless.sources`.

The headline test is `test_renames_resolved_by_field_id`: it runs the resolved
schema through core's own differ and asserts one `COLUMN_RENAMED` line where
every other source in the project would produce a removal plus an addition. The
rest covers the surface a user actually touches - URI splitting, the history
selectors, caching, and what the failures say.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import iceberg_tables
import pytest
from rowless.errors import BadSourceRef, SchemaNotFound
from rowless.ir.diff import ChangeKind, Severity, diff_schemas
from rowless.ir.dtypes import TimeUnit, datetime, int_, string
from rowless.ir.schema import Schema
from rowless.sources.base import SourceCache, SourceRef, resolve, source_for
from rowless_iceberg import catalog
from rowless_iceberg.catalog import CATALOG_ENV, TableRef, parse_table_ref
from rowless_iceberg.source import IcebergSource

FIRST, MIDDLE, LATEST = iceberg_tables.ORDERS_SNAPSHOTS


def _uri(tmp_path: Path, **options: object) -> str:
    """The orders fixture's URI, with extra options spelled as keywords."""
    return iceberg_tables.uri(tmp_path, options=options)


def _ref(tmp_path: Path, **options: object) -> SourceRef:
    """A parsed ref for the orders fixture, with extra URI options."""
    return SourceRef.parse(iceberg_tables.uri(tmp_path, options=options))


# --- Registration ------------------------------------------------------------


def test_the_source_is_registered_for_the_iceberg_scheme() -> None:
    """Pipeline code only ever writes a URI, so the entry point is the API."""
    found = source_for(SourceRef.parse("iceberg://sales.orders"))

    assert isinstance(found, IcebergSource)
    assert found.schemes == ("iceberg",)


def test_available_needs_only_pyiceberg() -> None:
    """Not "and a catalog is configured": `?metadata=` needs no catalog.

    Gating here would report the source as uninstalled to a CI job using the
    credential-free route, which is the route that matters most.
    """
    assert IcebergSource().available() is True


def test_available_does_not_raise_on_a_broken_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`rowless doctor` lists sources, and is run when things are broken.

    A half-removed distribution leaves a meta-path finder that raises rather
    than reporting the module absent, and `find_spec` passes that straight on.
    An uninstalled source has to read as unavailable, not as a crash in the
    command a user reaches for to find that out.
    """

    def broken(name: str, package: str | None = None) -> Any:
        raise ImportError(f"half-removed: {name}")

    monkeypatch.setattr("importlib.util.find_spec", broken)

    assert IcebergSource().available() is False


# --- Resolution --------------------------------------------------------------


def test_resolves_current_schema(tmp_path: Path) -> None:
    """A table resolves to its current schema with field IDs preserved."""
    schema = resolve(_uri(tmp_path))

    assert schema.name == "orders"
    assert tuple(schema) == ("order_id", "amount_cents", "order_status", "created_at", "tags")
    assert schema["order_id"].dtype == int_(64)
    assert schema["amount_cents"].dtype == int_(64)
    assert schema["created_at"].dtype == datetime(TimeUnit.MICROSECOND, "UTC")
    assert IcebergSource().field_ids(_ref(tmp_path)) == {
        "order_id": 1,
        "amount_cents": 2,
        "order_status": 3,
        "created_at": 5,
        "tags": 6,
    }


def test_renames_resolved_by_field_id(tmp_path: Path) -> None:
    """A renamed column is reported as a rename, not as drop plus add.

    The schema a pipeline was written against had `amount` as an int32 of
    dollars and `status`; the table now has `amount_cents` as an int64 and
    `order_status`, and has dropped `customer_email` while adding two columns.
    Both renames are reported as renames, the drop stays a drop, and the two
    additions stay additions - no pairing of a removal with an addition
    anywhere.
    """
    source = IcebergSource()
    ref = _ref(tmp_path)
    before = source.schema_at(ref, FIRST)
    after = source.resolve(ref)

    diff = diff_schemas(before, after)
    renamed = {
        change.column: change.renamed_to
        for change in diff.changes
        if change.kind is ChangeKind.COLUMN_RENAMED
    }
    removed = {change.column for change in diff.changes if change.kind is ChangeKind.COLUMN_REMOVED}
    added = {change.column for change in diff.changes if change.kind is ChangeKind.COLUMN_ADDED}

    assert renamed == {"amount": "amount_cents", "status": "order_status"}
    assert removed == {"customer_email"}
    assert added == {"created_at", "tags"}
    assert diff.severity is Severity.BREAKING


def test_a_retyped_rename_is_only_found_through_the_field_id(tmp_path: Path) -> None:
    """The control for the test above: the same two schemas, evidence stripped.

    `status` to `order_status` is a rename core would have found on its own -
    same position, same dtype - so the field IDs confirm a guess rather than
    replacing one. `amount` to `amount_cents` is the case that needs them: the
    column was promoted from int32 to int64 in the same commit, so the
    position-and-dtype rule does not fire, and the fallback rule needs exactly
    one unpaired column on each side where this table has two and three. Core
    is right to refuse it; Iceberg is the only source that can supply the
    answer.
    """
    source = IcebergSource()
    ref = _ref(tmp_path)
    before = _stripped(source.schema_at(ref, FIRST))
    after = _stripped(source.resolve(ref))

    diff = diff_schemas(before, after)
    guessed = {
        change.column: change.renamed_to
        for change in diff.changes
        if change.kind is ChangeKind.COLUMN_RENAMED
    }
    removed = {change.column for change in diff.changes if change.kind is ChangeKind.COLUMN_REMOVED}

    assert guessed == {"status": "order_status"}
    assert removed == {"amount", "customer_email"}


def test_rename_evidence_does_not_veto_the_heuristic(tmp_path: Path) -> None:
    """The limit of what `derived_from` buys, pinned so nobody oversells it.

    Field IDs are *positive* evidence. Core's own rules still run underneath
    them, so two schemas resolved from Iceberg metadata are not immune to a
    wrong pairing: here a dropped column and an unrelated one of the same type
    are reported as a rename nobody made, because core's position-and-dtype
    rule matches them and the metadata said nothing either way. The target even
    carries rename evidence of its own, naming a different column, and that
    does not stop it.

    Refusing that needs field IDs on *both* sides of the comparison rather than
    evidence on one, which is what `renames_since` has and `diff_schemas` does
    not. The two are asserted together so the difference stays visible.
    """
    source = IcebergSource()
    ref = _ref(tmp_path)
    resolved = source.resolve(ref)
    # One column each, both strings at position 0: field 4 was dropped, field 3
    # is a different column that happens to have kept its type. Nothing in the
    # metadata connects them.
    dropped = source.schema_at(ref, MIDDLE).select(("customer_email",))
    added = resolved.select(("order_status",))
    assert added["order_status"].derived_from == ("status",)

    guessed = [
        (change.column, change.renamed_to)
        for change in diff_schemas(dropped, added).changes
        if change.kind is ChangeKind.COLUMN_RENAMED
    ]

    assert guessed == [("customer_email", "order_status")]
    # The exact answer has no room for the question: a dropped column shares no
    # field ID with anything in the newer schema, so it cannot appear at all.
    assert source.renames_since(ref, MIDDLE) == {"status": "order_status"}


def _stripped(schema: Schema) -> Schema:
    """The same schema with its rename evidence removed."""
    from dataclasses import replace

    return replace(
        schema,
        fields=tuple(replace(field, derived_from=()) for field in schema.fields),
    )


def test_an_empty_relation_name_falls_back_to_the_uri(tmp_path: Path) -> None:
    """Core labels a schema from the URI's last segment; this one agrees."""
    schema = resolve(iceberg_tables.uri(tmp_path, table="warehouse.sales.orders_daily"))

    assert schema.name == "orders_daily"


# --- History selectors -------------------------------------------------------


@pytest.mark.parametrize(
    ("options", "columns"),
    [
        ({}, ("order_id", "amount_cents", "order_status", "created_at", "tags")),
        ({"snapshot": FIRST}, ("order_id", "amount", "status", "customer_email")),
        ({"snapshot": MIDDLE}, ("order_id", "amount_cents", "status", "customer_email")),
        ({"branch": "audit"}, ("order_id", "amount", "status", "customer_email")),
        ({"tag": "release-2024-11"}, ("order_id", "amount_cents", "status", "customer_email")),
    ],
)
def test_a_selector_resolves_the_schema_that_ran(
    options: dict[str, object],
    columns: tuple[str, ...],
    tmp_path: Path,
) -> None:
    """`?snapshot=`, `?branch=` and `?tag=` each pick a point in the history."""
    assert tuple(resolve(iceberg_tables.uri(tmp_path, options=options))) == columns


def test_two_selectors_are_refused_rather_than_ranked(tmp_path: Path) -> None:
    """A URI whose meaning depends on knowing which option wins is unreadable."""
    with pytest.raises(BadSourceRef) as caught:
        resolve(_uri(tmp_path, snapshot=FIRST, tag="release-2024-11"))

    message = str(caught.value)
    assert "more than one point in the table's history" in message
    assert "tag=release-2024-11" in message


def test_a_branch_name_in_the_snapshot_option_says_so(tmp_path: Path) -> None:
    """The likely mistake, and the message has to name the option to use."""
    with pytest.raises(BadSourceRef) as caught:
        resolve(_uri(tmp_path, snapshot="release-2024-11"))

    assert "?branch= or ?tag=" in str(caught.value)


def test_schema_at_and_renames_since_answer_from_the_metadata(tmp_path: Path) -> None:
    """No git checkout involved: the table carries both sides of the diff."""
    source = IcebergSource()
    ref = _ref(tmp_path)

    assert tuple(source.schema_at(ref, MIDDLE)) == (
        "order_id",
        "amount_cents",
        "status",
        "customer_email",
    )
    assert source.renames_since(ref, FIRST) == {
        "amount": "amount_cents",
        "status": "order_status",
    }
    assert source.renames_since(ref, LATEST) == {}


def test_renames_since_respects_the_selector(tmp_path: Path) -> None:
    """The pinned side of the comparison is whatever the URI resolves to.

    A URI pinned to a tag compares the snapshot against that tag, which is what
    makes this usable for "what changed between the two releases".
    """
    source = IcebergSource()

    pinned = _ref(tmp_path, tag="release-2024-11")

    assert source.renames_since(pinned, FIRST) == {"amount": "amount_cents"}


def test_a_historical_schema_carries_the_dtypes_of_its_own_time(tmp_path: Path) -> None:
    """Schema 0 had `status` undocumented and `amount` an int32, and says so."""
    schema = IcebergSource().schema_at(_ref(tmp_path), FIRST)

    assert schema["status"].dtype == string()
    assert schema["status"].description is None
    assert schema["amount"].dtype == int_(32)


# --- Fingerprinting and caching ----------------------------------------------


def test_the_fingerprint_is_the_metadata_location_and_the_schema(tmp_path: Path) -> None:
    """Iceberg commits by swapping a metadata pointer, so the location is the
    change token - no polling and no catalog-specific etag."""
    source = IcebergSource()
    ref = _ref(tmp_path)

    fingerprint = source.fingerprint(ref)

    assert iceberg_tables.ORDERS in fingerprint
    assert fingerprint.endswith(":2:")


def test_the_fingerprint_separates_two_uris_that_differ_only_in_options(
    tmp_path: Path,
) -> None:
    """The on-disk schema cache is keyed on the URI without its query.

    So `iceberg://sales.orders` and `iceberg://sales.orders?snapshot=...` share
    one cache file, and only the fingerprint keeps them apart. Getting this
    wrong would serve a historical schema as the current one.
    """
    source = IcebergSource()
    cache = SourceCache(root=str(tmp_path / "cache"))
    live = _ref(tmp_path)
    historical = _ref(tmp_path, snapshot=FIRST)

    assert source.fingerprint(live) != source.fingerprint(historical)

    cache.put(live, source.fingerprint(live), source.resolve(live))

    assert cache.get(historical, source.fingerprint(historical)) is None
    assert cache.get(live, source.fingerprint(live)) is not None


def test_a_cached_schema_keeps_its_rename_evidence(tmp_path: Path) -> None:
    """`derived_from` has to survive the JSON round trip, or the second run of
    `rowless check` loses the only exact rename evidence in the project."""
    source = IcebergSource()
    cache = SourceCache(root=str(tmp_path / "cache"))
    ref = _ref(tmp_path)
    fingerprint = source.fingerprint(ref)
    cache.put(ref, fingerprint, source.resolve(ref))

    cached = cache.get(ref, fingerprint)

    assert cached is not None
    assert cached["amount_cents"].derived_from == ("amount",)


def test_a_table_is_loaded_once_per_run(tmp_path: Path) -> None:
    """`fingerprint` then `resolve` must see one metadata version.

    Two reads could straddle a commit and cache a schema from one version
    under the fingerprint of another, which is a stale schema reported as fact.
    """
    source = IcebergSource()
    ref = _ref(tmp_path)

    first = catalog.load_table(ref)
    source.fingerprint(ref)
    source.resolve(ref)

    assert catalog.load_table(ref) is first


def test_rewriting_the_metadata_file_invalidates_the_cache(tmp_path: Path) -> None:
    """A regenerated metadata file has to be re-read, mtime being the token."""
    ref = _ref(tmp_path)
    first = catalog.load_table(ref)
    path = Path(str(ref.options["metadata"]))
    path.write_text(
        (iceberg_tables.FIXTURES / iceberg_tables.LEGACY).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    import os

    os.utime(path, (1_600_000_000, 1_600_000_000))

    assert catalog.load_table(ref) is not first
    assert tuple(IcebergSource().resolve(ref)) == ("event_id", "payload")


# --- URI splitting -----------------------------------------------------------


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        ("iceberg://prod.sales.orders", TableRef(identifier="sales.orders", catalog="prod")),
        ("iceberg://sales.orders", TableRef(identifier="sales.orders")),
        (
            "iceberg://sales.orders?catalog=prod",
            TableRef(identifier="sales.orders", catalog="prod"),
        ),
        (
            "iceberg://a.b.c.orders?catalog=prod",
            TableRef(identifier="a.b.c.orders", catalog="prod"),
        ),
        ("iceberg://deep.a.b.orders", TableRef(identifier="a.b.orders", catalog="deep")),
    ],
)
def test_a_path_splits_into_a_catalog_and_an_identifier(uri: str, expected: TableRef) -> None:
    """Three or more segments means the first is the catalog; `?catalog=` wins."""
    assert parse_table_ref(SourceRef.parse(uri)) == expected


def test_the_catalog_environment_variable_supplies_a_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """For a repository whose URIs are written without a catalog name."""
    monkeypatch.setenv(CATALOG_ENV, "prod")

    assert parse_table_ref(SourceRef.parse("iceberg://sales.orders")) == TableRef(
        identifier="sales.orders", catalog="prod"
    )
    # A name on the URI still wins over the environment.
    assert parse_table_ref(SourceRef.parse("iceberg://sales.orders?catalog=staging")) == TableRef(
        identifier="sales.orders", catalog="staging"
    )


def test_a_table_with_no_namespace_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """Iceberg tables live in a namespace; guessing one resolves the wrong table."""
    monkeypatch.delenv(CATALOG_ENV, raising=False)

    with pytest.raises(BadSourceRef) as caught:
        parse_table_ref(SourceRef.parse("iceberg://orders"))

    message = str(caught.value)
    assert "names a table with no namespace" in message
    assert "?metadata=" in message


def test_the_metadata_option_makes_the_path_a_label(tmp_path: Path) -> None:
    """The file is the address, so no catalog is read out of the path as well."""
    parsed = parse_table_ref(_ref(tmp_path))

    assert parsed.catalog is None
    assert parsed.metadata is not None
    assert parsed.identifier == "sales.orders"


# --- Failures ----------------------------------------------------------------


def test_an_unreadable_metadata_file_names_the_location(tmp_path: Path) -> None:
    """The usual mistake on the credential-free route is a wrong path."""
    missing = (tmp_path / "v9.metadata.json").as_posix()

    with pytest.raises(SchemaNotFound) as caught:
        resolve(f"iceberg://sales.orders?metadata={missing}")

    message = str(caught.value)
    assert missing in message
    assert "cannot read the Iceberg metadata" in message


def test_a_catalog_that_cannot_be_loaded_names_every_way_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A PyIceberg traceback does not tell a user which of four things to fix."""

    def boom(name: str | None = None, **properties: str | None) -> Any:
        raise ValueError("URI missing")

    monkeypatch.setattr("pyiceberg.catalog.load_catalog", boom)
    monkeypatch.delenv(CATALOG_ENV, raising=False)
    catalog.clear_cache()

    with pytest.raises(SchemaNotFound) as caught:
        resolve("iceberg://prod.sales.orders")

    message = str(caught.value)
    assert "cannot load prod" in message
    assert ".pyiceberg.yaml" in message
    assert "PYICEBERG_" in message
    assert CATALOG_ENV in message
    assert "?metadata=" in message


def test_a_missing_table_explains_how_the_uri_was_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The silent failure mode: a two-level namespace read as a catalog name.

    The user can see `prod.sales.orders` in their catalog, rowless looked for
    `sales.orders` in a catalog called `prod`, and nothing in PyIceberg's
    `NoSuchTableError` says which of the two happened.
    """

    class _Catalog:
        def load_table(self, identifier: str) -> Any:
            raise KeyError(identifier)

    monkeypatch.setattr("pyiceberg.catalog.load_catalog", lambda name=None, **kw: _Catalog())
    monkeypatch.delenv(CATALOG_ENV, raising=False)
    catalog.clear_cache()

    with pytest.raises(SchemaNotFound) as caught:
        resolve("iceberg://prod.sales.orders")

    message = str(caught.value)
    assert "rowless read 'iceberg://prod.sales.orders' as catalog 'prod'" in message
    assert "pass the catalog separately with ?catalog=" in message


def test_a_history_failure_names_the_source_it_came_from(tmp_path: Path) -> None:
    """A finding has to say which input it is about, not just what is wrong."""
    with pytest.raises(SchemaNotFound) as caught:
        resolve(_uri(tmp_path, snapshot=1))

    message = str(caught.value)
    assert message.startswith("iceberg://sales.orders:")
    assert "snapshot 1 is not in the table metadata" in message
