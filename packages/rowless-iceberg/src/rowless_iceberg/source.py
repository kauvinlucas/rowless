"""Schemas from Iceberg table metadata: `iceberg://catalog.db.table`.

Iceberg is the best-behaved source in the project, because schema evolution is
a first-class concept in the format:

* Field IDs are stable across renames, so a rename is unambiguous rather than
  guessed. Everywhere else rowless infers renames heuristically; here the
  format records them, and `rowless diff` can state them as fact.
* Schema history is retained, so a diff against a previous snapshot needs no
  git checkout at all.

Both of those make this the source where breaking-change detection is strongest,
which is worth saying in the docs: teams on Iceberg get more out of rowless than
teams on bare Parquet.

## What is read, and what is not

The table metadata and nothing else. No manifests, no manifest lists, no data
files - which means a petabyte table resolves in one object read, and that a
source which scanned rows would be a bug rather than a slow path. Nothing here
needs a query engine either: the schema lives in the metadata JSON, so this
source resolves a table that no engine installed on the machine can read.

## Options

| option | effect |
| --- | --- |
| `metadata=` | read this metadata file; no catalog, no credentials |
| `catalog=` | which configured catalog, instead of the URI's first segment |
| `snapshot=` | the schema as of a snapshot ID |
| `branch=` / `tag=` | the schema behind a named ref |

`snapshot`, `branch` and `tag` are mutually exclusive, and naming two of them
is refused rather than resolved by precedence: a URI whose meaning depends on
knowing which option wins is a URI nobody can read in review.

Where they come from is in `rowless_iceberg.catalog`; what the historical
schemas are good for is in `rowless_iceberg.history`.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, ClassVar

from rowless.errors import BadSourceRef, SchemaNotFound
from rowless.ir.schema import Schema
from rowless.sources.base import SourceRef

from rowless_iceberg import catalog, history

#: URI options that pick a point in the table's history. At most one.
SELECTORS: tuple[str, ...] = ("snapshot", "branch", "tag")


class IcebergSource:
    """Resolve `iceberg://` URIs through a PyIceberg catalog."""

    name: ClassVar[str] = "iceberg"
    schemes: ClassVar[tuple[str, ...]] = ("iceberg",)
    requires: ClassVar[str] = "pyiceberg[pyarrow]>=0.12"

    def available(self) -> bool:
        """True when pyiceberg imports and a catalog is configured.

        Only the first half, in the end: see `rowless_iceberg.catalog.available`
        for why requiring a configured catalog here would hide the
        credential-free `?metadata=` route behind a message about a missing
        install.
        """
        return catalog.available()

    def resolve(self, ref: SourceRef) -> Schema:
        """Load the table and map its current schema, keeping field IDs.

        Field IDs are kept in the sense that matters to a diff: a column whose
        ID has held other names carries them in `derived_from`, which is the
        evidence `rowless.ir.diff` trusts above its own heuristics. The IDs
        themselves are available from `field_ids`, outside the shared IR,
        because no other source has one to put there.
        """
        table = catalog.load_table(ref)
        with _blamed(ref):
            return history.to_ir(
                table,
                self._schema_id(table, ref),
                uri=ref.uri,
                name=ref.table_name,
            )

    def fingerprint(self, ref: SourceRef) -> str:
        """The current metadata location, which changes on every commit.

        Iceberg commits by writing a new metadata file and atomically swapping
        the pointer to it, so the location is a change token with no polling
        and no catalog-specific etag. The resolved schema ID and the selector
        travel with it because the on-disk schema cache is keyed on the URI
        without its options, so `?snapshot=111` and the live table would
        otherwise share one cache entry.
        """
        table = catalog.load_table(ref)
        with _blamed(ref):
            schema_id = self._schema_id(table, ref)
        location = getattr(table, "metadata_location", None) or ""
        return f"{location}:{schema_id}:{self._selector(ref)}"

    def schema_at(self, ref: SourceRef, snapshot_id: int) -> Schema:
        """The schema as of a historical snapshot.

        Lets `rowless diff` compare against what production actually ran, not
        just against another git revision.
        """
        table = catalog.load_table(ref)
        with _blamed(ref):
            return history.to_ir(
                table,
                history.schema_id_at_snapshot(table, snapshot_id),
                uri=f"{ref.uri}?snapshot={snapshot_id}",
                name=ref.table_name,
            )

    def renames_since(self, ref: SourceRef, snapshot_id: int) -> dict[str, str]:
        """Renames between a snapshot and now, resolved by field ID.

        Exact, not heuristic: the format keeps field identity across renames.
        A column added or dropped since the snapshot shares no field ID with
        anything and is left out, so this never reports a drop plus an add as a
        rename.
        """
        table = catalog.load_table(ref)
        with _blamed(ref):
            return history.renames_between(
                table,
                history.schema_id_at_snapshot(table, snapshot_id),
                self._schema_id(table, ref),
            )

    def field_ids(self, ref: SourceRef) -> dict[str, int]:
        """Column name to Iceberg field ID, for the resolved schema.

        An extra beyond the `SchemaSource` protocol, which core never calls.
        It is here because column identity is the one thing this format knows
        and the IR does not, and a caller reasoning about a rename across two
        revisions wants it.
        """
        table = catalog.load_table(ref)
        with _blamed(ref):
            return history.field_ids(table, self._schema_id(table, ref))

    # --- Selectors -----------------------------------------------------------

    def _schema_id(self, table: Any, ref: SourceRef) -> int:
        """Which retained schema this URI asks for."""
        option, value = self._selected(ref)
        if option is None:
            return history.current_schema_id(table)
        if option == "snapshot":
            return history.schema_id_at_snapshot(table, _snapshot_id(ref, value))
        return history.schema_id_at_ref(table, value)

    def _selected(self, ref: SourceRef) -> tuple[str | None, str]:
        """The one history selector on this URI, or `(None, "")` for none."""
        given = [
            (option, str(ref.options[option]).strip())
            for option in SELECTORS
            if str(ref.options.get(option) or "").strip()
        ]
        if len(given) > 1:
            named = ", ".join(f"{option}={value}" for option, value in given)
            raise BadSourceRef(
                f"{ref.uri} names more than one point in the table's history ({named}); "
                f"pass one of {', '.join(SELECTORS)}"
            )
        return given[0] if given else (None, "")

    def _selector(self, ref: SourceRef) -> str:
        """The selector as a cache-key fragment: `snapshot=111`, or empty."""
        option, value = self._selected(ref)
        return f"{option}={value}" if option is not None else ""


def _snapshot_id(ref: SourceRef, value: str) -> int:
    """Parse `?snapshot=`, which is a 64-bit integer in the format."""
    try:
        return int(value)
    except ValueError as exc:
        raise BadSourceRef(
            f"{ref.uri}: snapshot={value!r} is not a snapshot id. Iceberg snapshot ids are "
            "integers; a branch or tag name goes in ?branch= or ?tag="
        ) from exc


@contextmanager
def _blamed(ref: SourceRef) -> Iterator[None]:
    """Prefix a `SchemaNotFound` from the history layer with the URI.

    The history functions are given a table, not a ref, so their messages say
    what is wrong without saying which input said it. That is the right split -
    it keeps them testable with no URI in the way - but a finding has to name
    the source it came from.
    """
    try:
        yield
    except SchemaNotFound as exc:
        if str(exc).startswith(ref.uri):
            raise
        raise SchemaNotFound(f"{ref.uri}: {exc}") from exc
