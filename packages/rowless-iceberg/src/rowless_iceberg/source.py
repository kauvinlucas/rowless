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
"""

from __future__ import annotations

from typing import ClassVar

from rowless.ir.schema import Schema
from rowless.sources.base import SourceRef


class IcebergSource:
    """Resolve `iceberg://` URIs through a PyIceberg catalog."""

    name: ClassVar[str] = "iceberg"
    schemes: ClassVar[tuple[str, ...]] = ("iceberg",)
    requires: ClassVar[str] = "pyiceberg>=0.7"

    def available(self) -> bool:
        """True when pyiceberg imports and a catalog is configured."""
        raise NotImplementedError("TODO(iceberg)")

    def resolve(self, ref: SourceRef) -> Schema:
        """Load the table and map its current schema, keeping field IDs."""
        raise NotImplementedError("TODO(iceberg)")

    def fingerprint(self, ref: SourceRef) -> str:
        """The current metadata location, which changes on every commit."""
        raise NotImplementedError("TODO(iceberg)")

    def schema_at(self, ref: SourceRef, snapshot_id: int) -> Schema:
        """The schema as of a historical snapshot.

        Lets `rowless diff` compare against what production actually ran, not
        just against another git revision.
        """
        raise NotImplementedError("TODO(iceberg)")

    def renames_since(self, ref: SourceRef, snapshot_id: int) -> dict[str, str]:
        """Renames between a snapshot and now, resolved by field ID.

        Exact, not heuristic: the format keeps field identity across renames.
        """
        raise NotImplementedError("TODO(iceberg)")
