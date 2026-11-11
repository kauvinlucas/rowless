"""Iceberg schema source, via PyIceberg.

The only source in the project whose renames are facts rather than guesses:
Iceberg keeps a column's field ID across a rename, so `rowless diff` reports
`amount -> amount_cents` from the table metadata instead of pairing a removal
with an addition heuristically.

`IcebergSource` is registered under the `rowless.sources` entry-point group and
is reached through `rowless.sources.resolve`, so pipeline code only ever writes
a URI. The rest is exported for anything that wants column identity or schema
history directly - neither has a place in the shared IR, because no other
source can fill them in.
"""

from rowless_iceberg.dtypes import dtype_from_iceberg
from rowless_iceberg.history import (
    field_ids,
    name_history,
    renames_between,
    retained_schema_ids,
    schema_id_at_ref,
    schema_id_at_snapshot,
)
from rowless_iceberg.source import IcebergSource

__all__ = [
    "IcebergSource",
    "dtype_from_iceberg",
    "field_ids",
    "name_history",
    "renames_between",
    "retained_schema_ids",
    "schema_id_at_ref",
    "schema_id_at_snapshot",
]
