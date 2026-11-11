"""Schema history, snapshots, and renames resolved by field ID.

This is the module the package exists for. Everywhere else in rowless a rename
is a guess: `rowless.ir.diff` pairs a removed column with an added one by
lineage, then by position and dtype, then by there being only one of each, and
the last of those rules is weak enough that it can be turned off. Iceberg does
not need any of it. The format assigns every column a field ID at creation and
keeps it across renames, retypes and reorders, so "`amount` became
`amount_cents`" is a fact recorded in the table metadata.

Two things follow, and both are what make an Iceberg-backed repository get more
out of rowless than a Parquet-backed one:

* **A resolved schema carries its own rename evidence.** Each field's
  `derived_from` is filled in with the names that field ID used to have, so
  `rowless diff` reports one `COLUMN_RENAMED` line instead of a removal plus an
  addition - through its highest-trust rule, the one that reads `derived_from`,
  rather than through the heuristics below it. Nothing in core had to change
  for that; the evidence slot was already there, and this is the first source
  able to fill it truthfully.
* **A diff needs no git checkout.** Iceberg retains the schema of every
  retained snapshot, so `?snapshot=`, `?branch=` and `?tag=` resolve what
  production actually ran rather than what another revision of the repository
  says it should have. `renames_since` answers the same question directly.

## What is not recoverable

A snapshot's `schema-id` is optional in the format, and a table old enough to
have snapshots written without one does not record which schema they ran
against. That is reported rather than filled in with the current schema: the
whole value of this source is that its renames are facts, and one guessed
schema in a historical comparison would make every finding derived from it
suspect.
"""

from __future__ import annotations

from typing import Any

from rowless.errors import SchemaNotFound
from rowless.ir.schema import Field, Origin, Schema

from rowless_iceberg.dtypes import dtype_from_iceberg


def retained_schema_ids(table: Any) -> tuple[int, ...]:
    """Every schema ID the table still carries, oldest first.

    Ascending by ID rather than by any recorded order: Iceberg assigns schema
    IDs in sequence, so this is the order the schemas were added in, and it is
    the order a name history has to be read in.
    """
    return tuple(sorted(table.schemas()))


def current_schema_id(table: Any) -> int:
    """The schema ID a plain `iceberg://` URI resolves to."""
    return int(table.schema().schema_id)


def iceberg_schema(table: Any, schema_id: int) -> Any:
    """The PyIceberg schema for one ID.

    Raises:
        SchemaNotFound: when the table no longer retains that schema, listing
            the ones it does. Iceberg expires old metadata, so this is an
            ordinary outcome rather than a bug, and a user needs to be told
            which comparisons are still possible.
    """
    schemas = table.schemas()
    found = schemas.get(schema_id)
    if found is None:
        known = ", ".join(str(known_id) for known_id in sorted(schemas)) or "none"
        raise SchemaNotFound(
            f"the table does not retain schema {schema_id}; retained schema ids: {known}"
        )
    return found


def schema_id_at_snapshot(table: Any, snapshot_id: int) -> int:
    """The schema ID a snapshot was written against.

    Raises:
        SchemaNotFound: when the snapshot is gone, or when it records no
            schema ID. See the module docstring for why the second case is not
            answered with the current schema.
    """
    snapshot = table.snapshot_by_id(snapshot_id)
    if snapshot is None:
        raise SchemaNotFound(
            f"snapshot {snapshot_id} is not in the table metadata; "
            f"retained snapshots: {_snapshot_sample(table)}"
        )
    schema_id = getattr(snapshot, "schema_id", None)
    if schema_id is None:
        raise SchemaNotFound(
            f"snapshot {snapshot_id} records no schema id, so the schema it ran against is "
            "not recoverable from the metadata. Compare against a snapshot written by a "
            "newer writer, or against another git revision"
        )
    return int(schema_id)


def schema_id_at_ref(table: Any, name: str) -> int:
    """The schema ID behind a named branch or tag.

    Raises:
        SchemaNotFound: when no branch or tag has that name, listing the ones
            that do.
    """
    refs = table.refs()
    found = refs.get(name)
    if found is None:
        known = ", ".join(sorted(refs)) or "none"
        raise SchemaNotFound(f"the table has no branch or tag {name!r}; it has: {known}")
    return schema_id_at_snapshot(table, int(found.snapshot_id))


def to_ir(table: Any, schema_id: int, *, uri: str, name: str | None = None) -> Schema:
    """One Iceberg schema as an IR schema, with rename evidence attached.

    Args:
        table: the loaded PyIceberg table.
        schema_id: which of its retained schemas to map.
        uri: the source URI, recorded as each field's origin so a finding can
            name where the column came from.
        name: the relation name. Core labels it from the URI's last segment
            when this is left out.

    Returns:
        The IR schema. Every field carries the Iceberg `doc` as its
        description, nullability from the field's `required` flag, and the
        names its field ID previously had in `derived_from`.
    """
    schema = iceberg_schema(table, schema_id)
    history = name_history(table, through=schema_id)
    origin = Origin(kind="source", ref=uri, file=_metadata_location(table))
    fields = tuple(
        Field(
            name=field.name,
            dtype=dtype_from_iceberg(field.field_type),
            nullable=not field.required,
            origin=origin,
            description=field.doc,
            derived_from=_former_names(history, int(field.field_id), field.name),
        )
        for field in schema.fields
    )
    return Schema(fields=fields, name=name)


def field_ids(table: Any, schema_id: int) -> dict[str, int]:
    """Column name to Iceberg field ID, for the top-level columns of a schema.

    Not part of the `SchemaSource` protocol, and not folded into the IR either:
    a field ID is Iceberg's identity for a column and nothing else in rowless
    has one, so inventing a slot for it in the shared IR would leave every
    other source with an empty field nobody can fill. Exposed here instead, for
    anything that wants to reason about column identity across a rename -
    including the tests, which assert that IDs survive the mapping.
    """
    schema = iceberg_schema(table, schema_id)
    return {field.name: int(field.field_id) for field in schema.fields}


def name_history(table: Any, *, through: int) -> dict[int, tuple[str, ...]]:
    """Every top-level name each field ID has held, oldest first.

    Read over the retained schemas up to and including `through`, so a schema
    resolved at a historical snapshot sees only the history that existed then.
    Repeats are collapsed: a column renamed and renamed back has two names in
    its history, not three.
    """
    schemas = table.schemas()
    collected: dict[int, list[str]] = {}
    for schema_id in sorted(schemas):
        if schema_id > through:
            break
        for field in schemas[schema_id].fields:
            names = collected.setdefault(int(field.field_id), [])
            if field.name not in names:
                names.append(field.name)
    return {field_id: tuple(names) for field_id, names in collected.items()}


def renames_between(table: Any, before_schema_id: int, after_schema_id: int) -> dict[str, str]:
    """Old name to new name for every field ID present in both schemas.

    Exact, not heuristic: a field ID is the column's identity, so a differing
    name under the same ID is a rename and nothing else. A column added or
    dropped between the two schemas has no ID in common and does not appear
    here, which is what keeps this from reporting a drop plus an add as a
    rename the way a name-based comparison would.
    """
    before = {
        int(field.field_id): field.name for field in iceberg_schema(table, before_schema_id).fields
    }
    after = {
        int(field.field_id): field.name for field in iceberg_schema(table, after_schema_id).fields
    }
    return {
        before[field_id]: after[field_id]
        for field_id in sorted(before.keys() & after.keys())
        if before[field_id] != after[field_id]
    }


def _former_names(
    history: dict[int, tuple[str, ...]],
    field_id: int,
    current: str,
) -> tuple[str, ...]:
    """The names this field ID used to have, most recent first.

    Most recent first because that is the order a reader wants: the name the
    column had in the previous release is the one a stale pipeline is still
    using. `rowless.ir.diff` matches against any entry, so the order only
    affects what a human reads.
    """
    return tuple(name for name in reversed(history.get(field_id, ())) if name != current)


def _metadata_location(table: Any) -> str | None:
    """Where the table metadata was read from, for the field origin.

    A plain attribute read, guarded: every catalog PyIceberg ships sets it, but
    an origin is diagnostic commentary and losing it must not cost a schema.
    """
    location = getattr(table, "metadata_location", None)
    return str(location) if location else None


def _snapshot_sample(table: Any, limit: int = 5) -> str:
    """The most recent snapshot IDs, so a stale `?snapshot=` is fixable."""
    snapshots = [int(snapshot.snapshot_id) for snapshot in table.snapshots()]
    if not snapshots:
        return "none (the table has never been written to)"
    recent = sorted(snapshots, reverse=True)[:limit]
    more = ", ..." if len(snapshots) > limit else ""
    return ", ".join(str(snapshot_id) for snapshot_id in recent) + more
