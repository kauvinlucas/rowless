"""Schemas written by hand: inline mappings and JSON files. No dependencies.

The fallback that keeps rowless adoptable. A team with no Parquet, no dbt and
no catalog can still start by writing the schema of three tables into
`rowless.toml` and get a working check out of it.

Two file shapes are accepted, because both already exist in the wild:

    {"order_id": "int64", "amount": "int64"}

    {"name": "stg_orders",
     "fields": [{"name": "order_id", "dtype": "int64", "nullable": false}]}

The second is what `rowless snapshot` and the source cache write, so a
hand-written file and a generated one are the same thing.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar

from rowless.errors import SchemaNotFound
from rowless.ir.dtypes import DType
from rowless.ir.schema import Field, Origin, Schema
from rowless.ir.serde import DTypeSyntaxError, parse_dtype, schema_from_dict, schema_to_dict
from rowless.sources.base import SourceRef

#: Inline schemas registered by the loader before a check starts, keyed by the
#: name in `inline://<name>`. Populated from `[sources.inline]` in config,
#: which is the shape a brand-new user is most likely to write.
INLINE_SCHEMAS: dict[str, Schema] = {}


class JsonSchemaSource:
    """`json://schemas/orders.json` and `inline://name`, resolved from config."""

    name: ClassVar[str] = "json"
    schemes: ClassVar[tuple[str, ...]] = ("json", "inline")
    requires: ClassVar[str] = ""

    def available(self) -> bool:
        """Always true. This source is the one with no prerequisites."""
        return True

    def resolve(self, ref: SourceRef) -> Schema:
        """Read the JSON file, or look the name up in `[sources.inline]`."""
        if ref.scheme == "inline":
            try:
                return INLINE_SCHEMAS[ref.path]
            except KeyError as exc:
                known = ", ".join(sorted(INLINE_SCHEMAS)) or "none"
                raise SchemaNotFound(
                    f"no inline schema named {ref.path!r}; defined: {known}"
                ) from exc

        path = Path(ref.path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise SchemaNotFound(f"cannot read schema file {ref.path}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise SchemaNotFound(f"{ref.path} is not valid JSON: {exc}") from exc

        origin = Origin(kind="source", ref=ref.uri, file=str(path))
        return _schema_from_payload(payload, name=ref.table_name, origin=origin, where=ref.path)

    def fingerprint(self, ref: SourceRef) -> str:
        """Content hash."""
        if ref.scheme == "inline":
            schema = INLINE_SCHEMAS.get(ref.path)
            if schema is None:
                return ""
            # Hashed through the serialised form, not `hash()`: string hashing
            # is salted per process, and a cache key has to survive a restart.
            serialised = json.dumps(schema_to_dict(schema), sort_keys=True)
            return hashlib.sha256(serialised.encode("utf-8")).hexdigest()[:16]
        try:
            data = Path(ref.path).read_bytes()
        except OSError:
            return ""
        return hashlib.sha256(data).hexdigest()[:16]


def _schema_from_payload(
    payload: object,
    *,
    name: str,
    origin: Origin | None,
    where: str,
) -> Schema:
    """Build a schema from either accepted JSON shape."""
    if isinstance(payload, Mapping) and "fields" in payload:
        schema = schema_from_dict(dict(payload))
        schema = schema if schema.name else schema.relabel(name)
    elif isinstance(payload, Mapping):
        schema = schema_from_mapping(
            {str(column): str(dtype) for column, dtype in payload.items()}, name=name
        )
    else:
        found = type(payload).__name__
        raise SchemaNotFound(f"{where}: expected an object of columns, found {found}")
    if origin is None:
        return schema
    return Schema(
        fields=tuple(
            column
            if column.origin is not None
            else Field(
                name=column.name,
                dtype=column.dtype,
                nullable=column.nullable,
                origin=origin,
                description=column.description,
                derived_from=column.derived_from,
            )
            for column in schema.fields
        ),
        name=schema.name,
    )


def parse_dtype_string(text: str) -> DType:
    """Parse the compact dtype spelling used in config and JSON files.

    Examples: `int64`, `string`, `decimal(18,2)`, `datetime[us, UTC]`,
    `list[struct{a:int64,b:string}]`. The exact inverse of `DType.__str__`,
    and property-tested as such, so a dtype printed in a diagnostic can be
    pasted straight into a contract.

    The parser itself lives in `rowless.ir.serde`, next to the rest of the
    wire format; this is the name sources and config call.
    """
    return parse_dtype(text)


def schema_from_mapping(columns: Mapping[str, str], *, name: str | None = None) -> Schema:
    """Build a schema from name-to-dtype-string pairs."""
    fields: list[Field] = []
    for column, spelling in columns.items():
        try:
            dtype = parse_dtype_string(spelling)
        except DTypeSyntaxError as exc:
            raise SchemaNotFound(f"column {column!r}: {exc}") from exc
        fields.append(Field(name=column, dtype=dtype))
    return Schema(fields=tuple(fields), name=name)
