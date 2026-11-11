"""Schemas from a warehouse catalog snapshot. Extra: `rowless[warehouse]`.

Two modes, and the second one matters more:

* **Live**: query `information_schema.columns` over SQLAlchemy. Needs
  credentials, so it is unavailable in exactly the CI job that most needs the
  check to run.
* **Snapshot**: `rowless snapshot` writes the catalog to a JSON file that gets
  committed. CI then runs credential-free, which is the property that makes
  `whatbreaks` usable and is worth copying.
"""

from __future__ import annotations

import fnmatch
import hashlib
import importlib.util
import json
import os
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, ClassVar

from rowless.errors import SchemaNotFound
from rowless.ir.dtypes import (
    DType,
    TimeUnit,
    binary,
    boolean,
    date,
    datetime,
    decimal,
    float_,
    int_,
    string,
    time,
    unknown,
)
from rowless.ir.schema import Field, Origin, Schema
from rowless.sources.base import SourceRef

#: Where a committed snapshot is looked for, in order. The file is written by
#: `rowless snapshot` and read by CI, which has no credentials.
SNAPSHOT_PATHS: tuple[str, ...] = ("rowless-catalog.json", ".rowless/catalog.json")

#: Environment variable holding the connection string for live mode. A DSN is
#: a credential, so it is never read out of `rowless.toml`.
DSN_ENV = "ROWLESS_WAREHOUSE_DSN"

#: Version of the snapshot file format, so a stale file is reported rather
#: than misread.
SNAPSHOT_VERSION = 1


class InformationSchemaSource:
    """`warehouse://database.schema.table`, live or from a snapshot."""

    name: ClassVar[str] = "information_schema"
    schemes: ClassVar[tuple[str, ...]] = ("warehouse", "snapshot")
    requires: ClassVar[str] = "sqlalchemy>=2.0"

    def available(self) -> bool:
        """True with a snapshot file present, or a configured connection."""
        if _find_snapshot() is not None:
            return True
        return bool(os.environ.get(DSN_ENV)) and importlib.util.find_spec("sqlalchemy") is not None

    def resolve(self, ref: SourceRef) -> Schema:
        """Prefer the snapshot; fall back to a live query when configured."""
        snapshot = _find_snapshot()
        if snapshot is not None:
            schema = _from_snapshot(snapshot, ref.path)
            if schema is not None:
                return schema
            if ref.scheme == "snapshot":
                raise SchemaNotFound(f"{ref.path} is not in the snapshot at {snapshot}")

        dsn = os.environ.get(DSN_ENV)
        if not dsn:
            where = snapshot or "no snapshot file"
            raise SchemaNotFound(
                f"cannot resolve {ref.uri}: {where}, and {DSN_ENV} is not set. "
                "Run `rowless snapshot <dsn>` and commit the result so CI needs no credentials."
            )
        return _query_live(dsn, ref)

    def fingerprint(self, ref: SourceRef) -> str:
        """Snapshot file hash, or the catalog's own change counter when exposed."""
        snapshot = _find_snapshot()
        if snapshot is None:
            # A live catalog has no cheap change token, so nothing is cached:
            # a stale schema would be reported as fact.
            return ""
        try:
            return hashlib.sha256(Path(snapshot).read_bytes()).hexdigest()[:16]
        except OSError:
            return ""


def write_snapshot(dsn: str, target: str, *, include: tuple[str, ...] = ()) -> None:
    """Dump a catalog to a committable JSON file. Backs `rowless snapshot`.

    Writes types only, never data, never row counts: the file lands in a git
    repo and must be boring enough that nobody has to review it for leaks.
    """
    from sqlalchemy import create_engine, text

    engine = create_engine(dsn)
    query = text(
        "select table_catalog, table_schema, table_name, column_name, ordinal_position, "
        "data_type, is_nullable, numeric_precision, numeric_scale, datetime_precision "
        "from information_schema.columns "
        "order by table_catalog, table_schema, table_name, ordinal_position"
    )
    tables: dict[str, list[Field]] = {}
    with engine.connect() as connection:
        for row in connection.execute(query).mappings():
            qualified = ".".join(
                str(part)
                for part in (row["table_catalog"], row["table_schema"], row["table_name"])
                if part
            )
            if include and not any(_matches(qualified, pattern) for pattern in include):
                continue
            tables.setdefault(qualified, []).append(
                Field(
                    name=str(row["column_name"]),
                    dtype=sql_type_to_ir(
                        str(row["data_type"]),
                        precision=row["numeric_precision"],
                        scale=row["numeric_scale"],
                        datetime_precision=row["datetime_precision"],
                    ),
                    nullable=str(row["is_nullable"]).upper() != "NO",
                )
            )

    from rowless.ir.serde import schema_to_dict

    payload = {
        "snapshot_version": SNAPSHOT_VERSION,
        "tables": {
            name: schema_to_dict(Schema(fields=tuple(fields), name=name.split(".")[-1]))
            for name, fields in sorted(tables.items())
        },
    }
    destination = Path(target)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def sql_type_to_ir(
    data_type: str,
    *,
    precision: int | None = None,
    scale: int | None = None,
    datetime_precision: int | None = None,
) -> DType:
    """An `information_schema.columns.data_type` string to an IR dtype.

    Warehouses spell the same type several ways, so this is a lookup table
    rather than a parser. An unrecognised type is UNKNOWN with its spelling in
    the reason: a near-enough guess here would quietly validate a pipeline
    against a type the warehouse does not have.
    """
    name = data_type.strip().lower()
    base = name.split("(")[0].strip()
    if base in {"numeric", "decimal", "number", "bignumeric"}:
        return decimal(precision or 38, scale or 0)
    if base in {"timestamp with time zone", "timestamptz", "datetimeoffset"}:
        return datetime(_sql_unit(datetime_precision), "UTC")
    if base in {"timestamp", "timestamp without time zone", "datetime", "datetime2"}:
        return datetime(_sql_unit(datetime_precision))
    if base in {"time", "time without time zone", "time with time zone"}:
        return time(_sql_unit(datetime_precision))
    mapped = _SQL_TYPES.get(base)
    if mapped is not None:
        return mapped()
    return unknown(f"no IR mapping for SQL type {data_type!r}")


#: The spellings that cover Postgres, Snowflake, BigQuery, DuckDB and MySQL.
#: Values are constructors so each call returns a fresh dtype.
_SQL_TYPES: Mapping[str, Callable[[], DType]] = {
    "boolean": boolean,
    "bool": boolean,
    "bit": boolean,
    "smallint": lambda: int_(16),
    "int2": lambda: int_(16),
    "tinyint": lambda: int_(8),
    "integer": lambda: int_(32),
    "int": lambda: int_(32),
    "int4": lambda: int_(32),
    "bigint": lambda: int_(64),
    "int8": lambda: int_(64),
    "int64": lambda: int_(64),
    "hugeint": lambda: int_(128),
    "real": lambda: float_(32),
    "float4": lambda: float_(32),
    "double precision": lambda: float_(64),
    "double": lambda: float_(64),
    "float": lambda: float_(64),
    "float8": lambda: float_(64),
    "float64": lambda: float_(64),
    "character varying": string,
    "varchar": string,
    "character": string,
    "char": string,
    "text": string,
    "string": string,
    "nvarchar": string,
    "uuid": string,
    "json": string,
    "jsonb": string,
    "bytea": binary,
    "blob": binary,
    "bytes": binary,
    "varbinary": binary,
    "date": date,
}


def _sql_unit(datetime_precision: int | None) -> TimeUnit:
    """Map a declared fractional-second precision onto a time unit."""
    if datetime_precision is None:
        return TimeUnit.MICROSECOND
    if datetime_precision <= 0:
        return TimeUnit.SECOND
    if datetime_precision <= 3:
        return TimeUnit.MILLISECOND
    if datetime_precision <= 6:
        return TimeUnit.MICROSECOND
    return TimeUnit.NANOSECOND


def _find_snapshot() -> str | None:
    """The first committed catalog snapshot found, if any."""
    for candidate in SNAPSHOT_PATHS:
        if Path(candidate).is_file():
            return candidate
    return None


def _from_snapshot(path: str, table: str) -> Schema | None:
    """Read one table out of a snapshot, or None when it is not in there."""
    from rowless.ir.serde import schema_from_dict

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SchemaNotFound(f"cannot read catalog snapshot {path}: {exc}") from exc
    version = payload.get("snapshot_version")
    if version != SNAPSHOT_VERSION:
        raise SchemaNotFound(
            f"{path} is snapshot version {version}, this rowless reads {SNAPSHOT_VERSION}; "
            "re-run `rowless snapshot`"
        )
    tables = payload.get("tables", {})
    entry = tables.get(table) or _match_suffix(tables, table)
    if entry is None:
        return None
    schema = schema_from_dict(entry)
    origin = Origin(kind="source", ref=f"snapshot://{table}", file=path)
    return Schema(
        fields=tuple(
            Field(
                name=column.name,
                dtype=column.dtype,
                nullable=column.nullable,
                origin=origin,
                description=column.description,
                derived_from=column.derived_from,
            )
            for column in schema.fields
        ),
        name=schema.name or table.split(".")[-1],
    )


def _match_suffix(tables: Mapping[str, Any], table: str) -> Any | None:
    """Allow `warehouse://orders` to find `prod.public.orders`, when unambiguous.

    Ambiguity is refused rather than resolved by picking one: silently reading
    the wrong `orders` is the kind of error that survives review.
    """
    suffix = f".{table}"
    matches = [name for name in tables if name == table or name.endswith(suffix)]
    if len(matches) == 1:
        return tables[matches[0]]
    if len(matches) > 1:
        raise SchemaNotFound(
            f"{table!r} matches several tables in the snapshot: {', '.join(sorted(matches))}"
        )
    return None


def _query_live(dsn: str, ref: SourceRef) -> Schema:
    """Query `information_schema.columns` for one table."""
    from sqlalchemy import create_engine, text

    parts = ref.path.split(".")
    conditions = ["table_name = :table"]
    params: dict[str, Any] = {"table": parts[-1]}
    if len(parts) >= 2:
        conditions.append("table_schema = :schema")
        params["schema"] = parts[-2]
    if len(parts) >= 3:
        conditions.append("table_catalog = :catalog")
        params["catalog"] = parts[-3]

    query = text(
        "select column_name, data_type, is_nullable, numeric_precision, numeric_scale, "
        "datetime_precision from information_schema.columns "
        f"where {' and '.join(conditions)} order by ordinal_position"
    )
    engine = create_engine(dsn)
    origin = Origin(kind="source", ref=ref.uri)
    fields: list[Field] = []
    with engine.connect() as connection:
        for row in connection.execute(query, params).mappings():
            fields.append(
                Field(
                    name=str(row["column_name"]),
                    dtype=sql_type_to_ir(
                        str(row["data_type"]),
                        precision=row["numeric_precision"],
                        scale=row["numeric_scale"],
                        datetime_precision=row["datetime_precision"],
                    ),
                    nullable=str(row["is_nullable"]).upper() != "NO",
                    origin=origin,
                )
            )
    if not fields:
        raise SchemaNotFound(f"{ref.path} is not in the catalog at {DSN_ENV}")
    return Schema(fields=tuple(fields), name=parts[-1])


def _matches(qualified: str, pattern: str) -> bool:
    """Glob match for `--include`, on the fully qualified table name."""
    return fnmatch.fnmatch(qualified, pattern) or fnmatch.fnmatch(qualified.split(".")[-1], pattern)


def snapshot_tables(path: str) -> Iterable[str]:
    """Table names in a snapshot file. Used by `rowless doctor`."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()
    return tuple(payload.get("tables", {}))
