"""Schemas from Parquet footers. The spike's only source. Extra: `rowless[parquet]`.

Reads the footer and nothing else, so a terabyte dataset resolves in one range
request. Directory datasets resolve from the first file, with a mismatch across
files reported rather than merged silently.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, ClassVar

from rowless.errors import SchemaNotFound
from rowless.ir.dtypes import (
    DType,
    TimeUnit,
    any_,
    binary,
    boolean,
    date,
    datetime,
    decimal,
    duration,
    float_,
    int_,
    list_,
    map_,
    null,
    string,
    struct,
    time,
    uint,
    unknown,
)
from rowless.ir.schema import Field, Origin, Schema
from rowless.sources.base import SourceRef

#: Parquet files are read through pyarrow's filesystem layer, so anything it
#: can open resolves without rowless knowing about object stores.
_REMOTE_SCHEMES = frozenset({"s3", "gs", "gcs", "az", "abfs"})


class ParquetSource:
    """Resolve `file://`, `s3://`, `gs://`, `az://` and bare paths."""

    name: ClassVar[str] = "parquet"
    schemes: ClassVar[tuple[str, ...]] = ("file", "s3", "gs", "az", "")
    requires: ClassVar[str] = "pyarrow>=16.0"

    def available(self) -> bool:
        """True when pyarrow imports."""
        return importlib.util.find_spec("pyarrow") is not None

    def resolve(self, ref: SourceRef) -> Schema:
        """Read the footer and map the Arrow schema into the IR."""
        import pyarrow.parquet as pq

        target = self._locate(ref)
        try:
            arrow_schema = pq.read_schema(target)
        except Exception as exc:
            raise SchemaNotFound(f"cannot read Parquet footer at {target}: {exc}") from exc
        origin = Origin(kind="source", ref=ref.uri, file=str(target))
        return arrow_to_ir(arrow_schema, name=ref.table_name, origin=origin)

    def fingerprint(self, ref: SourceRef) -> str:
        """Object etag where the filesystem exposes one, else size plus mtime."""
        if ref.scheme in _REMOTE_SCHEMES:
            import pyarrow.fs as fs

            handle, path = fs.FileSystem.from_uri(ref.uri)
            info = handle.get_file_info(path)
            return f"{info.size}:{info.mtime_ns}"
        target = Path(self._locate(ref))
        try:
            stat = target.stat()
        except OSError:
            return ""
        return f"{stat.st_size}:{stat.st_mtime_ns}"

    def _locate(self, ref: SourceRef) -> str:
        """The file whose footer is read.

        A directory dataset resolves from its first file. Parquet allows the
        files of one dataset to disagree, and merging them quietly is how a
        tool ends up reporting a schema no file actually has; the mismatch is
        caught on read instead.
        """
        if ref.scheme in _REMOTE_SCHEMES:
            return ref.uri
        path = Path(ref.path)
        if not path.is_dir():
            return str(path)
        candidates = sorted(path.glob("**/*.parquet"))
        if not candidates:
            raise SchemaNotFound(f"no .parquet files under {path}")
        return str(candidates[0])


def arrow_to_ir(
    arrow_schema: Any,
    *,
    name: str | None = None,
    origin: Origin | None = None,
) -> Schema:
    """Arrow schema to rowless IR.

    Shared by every adapter with an Arrow bridge (Polars, DuckDB, pandas with
    pyarrow dtypes), so the Arrow mapping is written and tested once.
    """
    fields = tuple(
        Field(
            name=column.name,
            dtype=arrow_dtype_to_ir(column.type),
            nullable=column.nullable,
            origin=origin,
        )
        for column in arrow_schema
    )
    return Schema(fields=fields, name=name)


def arrow_dtype_to_ir(arrow_type: Any) -> DType:
    """One Arrow type to one IR dtype, keeping every parameter.

    Returns UNKNOWN for Arrow types with no IR equivalent, never a near-enough
    substitute: an extension type reported as `string` would make a diff look
    clean when it is not.
    """
    import pyarrow as pa

    if pa.types.is_boolean(arrow_type):
        return boolean()
    if pa.types.is_signed_integer(arrow_type):
        return int_(arrow_type.bit_width)
    if pa.types.is_unsigned_integer(arrow_type):
        return uint(arrow_type.bit_width)
    if pa.types.is_floating(arrow_type):
        return float_(arrow_type.bit_width)
    if pa.types.is_decimal(arrow_type):
        return decimal(arrow_type.precision, arrow_type.scale)
    if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        return string()
    if pa.types.is_binary(arrow_type) or pa.types.is_large_binary(arrow_type):
        return binary()
    if pa.types.is_fixed_size_binary(arrow_type):
        return binary()
    if pa.types.is_date(arrow_type):
        return date()
    if pa.types.is_time(arrow_type):
        return time(_arrow_unit(arrow_type.unit))
    if pa.types.is_timestamp(arrow_type):
        return datetime(_arrow_unit(arrow_type.unit), arrow_type.tz)
    if pa.types.is_duration(arrow_type):
        return duration(_arrow_unit(arrow_type.unit))
    if pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type):
        return list_(arrow_dtype_to_ir(arrow_type.value_type))
    if pa.types.is_fixed_size_list(arrow_type):
        return list_(arrow_dtype_to_ir(arrow_type.value_type))
    if pa.types.is_map(arrow_type):
        return map_(arrow_dtype_to_ir(arrow_type.key_type), arrow_dtype_to_ir(arrow_type.item_type))
    if pa.types.is_struct(arrow_type):
        return struct({child.name: arrow_dtype_to_ir(child.type) for child in arrow_type})
    if pa.types.is_dictionary(arrow_type):
        # A dictionary of strings is an enum with the variants left in the
        # data, which the footer does not carry. ANY would claim too much and
        # string would lose the closed set, so the value type is used and a
        # note is left to the adapter that knows better.
        return arrow_dtype_to_ir(arrow_type.value_type)
    if pa.types.is_null(arrow_type):
        return null()
    if isinstance(arrow_type, pa.ExtensionType):
        return any_()
    return unknown(f"no IR mapping for Arrow type {arrow_type}")


def _arrow_unit(unit: str) -> TimeUnit:
    """Arrow's unit spelling to the IR's."""
    return {
        "s": TimeUnit.SECOND,
        "ms": TimeUnit.MILLISECOND,
        "us": TimeUnit.MICROSECOND,
        "ns": TimeUnit.NANOSECOND,
    }[unit]
