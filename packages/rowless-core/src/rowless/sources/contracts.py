"""Schemas from declared data contracts: Pandera, dataframely, patito.

These are the cheapest and most trustworthy sources, because somebody wrote the
schema down on purpose. They also run in both directions: a contract is an
input schema, and it is the expected value `rowless check` compares a computed
output against.

Extra: `rowless[contracts]`. dataframely and patito are imported lazily and
need no extra of their own, since a user who has one declares it directly.

The three libraries model columns differently and move fast, so the conversion
is deliberately duck-typed: rowless looks for a `to_schema()`, then a
`columns` mapping, then a `dtypes` mapping, and says exactly what it looked for
when it finds none. A wrong guess about a contract's dtype would be a false
pass, which is worse than a readable failure.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from rowless.errors import BadSourceRef, SchemaNotFound
from rowless.ir.dtypes import DType, unknown
from rowless.ir.schema import Field, Origin, Schema
from rowless.sources.base import SourceRef


class _ModelSource:
    """Shared behaviour: import `module:Model`, convert its fields to IR."""

    name: ClassVar[str] = ""
    schemes: ClassVar[tuple[str, ...]] = ()
    requires: ClassVar[str] = ""
    module: ClassVar[str] = ""
    """Import name of the modelling library, checked by `available`."""

    def available(self) -> bool:
        """True when the modelling library imports."""
        if not self.module:
            return False
        try:
            return importlib.util.find_spec(self.module) is not None
        except (ImportError, ValueError):
            return False

    def resolve(self, ref: SourceRef) -> Schema:
        """Import the model and convert it."""
        model = _import_model(ref)
        columns = _columns_of(model)
        if not columns:
            raise SchemaNotFound(
                f"{ref.path}: {type(model).__name__} exposes no columns; rowless looked for "
                "to_schema(), .columns and .dtypes"
            )
        origin = Origin(kind="contract", ref=ref.uri, file=_model_file(model))
        return Schema(
            fields=tuple(
                Field(name=column, dtype=dtype, origin=origin) for column, dtype in columns.items()
            ),
            name=ref.table_name,
        )

    def fingerprint(self, ref: SourceRef) -> str:
        """Hash of the defining source file: contracts change when code changes."""
        try:
            model = _import_model(ref)
        except (BadSourceRef, SchemaNotFound):
            return ""
        path = _model_file(model)
        if path is None:
            return ""
        try:
            return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
        except OSError:
            return ""


class PanderaSource(_ModelSource):
    """`pandera://my.module:OrdersSchema`. Reads a DataFrameModel's columns."""

    name: ClassVar[str] = "pandera"
    schemes: ClassVar[tuple[str, ...]] = ("pandera",)
    requires: ClassVar[str] = "pandera>=0.20"
    module: ClassVar[str] = "pandera"


class DataframelySource(_ModelSource):
    """`dataframely://my.module:Orders`. Polars-native contracts."""

    name: ClassVar[str] = "dataframely"
    schemes: ClassVar[tuple[str, ...]] = ("dataframely",)
    requires: ClassVar[str] = "dataframely>=1.0"
    module: ClassVar[str] = "dataframely"


class PatitoSource(_ModelSource):
    """`patito://my.module:Orders`. Pydantic-flavoured Polars contracts."""

    name: ClassVar[str] = "patito"
    schemes: ClassVar[tuple[str, ...]] = ("patito",)
    requires: ClassVar[str] = "patito>=0.8"
    module: ClassVar[str] = "patito"


def _import_model(ref: SourceRef) -> Any:
    """Import `module:Attribute` out of a contract URI."""
    module_path, separator, attribute = ref.path.partition(":")
    if not separator:
        raise BadSourceRef(
            f"contract URI {ref.uri!r} needs a module and a model, as in "
            f"{ref.scheme}://my.module:Orders"
        )
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise SchemaNotFound(f"cannot import {module_path}: {exc}") from exc
    try:
        return getattr(module, attribute)
    except AttributeError as exc:
        raise SchemaNotFound(f"{module_path} has no attribute {attribute!r}") from exc


def _columns_of(model: Any) -> dict[str, DType]:
    """Column name to IR dtype, from whichever shape the library offers."""
    to_schema = getattr(model, "to_schema", None)
    if callable(to_schema):
        try:
            declared = to_schema()
        except Exception as exc:
            raise SchemaNotFound(f"{model}.to_schema() failed: {exc}") from exc
        return _from_mapping(getattr(declared, "columns", {}))

    for attribute in ("columns", "dtypes", "column_types"):
        candidate = getattr(model, attribute, None)
        if callable(candidate):
            candidate = candidate()
        if isinstance(candidate, Mapping) and candidate:
            return _from_mapping(candidate)
    return {}


def _from_mapping(columns: Mapping[str, Any]) -> dict[str, DType]:
    """Convert a library's column mapping, column by column."""
    return {str(name): contract_dtype_to_ir(declared) for name, declared in columns.items()}


def contract_dtype_to_ir(declared: Any) -> DType:
    """Best-effort conversion of one contract column to an IR dtype.

    Tried in order: an IR dtype passed straight through, a `dtype` attribute
    to unwrap (Pandera's `Column`), an Arrow type, a Python annotation, and
    finally the printed spelling. Anything left over is UNKNOWN with the
    spelling in the reason, so `rowless explain` can show a user exactly what
    rowless could not read.
    """
    if isinstance(declared, DType):
        return declared

    for attribute in ("dtype", "type"):
        nested = getattr(declared, attribute, None)
        if nested is not None and not isinstance(nested, type):
            unwrapped = contract_dtype_to_ir(nested)
            if unwrapped.is_known:
                return unwrapped

    arrow = _as_arrow(declared)
    if arrow is not None:
        from rowless.sources.parquet import arrow_dtype_to_ir

        return arrow_dtype_to_ir(arrow)

    from rowless.udf.infer import annotation_to_ir

    if isinstance(declared, type):
        annotated = annotation_to_ir(declared)
        if annotated.is_known:
            return annotated

    spelled = _from_spelling(str(declared))
    if spelled is not None:
        return spelled
    return unknown(f"unreadable contract dtype {declared!r}")


def _as_arrow(declared: Any) -> Any | None:
    """An Arrow type for `declared`, when the library can produce one."""
    if importlib.util.find_spec("pyarrow") is None:
        return None
    import pyarrow as pa

    if isinstance(declared, pa.DataType):
        return declared
    for attribute in ("to_arrow", "pyarrow_dtype", "arrow_dtype"):
        candidate = getattr(declared, attribute, None)
        if candidate is None:
            continue
        resolved = candidate() if callable(candidate) else candidate
        if isinstance(resolved, pa.DataType):
            return resolved
    return None


#: Spellings that turn up in contracts, from pandas, numpy and Polars. Kept as
#: strings on purpose: core will not import an engine to read a type name.
_SPELLINGS: dict[str, str] = {
    "str": "string",
    "string": "string",
    "object": "string",
    "utf8": "string",
    "large_string": "string",
    "bool": "bool",
    "boolean": "bool",
    "int": "int64",
    "int8": "int8",
    "int16": "int16",
    "int32": "int32",
    "int64": "int64",
    "uint8": "uint8",
    "uint16": "uint16",
    "uint32": "uint32",
    "uint64": "uint64",
    "float": "float64",
    "float16": "float16",
    "float32": "float32",
    "float64": "float64",
    "double": "float64",
    "date": "date",
    "date32": "date",
    "date32[day]": "date",
    "datetime64[ns]": "datetime[ns]",
    "datetime64[us]": "datetime[us]",
    "datetime64[ms]": "datetime[ms]",
    "datetime64[s]": "datetime[s]",
    "timedelta64[ns]": "duration[ns]",
    "binary": "binary",
    "null": "null",
}


def _from_spelling(text: str) -> DType | None:
    """Map a printed type name onto the IR, or None when it is not recognised."""
    from rowless.ir.serde import DTypeSyntaxError, parse_dtype

    cleaned = text.strip().lower()
    mapped = _SPELLINGS.get(cleaned)
    if mapped is not None:
        return parse_dtype(mapped)
    try:
        return parse_dtype(cleaned)
    except DTypeSyntaxError:
        return None


def _model_file(model: Any) -> str | None:
    """File the model was defined in, for diagnostics and fingerprinting."""
    try:
        return inspect.getfile(model if inspect.isclass(model) else type(model))
    except (TypeError, OSError):
        return None
