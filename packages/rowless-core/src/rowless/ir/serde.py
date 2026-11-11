"""Serialising the IR, which is how every rowless process talks to another.

Two boundaries need this and both need it to be boring:

* The dry-run worker. Requests and results cross as IR JSON, never as pickles,
  because a pickled LazyFrame would need the engine library in the parent
  process and the parent deliberately does not have it.
* The caches under `.rowless/`: resolved source schemas, and the base-revision
  lineage graph that `rowless diff` reads instead of re-running a checkout.

Dtypes serialise as their printed spelling (`datetime[us, UTC]`), not as a
nested object. One representation for the wire, the cache, config files,
contracts and diagnostics means a user can copy a dtype out of an error message
and paste it into a schema file, and `test_dtype_string_round_trips` is what
keeps that true.
"""

from __future__ import annotations

from typing import Any

from rowless.ir.confidence import Confidence, Coverage
from rowless.ir.dtypes import (
    DType,
    Kind,
    TimeUnit,
    any_,
    binary,
    boolean,
    date,
    datetime,
    decimal,
    duration,
    enum,
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
from rowless.ir.schema import Field, Origin, Relation, Schema

_IDENT_START = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_")
_DIGITS = frozenset("0123456789")


class DTypeSyntaxError(ValueError):
    """A dtype string that is not in the IR's spelling.

    A plain `ValueError` subclass rather than a `RowlessError`: it is raised by
    the parser, and the caller (a config loader, a contract reader) is the one
    that knows which file and line to blame.
    """


# --- Dtypes ------------------------------------------------------------------


def parse_dtype(text: str) -> DType:
    """Parse the compact dtype spelling. The inverse of `DType.__str__`."""
    parser = _DTypeParser(text)
    dtype = parser.parse()
    parser.expect_end()
    return dtype


class _DTypeParser:
    """Recursive-descent parser for the dtype grammar.

    Hand-written, because the grammar is a dozen productions and a parser
    dependency in core would be one more thing between a user and a working
    install.
    """

    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0

    def parse(self) -> DType:
        """Parse one dtype at the current position."""
        self._skip_space()
        name = self._identifier()
        digits = self._digits()
        match name:
            case "bool":
                return boolean()
            case "int" | "uint" | "float":
                return self._sized(name, digits)
            case "decimal":
                return self._decimal()
            case "string":
                return string()
            case "binary":
                return binary()
            case "date":
                return date()
            case "time":
                return time(self._optional_unit())
            case "duration":
                return duration(self._optional_unit())
            case "datetime":
                return self._datetime()
            case "list":
                return list_(self._bracketed_single())
            case "map":
                return self._map()
            case "struct":
                return self._struct()
            case "enum":
                return enum(self._variants())
            case "null":
                return null()
            case "unknown":
                return unknown()
            case "any":
                return any_()
            case _:
                raise DTypeSyntaxError(f"unknown dtype {name!r} in {self.text!r}")

    def expect_end(self) -> None:
        """Fail when anything follows the dtype, rather than ignoring it."""
        self._skip_space()
        if self.pos != len(self.text):
            raise DTypeSyntaxError(f"trailing characters in {self.text!r} at {self.pos}")

    # --- Productions ---

    def _sized(self, name: str, digits: str) -> DType:
        """`int64`, `uint8`, `float32`: the family plus a width."""
        if not digits:
            raise DTypeSyntaxError(f"{name} needs a width, as in {name}64")
        bits = int(digits)
        if name == "int":
            return int_(bits)
        if name == "uint":
            return uint(bits)
        return float_(bits)

    def _decimal(self) -> DType:
        """`decimal(18,2)`."""
        self._expect("(")
        precision = int(self._digits() or "0")
        self._expect(",")
        self._skip_space()
        scale = int(self._digits() or "0")
        self._expect(")")
        return decimal(precision, scale)

    def _optional_unit(self) -> TimeUnit:
        """`[us]`, defaulting to microseconds when the spelling omits it."""
        self._skip_space()
        if not self._peek("["):
            return TimeUnit.MICROSECOND
        self._expect("[")
        unit = self._time_unit()
        self._expect("]")
        return unit

    def _datetime(self) -> DType:
        """`datetime[us]` or `datetime[us, Europe/London]`."""
        self._skip_space()
        if not self._peek("["):
            return datetime()
        self._expect("[")
        unit = self._time_unit()
        self._skip_space()
        tz: str | None = None
        if self._peek(","):
            self._expect(",")
            self._skip_space()
            tz = self._until("]")
        self._expect("]")
        return datetime(unit, tz or None)

    def _bracketed_single(self) -> DType:
        """`[<dtype>]`, for LIST."""
        self._expect("[")
        inner = self.parse()
        self._expect("]")
        return inner

    def _map(self) -> DType:
        """`map[<key>, <value>]`."""
        self._expect("[")
        key = self.parse()
        self._expect(",")
        value = self.parse()
        self._expect("]")
        return map_(key, value)

    def _struct(self) -> DType:
        """`struct{name:<dtype>,...}`."""
        self._expect("{")
        fields: dict[str, DType] = {}
        self._skip_space()
        if self._peek("}"):
            self._expect("}")
            return struct(fields)
        while True:
            self._skip_space()
            name = self._until(":")
            self._expect(":")
            fields[name.strip()] = self.parse()
            self._skip_space()
            if self._peek(","):
                self._expect(",")
                continue
            self._expect("}")
            return struct(fields)

    def _variants(self) -> tuple[str, ...]:
        """`{a,b,c}`, for ENUM."""
        self._expect("{")
        variants: list[str] = []
        self._skip_space()
        if self._peek("}"):
            self._expect("}")
            return ()
        while True:
            variants.append(self._until_any(",}").strip())
            if self._peek(","):
                self._expect(",")
                continue
            self._expect("}")
            return tuple(variants)

    # --- Lexing ---

    def _skip_space(self) -> None:
        while self.pos < len(self.text) and self.text[self.pos] == " ":
            self.pos += 1

    def _identifier(self) -> str:
        start = self.pos
        while self.pos < len(self.text) and self.text[self.pos] in _IDENT_START:
            self.pos += 1
        if start == self.pos:
            raise DTypeSyntaxError(f"expected a dtype name in {self.text!r} at {self.pos}")
        return self.text[start : self.pos]

    def _digits(self) -> str:
        start = self.pos
        while self.pos < len(self.text) and self.text[self.pos] in _DIGITS:
            self.pos += 1
        return self.text[start : self.pos]

    def _time_unit(self) -> TimeUnit:
        self._skip_space()
        raw = self._identifier()
        try:
            return TimeUnit(raw)
        except ValueError as exc:
            units = ", ".join(unit.value for unit in TimeUnit)
            raise DTypeSyntaxError(f"unknown time unit {raw!r}, expected one of {units}") from exc

    def _peek(self, char: str) -> bool:
        self._skip_space()
        return self.pos < len(self.text) and self.text[self.pos] == char

    def _expect(self, char: str) -> None:
        self._skip_space()
        if self.pos >= len(self.text) or self.text[self.pos] != char:
            found = self.text[self.pos : self.pos + 1] or "end of input"
            raise DTypeSyntaxError(f"expected {char!r} in {self.text!r}, found {found!r}")
        self.pos += 1

    def _until(self, char: str) -> str:
        return self._until_any(char)

    def _until_any(self, chars: str) -> str:
        start = self.pos
        while self.pos < len(self.text) and self.text[self.pos] not in chars:
            self.pos += 1
        if start == self.pos:
            raise DTypeSyntaxError(f"expected a name in {self.text!r} at {self.pos}")
        return self.text[start : self.pos]


# --- Records -----------------------------------------------------------------


def origin_to_dict(origin: Origin) -> dict[str, Any]:
    """Origin to a JSON object."""
    return {
        "kind": origin.kind,
        "ref": origin.ref,
        "file": origin.file,
        "line": origin.line,
        "column": origin.column,
    }


def origin_from_dict(payload: dict[str, Any] | None) -> Origin | None:
    """Origin from a JSON object, tolerating a missing one."""
    if not payload:
        return None
    return Origin(
        kind=payload.get("kind", "synthetic"),
        ref=payload.get("ref", ""),
        file=payload.get("file"),
        line=payload.get("line"),
        column=payload.get("column"),
    )


def field_to_dict(column: Field) -> dict[str, Any]:
    """Field to a JSON object, with the dtype as its printed spelling."""
    payload: dict[str, Any] = {
        "name": column.name,
        "dtype": str(column.dtype),
        "nullable": column.nullable,
    }
    if column.origin is not None:
        payload["origin"] = origin_to_dict(column.origin)
    if column.description is not None:
        payload["description"] = column.description
    if column.derived_from:
        payload["derived_from"] = list(column.derived_from)
    if column.dtype.reason is not None:
        # Carried alongside, not inside the spelling: `unknown` prints the same
        # either way, and the reason is what a user needs to read.
        payload["reason"] = column.dtype.reason
    return payload


def field_from_dict(payload: dict[str, Any]) -> Field:
    """Field from a JSON object."""
    dtype = parse_dtype(payload["dtype"])
    if dtype.kind is Kind.UNKNOWN and payload.get("reason"):
        dtype = unknown(payload["reason"])
    return Field(
        name=payload["name"],
        dtype=dtype,
        nullable=payload.get("nullable", True),
        origin=origin_from_dict(payload.get("origin")),
        description=payload.get("description"),
        derived_from=tuple(payload.get("derived_from", ())),
    )


def schema_to_dict(schema: Schema) -> dict[str, Any]:
    """Schema to a JSON object."""
    return {
        "name": schema.name,
        "fields": [field_to_dict(column) for column in schema.fields],
    }


def schema_from_dict(payload: dict[str, Any]) -> Schema:
    """Schema from a JSON object."""
    return Schema(
        fields=tuple(field_from_dict(column) for column in payload.get("fields", ())),
        name=payload.get("name"),
    )


def relation_to_dict(relation: Relation) -> dict[str, Any]:
    """Relation to a JSON object."""
    payload: dict[str, Any] = {
        "schema": schema_to_dict(relation.schema),
        "engine": relation.engine,
        "sort_keys": list(relation.sort_keys),
        "notes": list(relation.notes),
    }
    if relation.origin is not None:
        payload["origin"] = origin_to_dict(relation.origin)
    return payload


def relation_from_dict(payload: dict[str, Any]) -> Relation:
    """Relation from a JSON object."""
    return Relation(
        schema=schema_from_dict(payload["schema"]),
        engine=payload.get("engine", "unknown"),
        origin=origin_from_dict(payload.get("origin")),
        sort_keys=tuple(payload.get("sort_keys", ())),
        notes=tuple(payload.get("notes", ())),
    )


def coverage_to_dict(coverage: Coverage) -> dict[str, Any]:
    """Coverage to a JSON object."""
    return {
        "columns_total": coverage.columns_total,
        "columns_typed": coverage.columns_typed,
        "columns_traced": coverage.columns_traced,
        "confidence": coverage.confidence.name.lower(),
        "reasons": list(coverage.reasons),
    }


def coverage_from_dict(payload: dict[str, Any]) -> Coverage:
    """Coverage from a JSON object."""
    return Coverage(
        columns_total=payload.get("columns_total", 0),
        columns_typed=payload.get("columns_typed", 0),
        columns_traced=payload.get("columns_traced", 0),
        confidence=Confidence[payload.get("confidence", "unknown").upper()],
        reasons=tuple(payload.get("reasons", ())),
    )
