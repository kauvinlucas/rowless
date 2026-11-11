"""Dtype translation between DuckDB and the rowless IR.

DuckDB has the richest type system of the engines rowless targets, so most of
this file is a table. What is worth reading is the handful of places where the
two vocabularies genuinely disagree, and every one of them is written down here
rather than smoothed over:

* **`TIMESTAMP WITH TIME ZONE` is not a zone.** DuckDB stores an instant in UTC
  and renders it in the session's `TimeZone` setting; the column carries no
  named zone of its own. So `to_ir` reports `datetime[us, UTC]`, which is what
  the stored values actually are, and `from_ir` lowers a UTC datetime to
  `TIMESTAMPTZ`. A datetime with any *other* named zone has no DuckDB type and
  is refused rather than converted, because a guessed conversion shifts every
  value by hours and nothing downstream would notice.
* **`TIMESTAMPTZ` is microseconds only.** There is `TIMESTAMP_NS`, and there is
  `TIMESTAMPTZ`, and there is no type that is both. `datetime[ns, UTC]` is
  therefore refused too - see `UNSUPPORTED`, which the conformance suite reads.
* **`TIME` is microseconds, and has no zone.** There is one `TIME` type, so
  `time[us]` is the only IR time that round trips and every other resolution
  is refused rather than rounded - a `time[ms]` column lowered to `TIME` would
  read back as `time[us]` and be reported as a change the user never made.
  DuckDB's `TIME WITH TIME ZONE` carries a UTC offset per value, which the IR
  has no slot for, so it reads back as UNKNOWN.
* **`INTERVAL` is not a duration.** It is a (months, days, microseconds)
  triple, so "one month" is a calendar operation rather than a fixed length of
  time. The IR has only a fixed duration, and `duration[us]` is the closest
  honest reading of it; like `TIME`, every other resolution is refused rather
  than rounded. What the mapping loses is that an IR duration has no notion of
  months or days at all, so two `INTERVAL` values the IR calls equal may not
  be.
* **`DECIMAL` tops out at 38 digits.** `decimal(38,9)` is fine; anything wider
  is refused instead of being silently narrowed.
* **`ARRAY` loses its width.** A fixed-size `INTEGER[3]` becomes `list[int32]`,
  because the IR has no fixed-size variant and the element type is the part
  that breaks pipelines.
* **`UUID`, `BIT`, `BIGNUM` and `UNION` have no IR dtype.** All four read back
  as `Kind.UNKNOWN` with a reason, which poisons comparisons instead of passing
  them. `VARINT` is spelled `BIGNUM` by the type system, which is why the table
  keys on the type id rather than on the SQL name.
* **`ENUM` is a real closed variant set**, unlike a Polars `Categorical`, so it
  maps to the IR ENUM in both directions, variants and order included.

`from_ir` raises `UnsupportedDType` for everything it cannot lower, including
`UNKNOWN`. `to_ir` returns `UNKNOWN` for everything it cannot raise. Neither
substitutes a near-enough type, because a silently approximated dtype is the
bug that would destroy trust in the whole tool.

Types are built by handing DuckDB the SQL spelling and letting it parse
(`duckdb.sqltype`), rather than by calling the type constructors: the
constructors have moved between releases and `duckdb.enum_type` is not
implemented at all in 2.0, while the parser has accepted the same spellings
since 1.0.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rowless.errors import UnsupportedDType
from rowless.ir.dtypes import (
    DType,
    Kind,
    TimeUnit,
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

#: The only zone a DuckDB column can carry, because `TIMESTAMPTZ` stores
#: instants and renders them in the session zone. Compared case-insensitively:
#: a contract may well say `utc`.
UTC = "UTC"

#: Widest `DECIMAL` DuckDB has. Beyond this it has no type, not a slower one.
MAX_DECIMAL_PRECISION = 38

#: Time units `TIME` round trips. DuckDB has one `TIME` type and it is
#: microseconds, so anything else would come back as a dtype the user did not
#: declare.
TIME_UNITS: frozenset[TimeUnit] = frozenset({TimeUnit.MICROSECOND})

#: Time units `INTERVAL` round trips. Same single resolution as `TIME`.
INTERVAL_UNITS: frozenset[TimeUnit] = TIME_UNITS

#: Naive timestamp type per IR time unit. All four resolutions exist, which is
#: more than Polars offers and the reason `datetime[s]` survives here.
TIMESTAMP_BY_UNIT: dict[TimeUnit, str] = {
    TimeUnit.SECOND: "TIMESTAMP_S",
    TimeUnit.MILLISECOND: "TIMESTAMP_MS",
    TimeUnit.MICROSECOND: "TIMESTAMP",
    TimeUnit.NANOSECOND: "TIMESTAMP_NS",
}

#: Signed integer type per IR width.
INTEGER_BY_BITS: dict[int, str] = {
    8: "TINYINT",
    16: "SMALLINT",
    32: "INTEGER",
    64: "BIGINT",
    128: "HUGEINT",
}

#: Unsigned integer type per IR width.
UNSIGNED_BY_BITS: dict[int, str] = {
    8: "UTINYINT",
    16: "USMALLINT",
    32: "UINTEGER",
    64: "UBIGINT",
    128: "UHUGEINT",
}

#: Float type per IR width. DuckDB has no half-precision float.
FLOAT_BY_BITS: dict[int, str] = {32: "FLOAT", 64: "DOUBLE"}

#: Dtypes from the shared conformance corpus that DuckDB genuinely does not
#: have, declared so the suite checks that `from_ir` refuses them instead of
#: expecting a round trip that cannot happen. Read by
#: `rowless.testing.conformance.absent_dtypes` through the adapter.
UNSUPPORTED: tuple[DType, ...] = (datetime(TimeUnit.NANOSECOND, "Europe/London"),)


def to_ir(native: Any) -> DType:
    """Convert a DuckDB type to the IR, returning UNKNOWN when there is no mapping.

    Accepts both spellings rowless encounters: a `DuckDBPyType`, which is what
    a relation reports, and the SQL type string, which is what `DESCRIBE`
    returns.
    """
    dtype = _as_type(native)
    if dtype is None:
        qualified = f"{type(native).__module__}.{type(native).__qualname__}"
        return unknown(f"{qualified} is not a DuckDB type")

    identifier = str(getattr(dtype, "id", ""))
    simple = _SIMPLE_TO_IR.get(identifier)
    if simple is not None:
        return simple()
    match identifier:
        case "decimal":
            return _decimal_to_ir(dtype)
        case "list" | "array":
            return list_(to_ir(_child(dtype, "child")))
        case "struct":
            return _struct_to_ir(dtype)
        case "map":
            return map_(to_ir(_child(dtype, "key")), to_ir(_child(dtype, "value")))
        case "enum":
            return _enum_to_ir(dtype)
        case _:
            return unknown(f"no IR dtype for DuckDB {dtype}")


def from_ir(dtype: DType) -> Any:
    """Convert an IR dtype to a `DuckDBPyType`, raising `UnsupportedDType` for none.

    Goes through the DuckDB type parser, so a spelling this module gets wrong
    is an error here rather than a confusing failure in a `CREATE TABLE`
    hundreds of lines away.
    """
    import duckdb

    spec = type_spec(dtype)
    try:
        return duckdb.sqltype(spec)
    except Exception as exc:
        # The spec is built by this module, so this is a lowering bug rather
        # than anything the user did. It is still reported as an unsupported
        # dtype, because that is the thing the caller can act on.
        raise UnsupportedDType(f"DuckDB rejected the type {spec!r} for {dtype}: {exc}") from exc


def type_spec(dtype: DType) -> str:
    """The SQL spelling of an IR dtype, as a `CREATE TABLE` would write it."""
    match dtype.kind:
        case Kind.BOOL:
            return "BOOLEAN"
        case Kind.INT:
            return _by_width(dtype, INTEGER_BY_BITS, "signed integer")
        case Kind.UINT:
            return _by_width(dtype, UNSIGNED_BY_BITS, "unsigned integer")
        case Kind.FLOAT:
            return _by_width(dtype, FLOAT_BY_BITS, "float")
        case Kind.DECIMAL:
            return _decimal_spec(dtype)
        case Kind.STRING:
            return "VARCHAR"
        case Kind.BINARY:
            return "BLOB"
        case Kind.DATE:
            return "DATE"
        case Kind.TIME:
            return _temporal_spec(dtype, "TIME", TIME_UNITS)
        case Kind.DATETIME:
            return _datetime_spec(dtype)
        case Kind.DURATION:
            return _temporal_spec(dtype, "INTERVAL", INTERVAL_UNITS)
        case Kind.LIST:
            return f"{type_spec(_element(dtype))}[]"
        case Kind.STRUCT:
            return _struct_spec(dtype)
        case Kind.MAP:
            key, value = _pair(dtype)
            return f"MAP({type_spec(key)}, {type_spec(value)})"
        case Kind.ENUM:
            return _enum_spec(dtype)
        case Kind.NULL:
            # Quoted, because NULL is a keyword: `CREATE TABLE t (c NULL)` is a
            # parse error, `c "NULL"` is a column of the null type.
            return '"NULL"'
        case Kind.ANY:
            # The gradual-typing top type. The null type is the one DuckDB
            # coerces into everything else, which is as close as the engine
            # gets, and an ANY column is already counted against coverage.
            return '"NULL"'
        case _:
            detail = f" ({dtype.reason})" if dtype.reason else ""
            raise UnsupportedDType(f"DuckDB has no dtype for {dtype}{detail}")


def quote(identifier: str) -> str:
    """Quote a SQL identifier, doubling any embedded quote.

    Used for every name this adapter writes into SQL: a column called `order"id`
    is legal in the IR and has to survive reaching DuckDB.
    """
    escaped = identifier.replace('"', '""')
    return f'"{escaped}"'


def _as_type(native: Any) -> Any:
    """Return a DuckDB type for either spelling, or None for a foreign object."""
    import duckdb

    if isinstance(native, _type_class()):
        return native
    if isinstance(native, str):
        try:
            return duckdb.sqltype(native)
        except Exception:
            # Not a type DuckDB can parse. `to_ir` turns that into UNKNOWN with
            # the string quoted, which is more useful than a traceback.
            return None
    return None


def _type_class() -> type:
    """The `DuckDBPyType` class, which has moved between DuckDB releases.

    2.x exposes it from `duckdb.sqltypes`, 1.x from `duckdb.typing`, and
    neither re-exports it at the top level. The fallback is `type(...)` of a
    type DuckDB built itself, which works whatever the module layout is.
    """
    import duckdb

    for module_name in ("duckdb.sqltypes", "duckdb.typing"):
        try:
            module = __import__(module_name, fromlist=["DuckDBPyType"])
        except ImportError:
            continue
        found = getattr(module, "DuckDBPyType", None)
        if isinstance(found, type):
            return found
    return type(duckdb.sqltype("BIGINT"))


def _children(dtype: Any) -> dict[str, Any]:
    """The named children of a nested DuckDB type, or nothing when it has none.

    `DuckDBPyType.children` raises on a type that is not nested, which is why
    every caller here has already matched on the type id.
    """
    try:
        return {str(name): value for name, value in dtype.children}
    except Exception:
        return {}


def _child(dtype: Any, name: str) -> Any:
    """One child of a nested type, or a string that `to_ir` reports as UNKNOWN."""
    return _children(dtype).get(name, f"<DuckDB {dtype} reported no {name} type>")


def _decimal_to_ir(dtype: Any) -> DType:
    """Convert a DuckDB `DECIMAL`, which carries precision and scale as children."""
    children = _children(dtype)
    precision = children.get("precision")
    scale = children.get("scale")
    if not isinstance(precision, int) or not isinstance(scale, int):
        return unknown(f"DuckDB {dtype} did not report a precision and scale")
    return decimal(precision, scale)


def _struct_to_ir(dtype: Any) -> DType:
    """Convert a DuckDB `STRUCT`, field by field and in declared order."""
    return struct({name: to_ir(child) for name, child in _children(dtype).items()})


def _enum_to_ir(dtype: Any) -> DType:
    """Convert a DuckDB `ENUM`, keeping the variant set and its order."""
    variants = _children(dtype).get("values")
    if not isinstance(variants, list):
        return unknown(f"DuckDB {dtype} did not report its variants")
    return enum(tuple(str(variant) for variant in variants))


def _by_width(dtype: DType, table: dict[int, str], family: str) -> str:
    """Look a numeric type up by width, refusing a width DuckDB does not have."""
    bits = dtype.bits or 64
    found = table.get(bits)
    if found is None:
        widths = ", ".join(str(width) for width in sorted(table))
        raise UnsupportedDType(f"DuckDB has no {bits}-bit {family}; it has {widths}")
    return found


def _decimal_spec(dtype: DType) -> str:
    """Spell a `DECIMAL`, refusing one wider than DuckDB can represent."""
    if dtype.precision is None or dtype.scale is None:
        raise UnsupportedDType(f"DuckDB needs both a precision and a scale for {dtype}")
    if dtype.precision > MAX_DECIMAL_PRECISION:
        raise UnsupportedDType(
            f"DuckDB DECIMAL holds at most {MAX_DECIMAL_PRECISION} digits, and {dtype} needs "
            f"{dtype.precision}"
        )
    return f"DECIMAL({dtype.precision},{dtype.scale})"


def _temporal_spec(dtype: DType, name: str, allowed: frozenset[TimeUnit]) -> str:
    """Spell `TIME` or `INTERVAL`, refusing a resolution the type does not have.

    Both are microsecond types, and DuckDB offers no parameterised variant. A
    coarser unit would survive the lowering and come back as `us`, which reads
    downstream as a dtype change nobody made, so it is refused here instead.
    """
    unit = dtype.unit or TimeUnit.MICROSECOND
    if unit not in allowed:
        units = ", ".join(sorted(existing.value for existing in allowed))
        raise UnsupportedDType(
            f"DuckDB {name} is a microsecond type with no other resolution, so it cannot "
            f"represent {dtype}; it accepts {units}"
        )
    return name


def _datetime_spec(dtype: DType) -> str:
    """Spell a timestamp, with or without a zone.

    The zone is the whole difficulty. DuckDB has exactly one zoned timestamp
    type, it is microsecond resolution, and the instants in it are UTC.
    """
    unit = dtype.unit or TimeUnit.MICROSECOND
    if dtype.tz is None:
        return TIMESTAMP_BY_UNIT[unit]
    if dtype.tz.upper() != UTC:
        raise UnsupportedDType(
            f"DuckDB has no timestamp carrying a named zone, so it cannot represent {dtype}: "
            f"TIMESTAMP WITH TIME ZONE stores UTC instants and renders them in the session "
            f"zone. Convert the column to UTC, or declare it naive"
        )
    if unit is not TimeUnit.MICROSECOND:
        raise UnsupportedDType(
            f"DuckDB TIMESTAMP WITH TIME ZONE is microsecond resolution, so it cannot "
            f"represent {dtype}; TIMESTAMP_NS and TIMESTAMP_S carry no zone"
        )
    return "TIMESTAMP WITH TIME ZONE"


def _struct_spec(dtype: DType) -> str:
    """Spell a `STRUCT`, quoting every field name."""
    if len(dtype.names) != len(dtype.inner):
        raise UnsupportedDType(f"a DuckDB STRUCT needs a dtype per field name, got {dtype}")
    fields = ", ".join(
        f"{quote(name)} {type_spec(inner)}"
        for name, inner in zip(dtype.names, dtype.inner, strict=True)
    )
    return f"STRUCT({fields})"


def _enum_spec(dtype: DType) -> str:
    """Spell an `ENUM`, as an inline anonymous type.

    DuckDB accepts an inline `ENUM('a', 'b')` in a column definition as well as
    a `CREATE TYPE`, which keeps the materialized table self-contained.
    """
    if not dtype.names:
        raise UnsupportedDType("a DuckDB ENUM needs at least one variant, and none was reported")
    return f"ENUM({', '.join(_literal(variant) for variant in dtype.names)})"


def _literal(value: str) -> str:
    """A single-quoted SQL string literal, doubling any embedded quote."""
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


def _element(dtype: DType) -> DType:
    """The element type of a LIST, refusing one the adapter was not given."""
    if not dtype.inner:
        raise UnsupportedDType("a DuckDB list needs an element dtype, and none was reported")
    return dtype.inner[0]


def _pair(dtype: DType) -> tuple[DType, DType]:
    """The key and value types of a MAP, refusing a half-specified one."""
    if len(dtype.inner) != 2:
        raise UnsupportedDType("a DuckDB MAP needs both a key and a value dtype")
    return dtype.inner[0], dtype.inner[1]


#: DuckDB type ids with a parameter-free IR equivalent. Keyed by `id` rather
#: than by SQL name so that the aliases DuckDB accepts (`INT4`, `VARINT`) all
#: arrive at the same entry, and so the table can be built without importing
#: duckdb at module scope.
_SIMPLE_TO_IR: dict[str, Callable[[], DType]] = {
    "boolean": boolean,
    "tinyint": lambda: int_(8),
    "smallint": lambda: int_(16),
    "integer": lambda: int_(32),
    "bigint": lambda: int_(64),
    "hugeint": lambda: int_(128),
    "utinyint": lambda: uint(8),
    "usmallint": lambda: uint(16),
    "uinteger": lambda: uint(32),
    "ubigint": lambda: uint(64),
    "uhugeint": lambda: uint(128),
    "float": lambda: float_(32),
    "double": lambda: float_(64),
    "varchar": string,
    "blob": binary,
    "date": date,
    # Microseconds since midnight, with no zone. See the module docstring.
    "time": lambda: time(TimeUnit.MICROSECOND),
    "timestamp_s": lambda: datetime(TimeUnit.SECOND),
    "timestamp_ms": lambda: datetime(TimeUnit.MILLISECOND),
    "timestamp": lambda: datetime(TimeUnit.MICROSECOND),
    "timestamp_ns": lambda: datetime(TimeUnit.NANOSECOND),
    "timestamp with time zone": lambda: datetime(TimeUnit.MICROSECOND, UTC),
    # A (months, days, micros) triple read for its microsecond component.
    "interval": lambda: duration(TimeUnit.MICROSECOND),
    "null": null,
    "sqlnull": null,
    "time with time zone": lambda: unknown(
        "DuckDB TIME WITH TIME ZONE carries a UTC offset per value, which the IR has no slot for"
    ),
    "uuid": lambda: unknown("DuckDB UUID is a 128-bit identifier with no IR dtype"),
    "bit": lambda: unknown("DuckDB BIT is a bitstring, which is not the IR's opaque binary"),
    "bignum": lambda: unknown(
        "DuckDB BIGNUM (spelled VARINT in SQL) is an arbitrary-precision integer, and the IR's "
        "integers all have a declared width"
    ),
    "union": lambda: unknown("DuckDB UNION is a tagged union, which the IR cannot describe"),
    "any": lambda: unknown("DuckDB ANY is a binder placeholder, not a column dtype"),
}
