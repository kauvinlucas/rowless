"""SQL types and the rowless IR, in both directions.

This is the canonical mapping between `sqlglot.exp.DataType` and the IR, and it
is the only one in the workspace: `rowless_dbt.dtypes` parses a manifest's type
strings and then delegates the mapping here, which is what its module docstring
promised would happen once this adapter landed. One table, one set of
judgement calls, one place to fix a wrong answer.

`sqlglot` does the dialect work, which is the part this module must not
re-implement: Snowflake `number` arrives as `DECIMAL(38, 0)`, Snowflake `float`
as `DOUBLE`, BigQuery `float64` as `DOUBLE` and BigQuery `timestamp` as a
zone-aware one while its `datetime` is naive. What is left here is the mapping
onto the IR, and the decisions in it. Written down because each one decides
whether a rename or a widening is caught or missed:

* **A zone-aware timestamp is reported as UTC.** The IR tracks aware against
  naive, and SQL's `TIMESTAMP WITH TIME ZONE` is an instant, not a named zone.
  `timestamp without time zone` stays naive, which is the distinction that
  actually breaks pipelines.
* **A precision becomes a time unit.** `timestamp(3)` is millisecond
  resolution and `timestamp(6)` microsecond, and the same reading applies to
  `time(3)`. Where no precision is declared the IR default (microsecond) is
  used, because that is what the dialects that have a default use.
* **`decimal` with no precision is UNKNOWN, not a guess.** Where the dialect
  has a default, sqlglot has already filled it in; where it has not, the width
  is genuinely unknown and a guessed one would compare equal to something it
  is not. This is also what a decimal *expression* comes back as on older
  sqlglot releases, where arithmetic on a `DECIMAL(18, 2)` loses the
  parameters - see the README.
* **Semi-structured and exotic types are ANY.** `json`, `variant`, `super`,
  `uuid`, `geography` and the range types have no IR equivalent. ANY is the
  gradual-typing escape hatch: it lowers the coverage score and warns, rather
  than failing a check that might be correct or passing one that is not.
* **An enum widens to a string when its variants are not declared.** The IR
  accepts an enum value into a string slot, so that direction loses nothing.
  The reverse - lowering `enum{}` with no variants - is refused instead,
  because it would read back as a plain string and be reported downstream as a
  dtype change nobody made.
* **Anything unrecognised is UNKNOWN with the original spelling in the
  reason.** A column rowless cannot type poisons comparisons instead of
  passing them, and the reason is what tells a user which type to report.

## Lowering, and what it refuses

`from_ir` builds the `exp.DataType` programmatically rather than by parsing a
type string, for one concrete reason: `exp.DataType.build("UTINYINT")` is a
parse error in generic SQL and only resolves in a dialect that has unsigned
integers. Constructing the node directly means an unsigned column lowers the
same way whatever dialect the query is written in, which is what the catalogue
in `rowless_sql.adapter` needs to stay true.

Three lowerings are refused outright, and `UNSUPPORTED` is the list the
conformance suite holds this adapter to:

* `datetime[*, <named zone>]` - SQL's aware timestamp carries an offset, not a
  zone, so `datetime[ns, Europe/London]` would come back as
  `datetime[ns, UTC]`: every value shifted, silently.
* `duration` at anything other than microseconds - `INTERVAL` carries no
  resolution, so a millisecond duration would read back as a microsecond one.
* `float16` and `enum{}` - no SQL type for the first, nothing to round-trip
  for the second.
"""

from __future__ import annotations

from typing import Any

from rowless.errors import UnsupportedDType
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

#: Signed integers, by the sqlglot type name, with the width the IR records.
SIGNED_INTEGERS: dict[str, int] = {
    "TINYINT": 8,
    "SMALLINT": 16,
    "SMALLSERIAL": 16,
    "INT": 32,
    "MEDIUMINT": 32,
    "SERIAL": 32,
    "BIGINT": 64,
    "BIGSERIAL": 64,
    "INT128": 128,
}

#: Unsigned integers. MySQL spells these `int unsigned`; DuckDB spells them
#: `uinteger`, and sqlglot normalises both.
UNSIGNED_INTEGERS: dict[str, int] = {
    "UTINYINT": 8,
    "USMALLINT": 16,
    "UINT": 32,
    "UMEDIUMINT": 32,
    "UBIGINT": 64,
    "UINT128": 128,
}

#: Floats. The width is sqlglot's answer after dialect resolution, which is why
#: Snowflake `float` lands on DOUBLE and Postgres `float4` on FLOAT.
FLOATS: dict[str, int] = {"FLOAT": 32, "DOUBLE": 64}

#: Everything the warehouse calls a string.
STRINGS: frozenset[str] = frozenset(
    {
        "CHAR",
        "NCHAR",
        "VARCHAR",
        "NVARCHAR",
        "TEXT",
        "TINYTEXT",
        "MEDIUMTEXT",
        "LONGTEXT",
        "NAME",
    }
)

#: Everything the warehouse calls bytes.
BINARIES: frozenset[str] = frozenset(
    {"BINARY", "VARBINARY", "BLOB", "TINYBLOB", "MEDIUMBLOB", "LONGBLOB"}
)

#: Naive timestamps: no zone, however much precision.
NAIVE_TIMESTAMPS: frozenset[str] = frozenset(
    {"TIMESTAMP", "DATETIME", "DATETIME2", "SMALLDATETIME", "TIMESTAMPNTZ", "DATETIME64"}
)

#: Zone-aware timestamps, all reported as UTC.
AWARE_TIMESTAMPS: frozenset[str] = frozenset({"TIMESTAMPTZ", "TIMESTAMPLTZ"})

#: Timestamps whose unit is in the type name rather than in a precision.
FIXED_UNIT_TIMESTAMPS: dict[str, TimeUnit] = {
    "TIMESTAMP_S": TimeUnit.SECOND,
    "TIMESTAMP_MS": TimeUnit.MILLISECOND,
    "TIMESTAMP_NS": TimeUnit.NANOSECOND,
}

#: Types with no IR equivalent, reported as ANY with the spelling in the
#: reason. Semi-structured columns genuinely hold anything, and the exotic ones
#: are better reported as unconstrained than as a near-enough scalar.
OPAQUE: frozenset[str] = frozenset(
    {
        "JSON",
        "JSONB",
        "VARIANT",
        "OBJECT",
        "SUPER",
        "UUID",
        "GEOGRAPHY",
        "GEOMETRY",
        "POINT",
        "RING",
        "LINESTRING",
        "MULTILINESTRING",
        "POLYGON",
        "MULTIPOLYGON",
        "INET",
        "IPADDRESS",
        "IPPREFIX",
        "HLLSKETCH",
        "XML",
        "USERDEFINED",
        "INT4RANGE",
        "INT4MULTIRANGE",
        "INT8RANGE",
        "INT8MULTIRANGE",
        "NUMRANGE",
        "NUMMULTIRANGE",
        "TSRANGE",
        "TSMULTIRANGE",
        "TSTZRANGE",
        "TSTZMULTIRANGE",
        "DATERANGE",
        "DATEMULTIRANGE",
        "TIMETZ",
    }
)

#: The sqlglot type name every array-shaped type arrives under.
ARRAYS: frozenset[str] = frozenset({"ARRAY", "LIST", "NESTED"})

#: Time units a declared precision maps onto, coarsest first. A precision of 0
#: is whole seconds; anything above 6 digits is nanoseconds, which is as fine
#: as the IR goes.
_PRECISION_UNITS: tuple[tuple[int, TimeUnit], ...] = (
    (0, TimeUnit.SECOND),
    (3, TimeUnit.MILLISECOND),
    (6, TimeUnit.MICROSECOND),
)

#: Digits of precision written back out for each IR time unit, so a lowered
#: timestamp carries its resolution rather than the dialect's default.
_UNIT_PRECISION: dict[TimeUnit, int] = {
    TimeUnit.SECOND: 0,
    TimeUnit.MILLISECOND: 3,
    TimeUnit.MICROSECOND: 6,
    TimeUnit.NANOSECOND: 9,
}

#: IR integer width to the sqlglot type name, signed and unsigned.
_SIGNED_SPELLING: dict[int, str] = {
    8: "TINYINT",
    16: "SMALLINT",
    32: "INT",
    64: "BIGINT",
    128: "INT128",
}
_UNSIGNED_SPELLING: dict[int, str] = {
    8: "UTINYINT",
    16: "USMALLINT",
    32: "UINT",
    64: "UBIGINT",
    128: "UINT128",
}

#: IR float width to the sqlglot type name. There is no half-precision SQL
#: type, so `float16` is refused rather than widened.
_FLOAT_SPELLING: dict[int, str] = {32: "FLOAT", 64: "DOUBLE"}

#: Corpus dtypes SQL has no equivalent for. Read by the adapter's
#: `unsupported_dtypes()`, which is a claim the conformance suite then holds it
#: to: `from_ir` must raise for each of these rather than approximate it.
UNSUPPORTED: tuple[DType, ...] = (datetime(TimeUnit.NANOSECOND, "Europe/London"),)


def to_ir(native: Any, *, spelling: str | None = None) -> DType:
    """Map a `sqlglot.exp.DataType` onto the IR.

    Args:
        native: the parsed or annotated SQL type. Anything that is not a
            sqlglot type - including the `None` sqlglot leaves on an expression
            it could not annotate - comes back UNKNOWN rather than guessed at.
        spelling: the type as the warehouse spelled it, when a caller has it.
            Quoted verbatim in the reason of an ANY or UNKNOWN result, which is
            what tells a user which declaration to fix.

    Returns:
        The IR dtype, or UNKNOWN carrying the reason it could not be read.
    """
    name = _type_name(native)
    if name is None:
        return unknown(f"not a SQL type: {type(native).__qualname__}")
    quoted = spelling or name.lower()

    if name in SIGNED_INTEGERS:
        return int_(SIGNED_INTEGERS[name])
    if name in UNSIGNED_INTEGERS:
        return uint(UNSIGNED_INTEGERS[name])
    if name in FLOATS:
        return float_(FLOATS[name])
    if name.startswith("DECIMAL"):
        return _decimal(native, quoted)
    if name in STRINGS:
        return string()
    if name in BINARIES:
        return binary()
    if name == "BOOLEAN":
        return boolean()
    if name in {"DATE", "DATE32"}:
        return date()
    if name == "TIME":
        return time(_unit(native))
    if name in FIXED_UNIT_TIMESTAMPS:
        return datetime(FIXED_UNIT_TIMESTAMPS[name])
    if name in NAIVE_TIMESTAMPS:
        return datetime(_unit(native))
    if name in AWARE_TIMESTAMPS:
        # Aware is reported as UTC: SQL's aware timestamp is an instant, and
        # what matters downstream is that it is not naive.
        return datetime(_unit(native), "UTC")
    if name == "INTERVAL":
        return duration(TimeUnit.MICROSECOND)
    if name == "NULL":
        return null()
    if name in ARRAYS:
        return _array(native, quoted)
    if name == "STRUCT":
        return _struct(native, quoted)
    if name == "MAP":
        return _map(native, quoted)
    if name.startswith("ENUM"):
        return _enum(native)
    if name in OPAQUE:
        return any_().with_(reason=f"the SQL type {quoted} has no IR equivalent")
    if name == "UNKNOWN":
        return unknown(f"the warehouse reports {quoted} as an unknown type")
    return unknown(f"no IR dtype for the SQL type {quoted}")


def from_ir(dtype: DType) -> Any:
    """Lower an IR dtype to a `sqlglot.exp.DataType`.

    Raises:
        rowless.errors.UnsupportedDType: there is no SQL type whose round trip
            would return what it was given. Never approximated: a dtype that
            came back changed would be reported downstream as a change nobody
            made.
    """
    match dtype.kind:
        case Kind.BOOL:
            return _simple("BOOLEAN")
        case Kind.INT:
            return _simple(_width_spelling(dtype, _SIGNED_SPELLING, "signed integer"))
        case Kind.UINT:
            return _simple(_width_spelling(dtype, _UNSIGNED_SPELLING, "unsigned integer"))
        case Kind.FLOAT:
            return _simple(_width_spelling(dtype, _FLOAT_SPELLING, "float"))
        case Kind.DECIMAL:
            return _parameterised("DECIMAL", dtype.precision or 0, dtype.scale or 0)
        case Kind.STRING:
            return _simple("VARCHAR")
        case Kind.BINARY:
            return _simple("VARBINARY")
        case Kind.DATE:
            return _simple("DATE")
        case Kind.TIME:
            return _parameterised("TIME", _precision(dtype))
        case Kind.DATETIME:
            return _timestamp(dtype)
        case Kind.DURATION:
            return _interval(dtype)
        case Kind.LIST:
            return _nested("ARRAY", [from_ir(inner) for inner in dtype.inner])
        case Kind.MAP:
            return _nested("MAP", [from_ir(inner) for inner in dtype.inner])
        case Kind.STRUCT:
            return _nested("STRUCT", _field_defs(dtype))
        case Kind.ENUM:
            return _enum_type(dtype)
        case Kind.NULL:
            return _simple("NULL")
        case Kind.ANY:
            raise UnsupportedDType(
                "ANY is the gradual-typing escape hatch, not a SQL type; a column rowless could "
                "not type is materialized as the SQL null type and the run says so"
            )
        case _:
            reason = dtype.reason or "no reason given"
            raise UnsupportedDType(f"cannot lower an unresolved dtype to SQL ({reason})")


def parse(text: str, dialect: str = "") -> Any | None:
    """Parse a SQL type string, or return None when it cannot be read.

    Args:
        text: the type as a manifest, a catalog or a `CAST` spells it, e.g.
            `numeric(18,2)`.
        dialect: the sqlglot dialect to read it in. Tried first, then generic
            SQL: a type this dialect does not know may still parse as standard
            SQL, and a dialect rowless was given wrongly must not take every
            column with it.
    """
    from sqlglot import exp

    for attempt in ([dialect] if dialect else []) + [None]:
        try:
            return exp.DataType.build(text, dialect=attempt)
        except Exception:
            continue
    return None


def is_unannotated(native: Any) -> bool:
    """True when sqlglot resolved an expression without working out its type.

    The honest answer for one of these is an ANY column with a note naming
    what could not be typed, which is what `rowless_sql.resolve` produces.
    sqlglot leaves either `None` or the `UNKNOWN` type behind depending on the
    release and on where in an expression the gap is, so both count.
    """
    name = _type_name(native)
    return name is None or name == "UNKNOWN"


def _simple(name: str) -> Any:
    """A parameterless `exp.DataType`, built from the type name."""
    from sqlglot import exp

    return exp.DataType(this=exp.DataType.Type[name])


def _parameterised(name: str, *parameters: int) -> Any:
    """A type carrying integer parameters: `DECIMAL(18, 2)`, `TIMESTAMP(3)`."""
    from sqlglot import exp

    return exp.DataType(
        this=exp.DataType.Type[name],
        expressions=[
            exp.DataTypeParam(this=exp.Literal.number(parameter)) for parameter in parameters
        ],
    )


def _nested(name: str, inner: list[Any]) -> Any:
    """A type whose parameters are other types: `ARRAY<T>`, `MAP<K, V>`."""
    from sqlglot import exp

    return exp.DataType(this=exp.DataType.Type[name], expressions=inner, nested=True)


def _field_defs(dtype: DType) -> list[Any]:
    """Struct fields as the `ColumnDef` nodes sqlglot carries them in."""
    from sqlglot import exp

    return [
        exp.ColumnDef(this=exp.to_identifier(name), kind=from_ir(inner))
        for name, inner in zip(dtype.names, dtype.inner, strict=True)
    ]


def _enum_type(dtype: DType) -> Any:
    """`ENUM('a', 'b', 'c')`, refusing the variant set nobody declared."""
    from sqlglot import exp

    if not dtype.names:
        raise UnsupportedDType(
            "an enum with no variants cannot be lowered to SQL: it would read back as a plain "
            "string and be reported as a dtype change nobody made"
        )
    return exp.DataType(
        this=exp.DataType.Type.ENUM,
        expressions=[exp.Literal.string(variant) for variant in dtype.names],
    )


def _timestamp(dtype: DType) -> Any:
    """Timestamp, naive or UTC, refusing a named zone.

    SQL's `TIMESTAMP WITH TIME ZONE` stores an instant. Lowering a timestamp
    that carries `Europe/London` to it would move every value and keep no
    record that it had happened, so the dtype is refused instead.
    """
    if dtype.tz is None:
        return _parameterised("TIMESTAMP", _precision(dtype))
    if dtype.tz.upper() == "UTC":
        return _parameterised("TIMESTAMPTZ", _precision(dtype))
    raise UnsupportedDType(
        f"SQL has no timestamp carrying the zone {dtype.tz!r}: an aware SQL timestamp is an "
        "instant, so lowering this dtype would shift every value into UTC silently"
    )


def _interval(dtype: DType) -> Any:
    """`INTERVAL`, which carries no resolution, so only microseconds round-trip."""
    if (dtype.unit or TimeUnit.MICROSECOND) is not TimeUnit.MICROSECOND:
        raise UnsupportedDType(
            f"SQL INTERVAL carries no resolution, so {dtype} would read back as "
            "duration[us]; declare the column in microseconds, or as an integer count"
        )
    return _simple("INTERVAL")


def _width_spelling(dtype: DType, spellings: dict[int, str], family: str) -> str:
    """The SQL type name for a width, refusing one SQL does not have."""
    width = dtype.bits or 64
    name = spellings.get(width)
    if name is None:
        available = ", ".join(str(bits) for bits in sorted(spellings))
        raise UnsupportedDType(
            f"SQL has no {width}-bit {family}; the widths it has are {available}"
        )
    return name


def _precision(dtype: DType) -> int:
    """Digits of precision for an IR time unit."""
    return _UNIT_PRECISION[dtype.unit or TimeUnit.MICROSECOND]


def _type_name(native: Any) -> str | None:
    """The sqlglot type name (`VARCHAR`, `TIMESTAMPTZ`), or None."""
    this = getattr(native, "this", None)
    name = getattr(this, "name", None)
    return str(name) if isinstance(name, str) else None


def _decimal(native: Any, quoted: str) -> DType:
    """Fixed-point decimal, refusing the one with no declared width."""
    arguments = _integer_arguments(native)
    if not arguments:
        return unknown(
            f"{quoted} declares no precision, so rowless cannot say what fits in it; "
            "declare it as decimal(p,s) in the contract"
        )
    precision = arguments[0]
    scale = arguments[1] if len(arguments) > 1 else 0
    if scale > precision:
        return unknown(f"{quoted} has a scale larger than its precision")
    return decimal(precision, scale)


def _array(native: Any, quoted: str) -> DType:
    """List type, with its element type when the declaration carries one."""
    inner = _inner_types(native)
    if not inner:
        return unknown(f"{quoted} does not say what it holds")
    return list_(inner[0])


def _struct(native: Any, quoted: str) -> DType:
    """Struct type, field by field and in declaration order."""
    fields: dict[str, DType] = {}
    for position, argument in enumerate(getattr(native, "expressions", ())):
        kind = argument.args.get("kind") if hasattr(argument, "args") else None
        if kind is None:
            return unknown(f"{quoted} has a field rowless could not read")
        name = getattr(getattr(argument, "this", None), "name", None)
        fields[str(name) if name else f"f{position}"] = to_ir(kind)
    if not fields:
        return unknown(f"{quoted} does not say what fields it has")
    return struct(fields)


def _map(native: Any, quoted: str) -> DType:
    """Map type, which needs both halves to be useful."""
    inner = _inner_types(native)
    if len(inner) != 2:
        return unknown(f"{quoted} does not say what it maps")
    return map_(inner[0], inner[1])


def _enum(native: Any) -> DType:
    """Closed variant set when the type lists one, else a plain string.

    Widening an enum to a string is lossless in the direction that matters:
    the IR accepts an enum value into a string slot, and a variant set rowless
    does not know is better left open than invented.
    """
    variants = tuple(
        str(expression.name)
        for expression in getattr(native, "expressions", ())
        if getattr(expression, "name", None)
    )
    return enum(variants) if variants else string()


def _inner_types(native: Any) -> tuple[DType, ...]:
    """IR dtypes for a nested type's parameters."""
    found: list[DType] = []
    for expression in getattr(native, "expressions", ()):
        if _type_name(expression) is None:
            continue
        found.append(to_ir(expression))
    return tuple(found)


def _integer_arguments(native: Any) -> tuple[int, ...]:
    """The integer parameters of a type: the 18 and 2 of `numeric(18,2)`."""
    found: list[int] = []
    for expression in getattr(native, "expressions", ()):
        text = getattr(expression, "name", None)
        if not isinstance(text, str) or not text.strip().isdigit():
            continue
        found.append(int(text.strip()))
    return tuple(found)


def _unit(native: Any) -> TimeUnit:
    """Time unit for a temporal type, read off its declared precision."""
    arguments = _integer_arguments(native)
    if not arguments:
        return TimeUnit.MICROSECOND
    digits = arguments[0]
    for limit, unit in _PRECISION_UNITS:
        if digits <= limit:
            return unit
    return TimeUnit.NANOSECOND
