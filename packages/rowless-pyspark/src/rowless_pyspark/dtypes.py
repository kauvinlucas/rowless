"""Dtype translation between Spark and the rowless IR.

Spark's type system is the most opinionated of the engines rowless targets, and
the disagreements are not cosmetic. Every one of them is written down here
rather than smoothed over:

* **There are no unsigned integers.** `uint8` and `uint32` have no Spark
  equivalent at all. `ByteType` would lose the top half of a `uint8` range and
  `ShortType` would change the width, so both are refused outright and
  declared through `unsupported_dtypes()`.
* **`TimestampType` has no timezone of its own.** It is an instant, rendered in
  the *session's* timezone, so the IR dtype it maps to depends on what that
  zone is. The adapter pins the sessions it creates to UTC, which is what makes
  `datetime[us, UTC]` round trip; a named zone that is not the session's is
  refused rather than converted, because the conversion would move every value
  by hours. `TimestampNTZType` is the naive one, and rowless keeps naive and
  aware apart exactly as Spark does.
* **Timestamps are microseconds, always.** `datetime[ms]` and `datetime[ns]`
  have no Spark spelling; lowering either to `TimestampType` would change the
  resolution silently, so both are refused.
* **`TIME` is a Spark 4.1 type with a precision, and `TIMESTAMP` is not.**
  `TimeType(6)` is microseconds, `TimeType(3)` milliseconds, and the maximum is
  6 - so `time[ns]` is refused. On PySpark releases without `TimeType` at all,
  every `time` dtype is declared absent rather than approximated as a string.
* **A duration is a day-time interval, and its fields are a range rather than
  a resolution.** `DayTimeIntervalType` stores microseconds whatever its start
  and end fields say, so `to_ir` reports `duration[us]` for all of them and
  `from_ir` refuses any other unit.
* **`YearMonthIntervalType` is not a duration.** A month is not a fixed number
  of microseconds, so there is no IR dtype for it and `to_ir` returns UNKNOWN
  with that as the reason.
* **There is a real `MapType`.** Of the engines rowless targets, Spark and
  DuckDB are the only ones with a first-class map; Polars has to encode one as
  `list[struct{key, value}]`. A map round trips here with nothing lost.
* **There is no enum.** Spark has no closed variant set, and a plain string
  loses it, so `enum{...}` is declared absent.
* **`CHAR(n)`, `VARCHAR(n)` and a collated string are all strings.** The length
  limit and the collation are real to Spark and invisible to the IR, which is
  worth knowing when a contract passes here and would fail on an engine that
  models the width.
* **`VariantType` is refused, not guessed.** It holds arbitrary
  semi-structured values, so it becomes `Kind.UNKNOWN` with a reason, which
  poisons comparisons instead of passing them.

`from_ir` raises `UnsupportedDType` for everything it cannot lower, including
`UNKNOWN`. `to_ir` returns `UNKNOWN` for everything it cannot raise. Neither
substitutes a near-enough type, because a silently approximated dtype is the
bug that would destroy trust in the whole tool.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
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
from rowless.ir.schema import Field, Schema

from rowless_pyspark import spark as cluster

#: Time units Spark's `TIME` type can hold, as the precision it spells them
#: with. There is no nanosecond `TIME`, so `time[ns]` is refused.
TIME_PRECISION: dict[TimeUnit, int] = {
    TimeUnit.SECOND: 0,
    TimeUnit.MILLISECOND: 3,
    TimeUnit.MICROSECOND: 6,
}

#: Signed integer widths Spark has, by the type that holds each one.
INTEGER_TYPES: dict[int, str] = {8: "ByteType", 16: "ShortType", 32: "IntegerType", 64: "LongType"}

#: Spark's largest decimal. A `decimal(38,9)` fits; anything wider does not.
MAX_DECIMAL_PRECISION: int = 38


def to_ir(native: Any) -> DType:
    """Convert a Spark DataType to the IR, returning UNKNOWN when there is no mapping.

    Accepts both spellings PySpark uses: the class (`StringType`) and the
    instance (`DecimalType(18, 2)`).
    """
    dtype = _as_instance(native)
    if dtype is None:
        qualified = f"{type(native).__module__}.{type(native).__qualname__}"
        return unknown(f"{qualified} is not a Spark DataType")

    name = type(dtype).__name__
    simple = _SIMPLE_TO_IR.get(name)
    if simple is not None:
        return simple()
    match name:
        case "DecimalType":
            return decimal(int(dtype.precision), int(dtype.scale))
        case "TimestampType":
            # The zone is the session's, not the type's. See the module
            # docstring: reporting UTC for a session in another zone would be a
            # dtype rowless invented.
            return datetime(TimeUnit.MICROSECOND, cluster.session_timezone())
        case "TimeType":
            return _time_to_ir(dtype)
        case "DayTimeIntervalType":
            return duration(TimeUnit.MICROSECOND)
        case "ArrayType":
            return list_(to_ir(dtype.elementType))
        case "MapType":
            return map_(to_ir(dtype.keyType), to_ir(dtype.valueType))
        case "StructType":
            return struct({str(f.name): to_ir(f.dataType) for f in dtype.fields})
        case _:
            return unknown(f"no IR dtype for Spark {_spelled(dtype)}")


def from_ir(dtype: DType) -> Any:
    """Convert an IR dtype to Spark, raising `UnsupportedDType` when there is none."""
    import pyspark.sql.types as t

    match dtype.kind:
        case Kind.BOOL:
            return t.BooleanType()
        case Kind.INT:
            return _integer(dtype)
        case Kind.UINT:
            raise UnsupportedDType(
                f"Spark has no unsigned integer type, so there is nothing to lower {dtype} to; "
                "a signed type of the same width would lose the top half of its range"
            )
        case Kind.FLOAT:
            return _float(dtype)
        case Kind.DECIMAL:
            return _decimal(dtype)
        case Kind.STRING:
            return t.StringType()
        case Kind.BINARY:
            return t.BinaryType()
        case Kind.DATE:
            return t.DateType()
        case Kind.TIME:
            return _time(dtype)
        case Kind.DATETIME:
            return _timestamp(dtype)
        case Kind.DURATION:
            return _duration(dtype)
        case Kind.LIST:
            return t.ArrayType(from_ir(_element(dtype)), containsNull=True)
        case Kind.MAP:
            key, value = _pair(dtype)
            return t.MapType(from_ir(key), from_ir(value), valueContainsNull=True)
        case Kind.STRUCT:
            return struct_type(
                tuple(
                    Field(name=name, dtype=inner)
                    for name, inner in zip(dtype.names, dtype.inner, strict=True)
                )
            )
        case Kind.ENUM:
            raise UnsupportedDType(
                f"Spark has no enum type, so {dtype} has nothing to lower to; a plain string "
                "would lose the closed variant set the enum exists to declare"
            )
        case Kind.NULL:
            return t.NullType()
        case Kind.ANY:
            # The gradual-typing top type. `NullType` is the one Spark type that
            # coerces into anything else, which is as close as the engine gets,
            # and an ANY column is already counted against coverage.
            return t.NullType()
        case _:
            detail = f" ({dtype.reason})" if dtype.reason else ""
            raise UnsupportedDType(f"Spark has no dtype for {dtype}{detail}")


def struct_type(fields: Sequence[Field]) -> Any:
    """A `StructType` for IR fields, carrying nullability.

    Spark tracks nullability per field and propagates it through analysis, so
    an input column declared non-nullable is still non-nullable after a
    projection. It is one of the two engines rowless targets where a contract
    that declares a column `NOT NULL` can actually be checked.
    """
    import pyspark.sql.types as t

    return t.StructType(
        [
            t.StructField(field.name, from_ir(field.dtype), nullable=field.nullable)
            for field in fields
        ]
    )


def to_schema(native: Any, *, name: str | None = None) -> Schema:
    """Read a Spark `StructType` as an IR schema, nullability included."""
    return Schema(
        fields=tuple(
            Field(
                name=str(field.name),
                dtype=to_ir(field.dataType),
                nullable=bool(field.nullable),
            )
            for field in getattr(native, "fields", ())
        ),
        name=name,
    )


def unsupported_dtypes() -> tuple[DType, ...]:
    """Corpus dtypes Spark has no equivalent for, so the suite checks the refusal.

    Environment-dependent, the way the pandas adapter's absences depend on
    pyarrow: `TIME` arrived in Spark 4.1, and declaring `time[us]` present on a
    PySpark that has no `TimeType` would be as dishonest as approximating it as
    a string on one that does.
    """
    absent = [
        uint(8),
        uint(32),
        # Microseconds, always: a millisecond timestamp would change resolution
        # and a nanosecond one would lose precision.
        datetime(TimeUnit.MILLISECOND),
        datetime(TimeUnit.NANOSECOND, "Europe/London"),
        _ENUM_CORPUS,
    ]
    if not time_type_available():
        absent.append(time(TimeUnit.MICROSECOND))
    return tuple(absent)


def time_type_available() -> bool:
    """True when this PySpark has the `TIME` type, which arrived in Spark 4.1."""
    try:
        import pyspark.sql.types as t
    except Exception:  # pragma: no cover - only without pyspark
        return False
    return hasattr(t, "TimeType")


# --- Helpers -----------------------------------------------------------------


def _as_instance(native: Any) -> Any:
    """Return a Spark DataType instance for either spelling, or None for a foreign object."""
    try:
        import pyspark.sql.types as t
    except Exception:  # pragma: no cover - only without pyspark
        return None
    if isinstance(native, t.DataType):
        return native
    if isinstance(native, type) and issubclass(native, t.DataType):
        try:
            return native()
        except Exception:
            # A parameterised class with no usable default. There is nothing to
            # report about the class itself.
            return None
    return None


def _spelled(dtype: Any) -> str:
    """Spark's own spelling of a type, for a diagnostic."""
    try:
        return str(dtype.simpleString())
    except Exception:  # pragma: no cover - a type that will not render
        return type(dtype).__name__


def _time_to_ir(dtype: Any) -> DType:
    """Convert a Spark `TimeType`, whose precision is the IR time unit."""
    precision = int(getattr(dtype, "precision", 6))
    for unit, spelled in TIME_PRECISION.items():
        if spelled == precision:
            return time(unit)
    # Spark allows 0 to 6; the ones the IR has no unit for are the odd
    # precisions (1, 2, 4, 5), which carry a resolution no IR unit describes.
    return unknown(f"Spark TIME({precision}) has no IR time unit; the IR has s, ms, us and ns")


def _integer(dtype: DType) -> Any:
    """Return `ByteType`, `ShortType`, `IntegerType` or `LongType`, by width."""
    import pyspark.sql.types as t

    bits = dtype.bits or 64
    name = INTEGER_TYPES.get(bits)
    if name is None:
        raise UnsupportedDType(
            f"Spark has no {bits}-bit integer; it has 8, 16, 32 and 64. A 128-bit integer would "
            "have to become a decimal, which is a different type with different arithmetic"
        )
    return getattr(t, name)()


def _float(dtype: DType) -> Any:
    """Return `FloatType` or `DoubleType`; Spark has no half-precision float."""
    import pyspark.sql.types as t

    bits = dtype.bits or 64
    if bits == 32:
        return t.FloatType()
    if bits == 64:
        return t.DoubleType()
    raise UnsupportedDType(f"Spark has no {bits}-bit float; it has FloatType and DoubleType")


def _decimal(dtype: DType) -> Any:
    """Return a `DecimalType`, refusing one wider than Spark can store."""
    import pyspark.sql.types as t

    if dtype.precision is None or dtype.scale is None:
        raise UnsupportedDType(f"Spark needs both a precision and a scale for {dtype}")
    if dtype.precision > MAX_DECIMAL_PRECISION:
        raise UnsupportedDType(
            f"Spark decimals hold at most {MAX_DECIMAL_PRECISION} digits, and {dtype} needs "
            f"{dtype.precision}"
        )
    return t.DecimalType(dtype.precision, dtype.scale)


def _time(dtype: DType) -> Any:
    """Return a `TimeType` of the right precision, refusing what Spark lacks."""
    import pyspark.sql.types as t

    unit = dtype.unit or TimeUnit.MICROSECOND
    precision = TIME_PRECISION.get(unit)
    if precision is None:
        raise UnsupportedDType(
            f"Spark TIME holds at most microseconds, so there is nothing to lower {dtype} to"
        )
    found = getattr(t, "TimeType", None)
    if found is None:
        raise UnsupportedDType(
            f"this PySpark has no TIME type, so {dtype} has nothing to lower to; TIME arrived "
            "in Spark 4.1"
        )
    return found(precision)


def _timestamp(dtype: DType) -> Any:
    """Return `TimestampType` or `TimestampNTZType`, refusing a zone or unit Spark lacks."""
    import pyspark.sql.types as t

    unit = dtype.unit or TimeUnit.MICROSECOND
    if unit is not TimeUnit.MICROSECOND:
        raise UnsupportedDType(
            f"Spark timestamps are microseconds, so lowering {dtype} would change its "
            f"resolution from {unit.value} to us"
        )
    if dtype.tz is None:
        return t.TimestampNTZType()
    zone = cluster.session_timezone()
    if dtype.tz != zone:
        raise UnsupportedDType(
            f"Spark TIMESTAMP carries no zone of its own: it is an instant rendered in the "
            f"session timezone, which is {zone!r} here, so {dtype} cannot be lowered without "
            "shifting every value. Set spark.sql.session.timeZone to match, or declare the "
            "column naive"
        )
    return t.TimestampType()


def _duration(dtype: DType) -> Any:
    """Return a day-time interval, refusing a unit it cannot hold."""
    import pyspark.sql.types as t

    unit = dtype.unit or TimeUnit.MICROSECOND
    if unit is not TimeUnit.MICROSECOND:
        raise UnsupportedDType(
            f"a Spark day-time interval stores microseconds, so lowering {dtype} would change "
            f"its resolution from {unit.value} to us"
        )
    return t.DayTimeIntervalType(t.DayTimeIntervalType.DAY, t.DayTimeIntervalType.SECOND)


def _element(dtype: DType) -> DType:
    """The element type of a LIST, refusing one the adapter was not given."""
    if not dtype.inner:
        raise UnsupportedDType("a Spark ArrayType needs an element dtype, and none was reported")
    return dtype.inner[0]


def _pair(dtype: DType) -> tuple[DType, DType]:
    """The key and value types of a MAP, refusing a half-specified one."""
    if len(dtype.inner) != 2:
        raise UnsupportedDType("a map needs both a key and a value dtype")
    return dtype.inner[0], dtype.inner[1]


#: The enum the conformance corpus asks about, spelled once so the declared
#: absence and the test that checks it cannot drift apart.
_ENUM_CORPUS: DType = DType(Kind.ENUM, names=("a", "b", "c"))

#: Spark types with a parameter-free IR equivalent, keyed by class name so the
#: table can be built without importing pyspark at module scope.
_SIMPLE_TO_IR: dict[str, Callable[[], DType]] = {
    "BooleanType": boolean,
    "ByteType": lambda: int_(8),
    "ShortType": lambda: int_(16),
    "IntegerType": lambda: int_(32),
    "LongType": lambda: int_(64),
    "FloatType": lambda: float_(32),
    "DoubleType": lambda: float_(64),
    "StringType": string,
    # The length limit is real to Spark and invisible to the IR. See the module
    # docstring: a contract that passes here may fail on an engine that models
    # the width.
    "CharType": string,
    "VarcharType": string,
    "BinaryType": binary,
    "DateType": date,
    "TimestampNTZType": lambda: datetime(TimeUnit.MICROSECOND),
    "NullType": null,
    "YearMonthIntervalType": lambda: unknown(
        "a Spark year-month interval is a count of months, and a month is not a fixed number of "
        "microseconds, so the IR has no duration that describes it"
    ),
    "CalendarIntervalType": lambda: unknown(
        "a Spark calendar interval mixes months, days and microseconds, which the IR has no "
        "single duration for"
    ),
    "VariantType": lambda: unknown("a Spark Variant holds arbitrary semi-structured values"),
    "GeometryType": lambda: unknown("Spark geometry has no IR dtype"),
    "GeographyType": lambda: unknown("Spark geography has no IR dtype"),
}
