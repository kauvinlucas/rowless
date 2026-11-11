"""Dtype translation between Ibis and the rowless IR.

Ibis is a type system with a query builder attached, so most of this file is a
table. What is worth reading is the handful of places where the two
vocabularies genuinely disagree, and every one of them is written down here
rather than smoothed over:

* **There is no enum.** `ibis.expr.datatypes.Enum` is an *alias for* `String`:
  it carries no variant set, so an IR `enum{a,b,c}` lowered to it would read
  back as a plain string and be reported downstream as a dtype change nobody
  made. It is refused instead - see `UNSUPPORTED`, which the conformance suite
  reads through the adapter.
* **`Time` has no unit.** Ibis stores time-of-day in one unparameterised
  `Time`, so `to_ir` reports the IR default (`time[us]`) and `from_ir` accepts
  that unit and no other. The alternative - lowering `time[ms]` to `Time` -
  changes the resolution of a column silently, which is the one failure mode
  this project exists to prevent.
* **A timestamp carries a scale, not a unit.** Ibis counts fractional digits:
  0 is seconds, 3 milliseconds, 6 microseconds, 9 nanoseconds, and those are
  the four the IR can name. A scale ibis accepts and the IR cannot spell
  (`timestamp(4)`) reads back as UNKNOWN rather than being rounded to a
  neighbour. `scale=None` means "whatever the backend does", and it is read as
  microseconds, which is what the backends ibis targets overwhelmingly do.
* **A duration is an interval, and an interval is not always a duration.**
  Ibis intervals include calendar units - `Y`, `Q`, `M`, `W`, `D`, `h`, `m` -
  and a month is not a fixed length of time. Only the four fixed units map to
  the IR `duration`; the calendar ones read back as UNKNOWN.
* **There are no 128-bit integers.** `int128` and `uint128` are refused rather
  than narrowed, which would overflow in production and nowhere else.
* **`JSON`, `UUID`, `INET`, `MACADDR` and the geospatial types have no IR
  equivalent.** They read back as UNKNOWN with a reason. Mapping them to
  `string` would make a contract comparing them to a real string column pass.
* **Nullability lives on the dtype.** Ibis is the only engine rowless targets
  that tracks `!int64` through a projection, a filter and a union, so the
  adapter reads nullability off the dtype rather than reporting every column
  nullable. The IR keeps nullability on the field, so the two are joined up in
  `materialize` with `with_nullability` and read back in `describe`.
* **`Float16` exists**, unlike in Polars, so `float16` round trips here.

`from_ir` raises `UnsupportedDType` for everything it cannot lower, including
`UNKNOWN`. `to_ir` returns `UNKNOWN` for everything it cannot raise. Neither
substitutes a near-enough type, because a silently approximated dtype is the
bug that would destroy trust in the whole tool.
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

#: Fractional digits ibis counts, per IR time unit. `_UNIT_BY_SCALE` is the
#: inverse, derived rather than written out, so a unit cannot be added to one
#: direction only.
SCALE_BY_UNIT: dict[TimeUnit, int] = {
    TimeUnit.SECOND: 0,
    TimeUnit.MILLISECOND: 3,
    TimeUnit.MICROSECOND: 6,
    TimeUnit.NANOSECOND: 9,
}

_UNIT_BY_SCALE: dict[int, TimeUnit] = {scale: unit for unit, scale in SCALE_BY_UNIT.items()}

#: Interval units that are a fixed length of time. The rest of what ibis
#: accepts (`Y`, `Q`, `M`, `W`, `D`, `h`, `m`) is calendar arithmetic, which
#: the IR `duration` cannot describe.
FIXED_INTERVAL_UNITS: dict[str, TimeUnit] = {
    "s": TimeUnit.SECOND,
    "ms": TimeUnit.MILLISECOND,
    "us": TimeUnit.MICROSECOND,
    "ns": TimeUnit.NANOSECOND,
}

#: Integer widths ibis has. 128-bit integers are not among them.
INTEGER_BITS: frozenset[int] = frozenset({8, 16, 32, 64})

#: Float widths ibis has, which includes half precision.
FLOAT_BITS: frozenset[int] = frozenset({16, 32, 64})

#: Ibis types that exist, are perfectly usable in a pipeline, and have no IR
#: equivalent at all. Mapped to UNKNOWN with the reason, keyed by the ibis
#: class name so the table can be built without importing ibis at module scope.
_NO_IR_EQUIVALENT: dict[str, str] = {
    "JSON": "an ibis JSON column holds arbitrary documents, which the IR has no dtype for",
    "UUID": (
        "the IR has no UUID dtype, and reporting it as a string would make a contract "
        "comparing the two pass"
    ),
    "INET": "the IR has no network-address dtype",
    "MACADDR": "the IR has no MAC-address dtype",
}

#: Dtypes from the shared conformance corpus ibis genuinely does not have,
#: declared so the suite checks that `from_ir` refuses them instead of
#: expecting a round trip that cannot happen. Read by
#: `rowless.testing.conformance.absent_dtypes` through the adapter.
UNSUPPORTED: tuple[DType, ...] = (enum(("a", "b", "c")),)


def to_ir(native: Any) -> DType:
    """Convert an ibis dtype to the IR, returning UNKNOWN when there is no mapping.

    Accepts both spellings rowless encounters: a `DataType` instance, which is
    what a schema reports, and the string shorthand (`"int64"`,
    `"array<string>"`) a pipeline may have declared its table with.
    """
    dtype = _as_datatype(native)
    if dtype is None:
        qualified = f"{type(native).__module__}.{type(native).__qualname__}"
        return unknown(f"{qualified} is not an ibis dtype")

    name = type(dtype).__name__
    simple = _SIMPLE_TO_IR.get(name)
    if simple is not None:
        return simple()

    refused = _NO_IR_EQUIVALENT.get(name)
    if refused is not None:
        return unknown(refused)

    import ibis.expr.datatypes as dt

    if isinstance(dtype, dt.Decimal):
        return _decimal_to_ir(dtype)
    if isinstance(dtype, dt.Timestamp):
        return _timestamp_to_ir(dtype)
    if isinstance(dtype, dt.Interval):
        return _interval_to_ir(dtype)
    if isinstance(dtype, dt.Array):
        return list_(to_ir(dtype.value_type))
    if isinstance(dtype, dt.Map):
        return map_(to_ir(dtype.key_type), to_ir(dtype.value_type))
    if isinstance(dtype, dt.Struct):
        return struct({str(field): to_ir(inner) for field, inner in dtype.fields.items()})
    if isinstance(dtype, dt.GeoSpatial):
        # The geospatial hierarchy is deep enough that a subclass rowless has
        # not seen should still be refused rather than guessed at.
        return unknown("the IR has no geospatial dtypes")
    return unknown(f"no IR dtype for ibis {dtype}")


def from_ir(dtype: DType) -> Any:
    """Convert an IR dtype to ibis, raising `UnsupportedDType` when there is none."""
    import ibis.expr.datatypes as dt

    match dtype.kind:
        case Kind.BOOL:
            return dt.Boolean()
        case Kind.INT:
            return _integer(dtype, "Int")
        case Kind.UINT:
            return _integer(dtype, "UInt")
        case Kind.FLOAT:
            return _float(dtype)
        case Kind.DECIMAL:
            if dtype.precision is None or dtype.scale is None:
                raise UnsupportedDType(f"ibis needs both a precision and a scale for {dtype}")
            return dt.Decimal(dtype.precision, dtype.scale)
        case Kind.STRING:
            return dt.String()
        case Kind.BINARY:
            return dt.Binary()
        case Kind.DATE:
            return dt.Date()
        case Kind.TIME:
            return _time(dtype)
        case Kind.DATETIME:
            return dt.Timestamp(timezone=dtype.tz, scale=_scale(dtype))
        case Kind.DURATION:
            return dt.Interval(unit=_interval_unit(dtype))
        case Kind.LIST:
            return dt.Array(from_ir(_element(dtype)))
        case Kind.STRUCT:
            return dt.Struct(
                {name: from_ir(inner) for name, inner in zip(dtype.names, dtype.inner, strict=True)}
            )
        case Kind.MAP:
            key, value = _pair(dtype)
            return dt.Map(from_ir(key), from_ir(value))
        case Kind.ENUM:
            raise UnsupportedDType(
                f"ibis has no enum type for {dtype}: ibis.expr.datatypes.Enum is an alias for "
                "String and carries no variant set, so the variants would be lost. Declare the "
                "column as a string where the closed set is not something you need checked"
            )
        case Kind.NULL:
            return dt.Null()
        case Kind.ANY:
            # The gradual-typing top type. `Null` is the one ibis dtype that
            # casts into every other, which is as close as the engine gets, and
            # an ANY column is already counted against coverage.
            return dt.Null()
        case _:
            detail = f" ({dtype.reason})" if dtype.reason else ""
            raise UnsupportedDType(f"ibis has no dtype for {dtype}{detail}")


def with_nullability(native: Any, *, nullable: bool) -> Any:
    """Return `native` with its nullability set, since ibis keeps it on the dtype.

    The IR keeps nullability on the field, so the two have to be joined up
    somewhere; `materialize` does it here, and `describe` reads it back off the
    dtype.
    """
    try:
        return native.copy(nullable=nullable)
    except Exception:
        # A dtype that refuses to be copied is still a usable dtype; the column
        # is simply reported nullable, which is what every other adapter does.
        return native


def is_nullable(native: Any) -> bool:
    """Whether an ibis dtype admits nulls, defaulting to True when it will not say."""
    return bool(getattr(native, "nullable", True))


def _as_datatype(native: Any) -> Any:
    """Return an ibis `DataType` for either spelling, or None for a foreign object."""
    import ibis
    import ibis.expr.datatypes as dt

    if isinstance(native, dt.DataType):
        return native
    if isinstance(native, str):
        try:
            return ibis.dtype(native)
        except Exception:
            return None
    return None


def _decimal_to_ir(dtype: Any) -> DType:
    """Convert an ibis `Decimal`, refusing the unbounded-precision form."""
    precision = dtype.precision
    if precision is None:
        return unknown("an ibis Decimal with no precision carries no width to compare")
    return decimal(int(precision), int(dtype.scale or 0))


def _timestamp_to_ir(dtype: Any) -> DType:
    """Convert an ibis `Timestamp`, reading its scale as an IR time unit."""
    zone = dtype.timezone
    scale = dtype.scale
    if scale is None:
        # "The backend decides", and every backend ibis targets decides on
        # microseconds. Documented in the module docstring, because a contract
        # that declares `datetime[ns]` will notice.
        return datetime(TimeUnit.MICROSECOND, str(zone) if zone else None)
    unit = _UNIT_BY_SCALE.get(int(scale))
    if unit is None:
        return unknown(
            f"ibis timestamp scale {scale} has no IR time unit; the IR has seconds, "
            "milliseconds, microseconds and nanoseconds"
        )
    return datetime(unit, str(zone) if zone else None)


def _interval_to_ir(dtype: Any) -> DType:
    """Convert an ibis `Interval`, refusing the calendar units."""
    unit = getattr(dtype.unit, "value", dtype.unit)
    fixed = FIXED_INTERVAL_UNITS.get(str(unit))
    if fixed is None:
        return unknown(
            f"ibis interval unit {unit!r} is calendar arithmetic rather than a fixed length of "
            "time, which the IR duration cannot describe"
        )
    return duration(fixed)


def _integer(dtype: DType, prefix: str) -> Any:
    """Return `dt.Int32`, `dt.UInt8` and friends, by width."""
    import ibis.expr.datatypes as dt

    bits = dtype.bits or 64
    if bits not in INTEGER_BITS:
        raise UnsupportedDType(
            f"ibis has no {bits}-bit integer; it has 8, 16, 32 and 64. Narrowing a {bits}-bit "
            "column would overflow in production and nowhere else"
        )
    return getattr(dt, f"{prefix}{bits}")()


def _float(dtype: DType) -> Any:
    """Return `dt.Float16`, `dt.Float32` or `dt.Float64`, by width."""
    import ibis.expr.datatypes as dt

    bits = dtype.bits or 64
    if bits not in FLOAT_BITS:
        raise UnsupportedDType(f"ibis has no {bits}-bit float; it has 16, 32 and 64")
    return getattr(dt, f"Float{bits}")()


def _time(dtype: DType) -> Any:
    """Return the one ibis `Time`, refusing every resolution but the IR default.

    Ibis does not parameterise `Time`, so `to_ir` can only ever report
    `time[us]`. Lowering `time[ms]` here would therefore read back as a dtype
    change the user never made, which is worse than refusing it.
    """
    import ibis.expr.datatypes as dt

    unit = dtype.unit or TimeUnit.MICROSECOND
    if unit is not TimeUnit.MICROSECOND:
        raise UnsupportedDType(
            f"ibis has one unparameterised Time, so {dtype} cannot round trip: it would read "
            "back as time[us] and be reported as a change nobody made"
        )
    return dt.Time()


def _scale(dtype: DType) -> int:
    """Fractional digits for a DATETIME, refusing a unit ibis cannot count."""
    unit = dtype.unit or TimeUnit.MICROSECOND
    scale = SCALE_BY_UNIT.get(unit)
    if scale is None:  # pragma: no cover - every IR unit is in the table
        raise UnsupportedDType(f"ibis has no scale for {unit.value}-resolution timestamps")
    return scale


def _interval_unit(dtype: DType) -> str:
    """The ibis interval unit for a DURATION."""
    unit = dtype.unit or TimeUnit.MICROSECOND
    if unit.value not in FIXED_INTERVAL_UNITS:  # pragma: no cover - all four are fixed
        raise UnsupportedDType(f"ibis has no {unit.value}-resolution interval")
    return unit.value


def _element(dtype: DType) -> DType:
    """Return the element type of a LIST, refusing one the adapter was not given."""
    if not dtype.inner:
        raise UnsupportedDType("an ibis Array needs an element dtype, and none was reported")
    return dtype.inner[0]


def _pair(dtype: DType) -> tuple[DType, DType]:
    """Return the key and value types of a MAP, refusing a half-specified one."""
    if len(dtype.inner) != 2:
        raise UnsupportedDType("a map needs both a key and a value dtype")
    return dtype.inner[0], dtype.inner[1]


#: Ibis dtypes with a parameter-free IR equivalent, keyed by class name so that
#: this table can be built without importing ibis at module scope. `Enum` is
#: absent on purpose: it *is* `String` in ibis, so a column declared with it
#: arrives here as `String` already.
_SIMPLE_TO_IR: dict[str, Callable[[], DType]] = {
    "Boolean": boolean,
    "Int8": lambda: int_(8),
    "Int16": lambda: int_(16),
    "Int32": lambda: int_(32),
    "Int64": lambda: int_(64),
    "UInt8": lambda: uint(8),
    "UInt16": lambda: uint(16),
    "UInt32": lambda: uint(32),
    "UInt64": lambda: uint(64),
    "Float16": lambda: float_(16),
    "Float32": lambda: float_(32),
    "Float64": lambda: float_(64),
    "String": string,
    "Binary": binary,
    "Date": date,
    # Unit-less in ibis; see the module docstring.
    "Time": lambda: time(TimeUnit.MICROSECOND),
    "Null": null,
    "Unknown": lambda: unknown("ibis could not determine this column's dtype"),
}
