"""Dtype translation between Polars and the rowless IR.

Polars is the closest of the engines to the IR, so most of this file is a
table. The interesting part is the handful of places where the two vocabularies
genuinely disagree, and every one of them is written down here rather than
smoothed over:

* **`Time` has no unit.** Polars stores time-of-day as nanoseconds since
  midnight and does not parameterise the type, so `to_ir` reports the IR
  default (`time[us]`) and `from_ir` accepts any unit. A `time[ns]` contract
  still accepts the result, since the IR treats a finer declared slot as able
  to hold a coarser value.
* **There is no `Map`.** Arrow encodes a map as `list<struct<key, value>>` and
  so does Polars, so that is what `from_ir` emits and what `to_ir` reads back
  as a map. The cost is that a genuine `list[struct{key, value}]` column is
  reported as a map; the field names are the only signal either library has.
* **`Categorical` is a string.** Polars dictionary-encodes it, but the values
  are strings and the variant set is open, so the IR dtype is `string`. Only
  `Enum`, which has a closed variant set, maps to `enum`.
* **`Array` loses its width.** A fixed-size list becomes `list`, because the IR
  has no fixed-size variant and the element type is the part that breaks
  pipelines.
* **Second resolution does not exist.** Polars accepts `ms`, `us` and `ns`
  only, so `datetime[s]` and `duration[s]` are refused rather than rounded.
* **`Object` and `Unknown` are refused, not guessed.** Both become
  `Kind.UNKNOWN` with a reason, which poisons comparisons instead of passing
  them.

`from_ir` raises `UnsupportedDType` for everything it cannot lower, including
`UNKNOWN`. `to_ir` returns `UNKNOWN` for everything it cannot raise. Neither
substitutes a near-enough type, because a silently approximated dtype is the
bug that would destroy trust in the whole tool.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, cast

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

#: Field names of the `struct` a map is encoded as. Arrow uses this convention,
#: and Polars reads an Arrow map into exactly this shape.
MAP_FIELDS: tuple[str, str] = ("key", "value")

#: Time units Polars has. `s` is missing, which is why `from_ir` refuses it.
POLARS_TIME_UNITS: frozenset[TimeUnit] = frozenset(
    {TimeUnit.MILLISECOND, TimeUnit.MICROSECOND, TimeUnit.NANOSECOND}
)


def to_ir(native: Any) -> DType:
    """Convert a Polars dtype to the IR, returning UNKNOWN when there is no mapping.

    Accepts both spellings Polars uses: the class (`pl.Int64`) and the instance
    (`pl.Datetime("us", "UTC")`).
    """
    import polars as pl

    dtype = _as_instance(native)
    if dtype is None:
        qualified = f"{type(native).__module__}.{type(native).__qualname__}"
        return unknown(f"{qualified} is not a Polars dtype")

    base = dtype.base_type()
    simple = _SIMPLE_TO_IR.get(base.__name__)
    if simple is not None:
        return simple()

    if base is pl.Decimal:
        return _decimal_to_ir(dtype)
    if base is pl.Datetime:
        return _datetime_to_ir(dtype)
    if base is pl.Duration:
        return duration(_unit_to_ir(getattr(dtype, "time_unit", None)))
    if base is pl.Enum:
        return enum(tuple(str(variant) for variant in getattr(dtype, "categories", ())))
    if base is pl.Struct:
        return _struct_to_ir(dtype)
    if base in (pl.List, pl.Array):
        return _list_to_ir(dtype)
    return unknown(f"no IR dtype for Polars {dtype}")


def from_ir(dtype: DType) -> Any:
    """Convert an IR dtype to Polars, raising `UnsupportedDType` when there is none."""
    import polars as pl

    match dtype.kind:
        case Kind.BOOL:
            return pl.Boolean
        case Kind.INT:
            return _integer(dtype, "Int")
        case Kind.UINT:
            return _integer(dtype, "UInt")
        case Kind.FLOAT:
            return _float(dtype)
        case Kind.DECIMAL:
            if dtype.precision is None or dtype.scale is None:
                raise UnsupportedDType(f"Polars needs both a precision and a scale for {dtype}")
            return pl.Decimal(dtype.precision, dtype.scale)
        case Kind.STRING:
            return pl.String
        case Kind.BINARY:
            return pl.Binary
        case Kind.DATE:
            return pl.Date
        case Kind.TIME:
            # Polars has one Time type, at nanosecond storage, so every IR unit
            # lowers to it. The quirk is in the module docstring.
            return pl.Time
        case Kind.DATETIME:
            return pl.Datetime(_polars_unit(dtype), dtype.tz)
        case Kind.DURATION:
            return pl.Duration(_polars_unit(dtype))
        case Kind.LIST:
            return pl.List(from_ir(_element(dtype)))
        case Kind.STRUCT:
            return pl.Struct(
                {name: from_ir(inner) for name, inner in zip(dtype.names, dtype.inner, strict=True)}
            )
        case Kind.MAP:
            key, value = _pair(dtype)
            return pl.List(pl.Struct({MAP_FIELDS[0]: from_ir(key), MAP_FIELDS[1]: from_ir(value)}))
        case Kind.ENUM:
            return pl.Enum(list(dtype.names))
        case Kind.NULL:
            return pl.Null
        case Kind.ANY:
            # The gradual-typing top type. `pl.Null` is the one dtype Polars
            # coerces into anything else, which is as close as the engine gets,
            # and an ANY column is already counted against coverage.
            return pl.Null
        case _:
            detail = f" ({dtype.reason})" if dtype.reason else ""
            raise UnsupportedDType(f"Polars has no dtype for {dtype}{detail}")


def _as_instance(native: Any) -> Any:
    """Return a Polars dtype instance for either spelling, or None for a foreign object."""
    import polars as pl

    if isinstance(native, pl.DataType):
        return native
    if isinstance(native, type) and issubclass(native, pl.DataType):
        try:
            return native()
        except Exception:
            # A parameterised class with no defaults (`pl.Enum`) cannot be
            # instantiated bare, and there is nothing to report about it.
            return None
    return None


def _decimal_to_ir(dtype: Any) -> DType:
    """Convert a Polars `Decimal`, refusing the unbounded-precision form."""
    precision = getattr(dtype, "precision", None)
    scale = getattr(dtype, "scale", None) or 0
    if precision is None:
        return unknown("Polars Decimal with no precision carries no width to compare")
    return decimal(int(precision), int(scale))


def _datetime_to_ir(dtype: Any) -> DType:
    """Convert a Polars `Datetime`, keeping the unit and the timezone apart."""
    zone = getattr(dtype, "time_zone", None)
    return datetime(_unit_to_ir(getattr(dtype, "time_unit", None)), str(zone) if zone else None)


def _struct_to_ir(dtype: Any) -> DType:
    """Convert a Polars `Struct`, field by field and in order."""
    fields = getattr(dtype, "fields", ())
    return struct({str(field.name): to_ir(field.dtype) for field in fields})


def _list_to_ir(dtype: Any) -> DType:
    """Convert a Polars `List` or `Array`, recognising the Arrow map encoding."""
    inner = to_ir(getattr(dtype, "inner", None))
    if inner.kind is Kind.STRUCT and inner.names == MAP_FIELDS:
        return map_(inner.inner[0], inner.inner[1])
    return list_(inner)


def _unit_to_ir(unit: str | None) -> TimeUnit:
    """Read a Polars time unit string as an IR `TimeUnit`, defaulting to microseconds."""
    try:
        return TimeUnit(unit) if unit else TimeUnit.MICROSECOND
    except ValueError:
        return TimeUnit.MICROSECOND


def _polars_unit(dtype: DType) -> Literal["ms", "us", "ns"]:
    """Return the Polars time unit for a temporal IR dtype, refusing second resolution."""
    unit = dtype.unit or TimeUnit.MICROSECOND
    if unit not in POLARS_TIME_UNITS:
        raise UnsupportedDType(
            f"Polars has no {unit.value}-resolution {dtype.kind.value}; it supports ms, us and ns"
        )
    # Narrowed by the membership check above: `POLARS_TIME_UNITS` is exactly
    # the three spellings Polars accepts.
    return cast('Literal["ms", "us", "ns"]', unit.value)


def _integer(dtype: DType, prefix: str) -> Any:
    """Return `pl.Int32`, `pl.UInt8` and friends, by width."""
    import polars as pl

    bits = dtype.bits or 64
    found = getattr(pl, f"{prefix}{bits}", None)
    if found is None:
        raise UnsupportedDType(
            f"this Polars version has no {prefix}{bits}; 128-bit integers need polars>=1.18"
        )
    return found


def _float(dtype: DType) -> Any:
    """Return `pl.Float32` or `pl.Float64`; Polars has no half-precision float."""
    import polars as pl

    bits = dtype.bits or 64
    if bits == 32:
        return pl.Float32
    if bits == 64:
        return pl.Float64
    raise UnsupportedDType(f"Polars has no {bits}-bit float; it has Float32 and Float64")


def _element(dtype: DType) -> DType:
    """Return the element type of a LIST, refusing one the adapter was not given."""
    if not dtype.inner:
        raise UnsupportedDType("a Polars List needs an element dtype, and none was reported")
    return dtype.inner[0]


def _pair(dtype: DType) -> tuple[DType, DType]:
    """Return the key and value types of a MAP, refusing a half-specified one."""
    if len(dtype.inner) != 2:
        raise UnsupportedDType("a map needs both a key and a value dtype")
    return dtype.inner[0], dtype.inner[1]


#: Polars dtypes with a parameter-free IR equivalent, keyed by class name so
#: that this table can be built without importing Polars at module scope.
_SIMPLE_TO_IR: dict[str, Callable[[], DType]] = {
    "Boolean": boolean,
    "Int8": lambda: int_(8),
    "Int16": lambda: int_(16),
    "Int32": lambda: int_(32),
    "Int64": lambda: int_(64),
    "Int128": lambda: int_(128),
    "UInt8": lambda: uint(8),
    "UInt16": lambda: uint(16),
    "UInt32": lambda: uint(32),
    "UInt64": lambda: uint(64),
    "UInt128": lambda: uint(128),
    "Float32": lambda: float_(32),
    "Float64": lambda: float_(64),
    "String": string,
    "Binary": binary,
    "Date": date,
    # Unit-less in Polars; see the module docstring.
    "Time": lambda: time(TimeUnit.MICROSECOND),
    "Null": null,
    # Dictionary-encoded strings with an open variant set. `Enum` is the closed
    # one, and the only one that maps to the IR ENUM.
    "Categorical": string,
    "Object": lambda: unknown("Polars Object holds arbitrary Python values"),
    "Unknown": lambda: unknown("Polars could not resolve this column dtype"),
}
