"""Dtype translation between pandas and the rowless IR.

pandas has three dtype vocabularies at once, and an adapter has to read all
three: the numpy dtypes it started with (`int64`, `datetime64[us]`), the
extension dtypes it grew (`Int64`, `boolean`, `string`, `category`,
`datetime64[us, UTC]`), and the pyarrow-backed ones it gained in 2.0
(`int64[pyarrow]`, `decimal128(18, 2)[pyarrow]`). `to_ir` accepts any of them,
including a bare pyarrow type and the string spelling pandas itself takes in
`astype`, because a pipeline can produce any of them.

`from_ir` has to pick *one*, and which one it picks is the most consequential
decision in this package. Two tables, selected by `backend()`:

* **arrow** - `pd.ArrowDtype`, used when pyarrow is importable. It is the
  complete table: decimals, dates, times, binary and the null type all exist,
  so nothing in the conformance corpus has to be refused.
* **numpy** - the masked extension dtypes (`Int64`, `boolean`, `Float64`,
  `string`) where pandas has one, plain numpy where it does not
  (`datetime64[us]`, `timedelta64[us]`, `float16`). It is an incomplete table,
  and `absent_dtypes()` says exactly what is missing.

The masked spelling is preferred over plain numpy for a reason that is this
adapter's whole subject: a masked column holds NA without changing dtype, so
`Int64` survives `where`, `mask` and `reindex` where `int64` silently becomes
`float64`. Choosing the dtype that cannot drift is cheaper, and more honest,
than correcting the drift afterwards.

Where the vocabularies genuinely disagree, written down rather than smoothed
over:

* **There is no decimal, date, time, binary or null dtype without pyarrow.**
  pandas stores all five as `object`, which is a Python-object column carrying
  no type at all. `from_ir` refuses them rather than lowering a decimal to
  `float64` or a date to `datetime64[us]`: both would round-trip back as
  something else and be reported downstream as a dtype change nobody made.
  With pyarrow installed all five map exactly.
* **`object` is refused, not guessed.** It becomes `Kind.UNKNOWN` with a
  reason. An object column may hold `Decimal`, `date`, `str`, lists or all four
  at once, and the honest answer is that pandas does not know either.
* **`category` is an enum when its variants are strings.** A `CategoricalDtype`
  carries a closed, declared variant set, which is exactly `Kind.ENUM` - unlike
  a Polars `Categorical`, whose variant set is open and which is therefore a
  plain string. A categorical over non-string categories is reported as the
  dtype of its categories: the IR has no numeric enum, the value type is the
  part that breaks pipelines, and so the closed set is what is lost - here,
  where it can be read, rather than silently downstream.
* **pandas keeps a named timezone.** `datetime64[ns, Europe/London]` is a real
  pandas dtype, so this adapter round-trips the one corpus dtype DuckDB has to
  refuse outright.
* **Second resolution exists.** `datetime64[s]` and `timedelta64[s]` are real
  pandas dtypes, which is the other way round from Polars, where both are
  refused.
* **A tz-aware dtype is read before the generic numpy path**, because
  `DatetimeTZDtype.numpy_dtype` answers with the *naive* dtype. Reading it
  through that path would drop the zone, which is the silent one-hour bug this
  project exists to catch.

`from_ir` raises `UnsupportedDType` for everything it cannot lower, including
`UNKNOWN`. `to_ir` returns `UNKNOWN` for everything it cannot raise. Neither
substitutes a near-enough type.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal

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

#: Which `from_ir` table is in use. `"arrow"` needs pyarrow importable.
Backend = Literal["arrow", "numpy"]

#: Field names of the struct a pyarrow map is encoded as. The same convention
#: `rowless_polars` uses, so a map survives a trip through either engine.
MAP_FIELDS: tuple[str, str] = ("key", "value")

#: Integer widths pandas has, masked and numpy alike. 128-bit integers are
#: numpy territory pandas does not expose, so they are refused.
PANDAS_INT_WIDTHS: frozenset[int] = frozenset({8, 16, 32, 64})

#: Time units pandas accepts on a datetime or a timedelta. All four, which is
#: one more than Polars has.
PANDAS_TIME_UNITS: frozenset[TimeUnit] = frozenset(
    {TimeUnit.SECOND, TimeUnit.MILLISECOND, TimeUnit.MICROSECOND, TimeUnit.NANOSECOND}
)

#: Spellings of the `arrow_dtypes` option that mean "do not use pyarrow".
_OFF: frozenset[str] = frozenset({"false", "no", "off", "0", "numpy"})

#: Spellings that mean "use pyarrow if it is there".
_ON: frozenset[str] = frozenset({"true", "yes", "on", "1", "arrow"})

#: Backend chosen by the adapter from the `arrow_dtypes` option. `None` means
#: decide per call from what is installed.
_BACKEND: Backend | None = None


def arrow_available() -> bool:
    """True when pyarrow is importable, which is what unlocks the full table."""
    try:
        import pyarrow  # noqa: F401
    except Exception:
        # A pyarrow that fails to import - a missing native wheel, say - is a
        # pyarrow that cannot carry a dtype, whatever the metadata claims.
        return False
    return True


def resolve_backend(option: object) -> Backend:
    """Read the `arrow_dtypes` option into the backend that will actually be used.

    `None` or `"auto"` picks arrow when pyarrow is installed. An explicit
    request for arrow that cannot be satisfied answers `"numpy"`: this function
    reports what will happen, and the adapter is the one that says so in a
    note. Answering `"arrow"` here would make the fallback invisible.
    """
    wanted = "auto" if option is None else str(option).strip().lower()
    if wanted in _OFF:
        return "numpy"
    if wanted in _ON:
        return "arrow" if arrow_available() else "numpy"
    return "arrow" if arrow_available() else "numpy"


def use_backend(chosen: Backend | None) -> None:
    """Pin the `from_ir` table for this run, or `None` to decide per call."""
    global _BACKEND
    _BACKEND = chosen


def backend() -> Backend:
    """The table `from_ir` will use right now."""
    return _BACKEND if _BACKEND is not None else resolve_backend(None)


def absent_dtypes(*, prefer: Backend | None = None) -> tuple[DType, ...]:
    """Corpus dtypes the installed pandas has no equivalent for.

    Empty on the arrow backend. On the numpy backend it is the five families
    pandas can only hold as `object`, which is not holding them at all.

    Read by `PandasAdapter.unsupported_dtypes`, which is a claim the
    conformance suite then holds the adapter to, so it describes the pandas
    that is installed rather than pandas in general. `prefer` answers for a
    table other than the configured one, which is how the suite checks both.
    """
    if (prefer or backend()) == "arrow":
        return ()
    return (
        decimal(18, 2),
        decimal(38, 9),
        binary(),
        date(),
        time(TimeUnit.MICROSECOND),
        null(),
    )


# --- IR out ------------------------------------------------------------------


def to_ir(native: Any) -> DType:
    """Convert a pandas, numpy or pyarrow dtype to the IR.

    Accepts every spelling a pipeline can hand over: a numpy dtype, a pandas
    extension dtype, a bare pyarrow type, and the strings pandas takes in
    `astype` (`"Int64"`, `"datetime64[us, UTC]"`). Returns `UNKNOWN`, with a
    reason, for anything it cannot map.
    """
    import numpy as np
    import pandas as pd

    if isinstance(native, DType):
        # Something already translated. Cheap to allow, and it keeps
        # `to_ir(to_ir(x))` from being a trap for a caller chaining helpers.
        return native
    arrow = _arrow_type(native)
    if arrow is not None:
        return from_arrow(arrow)
    dtype = _as_pandas_dtype(native)
    if dtype is None:
        qualified = f"{type(native).__module__}.{type(native).__qualname__}"
        return unknown(f"{qualified} is not a pandas dtype")

    if isinstance(dtype, pd.DatetimeTZDtype):
        # Before the numpy path: a tz-aware dtype answers `numpy_dtype` with
        # the naive one, which would silently drop the zone.
        return datetime(_unit_to_ir(getattr(dtype, "unit", None)), str(dtype.tz))
    if isinstance(dtype, pd.CategoricalDtype):
        return _from_categorical(dtype)
    if isinstance(dtype, pd.ArrowDtype):
        return from_arrow(dtype.pyarrow_dtype)
    if isinstance(dtype, pd.StringDtype):
        return string()
    if isinstance(dtype, pd.PeriodDtype):
        return unknown("a pandas Period has no IR dtype: it is an interval, not an instant")
    if isinstance(dtype, pd.IntervalDtype):
        return unknown("a pandas Interval has no IR dtype")
    if isinstance(dtype, pd.SparseDtype):
        # The storage is sparse; the values are whatever they are, so reading
        # through to the subtype is the useful answer.
        return to_ir(dtype.subtype)
    numpy_dtype = getattr(dtype, "numpy_dtype", None)
    if isinstance(numpy_dtype, np.dtype):
        # Every masked extension dtype - `Int64`, `boolean`, `Float32` -
        # answers here, so the width and kind logic is written once.
        return _from_numpy(numpy_dtype)
    if isinstance(dtype, np.dtype):
        return _from_numpy(dtype)
    return unknown(f"no IR dtype for the pandas dtype {dtype!r}")


def _as_pandas_dtype(native: Any) -> Any:
    """Normalise anything pandas accepts as a dtype, or None for a foreign object."""
    import numpy as np
    import pandas as pd
    from pandas.api.types import pandas_dtype

    if isinstance(native, np.dtype | pd.api.extensions.ExtensionDtype):
        return native
    if isinstance(native, str) or (isinstance(native, type) and issubclass(native, np.generic)):
        try:
            return pandas_dtype(native)
        except (TypeError, ValueError):
            return None
    if isinstance(native, type) and issubclass(native, pd.api.extensions.ExtensionDtype):
        try:
            return native()
        except Exception:
            # A parameterised dtype class with no usable default - `ArrowDtype`
            # needs a type, `DatetimeTZDtype` needs a zone - carries nothing to
            # report, and guessing a parameter would invent a dtype.
            return None
    return None


def _from_numpy(dtype: Any) -> DType:
    """Convert a numpy dtype by kind and width."""
    import numpy as np

    bits = dtype.itemsize * 8
    match dtype.kind:
        case "b":
            return boolean()
        case "i":
            return int_(bits)
        case "u":
            return uint(bits)
        case "f":
            return float_(bits)
        case "M":
            return datetime(_unit_to_ir(np.datetime_data(dtype)[0]))
        case "m":
            return duration(_unit_to_ir(np.datetime_data(dtype)[0]))
        case "O":
            return unknown(
                "a pandas object column holds arbitrary Python values, so pandas cannot say "
                "what is in it either"
            )
        case _:
            return unknown(f"no IR dtype for the numpy dtype {dtype!r}")


def _from_categorical(dtype: Any) -> DType:
    """Convert a `CategoricalDtype`: an enum when its variants are strings."""
    categories = getattr(dtype, "categories", None)
    if categories is None:
        return unknown("a pandas Categorical with no declared categories carries no variant set")
    variants = list(categories)
    if all(isinstance(variant, str) for variant in variants):
        return enum(tuple(str(variant) for variant in variants))
    # The variant set is real but unrepresentable: the IR has no numeric enum.
    # The value type is what breaks a pipeline, so that is what is reported.
    return to_ir(categories.dtype)


def from_arrow(arrow: Any) -> DType:
    """Convert a pyarrow type to the IR.

    Public because `describe` reads an arrow-backed column's type directly,
    and because the parquet source in core speaks the same vocabulary.
    """
    import pyarrow as pa

    if pa.types.is_boolean(arrow):
        return boolean()
    if pa.types.is_signed_integer(arrow):
        return int_(arrow.bit_width)
    if pa.types.is_unsigned_integer(arrow):
        return uint(arrow.bit_width)
    if pa.types.is_floating(arrow):
        return float_(arrow.bit_width)
    if pa.types.is_decimal(arrow):
        return decimal(int(arrow.precision), int(arrow.scale))
    if _is_string(pa, arrow):
        return string()
    if _is_binary(pa, arrow):
        return binary()
    if pa.types.is_date(arrow):
        # date32 and date64 differ in storage width, not in what they hold.
        return date()
    if pa.types.is_time(arrow):
        return time(_unit_to_ir(arrow.unit))
    if pa.types.is_timestamp(arrow):
        return datetime(_unit_to_ir(arrow.unit), arrow.tz)
    if pa.types.is_duration(arrow):
        return duration(_unit_to_ir(arrow.unit))
    if pa.types.is_null(arrow):
        return null()
    if pa.types.is_dictionary(arrow):
        # Arrow's categorical. The type records the value type and not the
        # variants, so this is a string rather than an enum: pandas keeps its
        # closed variant set in `CategoricalDtype`, which is read above.
        return from_arrow(arrow.value_type)
    if pa.types.is_map(arrow):
        return map_(from_arrow(arrow.key_type), from_arrow(arrow.item_type))
    if pa.types.is_struct(arrow):
        return struct({field.name: from_arrow(field.type) for field in arrow})
    if _is_list(pa, arrow):
        inner = from_arrow(arrow.value_type)
        if inner.kind is Kind.STRUCT and inner.names == MAP_FIELDS:
            return map_(inner.inner[0], inner.inner[1])
        return list_(inner)
    return unknown(f"no IR dtype for the pyarrow type {arrow}")


def _is_string(pa: Any, arrow: Any) -> bool:
    """True for every arrow string spelling, including the view types."""
    return bool(
        pa.types.is_string(arrow)
        or pa.types.is_large_string(arrow)
        or _is_view(pa, arrow, "string")
    )


def _is_binary(pa: Any, arrow: Any) -> bool:
    """True for every arrow binary spelling, fixed width included."""
    return bool(
        pa.types.is_binary(arrow)
        or pa.types.is_large_binary(arrow)
        or pa.types.is_fixed_size_binary(arrow)
        or _is_view(pa, arrow, "binary")
    )


def _is_list(pa: Any, arrow: Any) -> bool:
    """True for every arrow list spelling. A fixed-size list loses its width."""
    return bool(
        pa.types.is_list(arrow)
        or pa.types.is_large_list(arrow)
        or pa.types.is_fixed_size_list(arrow)
        or _is_view(pa, arrow, "list")
    )


def _is_view(pa: Any, arrow: Any, family: str) -> bool:
    """True for arrow's view types, which older pyarrow releases do not have."""
    check = getattr(pa.types, f"is_{family}_view", None)
    return bool(check(arrow)) if callable(check) else False


def _arrow_type(native: Any) -> Any:
    """`native` as a pyarrow type when it is one, else None."""
    if not arrow_available():
        return None
    import pyarrow as pa

    return native if isinstance(native, pa.DataType) else None


def _unit_to_ir(unit: str | None) -> TimeUnit:
    """Read a pandas or arrow unit string as a `TimeUnit`, defaulting to microseconds."""
    try:
        return TimeUnit(unit) if unit else TimeUnit.MICROSECOND
    except ValueError:
        return TimeUnit.MICROSECOND


# --- IR in -------------------------------------------------------------------


def from_ir(dtype: DType, *, prefer: Backend | None = None) -> Any:
    """Convert an IR dtype to a pandas dtype, raising `UnsupportedDType` when there is none.

    Args:
        dtype: the IR dtype to lower.
        prefer: the table to use, overriding the configured backend. The tests
            pass it to exercise both tables in one process; the adapter does
            not, because a pipeline gets one answer per run.

    Returns:
        A dtype object that both `astype` and the `DataFrame` constructor take.
    """
    chosen = prefer or backend()
    if dtype.kind is Kind.ENUM:
        # One answer for both tables. `CategoricalDtype` is pandas' own closed
        # variant set and the only spelling `groupby` and `merge` understand as
        # one; an arrow dictionary would record the value type and lose the
        # variants.
        return _categorical(dtype)
    if chosen == "arrow":
        return _to_arrow_dtype(dtype)
    return _to_numpy_dtype(dtype)


def _categorical(dtype: DType) -> Any:
    """A `CategoricalDtype` over an IR ENUM's variants."""
    import pandas as pd

    if not dtype.names:
        raise UnsupportedDType("a pandas Categorical needs its variants, and none were reported")
    return pd.CategoricalDtype(list(dtype.names), ordered=False)


def _to_arrow_dtype(dtype: DType) -> Any:
    """Lower an IR dtype to `pd.ArrowDtype`, which is the complete table."""
    import pandas as pd

    return pd.ArrowDtype(_to_arrow(dtype))


def _to_arrow(dtype: DType) -> Any:
    """Lower an IR dtype to a pyarrow type."""
    import pyarrow as pa

    match dtype.kind:
        case Kind.BOOL:
            return pa.bool_()
        case Kind.INT:
            return _arrow_integer(dtype, signed=True)
        case Kind.UINT:
            return _arrow_integer(dtype, signed=False)
        case Kind.FLOAT:
            return _arrow_float(dtype)
        case Kind.DECIMAL:
            return _arrow_decimal(dtype)
        case Kind.STRING:
            return pa.string()
        case Kind.BINARY:
            return pa.binary()
        case Kind.DATE:
            return pa.date32()
        case Kind.TIME:
            return _arrow_time(dtype)
        case Kind.DATETIME:
            return pa.timestamp(_unit_string(dtype), dtype.tz)
        case Kind.DURATION:
            return pa.duration(_unit_string(dtype))
        case Kind.LIST:
            return pa.list_(_to_arrow(_element(dtype)))
        case Kind.STRUCT:
            return pa.struct(
                [
                    (name, _to_arrow(inner))
                    for name, inner in zip(dtype.names, dtype.inner, strict=True)
                ]
            )
        case Kind.MAP:
            key, value = _pair(dtype)
            return pa.map_(_to_arrow(key), _to_arrow(value))
        case Kind.NULL:
            return pa.null()
        case Kind.ANY:
            # The gradual-typing top type. Arrow's null type is the one pandas
            # coerces into anything else, which is as close as the engine gets,
            # and an ANY column is already counted against coverage.
            return pa.null()
        case _:
            raise _refusal(dtype)


def _to_numpy_dtype(dtype: DType) -> Any:
    """Lower an IR dtype to a masked extension dtype, or to numpy where there is none."""
    import numpy as np
    import pandas as pd

    match dtype.kind:
        case Kind.BOOL:
            return pd.BooleanDtype()
        case Kind.INT:
            return _masked_integer(dtype, prefix="Int")
        case Kind.UINT:
            return _masked_integer(dtype, prefix="UInt")
        case Kind.FLOAT:
            return _masked_float(dtype)
        case Kind.STRING:
            return pd.StringDtype()
        case Kind.DATETIME:
            unit = _unit_string(dtype)
            if dtype.tz is None:
                return np.dtype(f"datetime64[{unit}]")
            return pd.DatetimeTZDtype(unit=unit, tz=dtype.tz)
        case Kind.DURATION:
            return np.dtype(f"timedelta64[{_unit_string(dtype)}]")
        case Kind.ANY:
            # `object` is the only pandas dtype that accepts anything, and that
            # it carries no type is exactly why an ANY column is not verified.
            return np.dtype("O")
        case Kind.DECIMAL | Kind.BINARY | Kind.DATE | Kind.TIME | Kind.NULL:
            raise UnsupportedDType(
                f"pandas has no {dtype.kind.value} dtype without pyarrow: {dtype} would be an "
                "object column, which carries no dtype at all. Install pyarrow, or set "
                "[engine.pandas] arrow_dtypes = true, to map it exactly"
            )
        case Kind.LIST | Kind.STRUCT | Kind.MAP:
            raise UnsupportedDType(
                f"pandas has no nested dtype without pyarrow: {dtype} would be an object "
                "column holding Python lists or dicts, with no element type"
            )
        case _:
            raise _refusal(dtype)


def _refusal(dtype: DType) -> UnsupportedDType:
    """The refusal for a dtype neither table has, with the reason when there is one."""
    detail = f" ({dtype.reason})" if dtype.reason else ""
    return UnsupportedDType(f"pandas has no dtype for {dtype}{detail}")


def _masked_integer(dtype: DType, *, prefix: str) -> Any:
    """`pd.Int32Dtype()` and friends, by width."""
    import pandas as pd

    bits = dtype.bits or 64
    if bits not in PANDAS_INT_WIDTHS:
        raise UnsupportedDType(f"pandas has no {bits}-bit integer; it has 8, 16, 32 and 64")
    found = getattr(pd, f"{prefix}{bits}Dtype", None)
    if found is None:  # pragma: no cover - every supported pandas has these
        raise UnsupportedDType(f"this pandas has no {prefix}{bits}Dtype")
    return found()


def _masked_float(dtype: DType) -> Any:
    """`pd.Float32Dtype()` or `pd.Float64Dtype()`, with numpy for half precision."""
    import numpy as np
    import pandas as pd

    bits = dtype.bits or 64
    if bits == 16:
        # pandas has no masked half-precision dtype, and numpy's float16 is a
        # real column: NA arrives in it as NaN, which is what float64 does too.
        return np.dtype("float16")
    if bits == 32:
        return pd.Float32Dtype()
    if bits == 64:
        return pd.Float64Dtype()
    raise UnsupportedDType(f"pandas has no {bits}-bit float; it has 16, 32 and 64")


def _arrow_integer(dtype: DType, *, signed: bool) -> Any:
    """A pyarrow integer type by width."""
    import pyarrow as pa

    bits = dtype.bits or 64
    if bits not in PANDAS_INT_WIDTHS:
        raise UnsupportedDType(f"pyarrow has no {bits}-bit integer; it has 8, 16, 32 and 64")
    factory: Callable[[], Any] = getattr(pa, f"{'int' if signed else 'uint'}{bits}")
    return factory()


def _arrow_float(dtype: DType) -> Any:
    """A pyarrow float type by width."""
    import pyarrow as pa

    bits = dtype.bits or 64
    factory = {16: pa.float16, 32: pa.float32, 64: pa.float64}.get(bits)
    if factory is None:
        raise UnsupportedDType(f"pyarrow has no {bits}-bit float; it has 16, 32 and 64")
    return factory()


def _arrow_decimal(dtype: DType) -> Any:
    """A pyarrow decimal, in the width that holds the declared precision."""
    import pyarrow as pa

    if dtype.precision is None or dtype.scale is None:
        raise UnsupportedDType(f"pandas needs both a precision and a scale for {dtype}")
    precision, scale = int(dtype.precision), int(dtype.scale)
    if precision <= 38:
        return pa.decimal128(precision, scale)
    if precision <= 76:
        return pa.decimal256(precision, scale)
    raise UnsupportedDType(
        f"pyarrow decimals hold at most 76 digits, and {dtype} asks for {precision}"
    )


def _arrow_time(dtype: DType) -> Any:
    """A pyarrow time, which splits across two physical widths by unit."""
    import pyarrow as pa

    unit = dtype.unit or TimeUnit.MICROSECOND
    if unit in {TimeUnit.SECOND, TimeUnit.MILLISECOND}:
        return pa.time32(unit.value)
    return pa.time64(unit.value)


def _unit_string(dtype: DType) -> str:
    """The unit of a temporal dtype, refused when pandas does not have it."""
    unit = dtype.unit or TimeUnit.MICROSECOND
    if unit not in PANDAS_TIME_UNITS:  # pragma: no cover - the IR has no fifth unit
        raise UnsupportedDType(f"pandas has no {unit.value}-resolution {dtype.kind.value}")
    return unit.value


def _element(dtype: DType) -> DType:
    """The element type of a LIST, refusing one the adapter was not given."""
    if not dtype.inner:
        raise UnsupportedDType("a list needs an element dtype, and none was reported")
    return dtype.inner[0]


def _pair(dtype: DType) -> tuple[DType, DType]:
    """The key and value types of a MAP, refusing a half-specified one."""
    if len(dtype.inner) != 2:
        raise UnsupportedDType("a map needs both a key and a value dtype")
    return dtype.inner[0], dtype.inner[1]


def describe_dtypes(dtypes: Sequence[Any]) -> tuple[DType, ...]:
    """Translate a frame's dtypes, in order. The hot path in `describe`."""
    return tuple(to_ir(dtype) for dtype in dtypes)
