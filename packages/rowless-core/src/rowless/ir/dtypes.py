"""The normalised dtype algebra every adapter maps into and out of.

Engines disagree about almost everything: Polars has `Utf8`, pandas has
`object` and `string[pyarrow]`, DuckDB has `VARCHAR`, Spark has `StringType`.
The IR is the one vocabulary the rest of rowless reasons about, so comparing a
Polars output against a dbt model's declared columns is a comparison of two
`DType` values and nothing more.

Two rules keep the IR honest:

1. **Lossless where it matters.** Width, signedness, timezone, time unit and
   decimal precision are modelled explicitly, because every one of them is a
   real source of pipeline breakage.
2. **Never silently exact.** An adapter that cannot determine a dtype returns
   `UNKNOWN`, which poisons comparisons instead of passing them. `ANY` is the
   deliberate escape hatch (see `rowless.ir.confidence`) and compares equal to
   everything, like `Any` in mypy.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, Self


class Kind(StrEnum):
    """Dtype families. Parameters live on `DType`, not in the member name."""

    BOOL = "bool"
    INT = "int"
    UINT = "uint"
    FLOAT = "float"
    DECIMAL = "decimal"
    STRING = "string"
    BINARY = "binary"
    DATE = "date"
    TIME = "time"
    DATETIME = "datetime"
    DURATION = "duration"
    LIST = "list"
    STRUCT = "struct"
    MAP = "map"
    ENUM = "enum"
    NULL = "null"

    #: The adapter could not work out this dtype. Comparisons involving UNKNOWN
    #: are reported as unverified, never as safe.
    UNKNOWN = "unknown"

    #: Gradual typing escape hatch: matches any dtype, and lowers the coverage
    #: score of whatever produced it.
    ANY = "any"


class TimeUnit(StrEnum):
    """Resolution for TIME, DATETIME and DURATION."""

    SECOND = "s"
    MILLISECOND = "ms"
    MICROSECOND = "us"
    NANOSECOND = "ns"


#: Widths the IR accepts, per family. Anything else is an adapter lowering bug
#: rather than an exotic engine type, so the constructors reject it.
INT_WIDTHS: tuple[int, ...] = (8, 16, 32, 64, 128)
FLOAT_WIDTHS: tuple[int, ...] = (16, 32, 64)

#: Decimal digits an integer of each width holds, for the integer-into-decimal
#: check in `DType.accepts`.
_INT_DIGITS: dict[int, int] = {8: 3, 16: 5, 32: 10, 64: 19, 128: 39}

#: Ordered coarse to fine, so a slot can be asked whether it is fine enough to
#: hold another unit's values without rounding.
_UNIT_ORDER: dict[TimeUnit, int] = {
    TimeUnit.SECOND: 0,
    TimeUnit.MILLISECOND: 1,
    TimeUnit.MICROSECOND: 2,
    TimeUnit.NANOSECOND: 3,
}

_TEMPORAL_KINDS = frozenset({Kind.TIME, Kind.DATETIME, Kind.DURATION})
_INTEGER_KINDS = frozenset({Kind.INT, Kind.UINT})


@dataclass(frozen=True, slots=True)
class DType:
    """A single normalised dtype.

    Construct through the module-level helpers (`int_()`, `datetime()`, ...)
    rather than by hand; they validate the parameter combinations.
    """

    kind: Kind
    bits: int | None = None
    """Width for INT, UINT and FLOAT."""

    unit: TimeUnit | None = None
    """Resolution for TIME, DATETIME and DURATION."""

    tz: str | None = None
    """IANA timezone for DATETIME. `None` means naive, which is not the same
    thing as UTC and is a frequent source of silent breakage."""

    precision: int | None = None
    """Total digits for DECIMAL."""

    scale: int | None = None
    """Fractional digits for DECIMAL."""

    inner: tuple[DType, ...] = ()
    """Element types: one for LIST, two (key, value) for MAP, N for STRUCT."""

    names: tuple[str, ...] = ()
    """Field names for STRUCT, or the variant set for ENUM."""

    reason: str | None = field(default=None, compare=False)
    """Why an UNKNOWN dtype is unknown, carried through to diagnostics.

    Excluded from comparison deliberately: it is commentary, and two UNKNOWNs
    with different reasons are equally unusable. `matches` refuses both
    whatever is written here.
    """

    def __str__(self) -> str:
        """Render the compact spelling diagnostics quote and contracts accept.

        The exact inverse of `rowless.sources.literal.parse_dtype_string`, so a
        dtype printed in a finding can be pasted straight into a contract file.
        """
        match self.kind:
            case Kind.INT | Kind.UINT | Kind.FLOAT:
                return f"{self.kind.value}{self.bits}"
            case Kind.DECIMAL:
                return f"decimal({self.precision},{self.scale})"
            case Kind.TIME | Kind.DURATION:
                return f"{self.kind.value}[{self._unit_str}]"
            case Kind.DATETIME:
                if self.tz is None:
                    return f"datetime[{self._unit_str}]"
                return f"datetime[{self._unit_str}, {self.tz}]"
            case Kind.LIST:
                return f"list[{_element(self)}]"
            case Kind.MAP:
                key, value = _pair(self)
                return f"map[{key}, {value}]"
            case Kind.STRUCT:
                body = ",".join(
                    f"{name}:{dtype}" for name, dtype in zip(self.names, self.inner, strict=False)
                )
                return f"struct{{{body}}}"
            case Kind.ENUM:
                return f"enum{{{','.join(self.names)}}}"
            case _:
                return self.kind.value

    @property
    def _unit_str(self) -> str:
        """Time unit spelling, defaulting to microseconds when one is missing."""
        return (self.unit or TimeUnit.MICROSECOND).value

    @property
    def is_known(self) -> bool:
        """False when this dtype, or anything nested in it, is UNKNOWN."""
        if self.kind is Kind.UNKNOWN:
            return False
        return all(inner.is_known for inner in self.inner)

    @property
    def is_any(self) -> bool:
        """True when this dtype, or anything nested in it, is ANY."""
        if self.kind is Kind.ANY:
            return True
        return any(inner.is_any for inner in self.inner)

    def matches(self, other: DType) -> bool:
        """Structural equality, except that ANY matches anything.

        UNKNOWN does *not* match itself: two dtypes nobody could determine are
        not evidence that a pipeline is sound.
        """
        if self.kind is Kind.UNKNOWN or other.kind is Kind.UNKNOWN:
            return False
        if self.kind is Kind.ANY or other.kind is Kind.ANY:
            return True
        if self.kind is not other.kind:
            return False
        parameters_agree = (
            self.bits == other.bits
            and self.unit == other.unit
            and self.tz == other.tz
            and self.precision == other.precision
            and self.scale == other.scale
            and self.names == other.names
        )
        if not parameters_agree or len(self.inner) != len(other.inner):
            return False
        return all(
            mine.matches(theirs) for mine, theirs in zip(self.inner, other.inner, strict=True)
        )

    def accepts(self, other: DType) -> bool:
        """True when a value of `other` can flow into a slot declared as `self`.

        Wider than `matches`: int32 -> int64 is accepted, the reverse is not.
        Used when checking a computed schema against a declared contract.
        """
        if self.kind is Kind.UNKNOWN or other.kind is Kind.UNKNOWN:
            return False
        if self.kind is Kind.ANY or other.kind is Kind.ANY:
            return True
        if other.kind is Kind.NULL:
            # An all-null column carries no type evidence against it, so it
            # fits any declared slot. It is still reported as NULL, because a
            # column that is entirely null is usually the bug a user wants.
            return True
        if self.matches(other):
            return True
        return self._accepts_widened(other)

    def _accepts_widened(self, other: DType) -> bool:
        """The widening half of `accepts`, kept separate to stay readable."""
        match self.kind:
            case Kind.INT:
                if other.kind is Kind.INT:
                    return _width(self) >= _width(other)
                # An unsigned value needs a bit of headroom to stay lossless.
                return other.kind is Kind.UINT and _width(self) > _width(other)
            case Kind.UINT:
                return other.kind is Kind.UINT and _width(self) >= _width(other)
            case Kind.FLOAT:
                if other.kind is Kind.FLOAT:
                    return _width(self) >= _width(other)
                # int32 fits float64 exactly; int64 does not, and rowless will
                # not pretend otherwise.
                return other.kind in _INTEGER_KINDS and _width(self) > _width(other)
            case Kind.DECIMAL:
                if other.kind is Kind.DECIMAL:
                    return _scale(self) >= _scale(other) and _integral(self) >= _integral(other)
                if other.kind in _INTEGER_KINDS:
                    return _integral(self) >= _INT_DIGITS.get(_width(other), 39)
                return False
            case Kind.STRING:
                # ENUM variants are strings. The reverse direction loses the
                # closed variant set, and is refused.
                return other.kind is Kind.ENUM
            case Kind.DATETIME:
                return (
                    other.kind is Kind.DATETIME
                    and self.tz == other.tz
                    and _resolution(self) >= _resolution(other)
                )
            case Kind.TIME | Kind.DURATION:
                return other.kind is self.kind and _resolution(self) >= _resolution(other)
            case Kind.LIST:
                return other.kind is Kind.LIST and _element(self).accepts(_element(other))
            case Kind.MAP:
                if other.kind is not Kind.MAP:
                    return False
                return all(
                    mine.accepts(theirs)
                    for mine, theirs in zip(_pair(self), _pair(other), strict=True)
                )
            case Kind.STRUCT:
                if other.kind is not Kind.STRUCT or self.names != other.names:
                    return False
                return all(
                    mine.accepts(theirs)
                    for mine, theirs in zip(self.inner, other.inner, strict=True)
                )
            case Kind.ENUM:
                return other.kind is Kind.ENUM and set(other.names) <= set(self.names)
            case _:
                return False

    def with_(self, **overrides: Any) -> Self:
        """Copy with parameters replaced. Adapters use this when lowering."""
        return replace(self, **overrides)


def _width(dtype: DType) -> int:
    """Declared width, defaulting to 64 when an adapter omitted it."""
    return dtype.bits or 64


def _scale(dtype: DType) -> int:
    """Decimal scale, defaulting to 0."""
    return dtype.scale or 0


def _integral(dtype: DType) -> int:
    """Digits left of the decimal point."""
    return (dtype.precision or 0) - _scale(dtype)


def _resolution(dtype: DType) -> int:
    """Time unit as an orderable number, coarse to fine."""
    return _UNIT_ORDER[dtype.unit or TimeUnit.MICROSECOND]


def _element(dtype: DType) -> DType:
    """LIST element type, or UNKNOWN when an adapter left it off."""
    return dtype.inner[0] if dtype.inner else unknown("list element type not reported")


def _pair(dtype: DType) -> tuple[DType, DType]:
    """MAP key and value types, padded with UNKNOWN."""
    padded = (*dtype.inner, unknown("map key/value not reported"), unknown())
    return padded[0], padded[1]


# --- Constructors ------------------------------------------------------------
# Deliberately plain functions: adapters call these a lot, and they read better
# than DType(Kind.INT, bits=64) at the call site.


def boolean() -> DType:
    """BOOL."""
    return DType(Kind.BOOL)


def int_(bits: int = 64) -> DType:
    """Signed integer of `bits` width (8, 16, 32, 64, 128)."""
    _check_width(Kind.INT, bits, INT_WIDTHS)
    return DType(Kind.INT, bits=bits)


def uint(bits: int = 64) -> DType:
    """Unsigned integer of `bits` width."""
    _check_width(Kind.UINT, bits, INT_WIDTHS)
    return DType(Kind.UINT, bits=bits)


def float_(bits: int = 64) -> DType:
    """IEEE float of `bits` width (16, 32, 64)."""
    _check_width(Kind.FLOAT, bits, FLOAT_WIDTHS)
    return DType(Kind.FLOAT, bits=bits)


def decimal(precision: int, scale: int) -> DType:
    """Fixed-point decimal."""
    if not 0 <= scale <= precision:
        raise ValueError(f"decimal needs precision >= scale >= 0, got ({precision},{scale})")
    return DType(Kind.DECIMAL, precision=precision, scale=scale)


def string() -> DType:
    """UTF-8 string, regardless of the engine's physical representation."""
    return DType(Kind.STRING)


def binary() -> DType:
    """Opaque bytes."""
    return DType(Kind.BINARY)


def date() -> DType:
    """Calendar date."""
    return DType(Kind.DATE)


def time(unit: TimeUnit = TimeUnit.MICROSECOND) -> DType:
    """Wall-clock time of day."""
    return DType(Kind.TIME, unit=unit)


def datetime(unit: TimeUnit = TimeUnit.MICROSECOND, tz: str | None = None) -> DType:
    """Timestamp. `tz=None` is naive, which rowless tracks as distinct from UTC."""
    return DType(Kind.DATETIME, unit=unit, tz=tz)


def duration(unit: TimeUnit = TimeUnit.MICROSECOND) -> DType:
    """Elapsed time."""
    return DType(Kind.DURATION, unit=unit)


def list_(inner: DType) -> DType:
    """Variable-length list of `inner`."""
    return DType(Kind.LIST, inner=(inner,))


def struct(fields: dict[str, DType]) -> DType:
    """Named, ordered nested fields."""
    return DType(Kind.STRUCT, names=tuple(fields), inner=tuple(fields.values()))


def map_(key: DType, value: DType) -> DType:
    """Key/value map."""
    return DType(Kind.MAP, inner=(key, value))


def enum(variants: tuple[str, ...]) -> DType:
    """Closed set of string variants (Polars `Enum`, DuckDB `ENUM`)."""
    return DType(Kind.ENUM, names=variants)


def null() -> DType:
    """All-null column with no further type information (pandas loves these)."""
    return DType(Kind.NULL)


def unknown(reason: str = "") -> DType:
    """A dtype the adapter could not resolve. `reason` is for diagnostics."""
    return DType(Kind.UNKNOWN, reason=reason or None)


def any_() -> DType:
    """The gradual-typing top type. Lowers coverage; never fails a check."""
    return DType(Kind.ANY)


def _check_width(kind: Kind, bits: int, allowed: tuple[int, ...]) -> None:
    """Reject a width no engine has, before it reaches a comparison."""
    if bits not in allowed:
        raise ValueError(f"{kind.value} width must be one of {allowed}, got {bits}")


# --- Lattice -----------------------------------------------------------------


def unify(left: DType, right: DType) -> DType:
    """Least dtype that accepts both operands.

    This is what `concat`, `union` and the branches of a conditional need.
    Falls back to UNKNOWN rather than inventing a coercion the engine would
    have refused.
    """
    if left.kind is Kind.UNKNOWN or right.kind is Kind.UNKNOWN:
        return unknown(left.reason or right.reason or "unified with an unresolved dtype")
    if left.kind is Kind.ANY or right.kind is Kind.ANY:
        return any_()
    if left.kind is Kind.NULL:
        return right
    if right.kind is Kind.NULL:
        return left
    if left.accepts(right):
        return left
    if right.accepts(left):
        return right
    return _promote(left, right)


def _promote(left: DType, right: DType) -> DType:
    """Neither side subsumes the other: promote only where engines agree."""
    kinds = {left.kind, right.kind}
    if kinds <= _INTEGER_KINDS:
        return _unify_integers(left, right)
    if kinds <= {Kind.INT, Kind.UINT, Kind.FLOAT}:
        widest = max(_width(left), _width(right))
        return float_(64) if widest > 32 else float_(32)
    if kinds == {Kind.DECIMAL}:
        scale = max(_scale(left), _scale(right))
        integral = max(_integral(left), _integral(right))
        return decimal(integral + scale, scale)
    if len(kinds) == 1 and left.kind in _TEMPORAL_KINDS:
        return _unify_temporal(left, right)
    if kinds == {Kind.LIST}:
        return _nested(list_(unify(_element(left), _element(right))))
    if kinds == {Kind.MAP}:
        key = unify(_pair(left)[0], _pair(right)[0])
        value = unify(_pair(left)[1], _pair(right)[1])
        return _nested(map_(key, value))
    if kinds == {Kind.STRUCT} and left.names == right.names:
        merged = {
            name: unify(mine, theirs)
            for name, mine, theirs in zip(left.names, left.inner, right.inner, strict=True)
        }
        return _nested(struct(merged))
    if kinds == {Kind.ENUM}:
        # The variant sets differ, and engines disagree about whether the union
        # is even legal. A plain string is the honest common type.
        return string()
    return unknown(f"no common dtype for {left} and {right}")


def _unify_integers(left: DType, right: DType) -> DType:
    """Signed and unsigned integer join, widening only where it stays lossless."""
    widest = max(_width(left), _width(right))
    if left.kind is right.kind:
        return int_(widest) if left.kind is Kind.INT else uint(widest)
    signed, unsigned = (left, right) if left.kind is Kind.INT else (right, left)
    width = max(_width(signed), _width(unsigned) * 2)
    if width not in INT_WIDTHS:
        return unknown(f"no signed integer wide enough for {signed} and {unsigned}")
    return int_(width)


def _unify_temporal(left: DType, right: DType) -> DType:
    """Temporal join: take the finer unit, but never reconcile timezones."""
    if left.kind is Kind.DATETIME and left.tz != right.tz:
        # Naive and aware are different values, not different spellings, and a
        # guessed conversion shifts data silently.
        return unknown(f"cannot unify timezones: {left} and {right}")
    finer = left.unit if _resolution(left) >= _resolution(right) else right.unit
    return left.with_(unit=finer)


def _nested(candidate: DType) -> DType:
    """Collapse a nested dtype whose elements failed to unify."""
    if candidate.is_known:
        return candidate
    return unknown(f"nested dtypes do not unify: {candidate}")
