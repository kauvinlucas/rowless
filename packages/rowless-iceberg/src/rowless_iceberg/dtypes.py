"""Iceberg types to the rowless IR.

Iceberg has the smallest type system of anything rowless reads, and it is
specified rather than per-engine: the spec names twelve primitives and three
nested types, every writer agrees on them, and `str()` on a PyIceberg type
returns the spec's own spelling (`long`, `timestamptz`, `decimal(18, 2)`).
So the mapping is a lookup table keyed on that spelling rather than a tree of
`isinstance` checks, which also means a PyIceberg release that adds a v3
primitive does not need a new import here to be *reported* - it lands in the
UNKNOWN branch with its spec name in the reason, instead of failing to import.

The judgement calls, written down because each one decides whether a diff is
believed:

* **`timestamptz` is UTC, not "aware".** Iceberg stores it as a UTC instant by
  definition, so this is the type's meaning and not a normalisation rowless
  chose. `timestamp` stays naive, which is the distinction that breaks
  pipelines.
* **`time` is microseconds.** The spec has one time type and it is microsecond
  resolution. There is no `time_ns`.
* **`uuid`, `geometry` and `geography` are ANY.** The IR has no equivalent.
  `string` would be worse than useless: a `string` to `uuid` retype would then
  read as no change at all. UNKNOWN would be worse in the other direction -
  most Iceberg tables have a UUID key, and marking every one of them
  UNVERIFIED forever is how a gate gets switched off. ANY is the escape hatch
  that lowers the coverage score and says why.
* **`fixed[L]` is binary, and the length is dropped.** Deliberate: the same
  column read through the Parquet footer resolves to `binary` too, and the two
  sources must not disagree about one file. Nothing is lost that Iceberg
  allows to change - the spec permits no promotion between fixed lengths, so a
  `fixed[16]` never becomes a `fixed[32]` in a valid table.
* **v3 `unknown` is NULL, not UNKNOWN.** They are opposites. Iceberg's
  `unknown` is a column whose every value is null, which the IR models exactly
  as `null()`; the IR's UNKNOWN means rowless could not work the type out.
* **Nested field nullability is not represented.** The IR models `nullable` on
  a top-level field only, so a struct subfield going required to optional is
  invisible here. It is a gap in the IR rather than in this mapping, and it is
  the one Iceberg change this source cannot report. See the README.
* **Anything unrecognised is UNKNOWN with the spec spelling in the reason.** A
  column rowless cannot type poisons comparisons rather than passing them.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from typing import Any

from rowless.ir.dtypes import (
    DType,
    TimeUnit,
    any_,
    binary,
    boolean,
    date,
    datetime,
    decimal,
    float_,
    int_,
    list_,
    map_,
    null,
    string,
    struct,
    time,
    unknown,
)

#: The Iceberg primitives with a fixed IR dtype, keyed by the spec's spelling.
#: Values are constructors so each call returns a fresh dtype. `decimal(p,s)`
#: and `fixed[L]` are parameterised and handled separately.
PRIMITIVES: Mapping[str, Callable[[], DType]] = {
    "boolean": boolean,
    # Iceberg has exactly two integer widths and no unsigned type, so these
    # two lines are the whole integer story.
    "int": lambda: int_(32),
    "long": lambda: int_(64),
    "float": lambda: float_(32),
    "double": lambda: float_(64),
    "date": date,
    "time": lambda: time(TimeUnit.MICROSECOND),
    "timestamp": lambda: datetime(TimeUnit.MICROSECOND),
    "timestamptz": lambda: datetime(TimeUnit.MICROSECOND, "UTC"),
    "timestamp_ns": lambda: datetime(TimeUnit.NANOSECOND),
    "timestamptz_ns": lambda: datetime(TimeUnit.NANOSECOND, "UTC"),
    "string": string,
    "binary": binary,
    # Format version 3: a column with one possible value, null. The IR's NULL
    # is the same thing, and `accepts` already lets it into any slot.
    "unknown": null,
}

#: Primitives with no IR equivalent, reported as ANY carrying the reason.
#: `geometry` and `geography` arrive parameterised by CRS, so the match is on
#: the leading word.
OPAQUE: frozenset[str] = frozenset({"uuid", "geometry", "geography"})

_DECIMAL = re.compile(r"^decimal\(\s*(\d+)\s*,\s*(\d+)\s*\)$")
_FIXED = re.compile(r"^fixed\[\s*\d+\s*\]$")


def dtype_from_iceberg(native: Any) -> DType:
    """Map one PyIceberg type onto one IR dtype.

    Args:
        native: a `pyiceberg.types.IcebergType`. Nested types are walked
            recursively; primitives are matched on their spec spelling.

    Returns:
        The IR dtype, or UNKNOWN carrying the Iceberg spelling that could not
        be mapped. Never a near-enough substitute.
    """
    nested = _nested(native)
    if nested is not None:
        return nested

    spelling = str(native)
    constructor = PRIMITIVES.get(spelling)
    if constructor is not None:
        return constructor()

    decimal_match = _DECIMAL.match(spelling)
    if decimal_match is not None:
        precision, scale = (int(group) for group in decimal_match.groups())
        if scale > precision:
            return unknown(f"the Iceberg type {spelling} has a scale larger than its precision")
        return decimal(precision, scale)

    if _FIXED.match(spelling):
        # The length is dropped on purpose: see the module docstring.
        return binary()

    if spelling.split("(")[0] in OPAQUE:
        return any_().with_(reason=f"the Iceberg type {spelling} has no IR equivalent")

    return unknown(f"no IR dtype for the Iceberg type {spelling!r}")


def _nested(native: Any) -> DType | None:
    """LIST, MAP or STRUCT, or None when this is not a nested type.

    Dispatched on the classes rather than on a spelling, because a nested
    type's `str()` embeds its element types and parsing that back would be a
    second, worse implementation of PyIceberg's own parser. These three
    classes have been in PyIceberg since its first release, so importing them
    costs nothing in version tolerance.
    """
    from pyiceberg.types import ListType, MapType, StructType

    if isinstance(native, ListType):
        return list_(dtype_from_iceberg(native.element_type))
    if isinstance(native, MapType):
        return map_(dtype_from_iceberg(native.key_type), dtype_from_iceberg(native.value_type))
    if isinstance(native, StructType):
        # Subfield `required` is dropped: the IR carries nullability on a
        # top-level field only.
        return struct({field.name: dtype_from_iceberg(field.field_type) for field in native.fields})
    return None
