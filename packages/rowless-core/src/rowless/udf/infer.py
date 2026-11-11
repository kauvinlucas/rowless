"""UDF return-type inference. Hard problem 05.

A UDF is a hole in the schema graph. Four ways to fill it, in descending order
of trust:

1. `return_dtype=` passed to `map_elements` / `apply`. The engine itself
   relies on this, so rowless should too.
2. A return annotation on the function.
3. Probing: call the function once with one synthetic value and inspect the
   result. Opt-in, since it runs user code with invented input.
4. Give up and return ANY, lowering coverage.

Probing is where this gets interesting: the synthetic value has to be
plausible enough not to raise. A function that does `x.split(",")[1]` needs a
string with a comma in it, so the generator is dtype-aware and tries a few
candidates before concluding the function is unprobeable.
"""

from __future__ import annotations

import datetime as dt
import decimal
import inspect
import types
import typing
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from rowless.ir.dtypes import (
    DType,
    Kind,
    any_,
    binary,
    boolean,
    date,
    datetime,
    duration,
    float_,
    int_,
    list_,
    map_,
    null,
    string,
    struct,
    time,
    unify,
)
from rowless.ir.schema import Schema

#: How many synthetic values are tried before a function is called unprobeable.
#: Three, because the useful variety is "boring", "structured" and "longer",
#: and a fourth attempt has never been the one that worked.
MAX_PROBE_ATTEMPTS = 3


class InferenceMethod(StrEnum):
    """How a return dtype was determined. Surfaced in diagnostics."""

    DECLARED = "declared"
    ANNOTATION = "annotation"
    PROBE = "probe"
    GAVE_UP = "gave_up"


@dataclass(frozen=True, slots=True)
class UdfSignature:
    """What rowless worked out about one UDF."""

    qualname: str
    returns: DType
    method: InferenceMethod
    probe_attempts: int = 0
    note: str | None = None


def infer_return_dtype(
    fn: Callable[..., Any],
    arg_schema: Schema,
    *,
    declared: DType | None = None,
    allow_probe: bool = False,
) -> UdfSignature:
    """Work out what `fn` returns, trying each method in order."""
    qualname = getattr(fn, "__qualname__", repr(fn))

    if declared is not None and declared.is_known:
        return UdfSignature(qualname=qualname, returns=declared, method=InferenceMethod.DECLARED)

    annotated = _return_annotation(fn)
    if annotated is not None:
        dtype = annotation_to_ir(annotated)
        if dtype.is_known and not dtype.is_any:
            return UdfSignature(
                qualname=qualname,
                returns=dtype,
                method=InferenceMethod.ANNOTATION,
            )

    if allow_probe:
        probed, attempts, note = _probe(fn, arg_schema)
        if probed is not None:
            return UdfSignature(
                qualname=qualname,
                returns=probed,
                method=InferenceMethod.PROBE,
                probe_attempts=attempts,
                note=note,
            )
        return UdfSignature(
            qualname=qualname,
            returns=any_(),
            method=InferenceMethod.GAVE_UP,
            probe_attempts=attempts,
            note=note,
        )

    return UdfSignature(
        qualname=qualname,
        returns=any_(),
        method=InferenceMethod.GAVE_UP,
        note="no return_dtype and no return annotation; enable probe_udfs to call it",
    )


def _probe(fn: Callable[..., Any], arg_schema: Schema) -> tuple[DType | None, int, str | None]:
    """Call `fn` with synthetic values until it returns something.

    Every attempt is one call with made-up input, which is why this is opt-in.
    The first exception is not the answer: a function that parses its argument
    fails on a boring value and succeeds on a structured one.
    """
    parameters = _positional_count(fn)
    dtypes = [column.dtype for column in arg_schema.fields] or [string()]
    last_error: str | None = None
    for attempt in range(MAX_PROBE_ATTEMPTS):
        arguments = [
            synthetic_value(dtypes[index % len(dtypes)], attempt) for index in range(parameters)
        ]
        try:
            produced = fn(*arguments)
        except Exception as exc:
            last_error = f"{type(exc).__qualname__}: {exc}"
            continue
        dtype = python_value_to_ir(produced)
        if dtype.is_known:
            return dtype, attempt + 1, None
        return None, attempt + 1, f"returned {type(produced).__qualname__}, which has no IR dtype"
    return None, MAX_PROBE_ATTEMPTS, f"every probe raised; last was {last_error}"


def _positional_count(fn: Callable[..., Any]) -> int:
    """How many positional arguments to synthesise. One when unknowable."""
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return 1
    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
        and parameter.default is inspect.Parameter.empty
    ]
    return max(len(positional), 1)


def _return_annotation(fn: Callable[..., Any]) -> object | None:
    """The function's return annotation, resolved from a string when possible."""
    try:
        hints = typing.get_type_hints(fn)
    except Exception:
        # A forward reference rowless cannot resolve is not worth failing over;
        # the raw annotation is tried instead, and ANY is the fallback.
        raw = getattr(fn, "__annotations__", {}).get("return")
        return raw
    return hints.get("return")


def synthetic_value(dtype: DType, attempt: int = 0) -> Any:
    """A plausible value for `dtype`, varied by `attempt`.

    Attempt 0 is boring ("x", 1, 2020-01-01). Later attempts get more
    structured, so a function that parses its input has a chance of succeeding
    before rowless declares it unprobeable.
    """
    match dtype.kind:
        case Kind.BOOL:
            return attempt % 2 == 0
        case Kind.INT | Kind.UINT:
            return (1, 42, 1_000_000)[min(attempt, 2)]
        case Kind.FLOAT:
            return (1.0, 3.5, 1e6)[min(attempt, 2)]
        case Kind.DECIMAL:
            return decimal.Decimal(("1", "12.34", "1234.5678")[min(attempt, 2)])
        case Kind.STRING:
            # Attempt 1 has a comma and attempt 2 a path separator, because
            # `x.split(",")[1]` and `x.rsplit("/")[-1]` are what UDFs do.
            return ("x", "a,b", "2020-01-01T00:00:00Z")[min(attempt, 2)]
        case Kind.BINARY:
            return (b"x", b"a,b", b"\x00\x01")[min(attempt, 2)]
        case Kind.DATE:
            return dt.date(2020, 1, 1)
        case Kind.TIME:
            return dt.time(12, 0, 0)
        case Kind.DATETIME:
            naive = dt.datetime(2020, 1, 1, 12, 0, 0)
            return naive if dtype.tz is None else naive.replace(tzinfo=dt.UTC)
        case Kind.DURATION:
            return dt.timedelta(days=1)
        case Kind.LIST:
            element = dtype.inner[0] if dtype.inner else string()
            return [synthetic_value(element, attempt)]
        case Kind.STRUCT:
            return {
                name: synthetic_value(inner, attempt)
                for name, inner in zip(dtype.names, dtype.inner, strict=False)
            }
        case Kind.MAP:
            key, value = (*dtype.inner, string(), string())[:2]
            return {synthetic_value(key, attempt): synthetic_value(value, attempt)}
        case Kind.ENUM:
            return dtype.names[0] if dtype.names else "x"
        case _:
            # NULL, UNKNOWN and ANY: there is no representative value, and
            # None is the only honest one.
            return None


def annotation_to_ir(annotation: object) -> DType:
    """Convert a Python type annotation to an IR dtype.

    Handles builtins, `datetime`, `decimal.Decimal`, and the engines' own
    scalar aliases. Anything else is ANY, not a guess.
    """
    if annotation is None or annotation is type(None):
        return null()
    if isinstance(annotation, DType):
        return annotation
    if isinstance(annotation, str):
        return _from_name(annotation)

    origin = typing.get_origin(annotation)
    if origin is not None:
        return _from_generic(origin, typing.get_args(annotation))
    if isinstance(annotation, type):
        scalar = _SCALARS.get(annotation)
        return scalar if scalar is not None else _from_name(annotation.__name__)
    return any_()


def _from_generic(origin: Any, args: Sequence[Any]) -> DType:
    """Convert `list[int]`, `dict[str, int]`, `int | None` and friends."""
    if origin in {types.UnionType, typing.Union}:
        candidates = [annotation_to_ir(arg) for arg in args if arg is not type(None)]
        if not candidates:
            return null()
        joined = candidates[0]
        for candidate in candidates[1:]:
            joined = unify(joined, candidate)
        return joined
    if origin in {list, set, frozenset, tuple, Sequence}:
        element = annotation_to_ir(args[0]) if args else any_()
        return list_(element)
    if origin in {dict, Mapping}:
        if len(args) == 2:
            return map_(annotation_to_ir(args[0]), annotation_to_ir(args[1]))
        return map_(string(), any_())
    return any_()


#: Python scalars with an exact IR equivalent. Width choices follow what every
#: engine does when it sees a Python object column: 64-bit everything.
_SCALARS: dict[type, DType] = {
    bool: boolean(),
    int: int_(64),
    float: float_(64),
    str: string(),
    bytes: binary(),
    bytearray: binary(),
    decimal.Decimal: DType(Kind.DECIMAL, precision=38, scale=9),
    dt.date: date(),
    dt.datetime: datetime(),
    dt.time: time(),
    dt.timedelta: duration(),
}

#: Scalar aliases the engines export, matched by name so core imports none of
#: them. `pl.Int64` and `np.int64` both arrive here as a name.
_ALIASES: dict[str, str] = {
    "int8": "int8",
    "int16": "int16",
    "int32": "int32",
    "int64": "int64",
    "uint8": "uint8",
    "uint16": "uint16",
    "uint32": "uint32",
    "uint64": "uint64",
    "float32": "float32",
    "float64": "float64",
    "bool_": "bool",
    "boolean": "bool",
    "str_": "string",
    "utf8": "string",
    "string": "string",
    "object": "any",
    "datetime64": "datetime[us]",
    "timedelta64": "duration[us]",
}


def _from_name(name: str) -> DType:
    """Last resort: match an annotation by name, for engine scalar aliases."""
    from rowless.ir.serde import DTypeSyntaxError, parse_dtype

    cleaned = name.strip().lower().removeprefix("np.").removeprefix("pl.")
    for candidate in (_ALIASES.get(cleaned), cleaned):
        if candidate is None:
            continue
        try:
            return parse_dtype(candidate)
        except DTypeSyntaxError:
            continue
    builtin = {"bool": boolean(), "int": int_(64), "float": float_(64), "str": string()}
    return builtin.get(cleaned, any_())


def python_value_to_ir(value: object) -> DType:
    """The IR dtype of a concrete Python value, for probe results.

    A heterogeneous list or dict unifies its elements, so a probe that returns
    `[1, "a"]` lands on UNKNOWN rather than on whichever element came first.
    """
    if value is None:
        return null()
    if isinstance(value, bool):
        return boolean()
    for python_type, dtype in _SCALARS.items():
        if type(value) is python_type:
            return dtype
    if isinstance(value, Mapping):
        if not value:
            return map_(string(), any_())
        keys = [python_value_to_ir(key) for key in value]
        values = [python_value_to_ir(item) for item in value.values()]
        if all(isinstance(key, str) for key in value):
            return struct({str(key): python_value_to_ir(item) for key, item in value.items()})
        return map_(_unify_all(keys), _unify_all(values))
    if isinstance(value, list | tuple | set | frozenset):
        elements = [python_value_to_ir(item) for item in value]
        return list_(_unify_all(elements) if elements else any_())
    if isinstance(value, dt.datetime):
        return datetime(tz="UTC" if value.tzinfo is not None else None)
    if isinstance(value, dt.date):
        return date()
    return _from_name(type(value).__name__)


def _unify_all(dtypes: Sequence[DType]) -> DType:
    """Join a list of dtypes, falling back to UNKNOWN as `unify` does."""
    if not dtypes:
        return any_()
    joined = dtypes[0]
    for dtype in dtypes[1:]:
        joined = unify(joined, dtype)
    return joined
