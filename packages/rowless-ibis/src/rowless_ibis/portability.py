"""Where one expression means different things on different backends.

This is the finding no other adapter in the project can produce. Every other
adapter answers "what is the schema of this pipeline"; ibis can answer "what is
the schema of this pipeline *on Snowflake*", because ibis carries one type
mapper per backend and those mappers are pure Python. Nothing has to be
installed and nothing has to be connected to: lowering a dtype through
`BigQueryType` and reading it back is a string round trip through sqlglot.

What that catches, with the four defaults:

* BigQuery has one integer type, so `int32` lands as `INT64` and comes back
  `int64`. A pipeline that declares a narrow integer in a contract and runs on
  BigQuery has a contract that cannot hold.
* BigQuery's `NUMERIC` is fixed at (38, 9), so `decimal(18, 2)` widens.
* A timestamp with a named zone is a `TIMESTAMP` on BigQuery, which stores UTC
  instants: the zone is gone, exactly as it is in DuckDB's `TIMESTAMPTZ`.
* SQLite has no array type at all, and no unsigned integers, so those columns
  come back unresolved rather than approximated.

A disagreement is reported when the dtype a backend would store does not
`match` the dtype ibis resolved. That is deliberately strict: `int32` widening
to `int64` is lossless for the values and still breaks a declared contract,
which is the thing a user wants to be told about before the migration rather
than after it.

Nullability is not part of this comparison. The backend type mappers work in
SQL type strings, which do not carry it, so a round trip through one would
report every column nullable and invent a disagreement that is not there.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from rowless.errors import RowlessError
from rowless.ir.dtypes import DType, unknown
from rowless.ir.schema import Field, Schema

#: Backends compared when a caller does not name any. Chosen for how far apart
#: their type systems are, not for popularity: these four between them cover
#: the integer collapse, the decimal widening, the timezone loss and the
#: outright absence of a type.
DEFAULT_BACKENDS: tuple[str, ...] = ("duckdb", "bigquery", "snowflake", "sqlite")


class UnknownBackend(RowlessError):
    """A backend name ibis has no type mapper for."""


def disagreements(
    schema: Schema,
    backends: Sequence[str] = DEFAULT_BACKENDS,
) -> dict[str, Schema]:
    """The schema each named backend would store, for the ones that differ.

    Args:
        schema: the schema ibis resolved, which is the reference.
        backends: ibis backend names, as `ibis.connect` spells them.

    Returns:
        Backend name to the schema that backend's type system would hold, for
        every named backend whose answer differs from the reference. A backend
        that agrees about every column is left out, so an empty mapping means
        "this expression is portable across the backends you asked about".

    Raises:
        UnknownBackend: a name ibis has no type mapper for, which is a typo
            rather than a portability finding.
    """
    mappers = {name: type_mapper(name) for name in backends}
    found: dict[str, Schema] = {}
    for name, mapper in mappers.items():
        lowered = Schema(
            fields=tuple(
                Field(
                    name=field.name,
                    dtype=round_trip(field.dtype, mapper, backend=name),
                    nullable=field.nullable,
                    origin=field.origin,
                    description=field.description,
                    derived_from=field.derived_from,
                )
                for field in schema.fields
            ),
            name=schema.name,
        )
        if _differs(schema, lowered):
            found[name] = lowered
    return found


def type_mapper(backend: str) -> Any:
    """The ibis type mapper for one backend name.

    Read off `ibis.backends.sql.datatypes` by name rather than imported, so
    this works for every backend the installed ibis ships a mapper for -
    including ones released after this adapter was written - and so none of
    them has to be installed.
    """
    mappers = _mappers()
    found = mappers.get(backend.strip().lower())
    if found is None:
        raise UnknownBackend(
            f"ibis has no type mapper for a backend called {backend!r}; this ibis has "
            f"{', '.join(sorted(mappers))}"
        )
    return found


def round_trip(dtype: DType, mapper: Any, *, backend: str) -> DType:
    """The dtype `backend` would hold for an IR dtype, via its own SQL type.

    Returns UNKNOWN, with the reason, where the backend has no type for it at
    all. That is a portability finding in its own right: a column that cannot
    be stored is worth more attention than one that merely widens.
    """
    from rowless_ibis import dtypes as dtype_map

    try:
        native = dtype_map.from_ir(dtype)
    except RowlessError as exc:
        return unknown(f"ibis cannot express {dtype}: {exc}")
    try:
        rendered = mapper.to_string(native)
        recovered = mapper.from_string(rendered)
    except Exception as exc:
        return unknown(f"{backend} has no type for {dtype} ({type(exc).__name__}: {exc})")
    return dtype_map.to_ir(recovered)


def _differs(reference: Schema, lowered: Schema) -> bool:
    """True when any column's dtype changed on the way through a backend."""
    for field in reference.fields:
        theirs = lowered[field.name].dtype
        if not field.dtype.matches(theirs):
            return True
    return False


def _mappers() -> Mapping[str, Any]:
    """Backend name to ibis type mapper, built once per process."""
    global _CACHED
    if _CACHED is not None:
        return _CACHED
    from ibis.backends.sql import datatypes as sql_types

    found: dict[str, Any] = {}
    for name in dir(sql_types):
        if not name.endswith("Type"):
            continue
        candidate = getattr(sql_types, name)
        if not isinstance(candidate, type):
            continue
        if not hasattr(candidate, "to_string") or not hasattr(candidate, "from_string"):
            continue
        found[name.removesuffix("Type").lower()] = candidate
    _CACHED = found
    return found


_CACHED: dict[str, Any] | None = None
