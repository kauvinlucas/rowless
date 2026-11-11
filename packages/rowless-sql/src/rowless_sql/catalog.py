"""The catalogue of input schemas a query resolves against.

This adapter's equivalent of the DuckDB adapter's scratch database. DuckDB
creates an empty table per input and asks its own binder; there is no engine
here, so an input is a name plus the types sqlglot should answer with, and
`sqlglot.optimizer.qualify` resolves the query against exactly that.

Three rules, and each one is a decision about what to do when a name is
ambiguous rather than a detail of the data structure:

* **Names are matched on the final identifier.** dbt compiles `ref()` into a
  fully qualified `"db"."schema"."table"`, and `rowless-dbt` offers rowless
  every spelling of a node's name. A `MappingSchema` has to have one nesting
  depth throughout, and a flat one keyed by table name resolves a qualified
  reference as well as a bare one - which is what makes `FROM
  "analytics"."staging"."stg_orders"` and `FROM stg_orders` both resolve
  against one entry. So the keys here are the last identifier of each input,
  unquoted.
* **A collision between different schemas is reported, not resolved.** Two
  inputs whose final identifier is the same and whose schemas differ cannot
  both be `orders`, and picking one would resolve the query against the wrong
  table. The second is dropped and said so, which makes the query resolve
  wrongly far less often than it makes it fail to resolve - and a failure is
  reported as unverified rather than as a broken model.
* **`materialize` never collides at all.** A frame is registered under a name
  nobody else in the run is using, so two inputs that resolved from
  identically named sources stay two tables and a self-join is a self-join.
  The name is what `str()` of the frame renders, so an f-string SQL model
  refers to the right one without the pipeline having to know.

Nothing here imports sqlglot at module scope: an adapter has to be importable
by a user who has none of the engines it targets.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from rowless.ir.schema import Schema

#: Characters a dialect may quote an identifier with.
_QUOTES = '"`[]'

#: `SELECT ...;` - a trailing terminator is legal in a file and not in a
#: statement sqlglot is asked to parse on its own.
_TERMINATOR = re.compile(r";\s*$")

#: The name a frame is registered under when its schema has none. A query that
#: refers to it by this name still resolves, which is what makes an unnamed
#: input usable rather than only diagnosable.
DEFAULT_TABLE = "rowless_frame"


@dataclass(frozen=True, slots=True)
class Table:
    """One registered input: the name a query may use, and what it holds.

    Both halves of the schema are kept. The IR `Schema` is what `describe`
    answers with for the frame itself, nullability included - it is the input,
    so its declared nullability is the engine's own answer rather than a guess.
    The sqlglot column types are what a query is resolved against, and they are
    built once by the adapter, which is where a dtype SQL cannot express is
    degraded and reported.
    """

    name: str
    schema: Schema
    columns: Mapping[str, Any]
    """Column name to `sqlglot.exp.DataType`, in schema order."""


@dataclass(slots=True)
class Catalog:
    """Registered inputs for one dry run, keyed by table name.

    Mutable by design: `materialize` is called once per input before user code
    runs, and `run_sql` registers whatever names its caller offers. One
    catalogue per run, replaced rather than cleared - see
    `rowless_sql.session`.
    """

    tables: dict[str, Table] = field(default_factory=dict)

    def __contains__(self, name: object) -> bool:
        """True when a query may refer to this name, qualified or not."""
        return bare_name(str(name)) in self.tables

    def __len__(self) -> int:
        return len(self.tables)

    def __iter__(self) -> Iterator[Table]:
        """Registered tables, in registration order."""
        return iter(self.tables.values())

    def register(self, schema: Schema, columns: Mapping[str, Any]) -> Table:
        """Record a new input under a name no other table in this run is using.

        What `materialize` calls. The suffix on a colliding name is what keeps
        two inputs that resolved from sources of the same name distinguishable
        in the SQL a pipeline writes.
        """
        base = bare_name(schema.name or DEFAULT_TABLE) or DEFAULT_TABLE
        name = base
        attempt = 1
        while name in self.tables:
            name = f"{base}_{attempt}"
            attempt += 1
        table = Table(name=name, schema=schema, columns=dict(columns))
        self.tables[name] = table
        return table

    def add(self, name: str, schema: Schema, columns: Mapping[str, Any]) -> str | None:
        """Record an input under the name a caller asked for.

        What `run_sql` calls: the name is the one the SQL uses, so it is taken
        as given rather than made unique.

        Returns:
            None when the input was recorded or was already there under an
            identical schema, and a collision message when it was dropped
            because the name is taken by a different schema. The caller
            degrades the run with it, so a query that meant the other table is
            reported as unverified rather than resolved against the wrong one.
        """
        key = bare_name(name)
        existing = self.tables.get(key)
        if existing is not None and existing.schema != schema:
            return (
                f"two SQL inputs resolve to the table name {key!r} with different schemas "
                f"({name!r} is the second); rowless kept the first, so a query that meant the "
                "other one will not resolve"
            )
        if existing is None:
            self.tables[key] = Table(name=key, schema=schema, columns=dict(columns))
        return None

    def get(self, name: str) -> Table | None:
        """The table registered under `name`, qualified or not, or None."""
        return self.tables.get(bare_name(name))

    def names(self) -> tuple[str, ...]:
        """Every registered table name, sorted, for a diagnostic to quote."""
        return tuple(sorted(self.tables))

    def columns(self) -> tuple[str, ...]:
        """Every column name any registered input has, sorted and deduplicated.

        What a near-miss suggestion is drawn from: sqlglot reports an
        unresolved column by name and does not say what the alternatives were.
        """
        found = {column for table in self.tables.values() for column in table.schema}
        return tuple(sorted(found))

    def mapping(self, dialect: str = "") -> Any:
        """The `sqlglot.schema.MappingSchema` a query is resolved against.

        Built fresh per call rather than cached: a catalogue gains entries
        while a run is open, and a schema object that had been handed to
        sqlglot already would not see them.
        """
        from sqlglot.schema import MappingSchema

        return MappingSchema(
            {name: dict(table.columns) for name, table in self.tables.items()},
            dialect=dialect or None,
        )


def bare_name(name: str) -> str:
    """The final identifier of a possibly qualified, possibly quoted table name."""
    parts = [part.strip().strip(_QUOTES) for part in str(name).split(".")]
    final = [part for part in parts if part]
    return final[-1] if final else str(name)


def strip_terminator(sql: str) -> str:
    """Drop a trailing semicolon, which a compiled dbt model often carries."""
    return _TERMINATOR.sub("", sql.strip())


def tables_read(expression: Any) -> tuple[str, ...]:
    """Every table a query reads, by bare name, excluding its own CTEs.

    Read off sqlglot's scope tree rather than by walking for `exp.Table` nodes,
    because a reference to a CTE *is* an `exp.Table` and attributing one to a
    missing input is how a tool reports a model as broken when it is fine.
    Each scope knows which of its sources are tables and which are scopes of
    their own, and only the first kind has to be registered.
    """
    from sqlglot import exp
    from sqlglot.optimizer.scope import build_scope

    root = build_scope(expression)
    if root is None:
        # Not a query sqlglot can scope - a DDL statement, or something it
        # parsed into a shape with no SELECT in it. The caller has already
        # refused those; returning nothing keeps this honest rather than
        # guessing from the text.
        return ()
    found: list[str] = []
    for scope in root.traverse():
        for source in scope.sources.values():
            # `name` is the table's own identifier, with any catalog, schema
            # and alias left off, which is the key this catalogue uses.
            if isinstance(source, exp.Table) and source.name:
                found.append(str(source.name))
    return tuple(dict.fromkeys(found))
