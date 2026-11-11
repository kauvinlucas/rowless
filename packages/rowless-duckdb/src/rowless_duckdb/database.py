"""The scratch database: empty tables, and the questions asked of them.

Every rowless relation this adapter hands to user code is a real table in a
private in-memory DuckDB, with the declared schema and no rows. That is the
whole trick: the binder then answers questions about it with its own coercion
rules, and nothing has to be reimplemented.

The connection is configured to refuse the outside world:

    enable_external_access = false   no COPY, no read_csv, no httpfs
    lock_configuration     = true    and user code cannot turn it back on

That matters because DuckDB writes through its own C++ filesystem, which the
audit hook in `rowless.runtime.sandbox` never sees - a `COPY ... TO` would
otherwise go straight to disk from inside a dry run. Layer 1 cannot stop it and
layer 2 can, which is exactly the division of labour the sandbox module
describes. The database is in memory, so the writes it does permit land
nowhere.

One connection per run, rebuilt when the run id changes, closed with the run.
A dry run is one pipeline, and keeping the tables of the last one around would
let a missing-input bug resolve against a stale relation.

**Why `run_sql` rebuilds its inputs rather than registering them.** The inputs
arrive as engine objects, and a `DuckDBPyRelation` belongs to the connection
that made it: registering one from somebody else's connection would put the
view there instead of here. So each input is described and re-created as an
empty table in this database, under every name the SQL might refer to it by -
including the three-part `"db"."schema"."table"` that dbt compiles, which is
attached and schema-created as needed. Zero rows is what makes that free.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from rowless.errors import UnknownSchemaError

from rowless_duckdb import session
from rowless_duckdb.dtypes import quote

#: Settings the scratch connection is opened with. Strings rather than bools:
#: DuckDB's config dictionary takes the SQL spelling.
SCRATCH_CONFIG: Mapping[str, str] = {
    "enable_external_access": "false",
    "lock_configuration": "true",
}

#: The virtual table name `DESCRIBE` borrows when reading nullability off a
#: relation. Never referenced by the query it runs; it only has to not collide
#: with a real relation the query does mention.
DESCRIBE_ALIAS = "rowless_described"

#: Prefix for every name this module invents, so a user reading a resolved SQL
#: string can tell which relations rowless put there.
PREFIX = "rowless"

_CONNECTION: Any | None = None

#: Table names already used in this run, so two inputs called `orders` do not
#: become one table.
_TAKEN: set[str] = set()

#: Catalogs attached in this run, by name.
_ATTACHED: set[str] = set()


@dataclass(frozen=True, slots=True)
class Column:
    """One column of a resolved relation, as DuckDB reported it."""

    name: str
    dtype: Any
    """A `DuckDBPyType` when it came off a relation, a type string when it came
    out of `DESCRIBE`. `rowless_duckdb.dtypes.to_ir` reads both."""

    nullable: bool = True


def connection() -> Any:
    """The scratch connection for this run, created on first use."""
    global _CONNECTION

    if _CONNECTION is None:
        _CONNECTION = _connect()
    return _CONNECTION


def reset() -> None:
    """Close the scratch database and forget what was in it. A new run starts."""
    global _CONNECTION

    closing, _CONNECTION = _CONNECTION, None
    _TAKEN.clear()
    _ATTACHED.clear()
    if closing is None:
        return
    # A connection that will not close is a leak of a few kilobytes in a
    # process that is about to exit. Failing the run over it would be worse.
    with suppress(Exception):
        closing.close()


session.on_reset(reset)


def empty_table(columns: Sequence[Column], *, name_hint: str | None = None) -> Any:
    """Create a table with these columns and no rows, and return it as a relation.

    The returned object is a genuine `DuckDBPyRelation`: user code calls
    `project`, `filter` and `aggregate` on it, and DuckDB binds every one of
    them against the declared schema.
    """
    if not columns:
        raise UnknownSchemaError(
            "DuckDB has no zero-column relation, so a schema with no columns cannot be "
            "materialized for it"
        )
    name = _unique_name(name_hint)
    conn = connection()
    with session.own_work():
        conn.execute(f"CREATE TABLE {quote(name)} ({_column_list(columns)})")
        return conn.table(name)


def create_relation(path: str, columns: Sequence[Column]) -> None:
    """Create an empty table at `path`, attaching and creating what it needs.

    `path` is a name a SQL string might refer to a relation by, so it may be
    bare (`stg_orders`), schema-qualified, or the fully qualified and quoted
    form dbt compiles `ref()` into. The catalog is attached in memory and the
    schema created when the path asks for them, because a compiled dbt model
    names relations that only exist in the warehouse.
    """
    if not columns:
        return
    parts = split_identifier(path)
    if not parts:
        return
    conn = connection()
    with session.own_work():
        if len(parts) >= 3:
            _attach(conn, parts[-3])
        if len(parts) >= 2:
            schema = ".".join(quote(part) for part in parts[-3:-1])
            conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
        qualified = ".".join(quote(part) for part in parts[-3:])
        conn.execute(f"CREATE OR REPLACE TABLE {qualified} ({_column_list(columns)})")


def describe_relation(relation: Any) -> tuple[Column, ...]:
    """Read the schema DuckDB computed for a relation, without executing it.

    Names and types come off the relation itself, so no type string is parsed
    on the way. Nullability needs `DESCRIBE`, which is the one thing a relation
    will not tell you directly; when that cannot be had every column is
    reported nullable rather than guessed at.
    """
    with session.own_work():
        names = tuple(str(name) for name in relation.columns)
        types = tuple(relation.types)
    if len(names) != len(types):  # pragma: no cover - DuckDB reports them together
        raise UnknownSchemaError(
            f"DuckDB reported {len(names)} column names and {len(types)} types for the same "
            "relation"
        )
    nullable = _nullability(relation, expected=len(names))
    return tuple(
        Column(name=name, dtype=dtype, nullable=flag)
        for name, dtype, flag in zip(names, types, nullable, strict=True)
    )


def describe_query(sql: str) -> tuple[Column, ...]:
    """Resolve a SQL string against this database with `DESCRIBE`.

    `DESCRIBE` binds the query and reports its projection without running it,
    which is the credential-free half of what a warehouse would charge for. A
    statement that is not a query - a `CREATE TABLE AS`, say - fails to
    describe rather than executing, which is the behaviour we want from
    something handed arbitrary SQL.
    """
    conn = connection()
    with session.own_work():
        rows = conn.execute(f"DESCRIBE {strip_terminator(sql)}").fetchall()
    return tuple(
        Column(name=str(row[0]), dtype=str(row[1]), nullable=_nullable_flag(row[2])) for row in rows
    )


def serialize_sql(sql: str) -> Mapping[str, Any]:
    """Parse `sql` with DuckDB's own parser and return the AST as JSON.

    `json_serialize_sql` is how this adapter reads a select list without
    depending on sqlglot: the engine that will bind the query is also the one
    that parses it for lineage, so the two cannot disagree about what the SQL
    says. A parse failure comes back as `{"error": true, ...}` rather than
    raising, and the caller treats that as "no lineage".

    Nothing here escapes as an exception, and that is deliberate: lineage is a
    capability rather than a requirement, so a DuckDB that will not answer
    costs the run its traced ratio and nothing else. The parameter is cast
    because 1.0 will not accept an untyped placeholder for this function.
    """
    import json

    conn = connection()
    with session.own_work():
        try:
            rows = conn.execute("SELECT json_serialize_sql(CAST(? AS VARCHAR))", [sql]).fetchall()
        except Exception:
            return {"error": True}
    if not rows or not rows[0] or not rows[0][0]:  # pragma: no cover - always one row
        return {"error": True}
    try:
        parsed: Any = json.loads(str(rows[0][0]))
    except ValueError:  # pragma: no cover - DuckDB emits JSON or an error flag
        return {"error": True}
    return parsed if isinstance(parsed, dict) else {"error": True}


def relation_sql(relation: Any) -> str | None:
    """The SQL text behind a relation, or None when it will not say.

    DuckDB builds this from the relation tree, so it is the engine's own
    rendering of what the user composed and the lineage parser can read it.
    """
    with session.own_work():
        try:
            found = relation.sql_query()
        except Exception:
            return None
    return str(found) if found else None


def split_identifier(path: str) -> tuple[str, ...]:
    """Split a possibly-quoted relation name into its parts.

    `"db"."schema"."table"` and ``db.schema.table`` and `` `db`.`table` `` all
    name the same thing as far as a registration is concerned. Dialects quote
    differently, and dbt hands over whichever its adapter produced.
    """
    parts: list[str] = []
    current: list[str] = []
    quoting: str | None = None
    for character in path:
        if quoting is not None:
            if character == quoting:
                quoting = None
            else:
                current.append(character)
            continue
        if character in '"`[':
            quoting = "]" if character == "[" else character
            continue
        if character == ".":
            parts.append("".join(current).strip())
            current = []
            continue
        current.append(character)
    parts.append("".join(current).strip())
    return tuple(part for part in parts if part)


def strip_terminator(sql: str) -> str:
    """Drop trailing whitespace and semicolons, which `DESCRIBE` will not take."""
    return sql.strip().rstrip(";").strip()


def statement_types(sql: str) -> tuple[str, ...]:
    """The kind of each statement in `sql`, as DuckDB's parser classifies it.

    Used to recognise a write before running it. Parsing is not binding, so
    this says nothing about whether the statement would succeed - only what it
    is. An unparseable string has no statements, and is left to fail where it
    would have failed anyway.
    """
    import duckdb

    with session.own_work():
        try:
            statements = duckdb.extract_statements(sql)
        except Exception:
            return ()
    return tuple(str(getattr(statement, "type", "")).rpartition(".")[2] for statement in statements)


def _connect() -> Any:
    """Open the scratch connection, with as much of the lockdown as it accepts.

    `lock_configuration` has been in DuckDB since well before the declared
    floor, but a setting that a future release renames must not stop the
    adapter working: each one is dropped in turn rather than failing the run.
    """
    import duckdb

    # Typed loosely: DuckDB widened the value type of its config mapping
    # across the supported range, and this dictionary only ever holds strings.
    settings: dict[str, Any] = dict(SCRATCH_CONFIG)
    with session.own_work():
        for _ in range(len(settings) + 1):
            try:
                return duckdb.connect(":memory:", config=settings)
            except Exception:
                if not settings:
                    raise
                dropped, _ = settings.popitem()
                session.note(
                    f"this DuckDB does not accept the {dropped} setting, so the scratch "
                    "database is less locked down than usual",
                    once_for=f"config:{dropped}",
                )
    raise UnknownSchemaError("DuckDB would not open an in-memory database")  # pragma: no cover


def _attach(conn: Any, catalog: str) -> None:
    """Attach an in-memory catalog under `catalog`, once per run."""
    if catalog in _ATTACHED:
        return
    # Already attached, or a name DuckDB will not take. Either way the CREATE
    # that follows reports the real problem.
    with suppress(Exception):
        conn.execute(f"ATTACH ':memory:' AS {quote(catalog)}")
    _ATTACHED.add(catalog)


def _column_list(columns: Sequence[Column]) -> str:
    """The column definitions of a `CREATE TABLE`, with nullability declared.

    DuckDB is the only engine here that tracks `NOT NULL` through a projection,
    so declaring it is not decoration: it is what lets `rowless diff` report a
    nullability change.
    """
    return ", ".join(
        f"{quote(column.name)} {column.dtype}{'' if column.nullable else ' NOT NULL'}"
        for column in columns
    )


def _unique_name(hint: str | None) -> str:
    """A fresh table name, derived from `hint` when it gives one.

    Naming the table after the relation means the SQL DuckDB renders for a
    pipeline mentions `stg_orders` rather than `rowless_1`, which is what a
    user reads in a finding.
    """
    base = _sanitised(hint) or f"{PREFIX}_relation"
    candidate = base
    suffix = 2
    while candidate in _TAKEN:
        candidate = f"{base}_{suffix}"
        suffix += 1
    _TAKEN.add(candidate)
    return candidate


def _sanitised(hint: str | None) -> str:
    """`hint` reduced to something safe to interpolate, or empty."""
    if not hint:
        return ""
    kept = "".join(character if character.isalnum() else "_" for character in hint).strip("_")
    if not kept:
        return ""
    return kept if kept[0].isalpha() or kept[0] == "_" else f"{PREFIX}_{kept}"


def _nullability(relation: Any, *, expected: int) -> tuple[bool, ...]:
    """Per-column nullability, read with `DESCRIBE` through the relation itself.

    `relation.query` runs SQL on the connection the relation belongs to, which
    is how this works without a connection handle and whichever database the
    relation came from. The query describes the relation's own SQL, because
    describing the virtual table instead loses the `NOT NULL` that the point
    of asking.
    """
    sql = relation_sql(relation)
    if sql is None:
        return (True,) * expected
    with session.own_work():
        try:
            described = relation.query(DESCRIBE_ALIAS, f"DESCRIBE {strip_terminator(sql)}")
            rows = described.fetchall() if described is not None else []
        except Exception:
            # Nullability is the one thing here that is nice to have. A
            # relation whose SQL will not re-bind - one built over a registered
            # Python object, say - reports every column nullable.
            return (True,) * expected
    if len(rows) != expected:
        return (True,) * expected
    return tuple(_nullable_flag(row[2]) for row in rows)


def _nullable_flag(value: Any) -> bool:
    """Read the `null` column of a `DESCRIBE` row: `NO` means not nullable."""
    return str(value).strip().upper() != "NO"


def columns_of(relation: Any) -> Iterator[str]:
    """The column names of a relation, or nothing when it will not say.

    Used for the near-miss suggestion on a missing column: the relation that
    was projected knows the valid names even when the error message does not
    list them.
    """
    with session.own_work():
        try:
            names = list(relation.columns)
        except Exception:
            return
    yield from (str(name) for name in names)
