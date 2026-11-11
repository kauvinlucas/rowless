"""The Python-to-SQL seam, resolved with no database behind it.

`SupportsSql` is what lets a SQL model and a Python pipeline land in one
lineage graph. DuckDB implements it by creating empty tables and asking its own
binder; this adapter cannot do that, because ibis has no engine of its own -
and it turns out not to need one. `ibis.parse_sql` reads a query against a
catalog of schemas and returns an ordinary unbound ibis expression, so the
output schema comes out of ibis's own type rules with no credentials, no
warehouse and nothing executed.

That makes this the credential-free SQL path the roadmap wanted from
`rowless-sql`, reached from the other direction: sqlglot parses and resolves
the query (it arrives with ibis, which is built on it), and ibis types the
result. What comes back is a real expression, so the lineage walk in
`rowless_ibis.lineage` traces a SQL model's columns exactly as it traces a
Python pipeline's.

Two things are worth knowing about the catalog:

* **Names are matched on the final identifier.** dbt compiles `ref()` into a
  fully qualified `"db"."schema"."table"`, and `rowless-dbt` offers rowless
  every spelling of a node's name. sqlglot resolves a qualified reference
  against a flat catalog keyed by table name, so the keys here are the last
  identifier of each input, unquoted - which matches the bare name, the
  unquoted qualified name and the quoted one, all three.
* **A collision is reported, not resolved.** Two inputs whose final identifier
  is the same and whose schemas differ cannot both be `orders`, and picking one
  would resolve the query against the wrong table. The second one is dropped
  and said so, which makes the query fail to resolve rather than resolve
  wrongly.

`ibis.parse_sql` is marked experimental upstream. It is used here anyway,
because the alternative is a second SQL resolver maintained in this repository,
and `UNSUPPORTED_SQL` records what it cannot read so a failure reports as
unverified rather than as a broken pipeline.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from rowless.errors import UnknownSchemaError

from rowless_ibis import errors

#: Characters a dialect may quote an identifier with.
_QUOTES = '"`[]'

#: `SELECT ...;` - a trailing terminator is legal in a file and not in a
#: statement sqlglot is asked to parse on its own.
_TERMINATOR = re.compile(r";\s*$")


def strip_terminator(sql: str) -> str:
    """Drop a trailing semicolon, which a compiled dbt model often carries."""
    return _TERMINATOR.sub("", sql.strip())


def catalog_for(schemas: Mapping[str, Any]) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Build the parse catalog for `schemas`, keyed the way sqlglot resolves names.

    Args:
        schemas: input name (bare, qualified, or quoted and qualified) to the
            ibis schema rowless resolved for it.

    Returns:
        The catalog to hand to `ibis.parse_sql`, and one note per input that
        was dropped because its name collided with another input's.
    """
    catalog: dict[str, Any] = {}
    collisions: list[str] = []
    for name, schema in schemas.items():
        key = bare_name(name)
        existing = catalog.get(key)
        if existing is None:
            catalog[key] = schema
            continue
        if existing != schema:
            collisions.append(
                f"two SQL inputs resolve to the table name {key!r} with different schemas "
                f"({name!r} is the second); rowless kept the first, so a query that meant the "
                "other one will not resolve"
            )
    return catalog, tuple(collisions)


def bare_name(name: str) -> str:
    """The final identifier of a possibly qualified, possibly quoted table name."""
    parts = [part.strip().strip(_QUOTES) for part in str(name).split(".")]
    final = [part for part in parts if part]
    return final[-1] if final else str(name)


def resolve(statement: str, catalog: Mapping[str, Any], *, dialect: str | None = None) -> Any:
    """Parse `statement` against `catalog` and return the ibis expression for it.

    Raises:
        rowless.errors.ColumnNotFound: the query reads a column no input has.
        rowless.errors.UnknownSchemaError: the query is one ibis's parser
            cannot read, so its schema is unverified rather than wrong.
    """
    import ibis

    available = sorted({column for schema in catalog.values() for column in _columns(schema)})
    try:
        return ibis.parse_sql(statement, dict(catalog), dialect=dialect)
    except Exception as exc:
        translated = errors.translate(exc) or errors.translate_sql_error(exc, available=available)
        if translated is not None:
            raise translated from exc
        registered = ", ".join(sorted(catalog)) or "none"
        raise errors.SqlNotResolvable(
            f"ibis could not resolve this query without a backend "
            f"({type(exc).__name__}: {exc}); the tables rowless registered for it were: "
            f"{registered}. A dialect's own syntax needs [engine.ibis] dialect set in "
            "rowless.toml for sqlglot to read it"
        ) from exc


def named_relations(expr: Any) -> dict[str, Any]:
    """Every named relation an expression reads from, as name to ibis schema.

    Used by the `Table.sql()` interception: a query handed to `.sql()` names its
    table the way the table is named - the alias from `Table.alias(...)`, or the
    table's own name - so those are the names the catalog has to answer to.
    """
    import ibis.expr.operations as ops
    import ibis.expr.types as ir

    if not isinstance(expr, ir.Expr):
        return {}
    found: dict[str, Any] = {}
    for node in expr.op().find(ops.Relation):
        name = getattr(node, "name", None)
        schema = getattr(node, "schema", None)
        if isinstance(name, str) and name and schema is not None and name not in found:
            found[name] = schema
    return found


def schema_of(value: Any) -> Any:
    """The ibis schema of an input to `run_sql`, or None when it is not one."""
    import ibis.expr.types as ir

    if isinstance(value, ir.Table):
        return value.schema()
    return None


def require_schema(value: Any, name: str) -> Any:
    """The ibis schema of `value`, refusing an input this adapter cannot read."""
    schema = schema_of(value)
    if schema is None:
        qualified = f"{type(value).__module__}.{type(value).__qualname__}"
        raise UnknownSchemaError(
            f"the ibis adapter cannot register {qualified} as the SQL input {name!r}; pass an "
            "ibis table or a rowless Schema"
        )
    return schema


def _columns(schema: Any) -> tuple[str, ...]:
    """Column names of an ibis schema, for the near-miss suggestion."""
    return tuple(str(name) for name in getattr(schema, "names", ()))
