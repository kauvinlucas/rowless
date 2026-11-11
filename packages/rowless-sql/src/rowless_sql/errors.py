"""Turning sqlglot failures into rowless diagnostics.

sqlglot fails in two different registers, and the whole value of this module is
keeping them apart:

* **the query says something wrong.** `SELECT amount_cents FROM orders` against
  a table that renamed the column is the flagship finding, and it has to arrive
  as `ColumnNotFound` with the column named and a near miss suggested. sqlglot
  reports it as `OptimizeError: Column 'amount_cents' could not be resolved`,
  with no list of what the table *does* have, so the alternatives come from the
  catalogue rather than from the message.
* **rowless could not read the query.** A `ParseError` is not evidence that a
  model is broken: valid Snowflake SQL fails to parse as generic SQL in exactly
  this way, and a dialect that was not named is the most common reason a dbt
  model will not read. Those are `UnknownSchemaError`s, so the result is
  reported as unverified - rowless does not know what the query returns - and
  the message says which dialect to set.

Two findings here are produced by this adapter rather than translated, because
sqlglot's own answer conflates them with something else:

* **an ambiguous column.** `SELECT created_at FROM orders JOIN customers` is
  reported by sqlglot with the same "could not be resolved" message as a column
  that exists nowhere. They are different problems with different fixes, so the
  catalogue is asked which tables have a column of that name, and a column two
  inputs both have is reported as ambiguous. Newer sqlglot releases do
  distinguish the two, and that message is recognised as well.
* **a set operation whose sides disagree.** sqlglot resolves `SELECT a FROM t
  UNION ALL SELECT b, c FROM u` without complaint and reports the left side's
  columns. A union of two different shapes is not a query any engine will run,
  so `rowless_sql.resolve` checks the arity itself and reports it here.

Anything this module cannot classify is **not** translated. It propagates, and
core reports it under `PIPELINE_RAISED` with the sqlglot message as the detail.
Mislabelling a failure the adapter does not understand would be worse than
letting core say so plainly.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import NoReturn

from rowless.diagnostics.codes import Code
from rowless.errors import (
    ColumnNotFound,
    RowlessError,
    SchemaError,
    UnknownSchemaError,
)

#: `Column 'amount_cents' could not be resolved.` The name arrives quoted on
#: its own in current releases and double-quoted inside the quotes on older
#: ones (`Column '"amount_cents"'`), so the identifier quoting is stripped
#: after the match rather than written into the pattern.
_UNRESOLVED_COLUMN = re.compile(r"Column '([^']*)' could not be resolved")

#: `Ambiguous column 'created_at' (Line: 1, Col: 8)`, which newer sqlglot
#: releases report in place of the message above when a name is in two sources.
_AMBIGUOUS_COLUMN = re.compile(r"Ambiguous column '([^']*)'")

#: `Unknown column: amount_cents`, and the ORDER BY form of it.
_UNKNOWN_COLUMN = re.compile(r"Unknown (?:output )?column: ([^\s,]+)")

#: `Unknown table: orders`
_UNKNOWN_TABLE = re.compile(r"Unknown table: ([^\s,]+)")

#: `Cannot resolve 'amount' due to missing source schemas`
_MISSING_SOURCES = re.compile(r"Cannot resolve '([^']*)' due to missing source schemas")

#: `Invalid set operation due to column mismatch: SELECT ...`
_SET_OPERATION = re.compile(r"Invalid set operation due to column mismatch")

#: `Cannot automatically join: created_at`
_CANNOT_JOIN = re.compile(r"Cannot automatically join: ([^\s,]+)")

#: sqlglot underlines the offending token with ANSI escapes, which belong in a
#: terminal that printed the exception and not in a rowless finding.
_ANSI = re.compile(r"\x1b\[[0-9;]*m")

#: Characters a dialect may quote an identifier with, stripped off a name
#: lifted out of a message.
_QUOTES = '"`[]'


class AmbiguousColumn(SchemaError):
    """A column name two of the query's sources both have.

    Reported under `AMBIGUOUS_COLUMN` rather than `COLUMN_NOT_FOUND`: the
    column exists, and the fix is to qualify the reference rather than to go
    looking for a rename.
    """

    code = Code.AMBIGUOUS_COLUMN


class SetOperationMismatch(SchemaError):
    """A `UNION`, `EXCEPT` or `INTERSECT` whose sides are different shapes."""

    code = Code.DTYPE_MISMATCH


class SqlNotParsable(UnknownSchemaError):
    """sqlglot could not read the query, which is not proof that it is wrong.

    A dialect's own syntax fails this way when the dialect was not named, so
    this reads as unverified: rowless has no idea what the query returns, and
    saying the model is broken would be a false positive on valid SQL.
    """


class TableNotRegistered(UnknownSchemaError):
    """The query reads a table nothing registered a schema for.

    Unverified rather than an error for the same reason: the table very likely
    exists in the warehouse, and rowless resolves queries against the schemas
    it was given. The message names what *was* registered, which is what makes
    a missing `ref()` or a `[sources]` rewrite findable.
    """


class UnnamedProjection(UnknownSchemaError):
    """The query projects an expression nobody aliased.

    sqlglot names one of those `_col_0`, which is its own placeholder rather
    than any warehouse's answer: DuckDB would call the column `sum(amount)`,
    Spark `_col0` and Postgres `sum`. Reporting a name no engine uses is worse
    than reporting none, so the query reads as unverified and the message says
    to alias the expression.
    """


class StatementNotAQuery(UnknownSchemaError):
    """Something other than a query was handed to the SQL seam.

    A dry run resolves what a query returns; an `INSERT` or a `CREATE TABLE`
    has no output schema to report, and rowless will not pretend the statement
    was analysed.
    """


def translate(
    exc: BaseException,
    *,
    available: Sequence[str] = (),
    ambiguous: Mapping[str, Sequence[str]] | None = None,
    registered: Sequence[str] = (),
    dialect: str = "",
) -> RowlessError | None:
    """Return the rowless error for a sqlglot exception, or None to leave it alone.

    Args:
        exc: the exception sqlglot raised while parsing, qualifying or
            annotating a query.
        available: every column the registered inputs have, for the near-miss
            suggestion. sqlglot does not list them in its message, so a
            `ColumnNotFound` without this is half a finding.
        ambiguous: column name to the registered inputs that have one, for
            the names more than one input has. A column in here is reported as
            ambiguous rather than missing, and the tables are named in the
            message.
        registered: the table names rowless registered, quoted in the message
            when the query reads one it did not.
        dialect: the dialect the query was parsed in, named in a parse failure
            so the fix is in the message.

    Returns:
        A `RowlessError` when the failure is one this adapter can classify, and
        None when it is not - which means the caller should let the original
        exception propagate.
    """
    if isinstance(exc, RowlessError):
        # Already translated, further in. Re-wrapping would lose the column
        # name the inner one carries.
        return None
    if not is_sqlglot_error(exc):
        return None

    message = _clean(str(exc))
    if _is_parse_error(exc):
        return SqlNotParsable(
            f"sqlglot could not parse this query: {message}. {_dialect_hint(dialect)}"
        )

    unknown_table = _UNKNOWN_TABLE.search(message)
    if unknown_table is not None:
        return _no_such_table(_unquote(unknown_table.group(1)), registered)
    missing_sources = _MISSING_SOURCES.search(message)
    if missing_sources is not None:
        return TableNotRegistered(
            f"sqlglot could not attribute the column {_unquote(missing_sources.group(1))!r} "
            "because one of this query's sources has no registered schema. "
            f"Registered: {_listed(registered)}"
        )
    if _SET_OPERATION.search(message) is not None:
        return SetOperationMismatch(
            f"this query's set operation combines two different shapes: sqlglot reports {message}"
        )
    cannot_join = _CANNOT_JOIN.search(message)
    if cannot_join is not None:
        return SqlNotParsable(
            f"sqlglot could not resolve a join in this query ({message}); its schema is "
            "reported as unverified rather than guessed at"
        )

    shared = dict(ambiguous or {})
    ambiguity = _AMBIGUOUS_COLUMN.search(message)
    if ambiguity is not None:
        column = _unquote(ambiguity.group(1))
        return _ambiguous(column, shared.get(column, ()))
    unresolved = _UNRESOLVED_COLUMN.search(message) or _UNKNOWN_COLUMN.search(message)
    if unresolved is not None:
        column = _unquote(unresolved.group(1))
        if column in shared:
            return _ambiguous(column, shared[column])
        return ColumnNotFound(column, available=tuple(available))
    return None


def raise_translated(
    exc: BaseException,
    *,
    available: Sequence[str] = (),
    ambiguous: Mapping[str, Sequence[str]] | None = None,
    registered: Sequence[str] = (),
    dialect: str = "",
) -> NoReturn:
    """Re-raise `exc` as a rowless error when it is one, else as it is.

    The one-line form used at every call site that parses, qualifies or
    annotates a query.
    """
    translated = translate(
        exc,
        available=available,
        ambiguous=ambiguous,
        registered=registered,
        dialect=dialect,
    )
    if translated is not None:
        raise translated from exc
    raise exc


def is_sqlglot_error(exc: BaseException) -> bool:
    """True when `exc` came out of sqlglot rather than out of user code.

    Checked by class hierarchy first and by module second, so a user exception
    that happens to share a name with one of sqlglot's is not mistaken for a
    resolution failure, and a sqlglot error class this adapter has never heard
    of is still recognised.
    """
    try:
        from sqlglot.errors import SqlglotError
    except ImportError:  # pragma: no cover - only without sqlglot
        return False
    if isinstance(exc, SqlglotError):
        return True
    return type(exc).__module__.startswith("sqlglot")


def no_such_table(name: str, registered: Sequence[str]) -> TableNotRegistered:
    """The finding for a query that reads a table rowless has no schema for."""
    return _no_such_table(name, registered)


def _no_such_table(name: str, registered: Sequence[str]) -> TableNotRegistered:
    """Build the missing-table finding, naming what is registered instead."""
    return TableNotRegistered(
        f"this query reads the table {name!r}, and nothing registered a schema for it. "
        f"Registered: {_listed(registered)}. A dbt model gets these from its `ref()`s; a "
        "Python pipeline from the inputs it was given"
    )


def _ambiguous(column: str, holders: Sequence[str]) -> AmbiguousColumn:
    """Build the ambiguous-column finding, naming the tables that have it."""
    where = _listed(holders) if holders else "more than one of this query's sources"
    return AmbiguousColumn(
        f"the column {column!r} is ambiguous: {where} each have one, so SQL cannot tell which "
        f"this query means. Qualify the reference, e.g. {_qualified(column, holders)}"
    )


def _qualified(column: str, holders: Sequence[str]) -> str:
    """A worked example of the fix, using the first table that has the column."""
    first = next(iter(holders), "t")
    return f"{first}.{column}"


def _listed(names: Sequence[str]) -> str:
    """Names for a message, or an honest word for having none."""
    return ", ".join(names) if names else "nothing"


def _dialect_hint(dialect: str) -> str:
    """What to try next after a parse failure, which depends on the dialect."""
    if dialect:
        return (
            f"It was parsed as {dialect}, which is what [engine.sql] dialect says; a query "
            "written for another warehouse needs that setting changed"
        )
    return (
        "It was parsed as generic SQL. A dialect's own syntax needs [engine.sql] dialect set in "
        "rowless.toml - or, for a dbt project, it is read from the manifest's adapter"
    )


def _is_parse_error(exc: BaseException) -> bool:
    """True for a sqlglot parse or tokenise failure."""
    try:
        from sqlglot.errors import ParseError, TokenError
    except ImportError:  # pragma: no cover - only without sqlglot
        return False
    return isinstance(exc, ParseError | TokenError)


def _clean(message: str) -> str:
    """One line, with sqlglot's terminal highlighting taken out."""
    return re.sub(r"\s+", " ", _ANSI.sub("", message)).strip().rstrip(".")


def _unquote(name: str) -> str:
    """A column or table name as the schema knows it, with quoting stripped."""
    return name.strip().strip(_QUOTES)
