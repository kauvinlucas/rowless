"""Resolving a query's output schema with no database behind it.

The heart of the adapter. `sqlglot.optimizer.qualify` resolves every identifier
in a query against the registered input schemas - expanding `SELECT *`,
attributing an unqualified column to the source that has it, and failing
loudly when nothing does - and `annotate_types` then types the projection
using sqlglot's own dialect-aware rules. No engine runs, nothing is connected
to, and no credential is needed, which is the property that makes this adapter
the one that works in a CI job with no warehouse access.

What this module adds on top of sqlglot, and why each piece is here rather
than left to the library:

* **Set operations are unified through the IR lattice.** sqlglot reports a
  union's columns as the left-hand side's, types included. That is right for
  the names and wrong for the types: a `VARCHAR` column unioned with a
  `BIGINT` one is not a `VARCHAR`. Each side is resolved separately and the
  positions are combined with `rowless.ir.dtypes.unify`, which widens where
  the engines agree and reports UNKNOWN where no common type exists. A
  mismatch in the *number* of columns is reported as a finding, because
  sqlglot resolves that query without complaint and no engine would run it.
* **An unannotated expression becomes an ANY column with a note.** sqlglot
  annotates what it knows and leaves `UNKNOWN` behind for the rest - a
  warehouse-specific function, a UDF, anything its rules do not cover. ANY is
  the gradual-typing answer: it lowers the typed-coverage score and warns,
  rather than failing a check that may well be correct. The note names the
  function, so a user can see which one cost them the column.
* **A duplicate output name is refused.** `SELECT *` over a join whose tables
  share a column name produces that name twice. SQL allows it; a rowless
  schema is addressed by name and so is everything downstream, so the
  duplicate is reported with the fix in the message.
* **A table nothing registered is reported before qualification.** sqlglot
  resolves `SELECT nope.amount FROM nope` happily, because a qualified column
  against an unknown table is syntactically fine. Checking the catalogue first
  turns that into a finding that names the table, instead of a schema that
  reports a column nobody can vouch for.
* **A folded identifier gets its declared spelling back.** Qualification
  normalises unquoted identifiers the way the dialect does - lower for
  Postgres and generic SQL, upper for Snowflake - so an input column declared
  `CustomerID` comes back `customerid` from `SELECT *`. Comparing that against
  the schema it came from would report a rename nobody made, so an output name
  that matches exactly one input column case-insensitively is reported under
  the input's own spelling. An *alias* the query wrote stays folded, because
  that is genuinely what the warehouse will call the column.

The one thing this module cannot do is **type checking**. sqlglot has no
binder: `SELECT amount + status` annotates as `BIGINT` and nothing complains,
where DuckDB refuses the expression outright. That is the trade this adapter
exists to make, it is written down in the adapter's docstring and asserted by
a conformance case, and it is why `rowless check` prefers a real binder when
one is installed.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from rowless.errors import DuplicateColumn
from rowless.ir.dtypes import DType, any_, unify

from rowless_sql import dtypes as dtype_map
from rowless_sql import errors
from rowless_sql.catalog import Catalog, strip_terminator, tables_read

#: The name qualification gives a projection nobody aliased. sqlglot's own
#: placeholder, not any warehouse's answer - see `_require_names`.
_PLACEHOLDER_NAME = re.compile(r"^_col_\d+$")


@dataclass(frozen=True, slots=True)
class Column:
    """One resolved output column, and anything rowless had to say about it."""

    name: str
    dtype: DType

    note: str | None = None
    """Why this column is ANY, or how its dtype was arrived at. Surfaced by
    `rowless explain`."""

    lookup: str | None = None
    """The name sqlglot gave this column, when `name` is an input's declared
    spelling instead. Lineage looks the column up by this one, since that is
    what the resolved projection calls it."""

    @property
    def resolved_name(self) -> str:
        """The name the resolved query uses, which is what lineage is keyed by."""
        return self.lookup or self.name

    @property
    def restored(self) -> bool:
        """True when a folded identifier was reported under its declared spelling."""
        return self.lookup is not None


def parse(sql: str, *, dialect: str = "") -> Any:
    """Parse one SQL statement into a sqlglot expression.

    Args:
        sql: the query. A trailing semicolon is dropped, which a compiled dbt
            model often carries.
        dialect: the sqlglot dialect to read it in. See `rowless_sql.dialects`.

    Returns:
        The parsed `sqlglot.exp.Expression`.

    Raises:
        rowless_sql.errors.SqlNotParsable: sqlglot could not read it, which is
            reported as unverified rather than as a broken query.
    """
    import sqlglot

    statement = strip_terminator(sql)
    if not statement:
        raise errors.SqlNotParsable("rowless was given an empty SQL string to resolve")
    try:
        parsed = sqlglot.parse_one(statement, dialect=dialect or None)
    except Exception as exc:
        errors.raise_translated(exc, dialect=dialect)
    if parsed is None:  # pragma: no cover - sqlglot raises rather than returning None
        raise errors.SqlNotParsable(f"sqlglot parsed nothing from {statement[:60]!r}")
    return parsed


def require_query(expression: Any) -> Any:
    """Return the query `expression` is, refusing anything that is not one.

    A dry run resolves what a query returns. An `INSERT`, a `CREATE TABLE AS`
    or a `MERGE` has no output schema to report, and resolving the select
    buried inside one would report a schema for a statement whose job is to
    write - so it is refused, and the refusal reads as unverified.
    """
    from sqlglot import exp

    unwrapped = expression.unnest() if isinstance(expression, exp.Subquery) else expression
    if isinstance(unwrapped, exp.Query):
        return unwrapped
    kind = type(expression).__name__.upper()
    raise errors.StatementNotAQuery(
        f"rowless resolves queries, and this is a {kind} statement: a dry run has no output "
        "schema to report for something whose job is to write"
    )


def resolved_expression(expression: Any, catalog: Catalog, *, dialect: str = "") -> Any:
    """Qualify and type-annotate a copy of a query.

    Copied before anything touches it: qualification rewrites identifiers in
    place, and the expression may be an object the pipeline still holds.

    This is also what lineage is read off - `sqlglot.lineage` resolves an
    unqualified column against the schema itself, and handing it the
    already-qualified expression keeps the two from disagreeing about which
    source a name came from.

    Raises:
        rowless.errors.ColumnNotFound: the query reads a column no input has.
        rowless_sql.errors.AmbiguousColumn: an unqualified column more than
            one source has.
        rowless_sql.errors.TableNotRegistered: the query reads a table nothing
            registered.
        rowless_sql.errors.SqlNotParsable: sqlglot could not resolve it, which
            is reported as unverified.
    """
    from sqlglot.optimizer.annotate_types import annotate_types
    from sqlglot.optimizer.qualify import qualify

    query = require_query(expression)
    for name in tables_read(query):
        if name not in catalog:
            raise errors.no_such_table(name, catalog.names())

    schema = catalog.mapping(dialect)
    try:
        qualified = qualify(query.copy(), schema=schema, dialect=dialect or None)
        return annotate_types(qualified, schema=schema, dialect=dialect or None)
    except Exception as exc:
        errors.raise_translated(
            exc,
            available=catalog.columns(),
            ambiguous=ambiguous_columns(catalog),
            registered=catalog.names(),
            dialect=dialect,
        )


def columns_of(resolved: Any, catalog: Catalog) -> tuple[Column, ...]:
    """Read the output columns off an already-resolved query.

    Args:
        resolved: the expression `resolved_expression` returned.
        catalog: the inputs it was resolved against, for restoring a declared
            identifier spelling the dialect folded.

    Returns:
        One `Column` per output column, in projection order.

    Raises:
        rowless.errors.DuplicateColumn: two output columns share a name.
        rowless_sql.errors.SetOperationMismatch: the sides of a set operation
            are different shapes.
        rowless_sql.errors.UnnamedProjection: the query projects an expression
            nobody aliased, so no engine-neutral name for it exists.
    """
    columns = _restored(_projection(resolved), declared_spellings(catalog))
    _require_names(columns)
    _refuse_duplicates(columns)
    return columns


def _projection(resolved: Any) -> tuple[Column, ...]:
    """The output columns of a resolved query, set operations included."""
    from sqlglot import exp

    # `SetOperation` is the base class of Union, Except and Intersect across
    # the whole supported range. Read by name rather than imported so that a
    # release which renames it degrades to handling plain unions - the common
    # case by far - instead of failing to import the adapter at all.
    set_operation = getattr(exp, "SetOperation", exp.Union)
    if isinstance(resolved, set_operation):
        return _set_operation(resolved)
    if isinstance(resolved, exp.Subquery):  # pragma: no cover - unwrapped by require_query
        return _projection(resolved.unnest())
    return tuple(_column(projection) for projection in resolved.selects)


def _set_operation(resolved: Any) -> tuple[Column, ...]:
    """Combine the two sides of a UNION, EXCEPT or INTERSECT.

    Names come from the left-hand side, which is what every SQL engine does.
    Types are unified through the IR lattice: `int32` and `int64` widen to
    `int64`, while a string and an integer have no common type and come back
    UNKNOWN rather than as whichever side was written first.
    """
    left = _projection(resolved.left)
    right = _projection(resolved.right)
    if len(left) != len(right):
        raise errors.SetOperationMismatch(
            f"this set operation has {len(left)} column(s) on the left and {len(right)} on the "
            f"right ({', '.join(column.name for column in left) or 'none'} against "
            f"{', '.join(column.name for column in right) or 'none'}), so no engine will run it"
        )
    combined: list[Column] = []
    for mine, theirs in zip(left, right, strict=True):
        dtype = unify(mine.dtype, theirs.dtype)
        note = mine.note or theirs.note
        if not dtype.is_known and mine.dtype.is_known and theirs.dtype.is_known:
            note = (
                f"the set operation combines {mine.dtype} and {theirs.dtype} in column "
                f"{mine.name!r}, which have no common dtype; rowless reports it as unresolved "
                "rather than taking the left-hand side's answer the way SQL's own rules do"
            )
        combined.append(Column(name=mine.name, dtype=dtype, note=note))
    return tuple(combined)


def _column(projection: Any) -> Column:
    """One select-list item as an output column.

    An item sqlglot could not type becomes ANY with the expression named, per
    `docs/confidence.md`: a function whose return type sqlglot does not know is
    a column rowless cannot vouch for, and ANY is how that is reported without
    failing a check that is probably fine.
    """
    from sqlglot import exp

    if isinstance(projection, exp.Star) or projection.is_star:
        # Qualification expands every star it can resolve, so one that is
        # still here belongs to a source with no schema - and
        # `resolved_expression` has already refused those. Reaching this means
        # sqlglot resolved something this adapter cannot read.
        raise errors.SqlNotParsable(
            "this query still projects a star after qualification, so rowless cannot say which "
            "columns it produces"
        )
    name = str(projection.alias_or_name)
    native = projection.type
    if dtype_map.is_unannotated(native):
        return Column(
            name=name,
            dtype=any_(),
            note=(
                f"sqlglot has no annotated return type for {_expression_text(projection)}, so "
                f"the column {name!r} is reported as ANY rather than guessed. A real binder - "
                "rowless-duckdb - types this one"
            ),
        )
    return Column(name=name, dtype=dtype_map.to_ir(native))


def _restored(
    columns: Sequence[Column],
    spellings: Mapping[str, str],
) -> tuple[Column, ...]:
    """Put a folded identifier back under the spelling its input declared."""
    found: list[Column] = []
    for column in columns:
        declared = spellings.get(column.name.casefold())
        if declared is None or declared == column.name:
            found.append(column)
            continue
        found.append(replace(column, name=declared, lookup=column.name))
    return tuple(found)


def declared_spellings(catalog: Catalog) -> dict[str, str]:
    """Case-folded column name to the one declared spelling that produced it.

    A name two inputs spell differently (`CustomerID` and `customerid`) is
    left out: there is no single declared spelling to restore, and picking one
    would be a guess about which table the column came from.
    """
    seen: dict[str, set[str]] = {}
    for table in catalog:
        for column in table.schema:
            seen.setdefault(column.casefold(), set()).add(column)
    return {folded: next(iter(names)) for folded, names in seen.items() if len(names) == 1}


def _expression_text(projection: Any) -> str:
    """What to name in a note about an untyped column.

    The function behind the column where there is one, because that is the
    thing a user can look up or cast; the whole expression otherwise, trimmed
    so a note stays a line.
    """
    from sqlglot import exp

    unaliased = projection.unalias() if isinstance(projection, exp.Alias) else projection
    for node in unaliased.find_all(exp.Func):
        if isinstance(node, exp.Anonymous):
            return f"{node.name}()"
        name = node.sql_name()
        if name:
            return f"{name}()"
    return _trimmed(str(unaliased.sql()))


def _trimmed(text: str, limit: int = 60) -> str:
    """One line of SQL, short enough to read inside a sentence."""
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else f"{collapsed[: limit - 1]}..."


def ambiguous_columns(catalog: Catalog) -> dict[str, tuple[str, ...]]:
    """Column names more than one registered input has, and which inputs those are.

    What tells an ambiguous reference apart from a missing one, since sqlglot
    reports both the same way on the oldest releases this adapter supports -
    and what lets the finding name the tables a user has to choose between.
    """
    holders: dict[str, list[str]] = {}
    for table in catalog:
        for column in table.schema:
            holders.setdefault(column, []).append(table.name)
    return {column: tuple(sorted(tables)) for column, tables in holders.items() if len(tables) > 1}


def _require_names(columns: Sequence[Column]) -> None:
    """Refuse a query whose projection sqlglot had to name for it.

    A rowless policy rather than a resolution step: sqlglot is perfectly
    willing to call an unaliased expression `_col_0`, and rowless will not
    report a column under a name no warehouse would use. The alternative -
    reporting `_col_0` as the column name - produces a contract that cannot
    hold against any engine, and a diff that claims a rename every time a
    model gains a column before the unaliased one.
    """
    unnamed = [column.name for column in columns if _PLACEHOLDER_NAME.match(column.name)]
    if not unnamed:
        return
    raise errors.UnnamedProjection(
        f"this query projects {len(unnamed)} expression(s) with no alias, and what a warehouse "
        "calls such a column is engine-specific - DuckDB says sum(amount), Spark says _col0, "
        "Postgres says sum - so rowless cannot say what this query returns. Alias them"
    )


def _refuse_duplicates(columns: Sequence[Column]) -> None:
    """Report two output columns of the same name, which SQL allows and rowless cannot.

    `SELECT *` over a join whose tables share a column name produces the name
    twice. SQL is content with that - a later reference is simply ambiguous -
    but a rowless schema is addressed by name, and so is everything downstream
    of it: a contract, a diff, a lineage edge. Reporting the duplicate is the
    only honest option, and the message names the fix.
    """
    seen: set[str] = set()
    for column in columns:
        if column.name in seen:
            raise DuplicateColumn(
                f"this query produces two columns named {column.name!r}. SQL keeps both columns "
                "of a colliding name after a join rather than renaming one, so nothing "
                "downstream can refer to either; alias one of them, or select the key from one "
                "side only"
            )
        seen.add(column.name)
