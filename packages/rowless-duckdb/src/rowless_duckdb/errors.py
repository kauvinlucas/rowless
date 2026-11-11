"""Turning DuckDB failures into rowless diagnostics.

DuckDB binds eagerly, which is the nicest property this adapter gets for free:
`rel.project("amount_cents")` raises at the line that wrote it rather than at
the end of the dry run. The binder also prints the information a finding needs -
the column it could not resolve, and the ones it could - so this module is
mostly a parser for those messages.

What matters here:

* the missing column name is lifted out of the message, so `ColumnNotFound`
  carries it and rowless can suggest the near miss (`amount` ->
  `amount_cents`, the rename case from the proposal);
* the candidate bindings DuckDB lists are parsed too, because they are what
  makes the suggestion possible. 1.x qualifies them (`"orders.amount"`) and
  2.x does not, so the table prefix is stripped either way;
* an ambiguous reference after a join is reported as `DuplicateColumn`, because
  that is what it is: two columns of the same name reached the same scope, and
  the query is not wrong so much as unanswerable;
* a write refused by a read-only database becomes `SideEffectBlocked`, since
  rowless opened it read-only on purpose and the user needs to be told that
  rather than shown an engine error;
* anything this module cannot classify is **not** translated. It propagates as
  the DuckDB exception, and core reports it under `PIPELINE_RAISED` with the
  engine message as the detail. Mislabelling an error the adapter does not
  understand would be worse than letting core say so plainly.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import NoReturn

from rowless.errors import (
    ColumnNotFound,
    DTypeMismatch,
    DuplicateColumn,
    RowlessError,
    SchemaNotFound,
    SideEffectBlocked,
    UnknownSchemaError,
)

#: `Binder Error: Referenced column "amount_cents" not found in FROM clause!`
_MISSING_COLUMN = re.compile(r'[Rr]eferenced column "([^"]*)" not found')

#: `Binder Error: Referenced table "orders" not found!`
_MISSING_TABLE = re.compile(r'[Rr]eferenced table "([^"]*)" not found')

#: `Candidate bindings: "orders.amount", "orders.created_at"`, or the same list
#: unqualified. Stops at the end of the line, because DuckDB appends the query
#: text and a caret afterwards.
_CANDIDATES = re.compile(r"[Cc]andidate bindings?:([^\n]*)")

#: `Ambiguous reference to column name "created_at" (use: ...)`
_AMBIGUOUS = re.compile(r'[Aa]mbiguous reference to column name "([^"]*)"')

#: `Column "amount" referenced that exists in the SELECT clause - but this
#: column cannot be referenced before it is defined`. DuckDB says this when the
#: name is not a column of the FROM clause *and* something in the same select
#: list is aliased to it - `sum(amount) AS amount` over a table where `amount`
#: was renamed. The column is still missing; the alias is only what DuckDB
#: found to talk about instead.
_SHADOWED_BY_ALIAS = re.compile(r'[Cc]olumn "([^"]*)" referenced that exists in the SELECT clause')

#: `Table with name nope does not exist!`, and the same for the other things a
#: query can name that rowless is responsible for registering. A catalog error
#: about anything else - a scalar function, a type, a macro - is a mistake in
#: the pipeline rather than an input rowless failed to resolve, and is left for
#: core to report under `PIPELINE_RAISED`. The distinction decides whether the
#: finding reads as "rowless did not check this" or as "this is wrong".
_UNKNOWN_RELATION = re.compile(
    r"(?:Table|View|Schema|Catalog)(?: with name)? \"?([^\s\"]+)\"? does not exist"
)

_QUOTED = re.compile(r'"([^"]*)"')

#: DuckDB appends the offending SQL and a caret after a blank line, or after a
#: `LINE n:` marker. Useful when debugging a query, noise in a one-line finding.
_QUERY_DUMP = re.compile(r"\n\s*\n|\nLINE \d+:")

#: Message fragments that mean a statement was refused because the database or
#: the filesystem was opened read-only by rowless rather than by the user.
_BLOCKED_FRAGMENTS: tuple[str, ...] = (
    "attached in read-only mode",
    "file system operations are disabled",
    "cannot be changed in read-only mode",
    "read-only mode",
)


def translate(
    exc: BaseException,
    *,
    relation: str | None = None,
    available: Sequence[str] = (),
) -> RowlessError | None:
    """Return the rowless error for a DuckDB exception, or None to leave it alone.

    Args:
        exc: the exception DuckDB raised while binding or executing.
        relation: the schema name to quote in the finding, when one is known.
        available: the columns the relation actually has, for the near-miss
            suggestion. Passed by the caller because it already knows them;
            the candidate bindings in the message are used when it does not.

    Returns:
        A `RowlessError` when the failure is one this adapter can classify, and
        None when it is not. A None result means the caller should let the
        original exception propagate.
    """
    if not is_duckdb_error(exc):
        # Dispatch below is partly by class name, which survives DuckDB moving
        # its exception classes between modules. The hierarchy check first
        # means a user exception that happens to share a name with one of them
        # is not mistaken for an engine failure.
        return None

    name = type(exc).__name__
    message = headline(exc)

    if _is_blocked_write(message):
        return SideEffectBlocked(
            f"{message}. rowless opened this database read-only and disabled file access, so a "
            "dry run cannot write to it; the writes a pipeline attempts are reported instead"
        )
    if name == "BinderException":
        return _binder_error(message, relation=relation, available=available)
    if name == "CatalogException":
        return _catalog_error(message)
    if name in {"ConversionException", "TypeMismatchException", "OutOfRangeException"}:
        return DTypeMismatch(message)
    if name in {"ParserException", "SyntaxException"}:
        return UnknownSchemaError(f"DuckDB could not parse this SQL: {message}")
    if name == "InvalidInputException" and "does not exist" in message:
        return SchemaNotFound(message)
    return None


def is_duckdb_error(exc: BaseException) -> bool:
    """True when `exc` came out of DuckDB rather than out of user code.

    Checked by class hierarchy rather than by name: every DuckDB exception
    subclasses `duckdb.Error`, and user code raising its own `BinderException`
    is not this.
    """
    try:
        import duckdb
    except ImportError:  # pragma: no cover - only without duckdb
        return False
    return isinstance(exc, duckdb.Error)


def headline(exc: BaseException) -> str:
    """The message without the query text and caret DuckDB appends."""
    text = str(exc).strip()
    return _QUERY_DUMP.split(text, maxsplit=1)[0].strip()


def candidates(message: str) -> tuple[str, ...]:
    """The column names DuckDB listed as valid, unqualified.

    1.x lists `"orders.amount"` and 2.x lists `"amount"`. The near-miss
    suggestion compares against the name the user wrote, which is unqualified
    in both, so the table prefix comes off here.
    """
    listed = _CANDIDATES.search(message)
    if listed is None:
        return ()
    quoted = _QUOTED.findall(listed.group(1))
    found = quoted or [part.strip() for part in listed.group(1).split(",")]
    return tuple(name.rpartition(".")[2] for name in found if name.strip())


def _binder_error(
    message: str,
    *,
    relation: str | None,
    available: Sequence[str],
) -> RowlessError | None:
    """Classify a `BinderException`, which covers most of what users hit."""
    missing = _MISSING_COLUMN.search(message) or _SHADOWED_BY_ALIAS.search(message)
    if missing is not None:
        return ColumnNotFound(
            missing.group(1),
            available=candidates(message) or tuple(available),
            relation=relation,
        )
    ambiguous = _AMBIGUOUS.search(message)
    if ambiguous is not None:
        return DuplicateColumn(
            f"{message}. DuckDB keeps both columns of a colliding name after a join, so a "
            "later reference cannot be resolved; qualify or alias one of them"
        )
    table = _MISSING_TABLE.search(message)
    if table is not None:
        return SchemaNotFound(
            f"{message}. A dry run only knows the relations rowless registered for it; "
            "declare this one as an input"
        )
    if "No function matches" in message or "explicit type casts" in message:
        return DTypeMismatch(message)
    # A binder error this adapter has no reading for. Core reports the engine
    # message under PIPELINE_RAISED, which says more than a guessed code.
    return None


def _catalog_error(message: str) -> RowlessError | None:
    """Classify a `CatalogException`: a name that is not in the catalogue.

    Only a missing *relation* is rowless's problem - that is an input nobody
    registered, so the pipeline was not really checked. A missing function or
    type is a mistake in the query, and reporting it as an unresolved source
    would read as "unchecked" when the right answer is "wrong".
    """
    relation = _UNKNOWN_RELATION.search(message)
    if relation is None:
        return None
    return SchemaNotFound(
        f"{message} A dry run only knows the relations rowless registered for it; declare "
        f"{relation.group(1)} as an input so its schema is resolved first"
    )


def _is_blocked_write(message: str) -> bool:
    """True when the failure is the sandbox refusing a write, not a bad query."""
    lowered = message.lower()
    return any(fragment in lowered for fragment in _BLOCKED_FRAGMENTS)


def raise_translated(
    exc: BaseException,
    *,
    relation: str | None = None,
    available: Sequence[str] = (),
) -> NoReturn:
    """Re-raise `exc` as a rowless error when it is one, else re-raise it as it is.

    The one-line form used at every call site that binds a DuckDB query.

    An unclassified DuckDB error is re-raised **without the engine's own stack
    frames**, which matters more than it looks. Core finds the line to put a
    finding on by walking the traceback for the deepest frame that is neither
    rowless nor an adapter, and DuckDB 2.x raises through a Python wrapper
    layer: left alone, every unclassified error would be reported against
    `site-packages/duckdb/_relation.py` instead of against the pipeline.
    Dropping the engine frames makes the raise happen here, so the deepest
    frame outside the adapter is the user's. The message is preserved, and the
    message is what core reports as the detail.
    """
    translated = translate(exc, relation=relation, available=available)
    if translated is not None:
        raise translated from exc
    raise exc.with_traceback(None)
