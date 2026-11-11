"""Turning Ibis failures into rowless diagnostics.

Ibis builds expressions eagerly, which is a gift: `orders.select("amount_cents")`
raises at the line that reads the column, so the traceback already points at
the code a user has to edit. What it raises, though, is an `IbisTypeError` with
a resolved-plan dump attached, and CI needs `ColumnNotFound` with the column
named and a near miss suggested.

What matters here:

* the column name and the valid column list are both lifted out of the
  message, which is what makes the near-miss suggestion possible (`amount` ->
  `amount_cents`, the rename case from the proposal);
* `Table.__getattr__` is the idiomatic way to read a column in ibis
  (`orders.amount`), and it fails with `AttributeError`, not with an ibis
  error. `MissingColumn` is a `ColumnNotFound` *and* an `AttributeError`, so
  the finding is the flagship one while `hasattr(table, "x")` and every
  library that probes an object with `getattr` keep working;
* ibis validation errors arrive with the whole expression tree printed inside
  them, so the plan dump is stripped and the type complaint kept;
* anything this module cannot classify is **not** translated. It propagates as
  the engine exception, and core reports it under `PIPELINE_RAISED` with the
  ibis message as the detail. Mislabelling an error the adapter does not
  understand would be worse than letting core say so plainly.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import NoReturn

from rowless.diagnostics.codes import Code
from rowless.errors import (
    ColumnNotFound,
    DTypeMismatch,
    DuplicateColumn,
    RowlessError,
    SchemaNotFound,
    UnknownSchemaError,
)

#: `Column 'amount' is not found in table. Existing columns: 'order_id', 'amount_cents'.`
_MISSING_COLUMN = re.compile(r"Column '([^']*)' is not found in table")
_EXISTING_COLUMNS = re.compile(r"Existing columns: (.*?)\.?$", re.MULTILINE)
_QUOTED = re.compile(r"'([^']*)'")

#: `Duplicate column name 'amount' in result set`
_DUPLICATE_COLUMN = re.compile(r"[Dd]uplicate column name '([^']*)'")

#: sqlglot, resolving the SQL seam: `Column 'amount' could not be resolved`
_UNRESOLVED_COLUMN = re.compile(r"[Cc]olumn '([^']*)' could not be resolved")

#: Ibis prints a resolved plan inside its validation errors: `StringConcat((r0
#: := UnboundTable: orders ...))`. The operation name in front of it is the
#: part worth keeping, along with the type complaint further down.
_OPERATION = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\(")
_TYPE_COMPLAINT = re.compile(r"(of type [^\n]*?is not [^\n]*)")

#: One `r0 := UnboundTable: orders` header plus the indented schema lines under
#: it. Deliberately not `re.DOTALL`: the complaint worth keeping sits *after*
#: the dump, and a greedy match would take it with the noise.
_RELATION_DUMP = re.compile(r"r\d+ := \w+: [^\n]*\n(?:[ \t]+[^\n]*\n|\n)*")

#: Messages that mean "this expression has no backend", which is the normal
#: state of every frame rowless hands a pipeline.
_UNBOUND_MARKERS: tuple[str, ...] = (
    "unbound table",
    "unbound tables",
    "no backend",
    "has no backend",
    "default backend",
)


class MissingColumn(ColumnNotFound, AttributeError):
    """A column read through attribute access that the table does not have.

    Both a `ColumnNotFound`, so it reports as the flagship finding with a
    near-miss suggestion, and an `AttributeError`, because that is what ibis
    raises and what every `hasattr` and `getattr(obj, name, default)` in the
    process below is written against. Dropping the `AttributeError` half would
    turn a probe for an optional attribute into a hard failure somewhere far
    away from this adapter.
    """


class ExecutionAttempted(UnknownSchemaError):
    """The pipeline tried to pull values out of a rowless frame.

    Reported under `ROW_DEPENDENT_CONTROL_FLOW`: a rowless frame has no rows,
    so whatever the pipeline would have computed from the result is a value
    production would not have produced. An `UnknownSchemaError`, so it reads as
    unverified rather than as a bug in the pipeline - the code is very likely
    correct, and rowless simply cannot check it.
    """

    code = Code.ROW_DEPENDENT_CONTROL_FLOW


class SqlNotResolvable(UnknownSchemaError):
    """A SQL string ibis could not resolve without a backend.

    `Table.sql()` normally asks the backend what a query returns. There is no
    backend in a dry run, so the adapter resolves the query through ibis's own
    SQL parser instead; when that cannot read the query, the result is
    unverified rather than wrong.
    """


def translate(
    exc: BaseException,
    *,
    relation: str | None = None,
    available: Sequence[str] = (),
) -> RowlessError | None:
    """Return the rowless error for an ibis exception, or None to leave it alone.

    Args:
        exc: the exception ibis raised while building or resolving an
            expression.
        relation: the schema name to quote in the finding, when one is known.
        available: the columns the table actually has, for the near-miss
            suggestion. Ibis lists them in the message itself, so this is only
            a fallback for the messages that do not.

    Returns:
        A `RowlessError` when the failure is one this adapter can classify, and
        None when it is not. A None result means the caller should let the
        original exception propagate.
    """
    if isinstance(exc, RowlessError):
        # Already translated, by a wrapper further in. Re-wrapping it would
        # lose the column name the inner one carries.
        return None
    if not is_ibis_error(exc):
        return None

    message = str(exc).strip()
    name = type(exc).__name__

    if _MISSING_COLUMN.search(message):
        return _column_not_found(message, relation=relation, available=available)
    duplicate = _DUPLICATE_COLUMN.search(message)
    if duplicate is not None:
        return DuplicateColumn(
            f"this expression produces two columns named {duplicate.group(1)!r}; ibis refuses a "
            "duplicate output name rather than keeping one of them. Alias one of the two"
        )
    if name == "TableNotFound":
        return SchemaNotFound(f"ibis cannot find the table {message}")
    if _is_unbound(message):
        return ExecutionAttempted(
            f"{_clean(message)} rowless frames are unbound ibis tables: they carry a schema and "
            "no backend, so nothing can be executed against them"
        )
    if _is_validation_error(exc):
        return DTypeMismatch(_signature_complaint(message))
    if name in {
        "RelationError",
        "IntegrityError",
        "IbisTypeError",
        "InputTypeError",
        "IbisInputError",
    }:
        return DTypeMismatch(_clean(message))
    return None


def translate_sql_error(
    exc: BaseException,
    *,
    available: Sequence[str] = (),
    relation: str | None = None,
) -> RowlessError | None:
    """Return the rowless error for a sqlglot failure, or None to leave it alone.

    Ibis parses SQL with sqlglot, so the SQL seam fails with sqlglot's
    exceptions rather than ibis's. The one worth translating is the unresolved
    column: `SELECT amount FROM orders` against a table that renamed it is the
    same finding as the Python one, and it deserves the same message.

    Recognised by module rather than by import, so this module still works in a
    process where sqlglot is not installed - it arrives with ibis, and the
    adapter should not fail differently if a wheel ever drops it.
    """
    if isinstance(exc, RowlessError):
        return None
    if not type(exc).__module__.startswith("sqlglot"):
        return None
    message = str(exc).strip()
    unresolved = _UNRESOLVED_COLUMN.search(message)
    if unresolved is not None:
        return ColumnNotFound(unresolved.group(1), available=tuple(available), relation=relation)
    # Everything else is left alone, a syntax error included. sqlglot refusing
    # to parse a query is not proof the query is wrong: a dialect's own syntax
    # fails the same way when the dialect was not named, and reporting valid
    # Snowflake SQL as broken would be worse than reporting it as unread.
    return None


def is_ibis_error(exc: BaseException) -> bool:
    """True when `exc` came out of ibis rather than out of user code.

    Checked by class hierarchy and by *module* rather than by class name, so a
    user exception that happens to share a name with one of ibis's is not
    mistaken for an engine failure. Three checks, because not everything ibis
    raises descends from `IbisError`: expression signatures fail with ibis's
    own `ValidationError`, and `TableNotFound` descends from `Exception`
    directly.
    """
    try:
        import ibis.common.exceptions as ibis_errors
    except ImportError:  # pragma: no cover - only without ibis
        return False
    if isinstance(exc, ibis_errors.IbisError):
        return True
    if _is_validation_error(exc):
        return True
    return type(exc).__module__.startswith("ibis.")


def raise_translated(
    exc: BaseException,
    *,
    relation: str | None = None,
    available: Sequence[str] = (),
) -> NoReturn:
    """Re-raise `exc` as a rowless error when it is one, else re-raise it as it is.

    The one-line form used at every call site that builds or resolves an ibis
    expression.
    """
    translated = translate(exc, relation=relation, available=available)
    if translated is not None:
        raise translated from exc
    raise exc


def missing_column(column: str, *, available: Sequence[str], relation: str | None) -> MissingColumn:
    """Build the finding for a column read through attribute access."""
    return MissingColumn(column, available=tuple(available), relation=relation)


def _is_validation_error(exc: BaseException) -> bool:
    """True for ibis's expression-signature validation errors."""
    try:
        from ibis.common.annotations import ValidationError
    except ImportError:  # pragma: no cover - only without ibis
        return False
    return isinstance(exc, ValidationError)


def _column_not_found(
    message: str,
    *,
    relation: str | None,
    available: Sequence[str] = (),
) -> ColumnNotFound:
    """Build the flagship finding out of an ibis missing-column message."""
    found = _MISSING_COLUMN.search(message)
    column = found.group(1) if found else "?"
    return ColumnNotFound(
        column,
        available=_existing(message) or tuple(available),
        relation=relation,
    )


def _existing(message: str) -> tuple[str, ...]:
    """The columns ibis listed as existing, which is what the near miss needs."""
    listed = _EXISTING_COLUMNS.search(message)
    if listed is None:
        return ()
    return tuple(_QUOTED.findall(listed.group(1)))


def _is_unbound(message: str) -> bool:
    """True when the message is ibis saying there is no backend to run against."""
    lowered = message.lower()
    return any(marker in lowered for marker in _UNBOUND_MARKERS)


def _clean(message: str) -> str:
    """One line, with the plan dumps ibis embeds in its messages taken out."""
    return re.sub(r"\s+", " ", _RELATION_DUMP.sub("... ", message)).strip()


def _signature_complaint(message: str) -> str:
    """The readable half of an ibis validation error.

    The full message is the operation, every input relation printed in full,
    and then the actual complaint. Only the first and last of those three are
    worth putting in a finding.
    """
    operation = _OPERATION.match(message)
    complaint = _TYPE_COMPLAINT.search(message)
    if operation is None and complaint is None:
        return _clean(message)
    subject = f"ibis refused the arguments to {operation.group(1)}" if operation else "ibis refused"
    if complaint is None:
        return f"{subject}: {_clean(message)[:200]}"
    return f"{subject}: a value {_clean(complaint.group(1))}"
