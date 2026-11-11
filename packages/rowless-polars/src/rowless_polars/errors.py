"""Turning Polars failures into rowless diagnostics.

Translating the engine exception is most of what this adapter does. Polars
already knows the useful part - the plan failed, and which column it failed on -
but it reports it as a traceback with a resolved plan dump attached, which is
not a finding a user can act on in CI.

What matters here:

* the column name is lifted out of the message, so `ColumnNotFound` carries it
  and `rowless` can suggest the near miss (`amount` -> `amount_cents`, the
  rename case from the proposal);
* the list of valid columns Polars prints is parsed too, because it is what
  makes the suggestion possible;
* anything this module cannot classify is **not** translated. It propagates as
  the engine exception, and core reports it under `PIPELINE_RAISED` with the
  Polars message as the detail. Mislabelling an error the adapter does not
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
    UnknownSchemaError,
)

#: `unable to find column "amount"; valid columns: ["order_id", "amount_cents"]`
_MISSING_COLUMN = re.compile(r'unable to find column "([^"]*)"')
_VALID_COLUMNS = re.compile(r"valid columns: \[(.*?)\]", re.DOTALL)
_QUOTED = re.compile(r'"([^"]*)"')

#: Polars appends a resolved-plan dump after a blank line. Useful when
#: debugging the adapter, noise in a one-line finding.
_PLAN_DUMP = re.compile(r"\n\s*\n")


class UnknownUdfReturn(UnknownSchemaError):
    """Polars could not work out what a UDF returns, and neither could rowless.

    An `UnknownSchemaError`, so it reports as unverified rather than as an
    error in the pipeline: the code may well be correct, and rowless simply has
    no way to know what comes out of it.
    """

    code = Code.UNKNOWN_UDF_RETURN


def translate(
    exc: BaseException,
    *,
    relation: str | None = None,
    available: Sequence[str] = (),
) -> RowlessError | None:
    """Return the rowless error for a Polars exception, or None to leave it alone.

    Args:
        exc: the exception Polars raised while resolving a plan.
        relation: the schema name to quote in the finding, when one is known.
        available: the columns the frame actually has, for the near-miss
            suggestion. Polars 1.0 reports a missing column as the bare name
            and later versions list the valid columns alongside it; passing
            what the caller already knows makes the suggestion independent of
            which version is installed.

    Returns:
        A `RowlessError` when the failure is one this adapter can classify, and
        None when it is not. A None result means the caller should let the
        original exception propagate.
    """
    if not is_polars_error(exc):
        # Dispatch below is by class name, which is readable and survives
        # Polars moving its exception classes around. The hierarchy check
        # first means a user exception that happens to share a name with one
        # of them is not mistaken for an engine failure.
        return None

    name = type(exc).__name__
    message = _headline(exc)

    if name == "ColumnNotFoundError":
        return _column_not_found(message, relation=relation, available=available)
    if name in {"StructFieldNotFoundError", "SchemaFieldNotFoundError"}:
        # The message is the bare field name, with no sibling list to suggest
        # from, so the finding says where it was looked for instead.
        return ColumnNotFound(_first_line(message), relation=relation or "a nested field")
    if name == "DuplicateError":
        return DuplicateColumn(message)
    if name == "SchemaError":
        return DTypeMismatch(message)
    if name == "InvalidOperationError":
        if "return type" in message or "return_dtype" in message:
            return UnknownUdfReturn(
                f"{message}. Pass return_dtype= to map_elements, annotate the function return "
                "type, or set probe_udfs = true to let rowless call it with one synthetic value"
            )
        return DTypeMismatch(message)
    return None


def is_polars_error(exc: BaseException) -> bool:
    """True when `exc` came out of Polars rather than out of user code.

    Checked by class hierarchy rather than by name: a Polars error subclasses
    `PolarsError`, and user code raising its own `ComputeError` is not this.
    """
    try:
        import polars.exceptions as polars_errors
    except ImportError:  # pragma: no cover - only without polars
        return False
    return isinstance(exc, polars_errors.PolarsError | polars_errors.PanicException)


def _column_not_found(
    message: str,
    *,
    relation: str | None,
    available: Sequence[str] = (),
) -> ColumnNotFound:
    """Build the flagship finding out of a Polars `ColumnNotFoundError`."""
    found = _MISSING_COLUMN.search(message)
    column = found.group(1) if found else _first_line(message).strip("'\"")
    return ColumnNotFound(
        column,
        available=_available(message) or tuple(available),
        relation=relation,
    )


def _available(message: str) -> tuple[str, ...]:
    """The columns Polars listed as valid, which is what the near miss needs."""
    listed = _VALID_COLUMNS.search(message)
    if listed is None:
        return ()
    return tuple(_QUOTED.findall(listed.group(1)))


def _headline(exc: BaseException) -> str:
    """The message without the resolved-plan dump Polars appends."""
    text = str(exc).strip()
    return _PLAN_DUMP.split(text, maxsplit=1)[0].strip()


def _first_line(message: str) -> str:
    """The first line of a message, for errors whose whole message is a name."""
    return message.splitlines()[0].strip() if message else "?"


def raise_translated(
    exc: BaseException,
    *,
    relation: str | None = None,
    available: Sequence[str] = (),
) -> NoReturn:
    """Re-raise `exc` as a rowless error when it is one, else re-raise it as it is.

    The one-line form used at every call site that resolves a Polars plan.
    """
    translated = translate(exc, relation=relation, available=available)
    if translated is not None:
        raise translated from exc
    raise exc
