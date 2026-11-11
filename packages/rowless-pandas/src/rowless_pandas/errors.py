"""Turning pandas failures into rowless diagnostics.

Harder here than in the other adapters, for one reason: **pandas reports a
missing column as a `KeyError`.** Polars raises `ColumnNotFoundError`, DuckDB
raises a `BinderException`, and both are unambiguous. A `KeyError` is what
every dictionary in Python raises, so translating one on sight would turn a
user's own `config["region"]` into a finding about a column that does not
exist.

So this module never classifies by exception type alone. `raised_in_pandas`
walks the traceback and asks where the exception actually came from: a
`KeyError` raised inside the pandas package is pandas failing to find a label,
and a `KeyError` raised anywhere else is the pipeline's own business and is
left to propagate. That check is the reason this file is worth its length.

The other half is the message. pandas spells a missing label five different
ways depending on which method was called:

    'amount'                                          df["amount"]
    "None of [Index(['amount'], dtype='str')] ..."     df[["amount"]]
    "['amount'] not found in axis"                     df.drop(...), df.rename(...)
    "None of ['amount'] are in the columns"            df.set_index(...)
    "Label(s) ['amount'] do not exist"                 df.agg(...)

Every one of them is parsed, because the column name is what makes the
near-miss suggestion possible - `amount` -> `amount_cents`, the rename case
from the proposal - and the suggestion is most of the value of the finding.
pandas never lists the valid columns, so the candidates come from the frame the
call was made on, which is why the wrappers in
`rowless_pandas.interception` pass them in.

Anything this module cannot classify is **not** translated. It propagates, and
core reports it under `PIPELINE_RAISED` with the pandas message attached.
Mislabelling an error the adapter does not understand would be worse than
letting core say plainly that the pipeline raised.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn

from rowless.diagnostics.codes import Code
from rowless.errors import (
    ColumnNotFound,
    DTypeMismatch,
    DuplicateColumn,
    RowDependentControlFlow,
    RowlessError,
    SchemaError,
)

#: `'amount'`, the bare label a `__getitem__` or a `loc` lookup raises with.
_BARE = re.compile(r"^['\"]?(?P<name>[^'\"\[\]]+)['\"]?$")

#: Every bracketed list pandas prints a missing label inside. One pattern per
#: message shape would be five patterns that all do the same thing; this finds
#: the quoted names inside the first bracketed group instead.
_BRACKETED = re.compile(r"\[(?P<body>[^\[\]]*)\]")
_QUOTED = re.compile(r"['\"]([^'\"]+)['\"]")

#: Messages that mean a label was not found, across the five spellings.
_NOT_FOUND = re.compile(
    r"not found in axis|not in index|are in the \[?columns|do not exist|None of",
    re.IGNORECASE,
)

#: A dtype the operation refuses, rather than a label it cannot find.
_DTYPE_REFUSED = re.compile(
    r"not supported for dtype|unsupported operand|could not convert|"
    r"can only use \.\w+ accessor|can only concatenate|cannot astype|invalid literal|"
    r"agg function failed|cannot perform|unsupported dtype",
    re.IGNORECASE,
)
"""Messages that mean a dtype was refused, across the supported pandas range.

Deliberately a message pattern and not an exception type. pandas raises a bare
`TypeError` for most of these and rewords it between releases: adding a string
column to an integer one is `can only concatenate str` on 2.2 and
`operation 'add' not supported for dtype` on 3.x. Both are the same finding.
"""

#: Where the pandas package lives, so a traceback frame can be attributed to it.
_PANDAS_ROOT: str | None = None


class JoinKeyMismatch(SchemaError):
    """A merge whose keys do not line up: no common column, or none usable.

    Its own class because core has the diagnostic code and no exception for it,
    and because this is the pandas finding users hit most after the missing
    column: `df.merge(other)` with no `on=` is a join on whatever the two
    frames happen to share, which is a schema change away from being a join on
    nothing.
    """

    code = Code.JOIN_KEY_MISMATCH


class EmptyReduction(RowDependentControlFlow, ValueError):
    """A reduction that needs a row to answer at all: `idxmax` on zero rows.

    A `RowDependentControlFlow`, so it reports as unverified rather than as an
    error in the pipeline: the code is fine, and it is the dry run that has no
    rows. Also a `ValueError`, which is what pandas itself raises here, so a
    pipeline that already catches one keeps behaving the way it does in
    production. `ColumnNotFound` is a `KeyError` in core for the same reason.
    """

    code = Code.ROW_DEPENDENT_CONTROL_FLOW


def pandas_root() -> str:
    """The directory the installed pandas lives in, cached after the first call."""
    global _PANDAS_ROOT
    if _PANDAS_ROOT is None:
        try:
            import pandas

            _PANDAS_ROOT = str(Path(str(pandas.__file__)).resolve().parent)
        except Exception:  # pragma: no cover - the wrappers only exist with pandas
            _PANDAS_ROOT = ""
    return _PANDAS_ROOT


def raised_in_pandas(exc: BaseException) -> bool:
    """True when `exc` came out of pandas rather than out of user code.

    The whole basis for translating a `KeyError`. Checked by file rather than
    by exception type, because the exception types pandas raises for a missing
    column are builtins that user code raises too. The *deepest* frame decides:
    a pipeline that calls pandas, which calls back into a user callback that
    raises, is the callback's failure and not pandas's.
    """
    root = pandas_root()
    if not root:
        return False
    deepest: str | None = None
    traceback = exc.__traceback__
    while traceback is not None:
        deepest = traceback.tb_frame.f_code.co_filename
        traceback = traceback.tb_next
    if deepest is None:
        return False
    try:
        resolved = str(Path(deepest).resolve())
    except OSError:  # pragma: no cover - a synthetic frame with no real file
        resolved = deepest
    return resolved.startswith(root)


def translate(
    exc: BaseException,
    *,
    relation: str | None = None,
    available: Sequence[str] = (),
) -> RowlessError | None:
    """Return the rowless error for a pandas exception, or None to leave it alone.

    Args:
        exc: the exception pandas raised.
        relation: the schema name to quote in the finding, when one is known.
        available: the columns the frame actually has, for the near-miss
            suggestion. pandas never lists them in the message, so the caller
            has to supply what it knows - which the interception wrappers do,
            because they have the receiver in hand.

    Returns:
        A `RowlessError` when the failure is one this adapter can classify, and
        None when it is not. None means the caller should let the original
        exception propagate to core.
    """
    if not raised_in_pandas(exc):
        return None

    import pandas as pd

    message = str(exc).strip().strip("\"'")

    if isinstance(exc, pd.errors.UndefinedVariableError):
        # `query` and `eval` resolve names against the frame, so an undefined
        # name is a missing column with a different spelling.
        return ColumnNotFound(
            _name_from_undefined(message), available=tuple(available), relation=relation
        )
    if isinstance(exc, KeyError):
        return _missing_label(message, relation=relation, available=available)
    if isinstance(exc, pd.errors.MergeError):
        return JoinKeyMismatch(message)
    if isinstance(exc, pd.errors.DuplicateLabelError | pd.errors.InvalidIndexError):
        return DuplicateColumn(message)
    if isinstance(exc, pd.errors.SpecificationError | pd.errors.DataError):
        return DTypeMismatch(message)
    if isinstance(exc, pd.errors.IntCastingNaNError):
        return DTypeMismatch(
            f"{message}. On a rowless frame there are no values to cast, so this is the "
            "declared dtype refusing the cast rather than the data"
        )
    if isinstance(exc, TypeError | ValueError | AttributeError) and _DTYPE_REFUSED.search(message):
        return DTypeMismatch(message)
    return None


def _missing_label(
    message: str,
    *,
    relation: str | None,
    available: Sequence[str],
) -> RowlessError | None:
    """Build the flagship finding out of a pandas `KeyError`.

    Returns None for a `KeyError` that came from pandas but is not about a
    label - a missing group, say - so it propagates rather than being reported
    as a column that does not exist.
    """
    names = _labels(message)
    if not names:
        return None
    # One finding per call, naming the first missing label: the suggestion is
    # per name, and a finding that lists three is a finding with no suggestion.
    return ColumnNotFound(names[0], available=tuple(available), relation=relation)


def _labels(message: str) -> tuple[str, ...]:
    """The labels pandas named in a `KeyError`, across every message shape."""
    bracketed = _BRACKETED.search(message)
    if bracketed is not None:
        quoted = _QUOTED.findall(bracketed.group("body"))
        if quoted:
            return tuple(quoted)
        # `[0]` and friends: a positional label, not a column name. Reported
        # as-is rather than dropped, since it is still what was looked up.
        body = bracketed.group("body").strip()
        return (body,) if body and not _NOT_FOUND.search(body) else ()
    bare = _BARE.match(message)
    if bare is not None and not _NOT_FOUND.search(message):
        return (bare.group("name"),)
    return ()


def _name_from_undefined(message: str) -> str:
    """The name in a `query` or `eval` failure: `name 'amount' is not defined`."""
    quoted = _QUOTED.search(message)
    return quoted.group(1) if quoted else message


def raise_translated(
    exc: BaseException,
    *,
    relation: str | None = None,
    available: Sequence[str] = (),
) -> NoReturn:
    """Re-raise `exc` as a rowless error when it is one, else re-raise it as it is.

    The one-line form used at every call site that touches pandas.
    """
    translated = translate(exc, relation=relation, available=available)
    if translated is not None:
        raise translated from exc
    raise exc
