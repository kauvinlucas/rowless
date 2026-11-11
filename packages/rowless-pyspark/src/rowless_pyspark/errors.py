"""Turning Spark failures into rowless diagnostics.

Spark is the best of the engines rowless targets at explaining itself, and this
module is mostly a matter of not throwing that away. Since 3.4 an
`AnalysisException` carries a machine-readable error condition and a parameter
map, so the column that could not be resolved and the near miss Spark already
computed arrive as data:

    condition  UNRESOLVED_COLUMN.WITH_SUGGESTION
    parameters {"objectName": "`amount_cents`", "proposal": "`amount`, `status`"}

That is the flagship finding from the proposal, assembled without parsing a
sentence. The regex fallbacks below exist for the cases that still arrive as
prose - a classic session on an older Spark, a condition this adapter has not
catalogued - and for nothing else.

What matters here:

* the condition is read from `getCondition()` on Spark 4 and `getErrorClass()`
  on 3.5, because the method was renamed and the adapter supports both;
* the near miss comes from Spark's own `proposal`, which is computed against
  the real plan and is therefore better than anything rowless could suggest
  from a column list;
* a Connect transport failure is **not** a pipeline error. A dropped
  connection or a refused endpoint is reported as an unresolved schema naming
  the endpoint, so a cluster that is down reads as "rowless could not check
  this" rather than as "your pipeline is broken";
* anything this module cannot classify is **not** translated. It propagates as
  the engine exception, and core reports it under `PIPELINE_RAISED` with the
  Spark message as the detail. Mislabelling an error the adapter does not
  understand would be worse than letting core say so plainly.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, NoReturn

from rowless.errors import (
    ColumnNotFound,
    DTypeMismatch,
    DuplicateColumn,
    RowlessError,
    UnknownSchemaError,
)

#: Conditions that mean a column or a nested field does not exist.
MISSING_COLUMN_CONDITIONS: tuple[str, ...] = (
    "UNRESOLVED_COLUMN",
    "UNRESOLVED_FIELD",
    "UNRESOLVED_MAP_KEY",
    "FIELD_NOT_FOUND",
    "NO_SUCH_STRUCT_FIELD",
)

#: Conditions that mean two columns of the same name, which a name-addressed
#: schema cannot describe and nothing downstream of one can refer to.
DUPLICATE_CONDITIONS: tuple[str, ...] = (
    "AMBIGUOUS_REFERENCE",
    "AMBIGUOUS_COLUMN_OR_FIELD",
    "AMBIGUOUS_COLUMN_REFERENCE",
    "COLUMN_ALREADY_EXISTS",
    "DUPLICATE_COLUMN",
    "DUPLICATED_FIELD_NAME_IN_ARROW_STRUCT",
)

#: Conditions that mean the operation received a dtype, or a column count, it
#: refuses. The union cases are in here because that is what they are: two
#: branches whose schemas do not line up.
DTYPE_CONDITIONS: tuple[str, ...] = (
    "DATATYPE_MISMATCH",
    "CANNOT_UP_CAST_DATATYPE",
    "CANNOT_CAST_DATATYPE",
    "INVALID_COLUMN_OR_FIELD_DATA_TYPE",
    "UNSUPPORTED_DATATYPE",
    "NUM_COLUMNS_MISMATCH",
    "INCOMPATIBLE_COLUMN_TYPE",
    "UNPIVOT_VALUE_DATA_TYPE_MISMATCH",
    "INVALID_USAGE_OF_STAR_OR_REGEX",
)

#: Conditions that mean rowless has nothing to check against, rather than that
#: the pipeline is wrong: a table the dry run never registered, or SQL Spark
#: could not parse. Both report as unverified.
UNRESOLVED_CONDITIONS: tuple[str, ...] = (
    "TABLE_OR_VIEW_NOT_FOUND",
    "PATH_NOT_FOUND",
    "SCHEMA_NOT_FOUND",
    "CATALOG_NOT_FOUND",
    "PARSE_SYNTAX_ERROR",
    "PARSE_EMPTY_STATEMENT",
    "UNSUPPORTED_FEATURE",
)

#: `cannot resolve 'amount_cents' given input columns: [order_id, status]`, the
#: shape a classic session on an older Spark reports in prose.
_LEGACY_MISSING = re.compile(
    r"cannot resolve ['\"`]?(?P<column>[^'\"`\s]+)['\"`]? given input columns",
    re.IGNORECASE,
)
_LEGACY_INPUT_COLUMNS = re.compile(
    r"given input columns:?\s*\[(?P<columns>[^\]]*)\]", re.IGNORECASE
)

#: Spark appends the failing plan after a `;` on its own line, and a SQLSTATE
#: before it. Useful when debugging the adapter, noise in a one-line finding.
_PLAN_DUMP = re.compile(r";?\s*\n\s*'?(?:Project|Aggregate|Join|Filter|Relation|Union)\b.*", re.S)
_SQLSTATE = re.compile(r"\s*SQLSTATE:\s*\S+\s*", re.I)
_BACKTICKED = re.compile(r"`([^`]*)`")


def translate(
    exc: BaseException,
    *,
    relation: str | None = None,
    available: Sequence[str] = (),
) -> RowlessError | None:
    """Return the rowless error for a Spark exception, or None to leave it alone.

    Args:
        exc: the exception Spark raised while analyzing a plan.
        relation: the schema name to quote in the finding, when one is known.
        available: the columns the frame actually has, for the near-miss
            suggestion. Only used when Spark did not supply a `proposal` of its
            own, which it does for most unresolved-column failures.

    Returns:
        A `RowlessError` when the failure is one this adapter can classify, and
        None when it is not. A None result means the caller should let the
        original exception propagate.
    """
    if _is_transport_error(exc):
        return UnknownSchemaError(
            f"the Spark Connect endpoint could not be reached, so nothing was analyzed: "
            f"{_headline(exc)}"
        )
    if not is_spark_error(exc):
        # Dispatch below is by error condition, which is Spark's own
        # vocabulary. The hierarchy check first means a user exception that
        # happens to look like one of them is not mistaken for an engine
        # failure.
        return None

    code = condition(exc)
    params = parameters(exc)
    message = _headline(exc)

    if _matches(code, MISSING_COLUMN_CONDITIONS) or _LEGACY_MISSING.search(message):
        return _column_not_found(code, params, message, relation=relation, available=available)
    if _matches(code, DUPLICATE_CONDITIONS):
        return DuplicateColumn(message)
    if _matches(code, DTYPE_CONDITIONS):
        return DTypeMismatch(message)
    if _matches(code, UNRESOLVED_CONDITIONS):
        return UnknownSchemaError(message)
    return None


def is_spark_error(exc: BaseException) -> bool:
    """True when `exc` came out of Spark rather than out of user code.

    Checked by class hierarchy rather than by name: every Spark error
    subclasses `PySparkException`, and user code raising its own
    `AnalysisException` is not this.
    """
    try:
        from pyspark.errors import PySparkException
    except ImportError:  # pragma: no cover - only without pyspark
        return False
    return isinstance(exc, PySparkException)


def condition(exc: BaseException) -> str:
    """Spark's machine-readable error condition, or the empty string.

    Renamed from `getErrorClass()` to `getCondition()` in Spark 4, and this
    adapter supports both, so both are tried.
    """
    for method in ("getCondition", "getErrorClass"):
        found = getattr(exc, method, None)
        if not callable(found):
            continue
        try:
            value = found()
        except Exception:  # pragma: no cover - third-party error objects
            continue
        if value:
            return str(value)
    return ""


def parameters(exc: BaseException) -> Mapping[str, str]:
    """The error's parameter map, which is where the column name lives."""
    found = getattr(exc, "getMessageParameters", None)
    if not callable(found):
        return {}
    try:
        params = found()
    except Exception:  # pragma: no cover - third-party error objects
        return {}
    if not isinstance(params, Mapping):
        return {}
    return {str(key): str(value) for key, value in params.items()}


def raise_translated(
    exc: BaseException,
    *,
    relation: str | None = None,
    available: Sequence[str] = (),
) -> NoReturn:
    """Re-raise `exc` as a rowless error when it is one, else re-raise it as it is.

    The one-line form used at every call site that analyzes a Spark plan.
    """
    translated = translate(exc, relation=relation, available=available)
    if translated is not None:
        raise translated from exc
    raise exc


# --- Helpers -----------------------------------------------------------------


def _column_not_found(
    code: str,
    params: Mapping[str, str],
    message: str,
    *,
    relation: str | None,
    available: Sequence[str],
) -> ColumnNotFound:
    """Build the flagship finding out of an unresolved-column failure.

    Spark names the column in `objectName` (or `fieldName` for a nested one)
    and lists the alternatives it computed in `proposal`. Those are better than
    anything this adapter could reconstruct, so they are preferred over both
    the message and the caller's column list.
    """
    column = _unquote(
        params.get("objectName")
        or params.get("fieldName")
        or params.get("columnName")
        or params.get("key")
        or _legacy_column(message)
    )
    proposal = _proposal(params) or tuple(available) or _legacy_available(message)
    where = relation or _unquote(params.get("columnPath", "")) or None
    nested = ("UNRESOLVED_FIELD", "FIELD_NOT_FOUND", "NO_SUCH_STRUCT_FIELD")
    if not where and _matches(code, nested):
        where = "a nested field"
    return ColumnNotFound(column or "?", available=proposal, relation=where)


def _proposal(params: Mapping[str, str]) -> tuple[str, ...]:
    """The alternatives Spark suggested, which is what the near miss needs."""
    raw = params.get("proposal") or params.get("availableFields") or ""
    if not raw:
        return ()
    quoted = tuple(_BACKTICKED.findall(raw))
    if quoted:
        return quoted
    return tuple(part.strip().strip("`'\"") for part in raw.split(",") if part.strip())


def _legacy_column(message: str) -> str:
    """The column name out of a prose `cannot resolve` message."""
    found = _LEGACY_MISSING.search(message)
    return found.group("column") if found else ""


def _legacy_available(message: str) -> tuple[str, ...]:
    """The input columns a prose message listed."""
    found = _LEGACY_INPUT_COLUMNS.search(message)
    if found is None:
        return ()
    listed = found.group("columns").split(",")
    return tuple(part.strip().strip("`'\"") for part in listed if part.strip())


def _matches(code: str, prefixes: Sequence[str]) -> bool:
    """True when an error condition is one of `prefixes`, or a subcondition of one.

    Spark's conditions are hierarchical - `UNRESOLVED_COLUMN.WITH_SUGGESTION` -
    and the part before the dot is the one worth dispatching on: a new
    subcondition in a later release should land on the same finding rather than
    falling through as unclassified.
    """
    if not code:
        return False
    head = code.split(".", maxsplit=1)[0]
    return head in set(prefixes)


def _headline(exc: BaseException) -> str:
    """The message without the plan dump and the SQLSTATE suffix."""
    text = str(exc).strip()
    text = _PLAN_DUMP.sub("", text)
    text = _SQLSTATE.sub(" ", text)
    return " ".join(text.split()).strip(" ;")


def _unquote(name: str | Any) -> str:
    """Strip the backticks Spark quotes an identifier with."""
    text = str(name or "").strip()
    quoted = _BACKTICKED.findall(text)
    if quoted:
        return ".".join(quoted) if len(quoted) > 1 else quoted[0]
    return text.strip("`'\"")


def _is_transport_error(exc: BaseException) -> bool:
    """True when the failure is the Connect channel rather than the plan.

    A cluster that is unreachable has to read as "rowless could not check
    this", not as a finding against the pipeline: the code may be perfectly
    correct, and blaming it would be the kind of false positive that gets a
    linter switched off.

    The exclusion matters as much as the inclusion. Under Connect, *every*
    server-side failure arrives as a subclass of the gRPC exception - an
    `AnalysisException` from a missing column is one too - so a plain class
    check would report a user's typo as a network problem. An error that is
    also an analysis or parse failure is therefore never transport, whether or
    not the server attached a condition to it.
    """
    try:
        from pyspark.errors import AnalysisException, ParseException, PySparkException
        from pyspark.errors.exceptions.connect import SparkConnectGrpcException
    except Exception:
        return False
    if isinstance(exc, AnalysisException | ParseException):
        return False
    if isinstance(exc, SparkConnectGrpcException):
        return not condition(exc)
    if isinstance(exc, PySparkException):
        return False
    # A raw gRPC failure, which does not subclass anything of Spark's: the
    # channel never got as far as a Spark error at all.
    return type(exc).__name__ in {"_InactiveRpcError", "_MultiThreadedRendezvous", "RpcError"}
