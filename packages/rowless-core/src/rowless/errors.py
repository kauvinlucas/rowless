"""Exception hierarchy.

Everything rowless raises descends from `RowlessError`, so a caller can catch
the tool's failures without swallowing an engine's. Each error maps to a
diagnostic code, which is what lets the CLI turn an exception into a readable
finding instead of a traceback.
"""

from __future__ import annotations

import difflib
from collections.abc import Iterable, Sequence

from rowless.diagnostics.codes import Code


def nearest_name(wanted: str, candidates: Iterable[str]) -> str | None:
    """The closest candidate name to `wanted`, or None when nothing is close.

    Lives with the errors because every near-miss suggestion rowless makes is
    attached to one. The cutoff is deliberately high: a wrong suggestion sends
    a user to edit the wrong line, which is worse than no suggestion at all.
    """
    matches = difflib.get_close_matches(wanted, list(candidates), n=1, cutoff=0.6)
    return matches[0] if matches else None


class RowlessError(Exception):
    """Base class. Carries the diagnostic code the CLI will report."""

    code: Code = Code.ADAPTER_ERROR


# --- User-facing: the pipeline is wrong --------------------------------------


class SchemaError(RowlessError):
    """A schema problem in the user's pipeline."""


class ColumnNotFound(SchemaError, KeyError):
    """A referenced column does not exist. The flagship finding.

    Also a `KeyError`, because `Schema` is a `Mapping` and `"amount" in schema`
    has to keep working. `__str__` is overridden so the message reads as a
    diagnostic rather than as `KeyError`'s bare repr.

    Carries the near-miss suggestion ("did you mean 'amount_cents'?"), because
    the rename case is the one in the proposal and a bare name is half an
    answer.
    """

    code = Code.COLUMN_NOT_FOUND

    def __init__(
        self,
        column: str,
        *,
        available: Sequence[str] = (),
        relation: str | None = None,
    ) -> None:
        self.column = column
        self.available = tuple(available)
        self.relation = relation
        self.suggestion = nearest_name(column, self.available)
        where = f" in {relation}" if relation else ""
        hint = f", did you mean {self.suggestion!r}?" if self.suggestion else ""
        self.message = f"ColumnNotFound {column!r}{where}{hint}"
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class DuplicateColumn(SchemaError):
    """Two columns with the same name after a join or rename."""

    code = Code.DUPLICATE_COLUMN


class DTypeMismatch(SchemaError):
    """An operation received a dtype the engine refuses."""

    code = Code.DTYPE_MISMATCH


class ContractViolation(SchemaError):
    """A computed schema does not satisfy a declared contract."""

    code = Code.CONTRACT_VIOLATION


# --- Lost confidence ---------------------------------------------------------


class UnknownSchemaError(RowlessError):
    """The schema could not be resolved. Never a pass; always reported."""

    code = Code.UNRESOLVED_SOURCE


class RowDependentControlFlow(UnknownSchemaError):
    """Execution depended on there being rows, so the result is not meaningful."""

    code = Code.ROW_DEPENDENT_CONTROL_FLOW


# --- Setup -------------------------------------------------------------------


class ConfigError(RowlessError):
    """Invalid `rowless.toml`."""

    code = Code.CONFIG_INVALID


class BadSourceRef(ConfigError):
    """Unparseable or unregistered source URI."""

    code = Code.BAD_SOURCE_REF


class SchemaNotFound(RowlessError):
    """A source resolved, but the table is not in it."""

    code = Code.SCHEMA_NOT_FOUND


class AdapterNotFound(RowlessError):
    """No adapter claims this object or engine name."""

    code = Code.ADAPTER_NOT_FOUND


class AmbiguousAdapter(AdapterNotFound):
    """Two adapters claimed the same object. An adapter bug, not a user error."""

    code = Code.ADAPTER_NOT_FOUND


class AdapterUnavailable(RowlessError):
    """The adapter exists but its engine library is missing. Names the extra."""

    code = Code.ADAPTER_UNAVAILABLE


class PluginLoadError(RowlessError):
    """An entry point failed to import."""

    code = Code.ADAPTER_ERROR


class UnsupportedDType(RowlessError):
    """An IR dtype has no equivalent in the target engine."""

    code = Code.IR_ROUND_TRIP_FAILED


# --- Sandbox and runtime -----------------------------------------------------


class SideEffectBlocked(RowlessError):
    """Sandboxed code attempted a write, a network call or a subprocess."""

    code = Code.SIDE_EFFECT_BLOCKED


class WorkerCrashed(RowlessError):
    """The dry-run subprocess died. Reported per pipeline; the run continues."""

    code = Code.WORKER_CRASHED


class WorkerTimeout(WorkerCrashed):
    """The dry run exceeded its limit, which usually means something executed."""

    code = Code.WORKER_TIMEOUT
