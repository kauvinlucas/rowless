"""Stable diagnostic codes.

Every finding has a code. Codes are public API: people put them in
`# rowless: ignore[RWL201]` comments and in CI allowlists, so they are append-
only and never recycled.

    RWL0xx  configuration and setup
    RWL1xx  schema resolution failures (the findings users came for)
    RWL2xx  lost confidence: unknown, unverified, degraded coverage
    RWL3xx  breaking changes, reported by `rowless diff`
    RWL4xx  sandbox and side effects
    RWL5xx  internal: adapter bugs, crashes, timeouts
"""

from __future__ import annotations

from enum import StrEnum


class Code(StrEnum):
    """The diagnostic code set."""

    # --- 0xx: setup ---
    NO_PIPELINES_FOUND = "RWL001"
    BAD_SOURCE_REF = "RWL002"
    ADAPTER_NOT_FOUND = "RWL003"
    ADAPTER_UNAVAILABLE = "RWL004"
    """The engine library is not installed. Suggests the right extra."""
    CONFIG_INVALID = "RWL005"
    SCHEMA_NOT_FOUND = "RWL006"

    # --- 1xx: schema resolution ---
    COLUMN_NOT_FOUND = "RWL101"
    """The headline diagnostic, and the one in the proposal's terminal mock."""
    DUPLICATE_COLUMN = "RWL102"
    DTYPE_MISMATCH = "RWL103"
    """An operation got a dtype the engine refuses: summing a string."""
    CONTRACT_VIOLATION = "RWL104"
    """The computed schema does not satisfy a declared contract."""
    JOIN_KEY_MISMATCH = "RWL105"
    AMBIGUOUS_COLUMN = "RWL106"
    """Post-join name collision the engine would resolve differently."""
    NULLABILITY_VIOLATION = "RWL107"
    PIPELINE_RAISED = "RWL108"
    """The pipeline raised, and no adapter recognised the exception. A real
    failure in the user's code, so it is an error rather than an internal one:
    reporting it as a rowless bug would blame the wrong party. An adapter that
    can classify it should translate it into a 1xx code instead."""

    # --- 2xx: lost confidence ---
    ROW_DEPENDENT_CONTROL_FLOW = "RWL201"
    """Hard problem 02: the dry run took a branch production would not."""
    UNKNOWN_UDF_RETURN = "RWL202"
    DYNAMIC_COLUMN_NAMES = "RWL203"
    """Column names computed at runtime, so lineage is necessarily partial."""
    COVERAGE_BELOW_THRESHOLD = "RWL204"
    UNRESOLVED_SOURCE = "RWL205"
    ANY_COLUMN_IN_OUTPUT = "RWL206"

    # --- 3xx: breaking changes ---
    BREAKING_COLUMN_REMOVED = "RWL301"
    BREAKING_COLUMN_RENAMED = "RWL302"
    BREAKING_DTYPE_CHANGED = "RWL303"
    RISKY_NULLABILITY_WIDENED = "RWL304"
    RISKY_TIMEZONE_CHANGED = "RWL305"
    UNVERIFIED_CHANGE = "RWL306"
    """Coverage too low to classify. Fails the gate by design."""

    # --- 4xx: sandbox ---
    SIDE_EFFECT_BLOCKED = "RWL401"
    NETWORK_BLOCKED = "RWL402"
    WRITE_RECORDED = "RWL403"
    """Informational: the pipeline writes here. Doubles as output lineage."""

    # --- 5xx: internal ---
    ADAPTER_ERROR = "RWL501"
    WORKER_CRASHED = "RWL502"
    WORKER_TIMEOUT = "RWL503"
    IR_ROUND_TRIP_FAILED = "RWL504"
    """An adapter's `to_ir`/`from_ir` disagree. Always an adapter bug."""


#: Codes meaning "rowless does not know", as opposed to "something is wrong".
#: They report at `Level.UNVERIFIED` and gate the run while
#: `fail_on_unverified` is on, which it is by default: a result nobody could
#: verify must not read as a pass.
UNVERIFIED_CODES: frozenset[Code] = frozenset(
    {
        Code.ROW_DEPENDENT_CONTROL_FLOW,
        Code.UNRESOLVED_SOURCE,
        Code.UNVERIFIED_CHANGE,
        # A pipeline rowless could not feed, or could not find an engine
        # for, is a pipeline rowless did not check - and an unchecked
        # pipeline looks exactly like a passing one.
        Code.SCHEMA_NOT_FOUND,
        Code.BAD_SOURCE_REF,
        Code.ADAPTER_UNAVAILABLE,
        Code.ADAPTER_NOT_FOUND,
    }
)


#: Codes that fail `rowless check` by default. Everything else is reported but
#: does not change the exit code, so a first run on a large repo is readable.
DEFAULT_ERROR_CODES: frozenset[Code] = frozenset(
    {
        Code.COLUMN_NOT_FOUND,
        Code.DUPLICATE_COLUMN,
        Code.DTYPE_MISMATCH,
        Code.CONTRACT_VIOLATION,
        Code.PIPELINE_RAISED,
        # Only ever emitted when a floor was configured, so including it here
        # keeps the default run reporting rather than blocking.
        Code.COVERAGE_BELOW_THRESHOLD,
        Code.JOIN_KEY_MISMATCH,
        Code.AMBIGUOUS_COLUMN,
        Code.BREAKING_COLUMN_REMOVED,
        Code.BREAKING_COLUMN_RENAMED,
        Code.BREAKING_DTYPE_CHANGED,
        Code.UNVERIFIED_CHANGE,
    }
)
