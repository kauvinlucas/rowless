"""The rowless runtime: run real code against rowless frames, safely.

`dryrun` orchestrates, `worker` isolates, `sandbox` contains side effects, and
`rowdep` catches logic that only makes sense when rows exist.
"""

from rowless.runtime.dryrun import DryRunRequest, DryRunResult, dry_run, dry_run_callable
from rowless.runtime.rowdep import RowDependency, RowDependencyLog
from rowless.runtime.sandbox import Policy, SideEffectLog, sandboxed

__all__ = [
    "DryRunRequest",
    "DryRunResult",
    "Policy",
    "RowDependency",
    "RowDependencyLog",
    "SideEffectLog",
    "dry_run",
    "dry_run_callable",
    "sandboxed",
]
