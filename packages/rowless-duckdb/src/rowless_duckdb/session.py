"""Per-run state, in one place so nothing else has to own a global.

The Polars adapter keeps this inside its interception module, because the only
thing that needs it is the patching. DuckDB needs it in three: the wrappers
have to know whether a dry run is open, the scratch database has to be rebuilt
when a new run starts, and the lineage parser has to borrow a connection
without being reported as user code. A module of its own keeps the import graph
acyclic - everything here depends on nothing in this package.

Two pieces of state, and the distinction between them is the whole point:

* **the run**, identified by `DryRunContext.run_id`. Changing it means a new
  dry run, and everything cached for the previous one is stale.
* **whose work is running**, tracked by `own_work()`. A wrapper that resolves a
  schema calls back into DuckDB, and those calls must not be reported as
  something the pipeline did.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from rowless.adapters.base import DryRunContext
from rowless.runtime.rowdep import active_log

#: True while this adapter is running its own bookkeeping: resolving a schema,
#: parsing SQL, reading nullability back. Keeps a patched method that the
#: adapter itself calls from being reported as a pipeline side effect.
_INSIDE: ContextVar[bool] = ContextVar("rowless_duckdb_inside", default=False)

_CONTEXT: DryRunContext | None = None
_RUN_ID: str | None = None

#: Called when a new run starts, so each module can drop what it cached for
#: the previous one. Registered at import time by the modules that cache.
_RESETS: list[Callable[[], None]] = []

#: Notes already written for this run, so a pipeline that writes in a loop says
#: so once rather than forty times.
_NOTED: set[str] = set()


def on_reset(callback: Callable[[], None]) -> None:
    """Register `callback` to run when a new dry run begins."""
    _RESETS.append(callback)


def begin(ctx: DryRunContext) -> None:
    """Bind this adapter to `ctx`, resetting per-run caches if the run changed."""
    global _CONTEXT, _RUN_ID

    _CONTEXT = ctx
    if ctx.run_id == _RUN_ID:
        return
    _RUN_ID = ctx.run_id
    _NOTED.clear()
    for reset in _RESETS:
        reset()


def context() -> DryRunContext | None:
    """The dry-run context the wrappers report into, if a run is open."""
    return _CONTEXT


def run_id() -> str | None:
    """The identifier of the run currently open, if any."""
    return _RUN_ID


def tracking() -> bool:
    """True while a dry run is executing user code.

    The gate on every wrapper in this package. Outside a dry run they call
    straight through, so a process that happens to have imported this adapter
    gets unmodified DuckDB.
    """
    return active_log() is not None and not _INSIDE.get()


@contextmanager
def own_work() -> Iterator[None]:
    """Mark the frames below as this adapter working, not as user code."""
    token = _INSIDE.set(True)
    try:
        yield
    finally:
        _INSIDE.reset(token)


def note(message: str, *, once_for: str | None = None) -> None:
    """Record adapter commentary on the open context, if there is one.

    `once_for` deduplicates: a pipeline that collects in a loop should say so
    once, and the key is what identifies "the same thing happening again".
    """
    ctx = _CONTEXT
    if ctx is None:
        return
    if once_for is not None:
        if once_for in _NOTED:
            return
        _NOTED.add(once_for)
    ctx.note(message)


def degrade(reason: str) -> None:
    """Flag that this run lost confidence, if a context is open to record it."""
    ctx = _CONTEXT
    if ctx is not None:
        ctx.degrade(reason)
