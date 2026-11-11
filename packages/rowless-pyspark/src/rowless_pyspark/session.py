"""Per-run state, in one place so nothing else has to own a global.

Four things in this package need it, which is why it is a module rather than a
few globals in `interception`: the wrappers have to know whether a dry run is
open, the Spark session cache has to know which run's scratch directory to
configure itself with, the writers need somewhere to report a side effect they
refused to perform, and the adapter's own calls back into Spark - analyzing a
plan, reading a schema back - must not be reported as something the pipeline
did.

Two pieces of state, and the distinction between them is the whole point:

* **the run**, identified by `DryRunContext.run_id`. Changing it means a new
  dry run, and everything cached for the previous one is stale. The Spark
  session deliberately is *not*: starting one is the expensive part, and it is
  reused across every run in a check.
* **whose work is running**, tracked by `own_work()`. A wrapper that analyzes a
  plan calls Spark, and that call is not the pipeline reading data.

Everything here depends on nothing else in this package, so the import graph
stays acyclic.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from rowless.adapters.base import DryRunContext
from rowless.runtime.rowdep import active_log

#: True while this adapter is running its own bookkeeping: analyzing a plan,
#: reading a schema back, registering a temporary view. Keeps a patched method
#: that the adapter itself calls from being reported as a pipeline side effect.
_INSIDE: ContextVar[bool] = ContextVar("rowless_pyspark_inside", default=False)

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


def options() -> Mapping[str, Any]:
    """The adapter's options from `[engine.pyspark]`, or nothing outside a run."""
    ctx = _CONTEXT
    return ctx.options if ctx is not None else {}


def option(name: str, default: Any = None) -> Any:
    """One `[engine.pyspark]` option, with a default for when it is not set."""
    return options().get(name, default)


def workdir() -> str | None:
    """The run's scratch directory: the only path the sandbox permits writes to."""
    ctx = _CONTEXT
    return ctx.workdir if ctx is not None else None


def tracking() -> bool:
    """True while a dry run is executing user code.

    The gate on every wrapper in this package. Outside a dry run they call
    straight through, so a process that happens to have imported this adapter
    gets unmodified PySpark.
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

    `once_for` deduplicates: a pipeline that writes in a loop should say so
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
