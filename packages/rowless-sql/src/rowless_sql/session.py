"""Per-run state, in one place so nothing else has to own a global.

Two things need it, which is why it is a module rather than a few globals
spread around the package: the adapter has to know whether a dry run is open -
that is what decides whether a `str` is SQL or an ordinary string - and the
catalogue of registered input schemas has to start empty when a new run
begins.

Two pieces of state, and the distinction between them is the point:

* **the run**, identified by `DryRunContext.run_id`. Changing it means a new
  dry run, and everything cached for the previous one is stale.
* **the catalogue**, which is this adapter's equivalent of DuckDB's scratch
  database: `materialize` records a schema under a table name, and `describe`
  resolves a query against what was recorded. It lives here rather than in
  `catalog` so that the lifetime rule - one catalogue per run - is written in
  the same place as the run.

What is *not* here is the "whose work is running" flag every other adapter in
the project needs. Those patch their engine, so a wrapper has to tell the
adapter resolving a schema from the pipeline reading data. Nothing is patched
here - there is nothing to intercept and no side effect to stop - so there is
nothing to mark.

Everything here depends on nothing else in this package except `catalog`, so
the import graph stays acyclic.
"""

from __future__ import annotations

from collections.abc import Callable

from rowless.adapters.base import DryRunContext

from rowless_sql.catalog import Catalog

_CONTEXT: DryRunContext | None = None
_RUN_ID: str | None = None

#: The catalogue for the open run: table name to the schema rowless resolved
#: for it. Replaced wholesale when a new run begins, never mutated between
#: runs, so a stale input cannot resolve a later query.
_CATALOG: Catalog = Catalog()

#: Called when a new run starts, so each module can drop what it cached for the
#: previous one. Registered at import time by the modules that cache.
_RESETS: list[Callable[[], None]] = []

#: Notes already written for this run, so a query that degrades the same way
#: forty times says so once.
_NOTED: set[str] = set()


def on_reset(callback: Callable[[], None]) -> None:
    """Register `callback` to run when a new dry run begins."""
    _RESETS.append(callback)


def begin(ctx: DryRunContext) -> None:
    """Bind this adapter to `ctx`, resetting per-run state if the run changed."""
    global _CONTEXT, _RUN_ID, _CATALOG

    _CONTEXT = ctx
    if ctx.run_id == _RUN_ID:
        return
    _RUN_ID = ctx.run_id
    _CATALOG = Catalog()
    _NOTED.clear()
    for reset in _RESETS:
        reset()


def context() -> DryRunContext | None:
    """The dry-run context this adapter reports into, if a run is open."""
    return _CONTEXT


def catalog() -> Catalog:
    """The catalogue of input schemas for the open run.

    What `materialize` writes to and what `describe` resolves against. A
    process with no run open has an empty one rather than none, so nothing has
    to check.
    """
    return _CATALOG


def sql_mode() -> bool:
    """True when a `str` handed to this adapter should be read as SQL.

    A SQL model is a string, which is the one frame type that collides with
    every other use of a string in Python. So this adapter claims `str` only
    while it has a dry run open: inside a run, a pipeline that returns a
    string is returning a query, and `describe` resolves it against the
    catalogue. Outside one, `handles("...")` is False and
    `rowless.adapters.registry.adapter_for` cannot route an arbitrary string
    here.
    """
    return _CONTEXT is not None


def note(message: str, *, once_for: str | None = None) -> None:
    """Record adapter commentary on the open context, if there is one.

    `once_for` deduplicates: a run that resolves forty models against the same
    untyped function should say so once, and the key is what identifies "the
    same thing happening again".
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
