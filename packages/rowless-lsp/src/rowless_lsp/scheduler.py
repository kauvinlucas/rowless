"""Debouncing: the difference between a language server and a space heater.

A dry run spawns a subprocess, imports the user's module and imports an engine.
Doing that per keystroke is not slow, it is pathological. So every re-check
goes through here, one pending job per document, and a job that is superseded
before it runs is cancelled rather than queued.

Cancellation is the part that matters. A queue would make the editor fall
further behind the faster somebody types, which is the one latency failure a
user cannot work around; dropping superseded work means the cost of typing a
line is one check, not one per character.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

#: Default quiet period before a re-check. Short enough to feel like a
#: response to the edit that stopped, long enough that normal typing produces
#: one check rather than one per word.
DELAY_S = 0.4


class Debouncer:
    """One pending job per key, each cancelling the last."""

    def __init__(self, delay: float = DELAY_S) -> None:
        self.delay = delay
        self._pending: dict[str, asyncio.Task[None]] = {}

    def schedule(self, key: str, job: Callable[[], Awaitable[None]]) -> None:
        """Run `job` after the delay, cancelling any job already pending for `key`."""
        self.cancel(key)
        self._pending[key] = asyncio.get_running_loop().create_task(self._run(key, job))

    def cancel(self, key: str) -> None:
        """Drop the pending job for one key, if there is one."""
        existing = self._pending.pop(key, None)
        if existing is not None and not existing.done():
            existing.cancel()

    def cancel_all(self) -> None:
        """Drop every pending job. Called on shutdown."""
        for key in list(self._pending):
            self.cancel(key)

    @property
    def pending(self) -> frozenset[str]:
        """Keys with work outstanding. Used by tests and by shutdown."""
        return frozenset(key for key, task in self._pending.items() if not task.done())

    async def drain(self) -> None:
        """Wait for everything pending to finish.

        For tests and for an orderly shutdown. A cancelled job is a normal
        outcome here, not an error, so the gather swallows it.
        """
        tasks = [task for task in self._pending.values() if not task.done()]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _run(self, key: str, job: Callable[[], Awaitable[None]]) -> None:
        """Wait out the quiet period, then run the job once."""
        try:
            await asyncio.sleep(self.delay)
            await job()
        except asyncio.CancelledError:
            raise
        finally:
            if self._pending.get(key) is asyncio.current_task():
                del self._pending[key]
