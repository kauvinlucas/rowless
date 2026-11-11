"""Failures this package raises, mapped onto core's diagnostic codes.

Both descend from `RowlessError`, so `rowless emit` reports them as findings
with an exit code rather than as a traceback: a catalog being unreachable is an
ordinary operational outcome, not a bug in rowless.
"""

from __future__ import annotations

from rowless.errors import AdapterUnavailable, RowlessError


class TransportNotConfigured(AdapterUnavailable):
    """There is nowhere to send events.

    An `AdapterUnavailable`, which is the same code a missing engine library
    gets, because it is the same situation: the plugin is installed and cannot
    run. The message names the variable to set, since that is the whole fix.
    """


class EmitFailed(RowlessError):
    """The transport refused or failed part-way through.

    Carries how many events made it, because a half-emitted graph is the one
    failure mode that leaves a catalog in a state somebody has to reason about.
    """

    def __init__(self, message: str, *, sent: int = 0, total: int = 0) -> None:
        self.sent = sent
        self.total = total
        progress = f" ({sent} of {total} event(s) sent)" if total else ""
        super().__init__(f"{message}{progress}")
