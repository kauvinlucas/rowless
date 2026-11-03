"""The language server: catch it in the editor, not in CI.

"In the editor" is the first clause of the proposal, and it is the part that
changes how the tool feels. A CI failure tells you after you pushed; a squiggle
under a column name tells you while you are typing it.

What makes this hard is not LSP, it is latency. A dry run per keystroke is not
viable, so the server:

* debounces, and only re-runs the pipelines whose file changed
* keeps resolved source schemas in memory for the session
* reuses one warm worker subprocess per engine rather than spawning per run,
  which is the single biggest win and the reason `rowless.runtime.worker` is
  designed around a request-response loop rather than one-shot invocation

Beyond diagnostics, the feature worth building is inline hints: the resolved
schema shown as an inlay at the end of a chain, so a developer can see what a
`group_by` produced without running anything.
"""

from __future__ import annotations

from typing import Any


def create_server() -> Any:
    """Build the pygls server with the rowless feature handlers attached."""
    raise NotImplementedError("TODO(lsp)")


def main() -> int:
    """Entry point for the `rowless-lsp` command. Speaks LSP over stdio."""
    raise NotImplementedError("TODO(lsp)")


# --- Feature handlers, in the order they are worth building ------------------


def on_did_save(params: Any) -> None:
    """Re-check the saved file and publish diagnostics. The first milestone."""
    raise NotImplementedError("TODO(lsp)")


def on_did_change(params: Any) -> None:
    """Debounced incremental re-check. Needs the warm-worker pool."""
    raise NotImplementedError("TODO(lsp)")


def on_inlay_hint(params: Any) -> list[Any]:
    """Show the resolved schema at the end of a dataframe expression chain."""
    raise NotImplementedError("TODO(lsp)")


def on_hover(params: Any) -> Any:
    """Show a column dtype, its origin, and its upstream lineage on hover."""
    raise NotImplementedError("TODO(lsp)")


def on_completion(params: Any) -> list[Any]:
    """Complete column names inside `pl.col(...)` and `df[...]` from the schema.

    The feature people will actually notice, and free once schemas resolve.
    """
    raise NotImplementedError("TODO(lsp)")


def on_code_action(params: Any) -> list[Any]:
    """Offer a fix for a near-miss column name, and an ignore comment."""
    raise NotImplementedError("TODO(lsp)")
