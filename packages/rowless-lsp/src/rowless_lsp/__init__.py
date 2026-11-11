"""Language server for rowless: schema diagnostics, completion and inline hints.

The package is a front end, not an adapter. Every finding it publishes comes
from `rowless.check.run`, the same call `rowless check` makes, so the editor
cannot have a second opinion about what is broken. What it adds is the warm
state that makes that answer usable interactively: one config per workspace,
one index per buffer, and the last check's schemas kept so completion and
hover answer without running anything.

    rowless-lsp            speak LSP over stdio, as an editor launches it
    rowless-lsp --tcp      listen on a socket, for attaching a debugger
"""

from rowless_lsp.documents import ColumnUse, DocumentIndex, IndexedFunction, Span, Use
from rowless_lsp.live import live_diagnostics
from rowless_lsp.server import RowlessLanguageServer, create_server, main
from rowless_lsp.session import Checked, Document, Resolved, Session

__all__ = [
    "Checked",
    "ColumnUse",
    "Document",
    "DocumentIndex",
    "IndexedFunction",
    "Resolved",
    "RowlessLanguageServer",
    "Session",
    "Span",
    "Use",
    "create_server",
    "live_diagnostics",
    "main",
]
