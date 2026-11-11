"""The language server: catch it in the editor, not in CI.

"In the editor" is the first clause of the proposal, and it is the part that
changes how the tool feels. A CI failure tells you after you pushed; a squiggle
under a column name tells you while you are typing it.

What makes this hard is not LSP, it is latency. A dry run imports the user's
module in a subprocess, so it cannot run per keystroke and it cannot run on an
unsaved buffer at all. The split that falls out of those two facts is the whole
design of this package:

* **on save**, a real check through `rowless.check.run` - the same call the CLI
  makes, so the editor and CI cannot disagree about what is broken
* **while typing**, the findings the buffer alone supports (`rowless_lsp.live`),
  plus the previous check's findings re-anchored onto the moved text
  (`rowless_lsp.anchors`) and dropped where their line is gone
* **everything else** - completion, hover, inlay hints, code actions - served
  from the last successful check's schemas, which is what makes them instant

Every handler here is thin. The work is in `session`, `live`, `documents` and
the four feature modules; this file is the protocol wiring and the thread
boundary, so that a second front end could reuse the rest of it.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from collections.abc import Mapping, Sequence
from typing import Any

from lsprotocol import types
from pygls.lsp.server import LanguageServer
from rowless import __version__ as rowless_version
from rowless.diagnostics.models import Diagnostic

from rowless_lsp.actions import code_actions
from rowless_lsp.completion import TRIGGER_CHARACTERS, complete
from rowless_lsp.convert import path_of, to_lsp, to_path, to_uri
from rowless_lsp.hints import inlay_hints
from rowless_lsp.hover import hover
from rowless_lsp.scheduler import DELAY_S, Debouncer
from rowless_lsp.session import Session

#: The server name clients show in their output panel.
NAME = "rowless"

#: Command clients can invoke to force a full workspace check, which is the
#: escape hatch for the case this server cannot see: a source table changed,
#: so nothing in the repository looks different and every schema is stale.
RECHECK = "rowless.recheck"

#: Debouncer key for the workspace-wide check, which is not per document.
WORKSPACE = "<workspace>"


class RowlessLanguageServer(LanguageServer):
    """A pygls server carrying one rowless session and one debouncer.

    The session holds every cached schema, so the server instance *is* the
    warm state: a test can build one, hand it a document and call a handler
    with no editor and no event loop in sight.
    """

    def __init__(
        self,
        *,
        root: str | None = None,
        delay: float = DELAY_S,
        name: str = NAME,
        version: str = rowless_version,
    ) -> None:
        # Full document sync, not incremental. Every change re-indexes the
        # whole buffer anyway - `ast.parse` has no incremental mode - so
        # applying a delta would be work done twice, and the notification
        # carrying the text is what lets a handler be driven without a client
        # attached.
        super().__init__(
            name=name,
            version=version,
            text_document_sync_kind=types.TextDocumentSyncKind.Full,
        )
        self._root = root
        self._session: Session | None = None
        self.debouncer = Debouncer(delay)

    @property
    def session(self) -> Session:
        """The workspace session, created on first use.

        Lazy because the workspace root arrives with `initialize`, and the
        config walk it triggers should not happen before the client has said
        where the repository is.
        """
        if self._session is None:
            self._session = Session(self._root or self._workspace_root())
        return self._session

    def _workspace_root(self) -> str:
        """The repository root, from the client's workspace or the process cwd."""
        try:
            root = self.workspace.root_path
        except (AttributeError, RuntimeError):  # pragma: no cover - no client attached
            root = None
        return root or os.getcwd()

    # --- Publishing ----------------------------------------------------------

    def publish(self, path: str) -> None:
        """Publish the current findings for one open document."""
        document = self.session.document(path)
        if document is None:
            return
        index = document.usable
        diagnostics = self.session.diagnostics(path)
        self.text_document_publish_diagnostics(
            types.PublishDiagnosticsParams(
                uri=to_uri(path),
                version=document.version or None,
                diagnostics=[to_lsp(diagnostic, index) for diagnostic in diagnostics],
            )
        )

    def publish_grouped(self, grouped: Mapping[str, list[Diagnostic]]) -> None:
        """Publish a check's findings to every file they landed in.

        A check of one Python file can report a finding against a dbt model,
        and a diagnostic published under the wrong URI is a diagnostic nobody
        sees. Open documents go through `publish` so that they pick up the
        buffer-only findings too; closed ones are published as they came back.
        """
        for key, diagnostics in grouped.items():
            document = self.session.document(key)
            if document is not None:
                self.publish(document.path)
                continue
            self.text_document_publish_diagnostics(
                types.PublishDiagnosticsParams(
                    uri=to_uri(path_of(diagnostics) or key),
                    diagnostics=[to_lsp(diagnostic) for diagnostic in diagnostics],
                )
            )
        for document in self.session.documents.values():
            if _key(document.path) not in grouped:
                self.publish(document.path)

    def clear(self, path: str) -> None:
        """Withdraw every finding for a file, as a close must."""
        self.text_document_publish_diagnostics(
            types.PublishDiagnosticsParams(uri=to_uri(path), diagnostics=[])
        )

    # --- Checking ------------------------------------------------------------

    async def check(self, path: str) -> None:
        """Run one file's dry runs off the event loop, then publish.

        In a thread because `rowless.check.run` is synchronous and spawns
        subprocesses: holding the loop for the length of a check would stop
        the server answering completion, which is the request a user is most
        likely to make while waiting for one.
        """
        grouped = await asyncio.to_thread(self.session.check, path)
        self.publish_grouped(grouped)

    async def check_workspace(self) -> None:
        """Check every pipeline under the configured roots, then publish."""
        grouped = await asyncio.to_thread(self.session.check_workspace)
        self.publish_grouped(grouped)


def create_server(*, root: str | None = None, delay: float = DELAY_S) -> RowlessLanguageServer:
    """Build the pygls server with the rowless feature handlers attached."""
    server = RowlessLanguageServer(root=root, delay=delay)

    server.feature(types.INITIALIZED)(on_initialized)
    server.feature(types.TEXT_DOCUMENT_DID_OPEN)(on_did_open)
    server.feature(types.TEXT_DOCUMENT_DID_CHANGE)(on_did_change)
    server.feature(types.TEXT_DOCUMENT_DID_SAVE)(on_did_save)
    server.feature(types.TEXT_DOCUMENT_DID_CLOSE)(on_did_close)
    server.feature(
        types.TEXT_DOCUMENT_COMPLETION,
        types.CompletionOptions(trigger_characters=list(TRIGGER_CHARACTERS)),
    )(on_completion)
    server.feature(types.TEXT_DOCUMENT_HOVER)(on_hover)
    server.feature(types.TEXT_DOCUMENT_INLAY_HINT)(on_inlay_hint)
    server.feature(
        types.TEXT_DOCUMENT_CODE_ACTION,
        types.CodeActionOptions(code_action_kinds=[types.CodeActionKind.QuickFix]),
    )(on_code_action)
    server.feature(types.WORKSPACE_DID_CHANGE_WATCHED_FILES)(on_watched_files_changed)
    server.command(RECHECK)(on_recheck)
    return server


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the `rowless-lsp` command. Speaks LSP over stdio."""
    parser = argparse.ArgumentParser(prog="rowless-lsp", description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="store_true", help="Print the version and exit.")
    parser.add_argument("--tcp", action="store_true", help="Listen on a socket instead of stdio.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2087)
    parser.add_argument(
        "--debounce",
        type=float,
        default=DELAY_S,
        metavar="SECONDS",
        help=f"Quiet period before a re-check. Default {DELAY_S:g}.",
    )
    options = parser.parse_args(list(argv) if argv is not None else None)
    if options.version:
        print(f"rowless-lsp {rowless_version}")
        return 0

    server = create_server(delay=options.debounce)
    if options.tcp:
        # For attaching a debugger or a second editor to a running server;
        # stdio is what an editor actually launches.
        server.start_tcp(options.host, options.port)
    else:
        server.start_io()
    return 0


# --- Feature handlers, in the order they are worth building ------------------


async def on_did_save(ls: RowlessLanguageServer, params: types.DidSaveTextDocumentParams) -> None:
    """Re-check the saved file and publish diagnostics. The first milestone.

    Debounced like everything else, because "save on focus change" turns one
    edit into several saves, and the second check of an unchanged file is pure
    waste.
    """
    path = _path(params.text_document.uri)
    if path is None:
        return
    ls.debouncer.schedule(f"check:{_key(path)}", lambda: ls.check(path))


async def on_did_change(
    ls: RowlessLanguageServer,
    params: types.DidChangeTextDocumentParams,
) -> None:
    """Debounced incremental re-publish from the buffer.

    No dry run: the module on disk is not what is in the buffer, and running
    the old file to report on the new one would be worse than saying nothing.
    What this does publish is current - the live checks read the buffer, and
    the carried-over findings are re-anchored or dropped.
    """
    path = _path(params.text_document.uri)
    text = _whole_document(params.content_changes)
    if path is None or text is None:
        return
    version = params.text_document.version or 0
    if ls.session.document(path) is None:
        ls.session.open(path, text, version)
    else:
        ls.session.update(path, text, version)

    async def republish() -> None:
        ls.publish(path)

    ls.debouncer.schedule(f"live:{_key(path)}", republish)


async def on_did_open(ls: RowlessLanguageServer, params: types.DidOpenTextDocumentParams) -> None:
    """Index a newly opened buffer and check it.

    The check is what warms every other feature: completion and hover have
    nothing to offer until a schema has resolved once.
    """
    path = _path(params.text_document.uri)
    if path is None:
        return
    ls.session.open(path, params.text_document.text, params.text_document.version)
    ls.debouncer.schedule(f"check:{_key(path)}", lambda: ls.check(path))


async def on_did_close(ls: RowlessLanguageServer, params: types.DidCloseTextDocumentParams) -> None:
    """Drop a closed buffer's state and withdraw its findings."""
    path = _path(params.text_document.uri)
    if path is None:
        return
    ls.debouncer.cancel(f"check:{_key(path)}")
    ls.debouncer.cancel(f"live:{_key(path)}")
    ls.session.close(path)
    ls.clear(path)


async def on_initialized(ls: RowlessLanguageServer, params: types.InitializedParams) -> None:
    """Check the whole workspace once, in the background.

    So that the problem list an editor shows on open is the one CI would
    produce, rather than only covering the files somebody happened to click.
    """
    _ = params
    ls.debouncer.schedule(WORKSPACE, ls.check_workspace)


def on_inlay_hint(
    ls: RowlessLanguageServer,
    params: types.InlayHintParams,
) -> list[types.InlayHint]:
    """Show the resolved schema at the end of a dataframe expression chain."""
    document = _document(ls, params.text_document.uri)
    if document is None:
        return []
    return inlay_hints(document, params.range)


def on_hover(ls: RowlessLanguageServer, params: types.HoverParams) -> types.Hover | None:
    """Show a column dtype, its origin, and its upstream lineage on hover."""
    document = _document(ls, params.text_document.uri)
    if document is None:
        return None
    return hover(document, params.position)


def on_completion(
    ls: RowlessLanguageServer,
    params: types.CompletionParams,
) -> types.CompletionList:
    """Complete column names inside `pl.col(...)` and `df[...]` from the schema.

    The feature people will actually notice, and free once schemas resolve.
    """
    document = _document(ls, params.text_document.uri)
    items = complete(document, params.position) if document is not None else []
    # Never incomplete: the candidate list is a resolved schema, so there is
    # no second page for the client to come back for.
    return types.CompletionList(is_incomplete=False, items=items)


def on_code_action(
    ls: RowlessLanguageServer,
    params: types.CodeActionParams,
) -> list[types.CodeAction]:
    """Offer a fix for a near-miss column name, and an ignore comment."""
    path = _path(params.text_document.uri)
    document = _document(ls, params.text_document.uri)
    if path is None or document is None:
        return []
    return code_actions(document, ls.session.diagnostics(path), params.range)


async def on_watched_files_changed(
    ls: RowlessLanguageServer,
    params: types.DidChangeWatchedFilesParams,
) -> None:
    """Re-read configuration and re-check when a config file changes.

    `rowless.toml` decides which engines run, which paths are excluded and
    which codes are errors. A cached config outliving the file is how a server
    ends up reporting findings the user has already configured away.
    """
    if any(_is_config(change.uri) for change in params.changes):
        ls.session.reload()
    ls.debouncer.schedule(WORKSPACE, ls.check_workspace)


async def on_recheck(ls: RowlessLanguageServer, *args: Any) -> None:
    """Force a full workspace check, ignoring every cache but the source cache.

    For the change this server cannot see: an upstream table was altered, so
    nothing in the repository looks different and every resolved schema is
    quietly out of date.
    """
    _ = args
    ls.session.reload()
    await ls.check_workspace()


# --- Plumbing ----------------------------------------------------------------


def _document(ls: RowlessLanguageServer, uri: str) -> Any:
    """The session's record for a URI, or None when it is not a local file."""
    path = _path(uri)
    return None if path is None else ls.session.document(path)


def _whole_document(changes: Sequence[Any]) -> str | None:
    """The buffer's new contents out of a change notification.

    The server syncs whole documents, so the last change carries the whole
    text. A client that sent a range change anyway gets None rather than a
    guess at what it meant.
    """
    for change in reversed(list(changes)):
        if getattr(change, "range", None) is not None:
            continue
        text = getattr(change, "text", None)
        if text is not None:
            return str(text)
    return None


def _path(uri: str) -> str | None:
    """A document URI as an absolute path, or None for a non-file URI."""
    path = to_path(uri)
    return os.path.abspath(path) if path else None


def _is_config(uri: str) -> bool:
    """True when a changed file is one rowless reads its settings from."""
    path = to_path(uri) or uri
    return os.path.basename(path) in {"rowless.toml", "pyproject.toml"}


def _key(path: str) -> str:
    """The canonical key for a path, matching the session's own."""
    return os.path.normcase(os.path.abspath(path))
