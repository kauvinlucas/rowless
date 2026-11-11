"""Server tests: drive the handlers directly, no editor involved.

The handlers are thin by design, so what these assert is the wiring: that a
notification reaches the session, that the right thing is published to the
right URI, and that the requests answer from cache rather than running a
check. Everything about *what* is found lives in the other test modules.

Publishing is captured rather than mocked out: the handler builds a real
`PublishDiagnosticsParams` and this records it, which is the closest thing to
an editor that does not need one.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from lsp_fixtures import (  # noqa: F401
    FIXED,
    PIPELINE,
    in_process,
    pipeline_path,
    project,
)
from lsprotocol import types
from rowless.diagnostics.codes import Code
from rowless.diagnostics.models import Diagnostic, Level
from rowless.ir.schema import Origin
from rowless_lsp.convert import path_of, to_path, to_uri
from rowless_lsp.server import (
    RECHECK,
    create_server,
    main,
    on_code_action,
    on_completion,
    on_did_change,
    on_did_close,
    on_did_open,
    on_did_save,
    on_hover,
    on_inlay_hint,
    on_recheck,
    on_watched_files_changed,
)


class Recorder:
    """A server that keeps what it would have published."""

    def __init__(self, root: str) -> None:
        self.server = create_server(root=root, delay=0.0)
        self.published: list[types.PublishDiagnosticsParams] = []
        # Swapped in rather than mocked: the handler still builds a real
        # `PublishDiagnosticsParams`, which is the part worth asserting.
        setattr(  # noqa: B010
            self.server, "text_document_publish_diagnostics", self._capture
        )

    def _capture(self, params: types.PublishDiagnosticsParams) -> None:
        """Stand in for the notification an attached client would receive."""
        self.published.append(params)

    def codes_for(self, path: str) -> list[str]:
        """The codes last published against one file."""
        uri = to_uri(path)
        for params in reversed(self.published):
            if params.uri == uri:
                return [str(diagnostic.code) for diagnostic in params.diagnostics]
        return []

    def uris(self) -> list[str]:
        """Every URI published to, in order."""
        return [params.uri for params in self.published]


@pytest.fixture
def recorder(project: Path, in_process: object) -> Recorder:
    """A server rooted in the fixture repository, with publishing captured."""
    _ = in_process
    return Recorder(str(project))


def drive(recorder: Recorder, handler: Callable[..., Any], params: Any) -> None:
    """Run a notification handler and let its debounced work finish."""

    async def exercise() -> None:
        await handler(recorder.server, params)
        await recorder.server.debouncer.drain()

    asyncio.run(exercise())


def opened(path: Path, text: str = PIPELINE) -> types.DidOpenTextDocumentParams:
    """The notification an editor sends when a file is opened."""
    return types.DidOpenTextDocumentParams(
        text_document=types.TextDocumentItem(
            uri=to_uri(str(path)), language_id="python", version=1, text=text
        )
    )


def saved(path: Path) -> types.DidSaveTextDocumentParams:
    """The notification an editor sends on save."""
    return types.DidSaveTextDocumentParams(
        text_document=types.TextDocumentIdentifier(uri=to_uri(str(path)))
    )


# --- Diagnostics -------------------------------------------------------------


def test_publishes_diagnostics_on_save(recorder: Recorder, pipeline_path: Path) -> None:
    """A file with a bad column reference yields one diagnostic on the right line."""
    drive(recorder, on_did_open, opened(pipeline_path))
    recorder.published.clear()
    drive(recorder, on_did_save, saved(pipeline_path))

    assert recorder.codes_for(str(pipeline_path)) == [Code.COLUMN_NOT_FOUND.value]
    published = recorder.published[-1]
    assert published.diagnostics[0].range.start.line == 10


def test_opening_a_file_checks_it(recorder: Recorder, pipeline_path: Path) -> None:
    """The check on open is what warms completion and hover."""
    drive(recorder, on_did_open, opened(pipeline_path))

    assert recorder.codes_for(str(pipeline_path)) == [Code.COLUMN_NOT_FOUND.value]
    document = recorder.server.session.document(str(pipeline_path))
    assert document is not None
    assert document.checked is not None


def test_a_fixed_file_republishes_an_empty_list(
    recorder: Recorder,
    pipeline_path: Path,
) -> None:
    """Withdrawing a finding needs an explicit empty publish, not silence."""
    drive(recorder, on_did_open, opened(pipeline_path))
    pipeline_path.write_text(FIXED, encoding="utf-8")
    drive(
        recorder,
        on_did_change,
        types.DidChangeTextDocumentParams(
            text_document=types.VersionedTextDocumentIdentifier(
                uri=to_uri(str(pipeline_path)), version=2
            ),
            content_changes=[types.TextDocumentContentChangeWholeDocument(text=FIXED)],
        ),
    )
    recorder.published.clear()
    drive(recorder, on_did_save, saved(pipeline_path))

    assert recorder.codes_for(str(pipeline_path)) == []


def test_a_change_publishes_without_running_a_dry_run(
    recorder: Recorder,
    pipeline_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The buffer is not what is on disk, so running the module would be a lie.

    What the change does publish is still current: the live checks read the
    buffer, and the carried findings are re-anchored onto it.
    """
    drive(recorder, on_did_open, opened(pipeline_path))

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("a buffer change must not trigger a dry run")

    monkeypatch.setattr("rowless.check.run", refuse)
    recorder.published.clear()
    drive(
        recorder,
        on_did_change,
        types.DidChangeTextDocumentParams(
            text_document=types.VersionedTextDocumentIdentifier(
                uri=to_uri(str(pipeline_path)), version=2
            ),
            content_changes=[
                types.TextDocumentContentChangeWholeDocument(text="# moved\n" + PIPELINE)
            ],
        ),
    )

    assert recorder.codes_for(str(pipeline_path)) == [Code.COLUMN_NOT_FOUND.value]
    assert recorder.published[-1].diagnostics[0].range.start.line == 11


def test_closing_a_file_withdraws_its_findings(
    recorder: Recorder,
    pipeline_path: Path,
) -> None:
    """A problem list that keeps a closed file's findings is a leak."""
    drive(recorder, on_did_open, opened(pipeline_path))
    recorder.published.clear()
    drive(
        recorder,
        on_did_close,
        types.DidCloseTextDocumentParams(
            text_document=types.TextDocumentIdentifier(uri=to_uri(str(pipeline_path)))
        ),
    )

    assert recorder.codes_for(str(pipeline_path)) == []
    assert recorder.server.session.document(str(pipeline_path)) is None


def test_a_config_change_rediscovers_and_rechecks(
    recorder: Recorder,
    project: Path,
    pipeline_path: Path,
) -> None:
    """Settings must take effect without restarting the editor."""
    drive(recorder, on_did_open, opened(pipeline_path))
    first = recorder.server.session.config
    (project / "rowless.toml").write_text(
        'roots = ["."]\nignore_codes = ["RWL101"]\n\n'
        '[sources]\n"lake://*" = "json://schemas/{name}.json"\n',
        encoding="utf-8",
    )
    recorder.published.clear()
    drive(
        recorder,
        on_watched_files_changed,
        types.DidChangeWatchedFilesParams(
            changes=[
                types.FileEvent(
                    uri=to_uri(str(project / "rowless.toml")),
                    type=types.FileChangeType.Changed,
                )
            ]
        ),
    )

    assert recorder.server.session.config is not first
    assert recorder.codes_for(str(pipeline_path)) == []


def test_the_recheck_command_runs_a_full_check(
    recorder: Recorder,
    pipeline_path: Path,
) -> None:
    """For the change this server cannot see: an upstream table was altered."""
    drive(recorder, on_did_open, opened(pipeline_path))
    recorder.published.clear()
    asyncio.run(on_recheck(recorder.server))

    assert recorder.codes_for(str(pipeline_path)) == [Code.COLUMN_NOT_FOUND.value]


# --- Requests ----------------------------------------------------------------


def test_completion_offers_schema_columns(recorder: Recorder, pipeline_path: Path) -> None:
    """Completion inside pl.col() returns the resolved input columns."""
    drive(recorder, on_did_open, opened(pipeline_path))
    line = PIPELINE.splitlines()[10]
    found = on_completion(
        recorder.server,
        types.CompletionParams(
            text_document=types.TextDocumentIdentifier(uri=to_uri(str(pipeline_path))),
            position=types.Position(line=10, character=line.index("amount")),
        ),
    )

    assert found.is_incomplete is False
    assert "amount_cents" in [item.label for item in found.items]


def test_completion_on_an_unknown_document_is_empty(recorder: Recorder) -> None:
    """A request for a file the server never saw must not raise."""
    found = on_completion(
        recorder.server,
        types.CompletionParams(
            text_document=types.TextDocumentIdentifier(uri="file:///nowhere.py"),
            position=types.Position(line=0, character=0),
        ),
    )

    assert found.items == []


def test_hover_answers_from_the_cached_schemas(
    recorder: Recorder,
    pipeline_path: Path,
) -> None:
    """No check runs for a hover; that is the point of keeping the schemas."""
    drive(recorder, on_did_open, opened(pipeline_path))
    line = PIPELINE.splitlines()[9]
    found = on_hover(
        recorder.server,
        types.HoverParams(
            text_document=types.TextDocumentIdentifier(uri=to_uri(str(pipeline_path))),
            position=types.Position(line=9, character=line.index("daily_revenue") + 1),
        ),
    )

    assert found is not None
    assert isinstance(found.contents, types.MarkupContent)
    assert "daily_revenue" in found.contents.value


def test_inlay_hints_come_back_for_the_parameters(
    recorder: Recorder,
    pipeline_path: Path,
) -> None:
    """The pipeline's output never resolved here, but its input did."""
    drive(recorder, on_did_open, opened(pipeline_path))
    hints = on_inlay_hint(
        recorder.server,
        types.InlayHintParams(
            text_document=types.TextDocumentIdentifier(uri=to_uri(str(pipeline_path))),
            range=types.Range(
                start=types.Position(line=0, character=0),
                end=types.Position(line=20, character=0),
            ),
        ),
    )

    assert "lake://stg_orders (3 cols)" in [hint.label for hint in hints]


def test_a_code_action_offers_the_fix(recorder: Recorder, pipeline_path: Path) -> None:
    """End to end: the finding on save becomes a one-keystroke rename."""
    drive(recorder, on_did_open, opened(pipeline_path))
    actions = on_code_action(
        recorder.server,
        types.CodeActionParams(
            text_document=types.TextDocumentIdentifier(uri=to_uri(str(pipeline_path))),
            range=types.Range(
                start=types.Position(line=10, character=0),
                end=types.Position(line=10, character=50),
            ),
            context=types.CodeActionContext(diagnostics=[]),
        ),
    )

    assert "Change 'amount' to 'amount_cents'" in [action.title for action in actions]
    assert "Ignore RWL101 on this line" in [action.title for action in actions]


# --- Wiring ------------------------------------------------------------------


def test_the_server_advertises_what_it_implements(project: Path) -> None:
    """A capability the client is not told about is a feature nobody gets."""
    server = create_server(root=str(project))
    features = server.protocol.fm.features

    assert types.TEXT_DOCUMENT_DID_SAVE in features
    assert types.TEXT_DOCUMENT_COMPLETION in features
    assert types.TEXT_DOCUMENT_HOVER in features
    assert types.TEXT_DOCUMENT_INLAY_HINT in features
    assert types.TEXT_DOCUMENT_CODE_ACTION in features
    assert RECHECK in server.protocol.fm.commands


def test_completion_declares_its_trigger_characters(project: Path) -> None:
    """A quote opens a column name, and the client has to re-ask on each letter."""
    server = create_server(root=str(project))
    options = server.protocol.fm.feature_options[types.TEXT_DOCUMENT_COMPLETION]

    assert '"' in options.trigger_characters
    assert "'" in options.trigger_characters


def test_a_non_file_uri_is_ignored(recorder: Recorder) -> None:
    """An untitled buffer cannot be imported, so there is nothing to check."""
    drive(
        recorder,
        on_did_open,
        types.DidOpenTextDocumentParams(
            text_document=types.TextDocumentItem(
                uri="untitled:Untitled-1", language_id="python", version=1, text=PIPELINE
            )
        ),
    )

    assert recorder.published == []
    assert recorder.server.session.documents == {}


def test_uris_round_trip_to_paths(pipeline_path: Path) -> None:
    """Every handler starts by doing this, and Windows drive letters are the
    reason it is worth asserting."""
    found = to_path(to_uri(str(pipeline_path)))

    assert found is not None
    assert Path(found) == pipeline_path


def test_the_server_syncs_whole_documents(project: Path) -> None:
    """Deliberate, not a default: every change re-indexes the whole buffer
    anyway, because `ast.parse` has no incremental mode."""
    server = create_server(root=str(project))

    assert server._text_document_sync_kind is types.TextDocumentSyncKind.Full


def test_findings_are_published_against_the_recorded_spelling() -> None:
    """Not the case-folded grouping key, which an editor would not match."""
    finding = Diagnostic(
        code=Code.COLUMN_NOT_FOUND,
        level=Level.ERROR,
        message="ColumnNotFound 'amount'",
        origin=Origin(kind="python", ref="x:y", file="C:/Repo/Revenue.py", line=1),
    )

    assert path_of([finding]) == "C:/Repo/Revenue.py"
    assert path_of([]) is None


def test_version_is_printed_without_starting_a_server(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`rowless-lsp --version` is the first thing anybody runs to check a
    client config, and it must not block waiting for LSP traffic on stdin."""
    code = main(["--version"])

    assert code == 0
    assert "rowless-lsp" in capsys.readouterr().out
