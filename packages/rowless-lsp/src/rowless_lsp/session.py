"""The warm state: what the server remembers between keystrokes.

`rowless check` is a batch tool. It discovers, resolves every source, dry-runs
every pipeline and exits, and doing that per keystroke is not a language
server, it is a space heater. This module is the difference: one `Config` per
workspace, one index per open buffer, and the last successful check's schemas
kept so that completion, hover and hints answer from memory instead of running
anything.

What is deliberately *not* cached is the dry run itself. A check imports the
user's module, so its answer is only true for the file on disk, and serving a
stale dry-run result as though it described the buffer is the one way this
server could lie. Results are kept for the features that read schemas, and
re-anchored rather than re-pointed when the buffer moves underneath them.

Everything a check produces goes through `rowless.check.run`, the same entry
point the CLI and the pytest plugin use. The editor must not be able to
disagree with CI about what is broken.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from rowless.check import Outcome, run
from rowless.config import Config
from rowless.diagnostics.models import Diagnostic, Report
from rowless.discovery.base import Pipeline
from rowless.ir.confidence import Coverage
from rowless.ir.schema import Schema
from rowless.lineage.graph import LineageGraph

from rowless_lsp.anchors import Anchored, anchor, reanchor
from rowless_lsp.convert import group_by_file, same_file
from rowless_lsp.documents import DocumentIndex
from rowless_lsp.live import live_diagnostics, suppress

#: The first quoted name in a diagnostic message, which is the column, table
#: or pipeline the finding is about.
_QUOTED = re.compile(r"'([^']+)'")


@dataclass(frozen=True, slots=True)
class Resolved:
    """One pipeline's resolved schemas, as the editor needs them.

    Keyed by function name rather than by import target: a cursor is in a
    function, and the editor never sees a module path.
    """

    inputs: Mapping[str, Schema] = field(default_factory=dict)
    """Input schema per parameter name."""

    uris: Mapping[str, str] = field(default_factory=dict)
    """Source URI per parameter name, which is the node id in the graph and
    the thing an inlay hint shows beside the parameter."""

    target: str = ""
    """The pipeline's import target, which is also its node id in the graph.
    Needed to walk lineage back from an output column."""

    output: Schema | None = None
    coverage: Coverage | None = None
    opaque: bool = False
    """True when the dry run could not see inside, so the output schema above
    is not evidence of anything and is reported as such."""


@dataclass(frozen=True, slots=True)
class Checked:
    """What one check left behind for the features to read."""

    report: Report
    resolved: Mapping[str, Resolved] = field(default_factory=dict)
    graph: LineageGraph = field(default_factory=LineageGraph)
    anchored: tuple[Anchored, ...] = ()
    """The findings for this file, each remembering the line it was on."""

    @property
    def inputs(self) -> Mapping[str, Mapping[str, Schema]]:
        """Input schemas by function, which is what the live check wants."""
        return {name: found.inputs for name, found in self.resolved.items()}

    @property
    def outputs(self) -> Mapping[str, Schema]:
        """Output schemas by function, for the functions that resolved one."""
        return {
            name: found.output
            for name, found in self.resolved.items()
            if found.output is not None and not found.opaque
        }


@dataclass(slots=True)
class Document:
    """One open buffer and everything known about it."""

    path: str
    text: str
    version: int = 0
    index: DocumentIndex = field(default_factory=lambda: DocumentIndex(path="", lines=()))

    parsed: DocumentIndex | None = None
    """The last index that actually parsed. A file being typed in does not
    parse most of the time, and completion that stops working mid-word is
    worse than completion based on the shape the file had a keystroke ago."""

    checked: Checked | None = None
    dirty: bool = False
    """True when the buffer has moved since the last check, which is what
    makes a dry-run finding provisional rather than current."""

    def replace(self, text: str, version: int) -> None:
        """Take new buffer contents and re-index."""
        self.text = text
        self.version = version
        self.index = DocumentIndex.of(self.path, text)
        if self.index.parsed:
            self.parsed = self.index
        self.dirty = True

    @property
    def usable(self) -> DocumentIndex:
        """The best index available: the current one, or the last that parsed."""
        if self.index.parsed or self.parsed is None:
            return self.index
        return self.parsed

    @property
    def lines(self) -> tuple[str, ...]:
        """The buffer's lines, for anchoring and for range narrowing."""
        return tuple(self.text.splitlines())

    def resolved(self, function: str) -> Resolved | None:
        """What the last check resolved for one function, if anything."""
        if self.checked is None:
            return None
        return self.checked.resolved.get(function)


class Session:
    """One workspace: its configuration, its open documents, its last results."""

    def __init__(self, root: str) -> None:
        self.root = os.path.abspath(root)
        self.documents: dict[str, Document] = {}
        self._config: Config | None = None

    # --- Configuration -------------------------------------------------------

    @property
    def config(self) -> Config:
        """The workspace config, discovered once and kept.

        Discovery walks the directory tree looking for `rowless.toml`, which is
        not something to repeat per keystroke. `reload` drops it when the file
        itself changes.
        """
        if self._config is None:
            self._config = Config.discover(self.root)
        return self._config

    def reload(self) -> None:
        """Forget the cached config. Called when a config file changes on disk."""
        self._config = None

    # --- Documents -----------------------------------------------------------

    def open(self, path: str, text: str, version: int = 0) -> Document:
        """Register a buffer and index it."""
        document = Document(path=os.path.abspath(path), text=text, version=version)
        document.index = DocumentIndex.of(document.path, text)
        if document.index.parsed:
            document.parsed = document.index
        self.documents[_key(path)] = document
        return document

    def update(self, path: str, text: str, version: int = 0) -> Document | None:
        """Take an edit. Returns None for a buffer the session never opened."""
        document = self.document(path)
        if document is None:
            return None
        document.replace(text, version)
        return document

    def close(self, path: str) -> None:
        """Drop a buffer and its cached results."""
        self.documents.pop(_key(path), None)

    def document(self, path: str) -> Document | None:
        """One open buffer by path."""
        return self.documents.get(_key(path))

    # --- Checking ------------------------------------------------------------

    def check(self, path: str) -> dict[str, list[Diagnostic]]:
        """Dry-run the pipelines in one file, and remember what came back.

        The findings come back grouped by the file they belong to, not by the
        file that was checked: a column renamed in a dbt model is reported
        against the model, and publishing it against the Python file would put
        the squiggle in the wrong document.
        """
        target = os.path.abspath(path)
        outcome = run(self.config, paths=[target])
        document = self.document(target)
        lines = document.lines if document is not None else _read_lines(target)
        grouped = group_by_file(outcome.report.diagnostics, fallback=target)
        if document is not None:
            document.checked = Checked(
                report=outcome.report,
                resolved=_resolve(outcome, target),
                graph=outcome.graph,
                anchored=anchor(grouped.get(_key(target), []), lines),
            )
            document.dirty = False
        return grouped

    def check_workspace(self) -> dict[str, list[Diagnostic]]:
        """Check everything under the configured roots, as `rowless check` does.

        Run once when the server starts, so the first hover in a file that was
        never opened still has schemas to show, and so the initial problem list
        is the same one CI would produce.
        """
        outcome = run(self.config)
        grouped = group_by_file(outcome.report.diagnostics)
        for key, document in self.documents.items():
            found = grouped.get(key, [])
            document.checked = Checked(
                report=outcome.report,
                resolved=_resolve(outcome, document.path),
                graph=outcome.graph,
                anchored=anchor(found, document.lines),
            )
            document.dirty = False
        return grouped

    def diagnostics(self, path: str) -> list[Diagnostic]:
        """The findings to publish for one buffer, right now.

        Three sources, in order of how much they are worth trusting:

        1. the dry run's own findings, re-anchored onto the current buffer and
           dropped where their line no longer exists
        2. the buffer-only findings, which are current by construction
        3. nothing else - an unresolvable position produces silence

        Both sets pass through the buffer's ignore comments, so adding one
        takes effect on the keystroke rather than on the next save.
        """
        document = self.document(path)
        if document is None:
            return []
        found: list[Diagnostic] = []
        if document.checked is not None:
            carried = (
                reanchor(document.checked.anchored, document.lines)
                if document.dirty
                else tuple(item.diagnostic for item in document.checked.anchored)
            )
            found.extend(suppress(carried, document.text))
        found.extend(self.live(path))
        return _deduped(found)

    def live(self, path: str) -> list[Diagnostic]:
        """The buffer-only findings for one document.

        Reads `index`, never the `usable` fallback. The other features fall
        back to the last index that parsed because a stale shape is better
        than a flicker, and a diagnostic is the one thing that cannot: a
        finding placed by stale line numbers lands on code that is fine, which
        is an accusation rather than a stale answer. A buffer that does not
        parse therefore reports nothing until it does.
        """
        document = self.document(path)
        if document is None or document.checked is None or not document.index.parsed:
            return []
        return list(
            live_diagnostics(
                document.index,
                document.text,
                schemas=document.checked.inputs,
                outputs=document.checked.outputs,
                config=self.config,
            )
        )


def _resolve(outcome: Outcome, path: str) -> dict[str, Resolved]:
    """Pull one file's resolved schemas out of a check outcome.

    Everything here is read off the graph and the results rather than from a
    second resolution pass: the source schemas are already nodes, and the
    output schema is already the pipeline's node. A separate path to the same
    schemas would be a second thing to keep in step.
    """
    coverage = {result.target.rpartition(":")[2]: result.coverage for result in outcome.results}
    resolved: dict[str, Resolved] = {}
    for pipeline in outcome.pipelines:
        if pipeline.file is None or not same_file(pipeline.file, path):
            continue
        node = outcome.graph.nodes.get(pipeline.target)
        resolved[pipeline.name] = Resolved(
            inputs=_input_schemas(pipeline, outcome.graph),
            uris={name: uri for name, uri in pipeline.inputs if uri is not None},
            target=pipeline.target,
            output=node.schema if node is not None else None,
            coverage=coverage.get(pipeline.name),
            opaque=node.opaque if node is not None else True,
        )
    return resolved


def _input_schemas(pipeline: Pipeline, graph: LineageGraph) -> dict[str, Schema]:
    """Parameter name to the schema its source resolved to."""
    schemas: dict[str, Schema] = {}
    for parameter, uri in pipeline.inputs:
        if uri is None:
            continue
        node = graph.nodes.get(uri)
        if node is not None:
            schemas[parameter] = node.schema
    return schemas


def _deduped(diagnostics: Sequence[Diagnostic]) -> list[Diagnostic]:
    """One finding per code, line and subject, keeping the first.

    The live check and the dry run can both reach the same missing column -
    that is the point, one previews the other - and publishing it twice makes
    the editor show two squiggles for one mistake. The subject is in the key
    because the two do not agree on how much they find: an engine raises on
    the first bad column in `select("amuont", "stauts")` and the live check
    sees both, so keying on the line alone would hide the second one.

    The dry run's own wording wins, since it is passed in first: it is the
    authoritative finding, and the live one is a preview of it.
    """
    seen: set[tuple[str, int | None, str]] = set()
    kept: list[Diagnostic] = []
    for diagnostic in diagnostics:
        origin = diagnostic.origin
        key = (
            diagnostic.code.value,
            origin.line if origin else None,
            _subject(diagnostic.message),
        )
        if key in seen:
            continue
        seen.add(key)
        kept.append(diagnostic)
    return kept


def _subject(message: str) -> str:
    """The first quoted name in a message: what the finding is about.

    Every finding that names a column quotes it, so this pairs one surface's
    wording with another's without either having to know the other exists.
    """
    found = _QUOTED.search(message)
    return found.group(1) if found is not None else message


def _read_lines(path: str) -> tuple[str, ...]:
    """A file's lines from disk, for a check of a document nobody has open."""
    try:
        with open(path, encoding="utf-8") as handle:
            return tuple(handle.read().splitlines())
    except OSError:
        return ()


def _key(path: str) -> str:
    """The canonical dictionary key for a path: absolute, case-folded on Windows."""
    return os.path.normcase(os.path.abspath(path))
