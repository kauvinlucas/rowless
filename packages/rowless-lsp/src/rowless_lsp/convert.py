"""rowless findings to LSP messages, and file URIs to paths.

One translation layer, so no feature handler builds a protocol object by hand.
`rowless.diagnostics.Diagnostic` already carries everything a surface needs -
that is stated as a requirement on the type - and this module is the proof of
it for the editor: nothing here reaches past a `Diagnostic` for information.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

from lsprotocol import types
from pygls import uris
from rowless.diagnostics.codes import UNVERIFIED_CODES
from rowless.diagnostics.models import Diagnostic, Level
from rowless.ir.schema import Field, Origin, Schema

from rowless_lsp.documents import DocumentIndex

#: The name shown beside every finding in an editor's problem list.
SOURCE = "rowless"

#: How a rowless level renders as an LSP severity.
#:
#: UNVERIFIED is the interesting one. The GitHub renderer promotes it to an
#: error because a CI annotation has nowhere else to say "this gates the
#: build". An editor gates nothing, and an error squiggle for "rowless does
#: not know" is how a language server gets switched off - so it is a warning
#: here, and the 2xx code carries the distinction. The code is visible in
#: every editor's problem list, which is where a user who wants to know the
#: difference goes looking.
SEVERITY: dict[Level, types.DiagnosticSeverity] = {
    Level.ERROR: types.DiagnosticSeverity.Error,
    Level.WARNING: types.DiagnosticSeverity.Warning,
    Level.UNVERIFIED: types.DiagnosticSeverity.Warning,
    Level.INFO: types.DiagnosticSeverity.Information,
}


def to_lsp(diagnostic: Diagnostic, index: DocumentIndex | None = None) -> types.Diagnostic:
    """One rowless finding as an LSP diagnostic.

    `index` is optional and only narrows the range: with it, the squiggle lands
    under the column literal the finding is about rather than under the whole
    line.
    """
    return types.Diagnostic(
        range=range_for(diagnostic, index),
        message=message_for(diagnostic),
        severity=SEVERITY.get(diagnostic.level, types.DiagnosticSeverity.Warning),
        code=diagnostic.code.value,
        source=SOURCE,
        related_information=_related(diagnostic) or None,
        # Carried so a code action can recover the finding without re-deriving
        # it from the message. `unverified` is in here because the severity
        # above flattens it into a warning.
        data={
            "code": diagnostic.code.value,
            "level": diagnostic.level.value,
            "suggestion": diagnostic.suggestion,
            "unverified": diagnostic.code in UNVERIFIED_CODES,
        },
    )


def message_for(diagnostic: Diagnostic) -> str:
    """The hover text for a squiggle: headline, cause, fix.

    The same three parts the terminal renderer prints, in the same order, so
    the editor and `rowless check` do not describe one finding two ways.
    """
    parts = [diagnostic.message]
    if diagnostic.detail:
        parts.append(diagnostic.detail)
    if diagnostic.suggestion:
        parts.append(diagnostic.suggestion)
    return "\n".join(parts)


def range_for(diagnostic: Diagnostic, index: DocumentIndex | None = None) -> types.Range:
    """Where the squiggle goes.

    Narrowest first: the column literal the finding names, then the column the
    origin recorded, then the whole flagged line. A finding with no line at
    all - a coverage score, say - lands at the top of the file, because a
    diagnostic an editor cannot place is a diagnostic nobody reads.
    """
    origin = diagnostic.origin
    if origin is None or origin.line is None:
        return _at(0, 0, 0)
    line = origin.line - 1
    if index is not None:
        narrowed = _literal_range(diagnostic, origin, index)
        if narrowed is not None:
            return narrowed
    if origin.column is not None:
        start = max(origin.column - 1, 0)
        return _at(line, start, start + 1)
    text = index.line_text(origin.line) if index is not None else ""
    if text.strip():
        return _at(line, len(text) - len(text.lstrip()), len(text))
    return _at(line, 0, 0)


def _literal_range(
    diagnostic: Diagnostic,
    origin: Origin,
    index: DocumentIndex,
) -> types.Range | None:
    """The span of the column literal a finding is about, when there is one.

    Matched by the name appearing in the message rather than by parsing the
    message: every finding about a column quotes its name, and looking for a
    name rowless already indexed on that line is a lookup instead of a parse.
    """
    if origin.line is None:
        return None
    for use in index.columns:
        if use.span.line != origin.line:
            continue
        if f"'{use.name}'" in diagnostic.message or f'"{use.name}"' in diagnostic.message:
            return _at(origin.line - 1, use.span.start, use.span.end)
    return None


def _related(diagnostic: Diagnostic) -> list[types.DiagnosticRelatedInformation]:
    """Secondary locations, which is where the cross-language payoff shows up.

    A Python finding whose related origin is a SQL model becomes a clickable
    second location in the editor: the column reference on one side, the model
    that renamed it on the other.
    """
    related: list[types.DiagnosticRelatedInformation] = []
    for origin in diagnostic.related:
        if not origin.file:
            continue
        line = max((origin.line or 1) - 1, 0)
        related.append(
            types.DiagnosticRelatedInformation(
                location=types.Location(uri=to_uri(origin.file), range=_at(line, 0, 0)),
                message=str(origin),
            )
        )
    return related


def group_by_file(
    diagnostics: Sequence[Diagnostic],
    *,
    fallback: str | None = None,
) -> dict[str, list[Diagnostic]]:
    """Findings bucketed by the file they belong to.

    A check of one file can report a finding located in another - that is the
    point of a cross-language graph - and a diagnostic published against the
    wrong URI is invisible. Findings with no file at all go to `fallback`,
    which is the document that triggered the check.
    """
    grouped: dict[str, list[Diagnostic]] = {}
    for diagnostic in diagnostics:
        origin = diagnostic.origin
        where = origin.file if origin is not None and origin.file else fallback
        if where is None:
            continue
        grouped.setdefault(os.path.normcase(os.path.abspath(where)), []).append(diagnostic)
    return grouped


def schema_markdown(schema: Schema, *, title: str | None = None) -> str:
    """A schema as a markdown table, for a hover popup."""
    lines = [f"**{title}**", ""] if title else []
    if not schema.fields:
        return "\n".join([*lines, "_no columns_"])
    lines.extend(("| column | type |", "| --- | --- |"))
    lines.extend(f"| `{column.name}` | `{column.dtype}` |" for column in schema.fields)
    return "\n".join(lines)


def field_markdown(column: Field, *, relation: str | None = None) -> str:
    """One column as hover text: type, nullability, and where it came from."""
    where = f" of `{relation}`" if relation else ""
    lines = [f"`{column.name}`: `{column.dtype}`{where}"]
    if not column.dtype.is_known or column.dtype.is_any:
        reason = column.dtype.reason or "the engine did not resolve this column"
        lines.append(f"- unverified: {reason}")
    lines.append(f"- nullable: {'yes' if column.nullable else 'no'}")
    if column.origin is not None:
        lines.append(f"- from: {column.origin}")
    if column.derived_from:
        lines.append(f"- derived from: {', '.join(f'`{name}`' for name in column.derived_from)}")
    if column.description:
        lines.append(f"- {column.description}")
    return "\n".join(lines)


def markdown(text: str) -> types.MarkupContent:
    """Hover content, as markdown."""
    return types.MarkupContent(kind=types.MarkupKind.Markdown, value=text)


def to_uri(path: str) -> str:
    """A filesystem path as a `file://` URI."""
    return uris.from_fs_path(os.path.abspath(path)) or path


def to_path(uri: str) -> str | None:
    """A `file://` URI as a filesystem path, or None for anything else.

    None rather than a guess: an editor can open an untitled buffer or a file
    inside an archive, and rowless cannot import either of those.
    """
    return uris.to_fs_path(uri)


def same_file(left: str, right: str) -> bool:
    """Whether two paths name one file, case-insensitively on Windows."""
    return os.path.normcase(os.path.abspath(left)) == os.path.normcase(os.path.abspath(right))


def position_of(span_line: int, character: int) -> types.Position:
    """A 1-based rowless line and 0-based character as an LSP position."""
    return types.Position(line=max(span_line - 1, 0), character=character)


def _at(line: int, start: int, end: int) -> types.Range:
    """A single-line range, 0-based, as the protocol wants it."""
    return types.Range(
        start=types.Position(line=line, character=start),
        end=types.Position(line=line, character=end),
    )


def path_of(diagnostics: Sequence[Diagnostic]) -> str | None:
    """The file a group of findings is about, with its original spelling.

    `group_by_file` keys by a case-folded path so that two spellings of one
    file are one group, which is what Windows requires. The key is therefore
    the wrong thing to publish against: an editor that opened a path spelled
    with a capital drive letter will not match diagnostics published against
    the lower-cased one. The findings themselves still carry the path as it
    was recorded, so that is what the URI is built from.
    """
    for diagnostic in diagnostics:
        origin = diagnostic.origin
        if origin is not None and origin.file:
            return origin.file
    return None
