"""Code actions: fix the name, or say "not now".

Two actions, which are the two things a developer does with a schema finding.
The first is the rename case the proposal opens with: the column is called
`amount_cents` now, and the fix is one edit rowless already worked out when it
built the suggestion. The second is the admission that the first run on a real
repository finds things nobody is fixing this week - `# rowless: ignore[...]`
exists for exactly that, and an editor that cannot write the comment for you
leaves people guessing at its spelling.

The ignore action is offered for every finding, including the ones rowless is
sure about. A checker with no way to say "not now" gets deleted rather than
configured, and suppressing a finding on one line is a smaller loss than
having the server turned off.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from lsprotocol import types
from rowless.config import IGNORE_PATTERN
from rowless.diagnostics.codes import Code
from rowless.diagnostics.models import Diagnostic
from rowless.errors import nearest_name

from rowless_lsp.convert import to_lsp, to_uri
from rowless_lsp.documents import Span
from rowless_lsp.session import Document

#: The quoted name inside a suggestion: `did you mean 'amount_cents'?`.
_SUGGESTED = re.compile(r"'([^']+)'")

#: Codes whose fix is a replacement of the column name under the cursor.
_RENAMEABLE: frozenset[Code] = frozenset({Code.COLUMN_NOT_FOUND, Code.AMBIGUOUS_COLUMN})


def code_actions(
    document: Document,
    diagnostics: Sequence[Diagnostic],
    within: types.Range,
) -> list[types.CodeAction]:
    """Every action available for the findings inside a range."""
    actions: list[types.CodeAction] = []
    for diagnostic in diagnostics:
        origin = diagnostic.origin
        if origin is None or origin.line is None or not _in_range(origin.line, within):
            continue
        fix = _rename_action(document, diagnostic)
        if fix is not None:
            actions.append(fix)
        actions.append(_ignore_action(document, diagnostic))
    return actions


def _rename_action(document: Document, diagnostic: Diagnostic) -> types.CodeAction | None:
    """Replace a near-miss column name with the one that exists.

    The replacement comes from the finding's own suggestion where there is one,
    because core computed it against the schema's actual column set, which is
    a better answer than anything this module could derive. Only when the
    finding carries no suggestion is the near miss recomputed here, from the
    schemas the session has.
    """
    if diagnostic.code not in _RENAMEABLE:
        return None
    origin = diagnostic.origin
    if origin is None or origin.line is None:
        return None
    use = _column_on(document, origin.line, diagnostic)
    if use is None:
        return None
    replacement = _suggested(diagnostic) or _nearest(document, use[0], origin.line)
    if replacement is None or replacement == use[0]:
        return None
    span = use[1]
    edit = types.TextEdit(
        range=types.Range(
            start=types.Position(line=span.line - 1, character=span.start),
            end=types.Position(line=span.line - 1, character=span.end),
        ),
        new_text=replacement,
    )
    return types.CodeAction(
        title=f"Change {use[0]!r} to {replacement!r}",
        kind=types.CodeActionKind.QuickFix,
        diagnostics=[to_lsp(diagnostic, document.usable)],
        is_preferred=True,
        edit=types.WorkspaceEdit(changes={to_uri(document.path): [edit]}),
    )


def _ignore_action(document: Document, diagnostic: Diagnostic) -> types.CodeAction:
    """Add or extend a `# rowless: ignore[...]` comment on the finding's line."""
    origin = diagnostic.origin
    line = origin.line if origin is not None and origin.line is not None else 1
    text = document.usable.line_text(line)
    edit = types.TextEdit(
        range=types.Range(
            start=types.Position(line=line - 1, character=0),
            end=types.Position(line=line - 1, character=len(text)),
        ),
        new_text=with_ignore(text, diagnostic.code),
    )
    return types.CodeAction(
        title=f"Ignore {diagnostic.code.value} on this line",
        kind=types.CodeActionKind.QuickFix,
        diagnostics=[to_lsp(diagnostic, document.usable)],
        edit=types.WorkspaceEdit(changes={to_uri(document.path): [edit]}),
    )


def with_ignore(line: str, code: Code) -> str:
    """One line of source with `code` added to its ignore comment.

    Extends an existing comment rather than adding a second one: two
    `# rowless: ignore[...]` comments on a line are not what the parser reads,
    and silently suppressing nothing is the worst outcome for this feature.
    """
    match = IGNORE_PATTERN.search(line)
    if match is None:
        return f"{line.rstrip()}  # rowless: ignore[{code.value}]"
    existing = [part.strip() for part in match.group(1).split(",") if part.strip()]
    if code.value in existing:
        return line
    merged = ",".join([*existing, code.value])
    return f"{line[: match.start(1)]}{merged}{line[match.end(1) :]}"


def _column_on(
    document: Document,
    line: int,
    diagnostic: Diagnostic,
) -> tuple[str, Span] | None:
    """The column literal a finding is about, with its span in the buffer."""
    for use in document.usable.columns:
        if use.span.line != line:
            continue
        if f"'{use.name}'" in diagnostic.message or f'"{use.name}"' in diagnostic.message:
            return use.name, use.span
    return None


def _suggested(diagnostic: Diagnostic) -> str | None:
    """The column name quoted inside a finding's suggestion, if there is one."""
    if not diagnostic.suggestion:
        return None
    found = _SUGGESTED.search(diagnostic.suggestion)
    return found.group(1) if found is not None else None


def _nearest(document: Document, name: str, line: int) -> str | None:
    """The closest column name in the schemas resolved for this function."""
    function = document.usable.function_at(line)
    if function is None:
        return None
    resolved = document.resolved(function.name)
    if resolved is None:
        return None
    available = {column for schema in resolved.inputs.values() for column in schema}
    return nearest_name(name, sorted(available))


def _in_range(line: int, within: types.Range) -> bool:
    """True when a 1-based line falls inside a 0-based LSP range."""
    return within.start.line + 1 <= line <= within.end.line + 1
