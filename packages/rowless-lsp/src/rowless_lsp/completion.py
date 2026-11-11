"""Column-name completion, from a schema rowless actually resolved.

The feature people notice. It is also the one with the least room for
cleverness: a completion list that includes a column which does not exist is
not a convenience, it is a way of introducing the exact bug this project
exists to catch. So the candidates come from a resolved schema or there are no
candidates.

The trigger is detected from the current line's text rather than from the
syntax tree, which is the one place in this package that does not go through
`DocumentIndex`. It has to be: the buffer does not parse while somebody is
typing `pl.col("` - the string is not closed yet - and a feature that needs a
parsed file would never fire at the moment it is wanted.
"""

from __future__ import annotations

import re

from lsprotocol import types
from rowless.ir.schema import Field

from rowless_lsp.convert import field_markdown, markdown
from rowless_lsp.documents import (
    CREATE_SELECTORS,
    READ_KEYWORDS,
    READ_SELECTORS,
    RENAME_SELECTORS,
    DocumentIndex,
)
from rowless_lsp.session import Document, Resolved

#: What the server advertises as completion triggers: the characters that open
#: a column name. The editor also re-asks on every letter typed inside the
#: string, which is what makes the prefix filtering work.
TRIGGER_CHARACTERS: tuple[str, ...] = ('"', "'", "[")

#: The identifier immediately before an opening bracket: the called name.
_CALLED = re.compile(r"(\w+)\s*$")

#: A keyword argument the cursor's string is the value of: `on="`, `by=["`.
_KEYWORD = re.compile(r"(\w+)\s*=\s*[\s\[{]*$")


class Trigger:
    """A cursor sitting inside a string where a column name belongs."""

    __slots__ = ("end", "keyword", "mapping_value", "prefix", "selector", "start")

    def __init__(
        self,
        *,
        selector: str,
        prefix: str,
        start: int,
        end: int,
        keyword: str | None = None,
        mapping_value: bool = False,
    ) -> None:
        self.selector = selector
        """The call the string is an argument of, or `[]` for a subscript."""
        self.prefix = prefix
        self.start = start
        """0-based character where the string's text starts."""
        self.end = end
        """0-based character where the string's text ends, so a completion
        replaces the whole literal rather than appending to half a name."""
        self.keyword = keyword
        """The keyword this string is the value of, when it is one."""
        self.mapping_value = mapping_value
        """True on the right of a `:` inside a mapping, which in a `rename`
        is the new name rather than an existing column."""

    @property
    def reads(self) -> bool:
        """True when the position names a column that is supposed to exist.

        Everything else is a place a *new* name is written - the right side of
        a rename, an `alias`, a `with_columns` keyword - and offering the
        existing column set there would suggest overwriting a column is the
        normal thing to do.
        """
        if self.mapping_value:
            return False
        if self.keyword is not None:
            return self.keyword in READ_KEYWORDS
        if self.selector in CREATE_SELECTORS:
            return False
        return (
            self.selector in READ_SELECTORS
            or self.selector in RENAME_SELECTORS
            or self.selector == "[]"
        )


def trigger_at(line: str, character: int) -> Trigger | None:
    """Whether a cursor is inside a column-name string, from one line of text.

    Quote handling is deliberately shallow: no escapes, no triple quotes, no
    strings continued across lines. A column name is a short literal written
    on one line, and every shape this misses degrades to no completion rather
    than to a wrong one.
    """
    cursor = min(max(character, 0), len(line))
    opened = _string_start(line, cursor)
    if opened is None:
        return None
    quote, start = opened
    enclosing = _enclosing(line[: start - 1])
    if enclosing is None:
        return None
    selector, inner, mapping_value = enclosing
    keyword = _KEYWORD.search(inner)
    return Trigger(
        selector=selector,
        prefix=line[start:cursor],
        start=start,
        end=_string_end(line, cursor, quote),
        keyword=keyword.group(1) if keyword is not None else None,
        mapping_value=mapping_value,
    )


def _enclosing(head: str) -> tuple[str, str, bool] | None:
    """The call or subscript a string belongs to, scanning right to left.

    Returns the called name, the text between its opening bracket and the
    string, and whether the string sits on the value side of a mapping entry.

    A bracket walk rather than one regex, because the construct that decides
    what a literal means is often not next to it: in
    `df.join(other, on="order_id")` the call is five tokens back, and in
    `df.rename({"amount": "amount_cents"})` the two literals in one mapping
    mean opposite things.
    """
    depth = 0
    brace_at: int | None = None
    for index in range(len(head) - 1, -1, -1):
        character = head[index]
        if character in ")]}":
            depth += 1
            continue
        if character not in "([{":
            continue
        if depth:
            depth -= 1
            continue
        if character == "{":
            # A dict literal. The call that gives it meaning is further out,
            # so remember where the mapping started and keep walking.
            brace_at = index
            continue
        if character == "[":
            before = head[:index].rstrip()
            if before and (before[-1].isalnum() or before[-1] in "_)]"):
                return "[]", head[index + 1 :], False
            # A bare list literal, which is transparent: every engine takes a
            # list of names wherever it takes one name.
            continue
        name = _CALLED.search(head[:index])
        return (
            name.group(1) if name is not None else "",
            head[index + 1 :],
            _is_mapping_value(head, brace_at),
        )
    return None


def _is_mapping_value(head: str, brace_at: int | None) -> bool:
    """True when the cursor is past the `:` of the current mapping entry."""
    if brace_at is None:
        return False
    entry = head[brace_at + 1 :]
    return ":" in entry[entry.rfind(",") + 1 :]


def complete(
    document: Document,
    position: types.Position,
    *,
    limit: int = 200,
) -> list[types.CompletionItem]:
    """Column names for the position, or an empty list.

    Empty is a real answer here and the common one: outside a column literal,
    in a function rowless does not check, or before the first successful
    resolve, there is nothing honest to offer.
    """
    line = position.line + 1
    text = _line_text(document, position.line)
    trigger = trigger_at(text, position.character)
    if trigger is None or not trigger.reads:
        return []
    resolved = _resolved_for(document, line)
    if resolved is None:
        return []
    candidates = _candidates(resolved, document.usable, line)
    edit = types.Range(
        start=types.Position(line=position.line, character=trigger.start),
        end=types.Position(line=position.line, character=trigger.end),
    )
    return [
        _item(name, column, edit=edit, relation=relation, rank=rank)
        for rank, (name, (column, relation)) in enumerate(list(candidates.items())[:limit])
    ]


def _candidates(
    resolved: Resolved,
    index: DocumentIndex,
    line: int,
) -> dict[str, tuple[Field | None, str | None]]:
    """Every column name that could legitimately be typed here.

    Three sources, narrowest meaning first: the inputs this pipeline reads,
    the schema its last dry run produced, and the names it creates itself. The
    third carries no dtype - nothing resolved one - and says so rather than
    inventing one.
    """
    found: dict[str, tuple[Field | None, str | None]] = {}
    for parameter, schema in resolved.inputs.items():
        relation = resolved.uris.get(parameter) or parameter
        for column in schema.fields:
            found.setdefault(column.name, (column, relation))
    if resolved.output is not None and not resolved.opaque:
        for column in resolved.output.fields:
            found.setdefault(column.name, (column, "this pipeline"))
    function = index.function_at(line)
    if function is not None:
        for name in sorted(function.created):
            found.setdefault(name, (None, "created in this pipeline"))
    return found


def _item(
    name: str,
    column: Field | None,
    *,
    edit: types.Range,
    relation: str | None,
    rank: int,
) -> types.CompletionItem:
    """One completion item: the name, its type, and where it comes from."""
    detail = str(column.dtype) if column is not None else "unresolved type"
    documentation = (
        field_markdown(column, relation=relation)
        if column is not None
        else f"created in this pipeline; no resolved type for `{name}`"
    )
    return types.CompletionItem(
        label=name,
        kind=types.CompletionItemKind.Field,
        detail=detail,
        documentation=markdown(documentation),
        text_edit=types.TextEdit(range=edit, new_text=name),
        # Schema order, not alphabetical: a wide table's useful columns are the
        # ones its author put first. Zero-padded because a client sorts
        # `sort_text` as a string, so `10` would otherwise come before `2`.
        sort_text=f"{rank:04d}",
    )


def _resolved_for(document: Document, line: int) -> Resolved | None:
    """The resolved schemas for the function at `line`.

    Falls back to the only checked function in the file when the cursor cannot
    be placed in one, which covers the usual single-pipeline module while it is
    being edited into a state that does not parse.
    """
    index = document.usable
    function = index.function_at(line)
    if function is not None:
        found = document.resolved(function.name)
        if found is not None and found.inputs:
            return found
    if document.checked is None:
        return None
    usable = [found for found in document.checked.resolved.values() if found.inputs]
    return usable[0] if len(usable) == 1 else None


def _line_text(document: Document, line: int) -> str:
    """One 0-based line of the buffer."""
    lines = document.lines
    return lines[line] if 0 <= line < len(lines) else ""


def _string_start(line: str, cursor: int) -> tuple[str, int] | None:
    """The quote character and text offset of the string the cursor is in."""
    quote: str | None = None
    start = 0
    for position, character in enumerate(line[:cursor]):
        if quote is None:
            if character in "\"'":
                quote, start = character, position + 1
        elif character == quote:
            quote = None
    return (quote, start) if quote is not None else None


def _string_end(line: str, cursor: int, quote: str) -> int:
    """Where the string's text ends: its closing quote, or the cursor."""
    closing = line.find(quote, cursor)
    return closing if closing != -1 else cursor
