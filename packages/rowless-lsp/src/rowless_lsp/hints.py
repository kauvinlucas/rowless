"""Inlay hints: the resolved schema, shown where the code is.

The feature the skeleton called out as the one worth building beyond
diagnostics. A `group_by().agg()` chain produces a schema nobody can read off
the source, and the usual way to find out what it is involves running the
pipeline on real data. The hint puts the answer at the end of the chain.

Two hints, and no more than two. An editor that annotates every line stops
being read, so the only places a hint appears are the two where a schema is
genuinely invisible:

* after a parameter: which source feeds it, and how wide it is
* at the end of a `return`: the columns and types that came out

A schema rowless did not verify is labelled as unverified rather than
omitted. A missing hint reads as "no information"; the point of this project
is that "rowless does not know" is information.
"""

from __future__ import annotations

from lsprotocol import types
from rowless.ir.schema import Schema

from rowless_lsp.convert import markdown, position_of, schema_markdown
from rowless_lsp.documents import IndexedFunction, Span
from rowless_lsp.session import Document, Resolved

#: Columns named in a return hint before it is truncated. Four fits a sidebar
#: and a split editor; the rest is one hover away in the tooltip.
MAX_COLUMNS = 4


def inlay_hints(document: Document, within: types.Range | None = None) -> list[types.InlayHint]:
    """Hints for the requested range, or for the whole buffer."""
    index = document.usable
    hints: list[types.InlayHint] = []
    for function in index.functions:
        if not function.checked or not _overlaps(function, within):
            continue
        resolved = document.resolved(function.name)
        if resolved is None:
            continue
        hints.extend(_parameter_hints(function, resolved))
        hints.extend(_return_hints(function, resolved))
    return hints


def _parameter_hints(function: IndexedFunction, resolved: Resolved) -> list[types.InlayHint]:
    """One hint per parameter: the source it reads and its column count."""
    hints: list[types.InlayHint] = []
    for parameter, span in zip(function.params, function.param_spans, strict=False):
        schema = resolved.inputs.get(parameter)
        if schema is None:
            continue
        uri = resolved.uris.get(parameter)
        label = f" {uri}" if uri else ""
        hints.append(
            types.InlayHint(
                position=position_of(span.line, span.end),
                label=f"{label} ({len(schema)} cols)".strip(),
                kind=types.InlayHintKind.Parameter,
                padding_left=True,
                tooltip=markdown(schema_markdown(schema, title=uri or parameter)),
            )
        )
    return hints


def _return_hints(function: IndexedFunction, resolved: Resolved) -> list[types.InlayHint]:
    """One hint per `return`: the schema the engine computed."""
    if resolved.output is None:
        return []
    label = _label(resolved)
    tooltip = _tooltip(resolved)
    return [
        types.InlayHint(
            position=position_of(span.line, span.end),
            label=label,
            kind=types.InlayHintKind.Type,
            padding_left=True,
            tooltip=markdown(tooltip),
        )
        for span in function.returns
        if _is_outermost(span, function)
    ]


def _label(resolved: Resolved) -> str:
    """The hint text: an arrow, then the columns, truncated."""
    if resolved.opaque or resolved.output is None:
        return "-> unverified"
    schema = resolved.output
    if not schema.fields:
        return "-> no columns"
    shown = [f"{column.name}: {column.dtype}" for column in schema.fields[:MAX_COLUMNS]]
    rest = len(schema.fields) - len(shown)
    suffix = f", +{rest} more" if rest > 0 else ""
    unverified = "" if _fully_typed(schema) else "  (partial)"
    return f"-> {', '.join(shown)}{suffix}{unverified}"


def _tooltip(resolved: Resolved) -> str:
    """The full schema behind a truncated hint, plus its confidence."""
    if resolved.output is None:
        return "rowless resolved no output schema for this pipeline."
    lines = [schema_markdown(resolved.output, title="output")]
    if resolved.opaque:
        lines.extend(("", "rowless could not see inside this pipeline."))
    coverage = resolved.coverage
    if coverage is not None and coverage.reasons:
        lines.append("")
        lines.extend(f"- {reason}" for reason in coverage.reasons)
    return "\n".join(lines)


def _fully_typed(schema: Schema) -> bool:
    """True when every column came back with a concrete dtype."""
    return all(column.dtype.is_known and not column.dtype.is_any for column in schema.fields)


def _is_outermost(span: Span, function: IndexedFunction) -> bool:
    """True when a return span is not inside another one on the same line.

    A function with several returns gets a hint on each, which is right: each
    one is an exit with the same resolved schema. What it must not do is stack
    two hints at the same position, which happens when a return expression
    ends where another does.
    """
    same = [other for other in function.returns if other.line == span.line]
    return not same or span.end >= max(other.end for other in same)


def _overlaps(function: IndexedFunction, within: types.Range | None) -> bool:
    """True when a function intersects the requested range.

    The client asks for the visible region, and answering with the whole file
    would send hints for code nobody is looking at.
    """
    if within is None:
        return True
    return function.line <= within.end.line + 1 and function.end_line >= within.start.line + 1
