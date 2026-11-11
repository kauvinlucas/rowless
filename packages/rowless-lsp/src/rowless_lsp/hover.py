"""Hover: what this column is, and where it came from.

Three things are worth hovering in a pipeline, and they answer three different
questions:

* a column name - what type is it, and which upstream column produced it
* a parameter - what does this input actually contain
* the function name - what came out, and how much of it rowless verified

The third is the honest one. `rowless explain` exists because a tool that
reports a schema without saying how sure it is will eventually be believed
when it should not be, and the hover over a pipeline name says the same thing
the command does: the confidence, and every reason it was lost.
"""

from __future__ import annotations

from lsprotocol import types
from rowless.ir.confidence import Confidence, Coverage
from rowless.ir.schema import Field
from rowless.lineage.graph import ColumnRef, LineageGraph

from rowless_lsp.convert import field_markdown, markdown, schema_markdown
from rowless_lsp.documents import ColumnUse, IndexedFunction
from rowless_lsp.session import Document, Resolved


def hover(document: Document, position: types.Position) -> types.Hover | None:
    """The popup for a position, or None when there is nothing resolved to say."""
    line = position.line + 1
    index = document.usable
    function = index.function_at(line)

    column = index.column_at(line, position.character)
    if column is not None and function is not None:
        return _column_hover(document, function, column)
    if function is None:
        return None
    if function.name_span.contains(line, position.character):
        return _function_hover(document, function)
    for parameter, span in zip(function.params, function.param_spans, strict=False):
        if span.contains(line, position.character):
            return _parameter_hover(document, function, parameter)
    return None


def _column_hover(
    document: Document,
    function: IndexedFunction,
    column: ColumnUse,
) -> types.Hover | None:
    """A column literal: its type, its origin, and what it was computed from."""
    resolved = document.resolved(function.name)
    if resolved is None:
        return None
    found, relation = _field(resolved, column)
    if found is None:
        return _unresolved_column(resolved, column)
    lines = [field_markdown(found, relation=relation)]
    upstream = _upstream(document, resolved, column.name)
    if upstream:
        lines.extend(("", "**upstream**", *upstream))
    return types.Hover(contents=markdown("\n".join(lines)))


def _unresolved_column(resolved: Resolved, column: ColumnUse) -> types.Hover | None:
    """A literal that no resolved schema holds.

    Said plainly rather than left blank: an empty hover on a column name reads
    as "rowless has nothing to say about this file", when the actual answer is
    "that column is not in the schema", which is a finding.
    """
    available = sorted({name for schema in resolved.inputs.values() for name in schema})
    if not available:
        return None
    return types.Hover(
        contents=markdown(
            "\n".join(
                [
                    f"`{column.name}` is not in any resolved input schema.",
                    "",
                    f"available: {', '.join(f'`{name}`' for name in available)}",
                ]
            )
        )
    )


def _parameter_hover(
    document: Document,
    function: IndexedFunction,
    parameter: str,
) -> types.Hover | None:
    """A pipeline parameter: the whole input schema, and which source it is."""
    resolved = document.resolved(function.name)
    if resolved is None:
        return None
    schema = resolved.inputs.get(parameter)
    if schema is None:
        uri = resolved.uris.get(parameter)
        detail = f"`{uri}` did not resolve" if uri else "no source is bound to this parameter"
        return types.Hover(
            contents=markdown(
                f"`{parameter}`: {detail}.\n\n"
                "Declare it with `@rowless.pipeline(inputs={...})` or a `[sources]` rewrite."
            )
        )
    title = resolved.uris.get(parameter) or parameter
    return types.Hover(contents=markdown(schema_markdown(schema, title=title)))


def _function_hover(document: Document, function: IndexedFunction) -> types.Hover:
    """A pipeline name: the output schema, and how much of it was verified."""
    if not function.checked:
        return types.Hover(
            contents=markdown(
                "\n".join(
                    [
                        f"rowless does not check `{function.name}`.",
                        "",
                        "No parameter or return annotation names an engine frame type, and "
                        "there is no `@rowless.pipeline` marker. Add either one and it joins "
                        "the next check.",
                    ]
                )
            )
        )
    resolved = document.resolved(function.name)
    if resolved is None:
        return types.Hover(
            contents=markdown(f"`{function.name}` has not been checked yet in this session.")
        )
    return types.Hover(contents=markdown("\n".join(_function_markdown(function, resolved))))


def _function_markdown(function: IndexedFunction, resolved: Resolved) -> list[str]:
    """The body of a pipeline hover: engine, schema, confidence, reasons."""
    engine = f" ({function.engine})" if function.engine else ""
    lines = [f"**{function.name}**{engine}", ""]
    if resolved.output is None or resolved.opaque:
        lines.append("rowless could not resolve an output schema for this pipeline.")
    else:
        lines.append(schema_markdown(resolved.output))
    coverage = resolved.coverage
    if coverage is not None:
        lines.extend(("", _coverage_line(coverage)))
        lines.extend(f"- {reason}" for reason in coverage.reasons)
    return lines


def _coverage_line(coverage: Coverage) -> str:
    """The one-line confidence summary, worded like the terminal's."""
    label = {
        Confidence.EXACT: "exact",
        Confidence.PARTIAL: "partial",
        Confidence.UNKNOWN: "unverified",
    }[coverage.confidence]
    return (
        f"confidence **{label}**: "
        f"{coverage.typed_ratio:.0%} typed, {coverage.traced_ratio:.0%} traced"
    )


def _field(resolved: Resolved, column: ColumnUse) -> tuple[Field | None, str | None]:
    """The schema field a literal refers to, and the relation it belongs to.

    The frame the chain is rooted in decides when it is known. Otherwise the
    inputs are searched in order and then the output, which is the same
    precedence the live check uses: a name that is in an input means the input's
    column, and a name only in the output is one the pipeline built.
    """
    if column.frame is not None:
        schema = resolved.inputs.get(column.frame)
        if schema is not None and column.name in schema:
            return schema[column.name], resolved.uris.get(column.frame) or column.frame
    for parameter, schema in resolved.inputs.items():
        if column.name in schema:
            return schema[column.name], resolved.uris.get(parameter) or parameter
    output = resolved.output
    if output is not None and not resolved.opaque and column.name in output:
        return output[column.name], "this pipeline"
    return None, None


def _upstream(document: Document, resolved: Resolved, column: str) -> list[str]:
    """Lineage for one output column, with the evidence behind each edge.

    The evidence is shown, not hidden: a `name_match` edge is a guess that two
    columns sharing a name are the same column, and a hover that presented it
    identically to a plan-walked edge would be overstating what rowless knows.
    """
    if document.checked is None or not resolved.target:
        return []
    graph: LineageGraph = document.checked.graph
    if resolved.target not in graph.nodes:
        return []
    edges = graph.edges_into(ColumnRef(resolved.target, column))
    lines = []
    for edge in edges:
        transform = f" via {edge.transform}" if edge.transform else ""
        lines.append(f"- `{edge.source}` ({edge.evidence.value}{transform})")
    return lines
