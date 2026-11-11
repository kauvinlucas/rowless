"""The cross-language column graph.

Nodes are relations (a source table, a Python function, a SQL model), edges are
column-to-column derivations. `rowless diff` walks it backwards from a changed
column to find everything that reads it, which is the blast-radius tree in the
proposal, and the thing existing tools stop short of because they only see one
language.

Edges come from whatever evidence is available, and are labelled with which:
Polars plan walking, sqlglot qualification, a dbt manifest, or nothing at all.
A node with no lineage evidence is recorded as opaque rather than omitted,
because an unknown gap in the graph must be visible in the coverage score.
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from rowless.ir.schema import Origin, Schema


class NodeKind(StrEnum):
    """What a node represents."""

    SOURCE = "source"
    """An external table. Has no upstream within the repo."""

    PYTHON = "python"
    SQL = "sql"
    DBT_MODEL = "dbt_model"
    SINK = "sink"
    """A write target, recovered from the sandbox's side-effect log."""


class Evidence(StrEnum):
    """How an edge was established. Shown in `rowless explain` output."""

    PLAN = "plan"
    """Walked the engine's resolved query plan. The strongest evidence."""

    SQL_QUALIFIED = "sql_qualified"
    """sqlglot resolved the column against known table schemas."""

    MANIFEST = "manifest"
    """Declared by dbt or another orchestrator."""

    NAME_MATCH = "name_match"
    """Inferred from an identical column name passing through. Weak, and
    excluded from breaking-change classification by default."""

    NONE = "none"
    """No evidence. The node is opaque; recorded to keep coverage honest."""


#: Evidence rowless will base a breaking-change claim on. A name match is
#: reported but does not gate, because two columns sharing a name is not proof
#: that one feeds the other.
TRUSTED_EVIDENCE: frozenset[Evidence] = frozenset(
    {Evidence.PLAN, Evidence.SQL_QUALIFIED, Evidence.MANIFEST}
)


@dataclass(frozen=True, slots=True)
class ColumnRef:
    """A column in a node: `(node, column)`."""

    node: str
    column: str

    def __str__(self) -> str:
        return f"{self.node}.{self.column}"


@dataclass(frozen=True, slots=True)
class Node:
    """One relation in the graph."""

    id: str
    kind: NodeKind
    schema: Schema
    engine: str | None = None
    origin: Origin | None = None
    opaque: bool = False
    """True when rowless could not see inside: an unprobed UDF, a dynamic
    select. Downstream edges from here are untrusted."""


@dataclass(frozen=True, slots=True)
class Edge:
    """A derivation from one input column to one output column."""

    source: ColumnRef
    target: ColumnRef
    evidence: Evidence
    transform: str | None = None
    """Short description: `sum`, `cast(int64)`, `dt.date`. Rendered in explain."""


@dataclass(slots=True)
class LineageGraph:
    """The whole repo's column graph for one revision."""

    nodes: dict[str, Node] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)

    def add_node(self, node: Node) -> None:
        """Insert or replace a node."""
        self.nodes[node.id] = node

    def add_edge(self, edge: Edge) -> None:
        """Insert an edge, erroring when either endpoint is unknown.

        Strict on purpose: an edge to a node nobody declared would make the
        blast radius look larger or smaller than it is, and either way the
        graph would stop being checkable.
        """
        for ref in (edge.source, edge.target):
            if ref.node not in self.nodes:
                known = ", ".join(sorted(self.nodes)) or "none"
                raise KeyError(
                    f"unknown node {ref.node!r} in edge {edge.source}->{edge.target}; "
                    f"known nodes: {known}"
                )
        self.edges.append(edge)

    def downstream(self, ref: ColumnRef, *, max_depth: int | None = None) -> Iterator[ColumnRef]:
        """Everything that reads `ref`, transitively. The blast radius."""
        yield from self._walk(ref, max_depth=max_depth, forward=True)

    def upstream(self, ref: ColumnRef, *, max_depth: int | None = None) -> Iterator[ColumnRef]:
        """Everything `ref` was computed from. Answers "where did this come from"."""
        yield from self._walk(ref, max_depth=max_depth, forward=False)

    def _walk(
        self,
        start: ColumnRef,
        *,
        max_depth: int | None,
        forward: bool,
    ) -> Iterator[ColumnRef]:
        """Breadth-first traversal, so results come out nearest-first.

        Nearest-first matters for output: the first line of a blast-radius tree
        should be the pipeline that reads the column directly, not a mart six
        hops away.
        """
        adjacency = self._adjacency(forward=forward)
        seen = {start}
        queue: deque[tuple[ColumnRef, int]] = deque([(start, 0)])
        while queue:
            current, depth = queue.popleft()
            if max_depth is not None and depth >= max_depth:
                continue
            for neighbour in adjacency.get(current, ()):
                if neighbour in seen:
                    continue
                seen.add(neighbour)
                yield neighbour
                queue.append((neighbour, depth + 1))

    def _adjacency(self, *, forward: bool) -> dict[ColumnRef, list[ColumnRef]]:
        """Edge index in one direction. Rebuilt per walk; graphs are small."""
        index: dict[ColumnRef, list[ColumnRef]] = {}
        for edge in self.edges:
            key, value = (edge.source, edge.target) if forward else (edge.target, edge.source)
            index.setdefault(key, []).append(value)
        return index

    def edges_into(self, ref: ColumnRef) -> list[Edge]:
        """Edges that produced `ref`, with their evidence. Backs `rowless explain`."""
        return [edge for edge in self.edges if edge.target == ref]

    def opaque_nodes(self) -> Iterator[Node]:
        """Nodes rowless could not see inside, for the coverage report."""
        return (node for node in self.nodes.values() if node.opaque)

    def paths_through_opaque(self, ref: ColumnRef) -> tuple[str, ...]:
        """Opaque nodes reachable downstream of `ref`.

        Reported next to a blast radius so a small-looking impact is not read
        as a safe one when rowless simply lost the trail.
        """
        reached = {ref.node, *(found.node for found in self.downstream(ref))}
        return tuple(sorted(node.id for node in self.opaque_nodes() if node.id in reached))

    def to_dot(self) -> str:
        """Graphviz output. Backs `rowless graph --format dot`."""
        lines = ["digraph rowless {", "  rankdir=LR;", "  node [shape=box];"]
        for node in self.nodes.values():
            style = ", style=dashed" if node.opaque else ""
            label = f"{node.id}\\n{node.kind.value}"
            lines.append(f'  "{node.id}" [label="{label}"{style}];')
        for edge in self.edges:
            label = edge.transform or edge.evidence.value
            lines.append(
                f'  "{edge.source.node}" -> "{edge.target.node}" '
                f'[label="{edge.source.column}->{edge.target.column}: {label}"];'
            )
        lines.append("}")
        return "\n".join(lines)

    def to_mermaid(self) -> str:
        """Mermaid output, for pasting into a PR comment or a doc."""
        lines = ["graph LR"]
        aliases = {node_id: f"n{index}" for index, node_id in enumerate(sorted(self.nodes))}
        for node_id, alias in aliases.items():
            lines.append(f"  {alias}[{node_id}]")
        for edge in self.edges:
            source = aliases.get(edge.source.node, edge.source.node)
            target = aliases.get(edge.target.node, edge.target.node)
            lines.append(f"  {source} -->|{edge.source.column}| {target}")
        return "\n".join(lines)

    def to_json(self) -> str:
        """Stable serialisation, cached under `.rowless/graph.json`.

        Lets `rowless diff` compare against a graph built on another revision
        without re-running the pipelines of that revision.
        """
        from rowless.ir.serde import origin_to_dict, schema_to_dict

        payload = {
            "version": 1,
            "nodes": [
                {
                    "id": node.id,
                    "kind": node.kind.value,
                    "schema": schema_to_dict(node.schema),
                    "engine": node.engine,
                    "origin": origin_to_dict(node.origin) if node.origin else None,
                    "opaque": node.opaque,
                }
                for node in sorted(self.nodes.values(), key=lambda node: node.id)
            ],
            "edges": [
                {
                    "source": [edge.source.node, edge.source.column],
                    "target": [edge.target.node, edge.target.column],
                    "evidence": edge.evidence.value,
                    "transform": edge.transform,
                }
                for edge in self.edges
            ],
        }
        return json.dumps(payload, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> LineageGraph:
        """Load a serialised graph, as `rowless diff` does for the base revision."""
        from rowless.ir.serde import origin_from_dict, schema_from_dict

        data: dict[str, Any] = json.loads(payload)
        graph = cls()
        for node in data.get("nodes", ()):
            graph.add_node(
                Node(
                    id=node["id"],
                    kind=NodeKind(node["kind"]),
                    schema=schema_from_dict(node["schema"]),
                    engine=node.get("engine"),
                    origin=origin_from_dict(node.get("origin")),
                    opaque=node.get("opaque", False),
                )
            )
        for edge in data.get("edges", ()):
            graph.add_edge(
                Edge(
                    source=ColumnRef(*edge["source"]),
                    target=ColumnRef(*edge["target"]),
                    evidence=Evidence(edge.get("evidence", Evidence.NONE.value)),
                    transform=edge.get("transform"),
                )
            )
        return graph

    def link_by_lineage(self, node_id: str, inputs: dict[str, str]) -> None:
        """Add edges from a node's `derived_from`, which adapters fill in.

        `inputs` maps a parameter name to the node it was bound to, which is
        how a column named in `derived_from` is attributed to the right
        upstream relation.
        """
        node = self.nodes[node_id]
        for column in node.schema.fields:
            for parent in column.derived_from:
                upstream, _, parent_column = parent.rpartition(".")
                source_node = inputs.get(upstream) or (
                    next(iter(inputs.values())) if len(inputs) == 1 else None
                )
                if source_node is None or source_node not in self.nodes:
                    continue
                self.add_edge(
                    Edge(
                        source=ColumnRef(source_node, parent_column or parent),
                        target=ColumnRef(node_id, column.name),
                        evidence=Evidence.PLAN,
                    )
                )
