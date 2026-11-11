"""Turning the rowless graph into OpenLineage events.

The whole module is one decision, made four times: **nothing here is a run
event**. OpenLineage's usual shape is a `RunEvent` with a START, a COMPLETE and
a run id, and that is exactly what rowless must never send. Nothing executed,
no rows moved, and a catalog that recorded this as a run would show a
production job finishing successfully at a time when all that happened was that
somebody saved a file. So the graph is emitted as *static lineage*: a
`DatasetEvent` carries a dataset's schema and lineage, a `JobEvent` carries a
job and what it reads and writes, and neither carries a run.

## What becomes what

| rowless node | event | carries |
| --- | --- | --- |
| a source (`NodeKind.SOURCE`) | `DatasetEvent` | schema, coverage, node facets |
| code (`python`, `sql`, `dbt_model`) | `JobEvent` | the job, its inputs, and its |
| | | output dataset with column lineage |
| a sink (`NodeKind.SINK`) | `DatasetEvent` | schema, and the lineage of the write |

Every non-source node is therefore both a job and a dataset. That is not a
workaround: the `columnLineage` facet lives on a dataset, and a Python pipeline
whose result is passed on in memory still has to be addressable as the thing
downstream columns were derived from. Where a pipeline does write a table, the
sink is its own dataset and gets the facet too, so the chain in the catalog is
`source -> pipeline -> table` with lineage on each hop.

## Facets are published once

An input dataset inside a `JobEvent` carries no facets. It has its own event -
every node in the graph gets exactly one event - and sending the same schema
twice invites a catalog to show two versions of it, one of them already stale.

## Dataset lineage is declared, column lineage is evidence

A job's inputs are the inputs the pipeline declared, so they are emitted
whatever rowless managed to resolve. A column-level edge is a claim about
derivation, so it is emitted only when something saw it: see
`facets.column_lineage_facet`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from rowless.ir.confidence import Coverage
from rowless.lineage.graph import LineageGraph, Node, NodeKind

from rowless_openlineage import facets
from rowless_openlineage.naming import DatasetName, Naming

if TYPE_CHECKING:  # pragma: no cover - import-time types only
    from openlineage.client.event_v2 import DatasetEvent, JobEvent

#: Node kinds that are code, and so become a job as well as a dataset.
CODE_KINDS: frozenset[NodeKind] = frozenset({NodeKind.PYTHON, NodeKind.SQL, NodeKind.DBT_MODEL})


def event_time() -> str:
    """Now, in UTC, RFC 3339. The time the lineage was *computed*.

    Not the time anything ran, because nothing ran. A static event still needs
    an event time - the spec requires it, and a catalog orders facet versions
    by it - so this is the honest reading of the field: when rowless looked.
    """
    return datetime.now(UTC).isoformat()


def build_events(
    graph: LineageGraph,
    coverage: Coverage,
    *,
    naming: Naming,
    at: str | None = None,
    include_name_match: bool = False,
) -> list[Any]:
    """Every event for one graph, in source-then-job-then-sink order.

    Deterministic throughout: nodes sorted by id, edges sorted inside each
    facet, one timestamp for the whole emission. Two runs over an unchanged
    repository produce identical bytes, which is what makes
    `rowless emit --dry-run` reviewable in a pull request and what stops a
    catalog from recording a version per invocation.
    """
    stamp = at or event_time()
    events: list[Any] = []
    for node in _nodes_of_kind(graph, lambda node: node.kind is NodeKind.SOURCE):
        events.append(
            dataset_event(
                graph,
                node,
                coverage,
                naming,
                at=stamp,
                include_name_match=include_name_match,
            )
        )
    for node in _nodes_of_kind(graph, lambda node: node.kind in CODE_KINDS):
        events.append(
            job_event(
                graph,
                node,
                coverage,
                naming,
                at=stamp,
                include_name_match=include_name_match,
            )
        )
    for node in _nodes_of_kind(graph, lambda node: node.kind is NodeKind.SINK):
        events.append(
            dataset_event(
                graph,
                node,
                coverage,
                naming,
                at=stamp,
                include_name_match=include_name_match,
            )
        )
    return events


def dataset_event(
    graph: LineageGraph,
    node: Node,
    coverage: Coverage,
    naming: Naming,
    *,
    at: str,
    include_name_match: bool,
) -> DatasetEvent:
    """A static dataset event: a schema, and lineage if the node has any."""
    from openlineage.client.event_v2 import DatasetEvent, StaticDataset

    name = naming.dataset(node)
    return DatasetEvent(
        eventTime=at,
        producer=facets.producer(),
        dataset=StaticDataset(
            namespace=name.namespace,
            name=name.name,
            facets=facets.dataset_facets(
                graph,
                node,
                coverage,
                naming,
                include_name_match=include_name_match,
            ),
        ),
    )


def job_event(
    graph: LineageGraph,
    node: Node,
    coverage: Coverage,
    naming: Naming,
    *,
    at: str,
    include_name_match: bool,
) -> JobEvent:
    """A static job event: one pipeline, what it reads, and what it produces."""
    from openlineage.client.event_v2 import InputDataset, Job, JobEvent, OutputDataset

    job = naming.job(node)
    output = naming.dataset(node)
    return JobEvent(
        eventTime=at,
        producer=facets.producer(),
        job=Job(
            namespace=job.namespace,
            name=job.name,
            facets=facets.job_facets(coverage),
        ),
        inputs=[
            InputDataset(namespace=upstream.namespace, name=upstream.name)
            for upstream in _input_names(graph, node, naming)
        ],
        outputs=[
            OutputDataset(
                namespace=output.namespace,
                name=output.name,
                facets=facets.dataset_facets(
                    graph,
                    node,
                    coverage,
                    naming,
                    include_name_match=include_name_match,
                ),
            )
        ],
    )


def _input_names(graph: LineageGraph, node: Node, naming: Naming) -> list[DatasetName]:
    """Datasets this node reads, deduplicated and sorted.

    Read off the edges rather than off the pipeline's declaration, because by
    the time a graph exists the two agree and the graph is what the rest of
    this package works from. A node whose dry run failed has no edges and so
    reads nothing, which is correct: rowless does not know what it read.
    """
    seen: dict[tuple[str, str], DatasetName] = {}
    for edge in graph.edges:
        if edge.target.node != node.id:
            continue
        upstream = graph.nodes.get(edge.source.node)
        if upstream is None:  # pragma: no cover - add_edge refuses unknown nodes
            continue
        name = naming.dataset(upstream)
        seen[(name.namespace, name.name)] = name
    return [seen[key] for key in sorted(seen)]


def _nodes_of_kind(graph: LineageGraph, predicate: Callable[[Node], bool]) -> Iterator[Node]:
    """Matching nodes, by id, so event order never depends on dict order."""
    for node_id in sorted(graph.nodes):
        node = graph.nodes[node_id]
        if predicate(node):
            yield node


def to_dicts(events: Sequence[Any]) -> list[dict[str, Any]]:
    """Serialise events the way the transports do.

    The same `Serde` the client uses, so what `rowless emit --dry-run` prints
    is what `rowless emit` would have sent - a dry run that printed a different
    shape from the real thing would be worse than no dry run.
    """
    from openlineage.client.serde import Serde

    return [Serde.to_dict(event) for event in events]
