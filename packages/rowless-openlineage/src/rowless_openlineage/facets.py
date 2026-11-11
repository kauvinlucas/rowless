"""The facets rowless puts on an event, and the two it had to invent.

Three of these are standard facets, which is the entire reason this package is
worth having: `schema` and `columnLineage` are what Marquez, DataHub, Atlan and
OpenMetadata already read, so rowless's graph arrives as lineage rather than as
a report nobody has a viewer for.

The other two are custom, and both exist to stop a catalog from presenting a
guess as a fact:

* **`rowless_coverage`** carries the coverage score of the run that produced
  the event. Without it a catalog cannot tell lineage rowless verified from
  lineage it partly inferred, and it would draw both with the same confident
  arrow. It also carries `executed: false`, because the one unrecoverable
  mistake here would be a catalog recording a dry run as a production job.
* **`rowless_node`** carries what rowless knows about the node that a standard
  facet has no slot for: which engine resolved it, where it is defined, whether
  rowless could see inside it at all, and how many edges were withheld for
  weak evidence.

Both are plain dicts rather than `BaseFacet` subclasses. The spec asks a custom
facet for two keys, `_producer` and `_schemaURL`, and nothing else; defining an
attrs class at import time would mean importing openlineage at module scope,
which `available()` has to work without. The schema URLs point at the JSON
schemas committed under `docs/spec/`, so a catalog that follows them finds a
real document.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

from rowless.ir.confidence import Confidence, Coverage
from rowless.ir.dtypes import Kind
from rowless.ir.schema import Field, Schema
from rowless.lineage.graph import TRUSTED_EVIDENCE, ColumnRef, Edge, LineageGraph, Node

from rowless_openlineage.naming import Naming

if TYPE_CHECKING:  # pragma: no cover - import-time types only
    from openlineage.client.facet_v2 import (
        DatasetFacet,
        JobFacet,
        column_lineage_dataset,
        schema_dataset,
    )

#: Facet keys. The first two are the spec's; the last two are rowless's own and
#: are prefixed, as the spec requires of a custom facet.
SCHEMA_FACET = "schema"
COLUMN_LINEAGE_FACET = "columnLineage"
COVERAGE_FACET = "rowless_coverage"
NODE_FACET = "rowless_node"

#: Where the custom facets' schemas are published.
_SPEC_BASE = "https://raw.githubusercontent.com/kauvinlucas/rowless/main/docs/spec"
COVERAGE_SCHEMA_URL = f"{_SPEC_BASE}/RowlessCoverageFacet.json#/$defs/RowlessCoverageFacet"
NODE_SCHEMA_URL = f"{_SPEC_BASE}/RowlessNodeFacet.json#/$defs/RowlessNodeFacet"

#: The producer URI template. Spelled the way the OpenLineage clients spell
#: theirs - repository tree at the producing version - so a catalog operator
#: reading `_producer` on an unfamiliar event lands on the code that sent it.
PRODUCER_TEMPLATE = (
    "https://github.com/kauvinlucas/rowless/tree/{version}/packages/rowless-openlineage"
)

#: What a reader of the job sees first in a catalog UI. The point of the
#: sentence is that nothing ran.
JOB_DOCUMENTATION = (
    "Static column lineage from a rowless dry run. The pipeline was executed "
    "against empty frames to resolve its schema; no rows were read or written, "
    "and this is not a record of a production run. See "
    "https://github.com/kauvinlucas/rowless."
)

#: Transform names that are aggregations, so the facet can say so instead of
#: calling every derivation a transformation. Matched against the leading
#: identifier of an adapter's transform label: `sum`, `mean(amount)`.
_AGGREGATIONS: frozenset[str] = frozenset(
    {
        "sum",
        "mean",
        "avg",
        "average",
        "min",
        "max",
        "count",
        "n_unique",
        "nunique",
        "median",
        "std",
        "var",
        "first",
        "last",
        "quantile",
        "agg",
    }
)

#: Spec vocabulary for `Transformation.type`. Every rowless edge is a
#: column-to-column derivation, which is what DIRECT means; rowless records no
#: indirect (filter, group-by, join-key) influences, so none are claimed.
_DIRECT = "DIRECT"


def producer() -> str:
    """The producer URI for this installation of rowless-openlineage."""
    try:
        installed = version("rowless-openlineage")
    except PackageNotFoundError:  # pragma: no cover - running from a source tree
        installed = "main"
    return PRODUCER_TEMPLATE.format(version=installed)


def schema_facet(schema: Schema) -> schema_dataset.SchemaDatasetFacet:
    """The standard `schema` facet, with rowless's own dtype spelling.

    `type` is the compact IR spelling (`int64`, `datetime[us, UTC]`), not the
    engine's: a column resolved through Polars and the same column resolved
    through DuckDB have to arrive in the catalog as the same type, or a
    cross-engine lineage graph reports a dtype change at every seam.
    """
    from openlineage.client.facet_v2 import schema_dataset

    return schema_dataset.SchemaDatasetFacet(
        fields=[_schema_field(field) for field in schema.fields],
        producer=producer(),
    )


def _schema_field(field: Field) -> schema_dataset.SchemaDatasetFacetFields:
    """One column, with struct members nested as the spec allows."""
    from openlineage.client.facet_v2 import schema_dataset

    return schema_dataset.SchemaDatasetFacetFields(
        name=field.name,
        type=str(field.dtype),
        description=field.description,
        fields=_struct_members(field),
    )


def _struct_members(field: Field) -> list[schema_dataset.SchemaDatasetFacetFields] | None:
    """Members of a struct column, so a catalog can expand it.

    Only one level is unfolded, and only for STRUCT. Anything deeper is already
    legible in the `type` string, and a catalog that cannot show nested fields
    still shows the whole dtype.

    None rather than an empty list for an ordinary column: the serialiser drops
    a null, so a scalar column arrives without the key instead of with an empty
    `fields` that a reader has to interpret.
    """
    from openlineage.client.facet_v2 import schema_dataset

    if field.dtype.kind is not Kind.STRUCT:
        return None
    return [
        schema_dataset.SchemaDatasetFacetFields(name=name, type=str(dtype))
        for name, dtype in zip(field.dtype.names, field.dtype.inner, strict=False)
    ]


def column_lineage_facet(
    graph: LineageGraph,
    node_id: str,
    naming: Naming,
    *,
    include_name_match: bool = False,
) -> column_lineage_dataset.ColumnLineageDatasetFacet | None:
    """The standard `columnLineage` facet for one node, or None when it has none.

    Weak evidence is withheld by default, and that is the one judgement in this
    module worth arguing about. A `NAME_MATCH` edge means two columns share a
    name and rowless never saw one feed the other; core already refuses to gate
    a CI run on it. Publishing it to a catalog is strictly worse than refusing
    to, because a catalog arrow has no "probably" in it and will be read as
    verified lineage by someone who was not there. The count of what was
    withheld travels in `rowless_node`, so the gap is visible rather than
    silent, and `include_name_match` publishes them for a team that would
    rather have a complete-looking graph and knows what it is buying.
    """
    from openlineage.client.facet_v2 import column_lineage_dataset

    node = graph.nodes[node_id]
    fields: dict[str, column_lineage_dataset.Fields] = {}
    for column in node.schema.fields:
        edges = _edges_for(graph, ColumnRef(node_id, column.name), include_name_match)
        if not edges:
            continue
        fields[column.name] = column_lineage_dataset.Fields(
            inputFields=[_input_field(graph, edge, naming) for edge in edges],
            # The field-level description and type predate per-input
            # transformations and cannot express a column with two differently
            # derived inputs. They carry the strongest-evidence input's, for
            # catalogs that read only the older shape; the exact answer is on
            # each input field.
            transformationDescription=_describe(edges[0]),
            transformationType=_subtype(edges[0]),
        )
    if not fields:
        return None
    return column_lineage_dataset.ColumnLineageDatasetFacet(
        fields=fields,
        producer=producer(),
    )


def _edges_for(graph: LineageGraph, ref: ColumnRef, include_name_match: bool) -> list[Edge]:
    """Edges into one column, strongest evidence first and deterministic.

    Sorted rather than left in insertion order: two runs of the same repository
    must produce byte-identical events, or `rowless emit --dry-run` cannot be
    diffed and a catalog records a change nobody made.
    """
    edges = [
        edge
        for edge in graph.edges_into(ref)
        if include_name_match or edge.evidence in TRUSTED_EVIDENCE
    ]
    return sorted(
        edges,
        key=lambda edge: (
            edge.evidence not in TRUSTED_EVIDENCE,
            edge.source.node,
            edge.source.column,
        ),
    )


def omitted_edges(graph: LineageGraph, node_id: str, *, include_name_match: bool) -> int:
    """How many edges into this node were withheld for weak evidence."""
    if include_name_match:
        return 0
    node = graph.nodes[node_id]
    return sum(
        1
        for column in node.schema.fields
        for edge in graph.edges_into(ColumnRef(node_id, column.name))
        if edge.evidence not in TRUSTED_EVIDENCE
    )


def _input_field(
    graph: LineageGraph,
    edge: Edge,
    naming: Naming,
) -> column_lineage_dataset.InputField:
    """One upstream column, named the way the rest of the catalog names it."""
    from openlineage.client.facet_v2 import column_lineage_dataset

    upstream = graph.nodes[edge.source.node]
    dataset = naming.dataset(upstream)
    return column_lineage_dataset.InputField(
        namespace=dataset.namespace,
        name=dataset.name,
        field=edge.source.column,
        transformations=[
            column_lineage_dataset.Transformation(
                type=_DIRECT,
                subtype=_subtype(edge),
                description=_describe(edge),
            )
        ],
    )


def _subtype(edge: Edge) -> str:
    """Spec subtype for a derivation: IDENTITY, AGGREGATION or TRANSFORMATION."""
    label = (edge.transform or "").strip().lower()
    if not label:
        return "IDENTITY"
    head = label.split("(", 1)[0].strip()
    if head in _AGGREGATIONS or head.rpartition(".")[2] in _AGGREGATIONS:
        return "AGGREGATION"
    return "TRANSFORMATION"


def _describe(edge: Edge) -> str:
    """Transform label plus the evidence it rests on.

    The evidence rides in the description because the spec has nowhere else for
    it, and because dropping it would publish a plan-walked derivation and a
    manifest-declared one as indistinguishable. A reader comparing two arrows
    in a catalog can then see which one the engine itself computed.
    """
    if edge.transform:
        return f"{edge.transform} (evidence: {edge.evidence.value})"
    return f"evidence: {edge.evidence.value}"


def coverage_facet(coverage: Coverage) -> dict[str, Any]:
    """The custom `rowless_coverage` facet. Rides on every event.

    On every event, not just the job's: a catalog may ingest one dataset event
    and drop the rest, and a schema that arrives with no coverage score beside
    it is a schema a reader will assume was verified.
    """
    return {
        "_producer": producer(),
        "_schemaURL": COVERAGE_SCHEMA_URL,
        "executed": False,
        "confidence": coverage.confidence.name.lower(),
        "verified": coverage.confidence is Confidence.EXACT,
        "columnsTotal": coverage.columns_total,
        "columnsTyped": coverage.columns_typed,
        "columnsTraced": coverage.columns_traced,
        "typedRatio": round(coverage.typed_ratio, 4),
        "tracedRatio": round(coverage.traced_ratio, 4),
        "reasons": list(coverage.reasons),
    }


def node_facet(node: Node, *, untraced: int, withheld: int) -> dict[str, Any]:
    """The custom `rowless_node` facet: what rowless knows that the spec lacks.

    `opaque` is the field that matters. A node rowless could not see inside is
    still published, because a missing node would make the graph look complete;
    the flag is how a catalog can tell a leaf that has no upstream from a leaf
    whose upstream rowless lost.
    """
    return {
        "_producer": producer(),
        "_schemaURL": NODE_SCHEMA_URL,
        "kind": node.kind.value,
        "engine": node.engine,
        "origin": str(node.origin) if node.origin is not None else None,
        "opaque": node.opaque,
        "columns": len(node.schema.fields),
        "untracedColumns": untraced,
        "withheldEdges": withheld,
    }


def job_facets(coverage: Coverage) -> dict[str, JobFacet]:
    """Facets on the job: what kind of job this is, and that it did not run."""
    from openlineage.client.facet_v2 import documentation_job, job_type_job

    facets: dict[str, Any] = {
        "jobType": job_type_job.JobTypeJobFacet(
            processingType="BATCH",
            integration="ROWLESS",
            jobType="DRY_RUN",
            producer=producer(),
        ),
        "documentation": documentation_job.DocumentationJobFacet(
            description=JOB_DOCUMENTATION,
            producer=producer(),
        ),
        COVERAGE_FACET: coverage_facet(coverage),
    }
    return facets


def dataset_facets(
    graph: LineageGraph,
    node: Node,
    coverage: Coverage,
    naming: Naming,
    *,
    include_name_match: bool,
    with_lineage: bool = True,
) -> dict[str, DatasetFacet]:
    """Every facet rowless puts on one dataset."""
    facets: dict[str, Any] = {
        SCHEMA_FACET: schema_facet(node.schema),
        COVERAGE_FACET: coverage_facet(coverage),
        NODE_FACET: node_facet(
            node,
            untraced=_untraced(graph, node, include_name_match=include_name_match),
            withheld=omitted_edges(graph, node.id, include_name_match=include_name_match),
        ),
    }
    if with_lineage:
        lineage = column_lineage_facet(
            graph,
            node.id,
            naming,
            include_name_match=include_name_match,
        )
        if lineage is not None:
            facets[COLUMN_LINEAGE_FACET] = lineage
    return facets


def _untraced(graph: LineageGraph, node: Node, *, include_name_match: bool) -> int:
    """Columns of `node` that reach the catalog with no lineage at all."""
    return sum(
        1
        for column in node.schema.fields
        if not _edges_for(graph, ColumnRef(node.id, column.name), include_name_match)
    )
