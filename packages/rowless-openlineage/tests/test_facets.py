"""Facet payloads, asserted against the spec's shape and against the schemas.

There is no bundled copy of the OpenLineage JSON schemas and these tests do not
reach the network, so "validates against the spec" is checked two ways:

* the standard facets are built by openlineage-python's own generated classes,
  so their field names and required arguments are the spec's by construction,
  and what is asserted here is the *content* rowless puts in them;
* the two custom facets are checked against the JSON schemas committed under
  `docs/spec/`, key for key in both directions, so a field added to the code
  and not to the schema - or the reverse - fails here rather than in somebody's
  catalog.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from openlineage.client.serde import Serde
from openlineage_graphs import (
    CUSTOMERS_URI,
    ORDERS_URI,
    PIPELINE_TARGET,
    coverage,
    opaque_graph,
    orders_graph,
)
from rowless.ir.confidence import Confidence, Coverage
from rowless.ir.dtypes import int_
from rowless.ir.schema import Field, Schema
from rowless.lineage.graph import ColumnRef, Edge, Evidence, LineageGraph, Node, NodeKind
from rowless_openlineage import facets
from rowless_openlineage.naming import Naming

SPEC_DIR = Path(__file__).resolve().parents[3] / "docs" / "spec"


def dict_of(facet: object) -> dict[str, Any]:
    """A facet as the transport would send it."""
    return dict(Serde.to_dict(facet))


# --- The schema facet --------------------------------------------------------


def test_schema_facet_uses_the_ir_dtype_spelling() -> None:
    """A column resolved by two engines must arrive as one type.

    If the facet carried the engine's own spelling, the same column would
    appear as `Int64` from Polars and `BIGINT` from DuckDB, and a cross-engine
    graph would report a dtype change at every seam in the pipeline.
    """
    payload = dict_of(facets.schema_facet(orders_graph().nodes[ORDERS_URI].schema))
    fields = {field["name"]: field["type"] for field in payload["fields"]}

    assert fields == {
        "order_id": "int64",
        "status": "string",
        "amount": "int64",
        "created_at": "datetime[us, UTC]",
    }


def test_schema_facet_unfolds_a_struct_and_keeps_descriptions() -> None:
    """A struct column is expandable in a catalog, not just a type string."""
    payload = dict_of(facets.schema_facet(orders_graph().nodes[CUSTOMERS_URI].schema))
    by_name = {field["name"]: field for field in payload["fields"]}

    assert by_name["country"]["description"] == "ISO 3166-1 alpha-2."
    assert by_name["address"]["type"].startswith("struct{")
    assert [member["name"] for member in by_name["address"]["fields"]] == ["city", "postcode"]


def test_schema_facet_carries_the_rowless_producer() -> None:
    """Every facet says what produced it, and it is not a real pipeline run."""
    payload = dict_of(facets.schema_facet(Schema(fields=(Field(name="a", dtype=int_(32)),))))

    assert "rowless" in payload["_producer"]


# --- Column lineage ----------------------------------------------------------


def test_column_lineage_has_the_spec_shape() -> None:
    """`fields` maps a column to its input fields, each fully addressed."""
    facet = facets.column_lineage_facet(orders_graph(), PIPELINE_TARGET, Naming())
    assert facet is not None
    payload = dict_of(facet)

    assert set(payload) >= {"fields", "_producer", "_schemaURL"}
    day = payload["fields"]["day"]
    assert day["inputFields"] == [
        {
            "namespace": "lake",
            "name": "stg_orders",
            "field": "created_at",
            "transformations": [
                {
                    "type": "DIRECT",
                    "subtype": "TRANSFORMATION",
                    "description": "dt.date (evidence: plan)",
                }
            ],
        }
    ]


def test_input_fields_are_named_like_every_other_dataset() -> None:
    """The upstream column is addressed by the catalog's name for the table."""
    facet = facets.column_lineage_facet(orders_graph(), PIPELINE_TARGET, Naming())
    assert facet is not None
    payload = dict_of(facet)

    assert payload["fields"]["country"]["inputFields"][0]["namespace"] == "s3://warehouse"
    assert payload["fields"]["country"]["inputFields"][0]["name"] == "customers/part-0.parquet"


def test_an_aggregation_is_not_published_as_a_plain_transformation() -> None:
    """`sum` is an AGGREGATION in the spec's vocabulary, and says so."""
    facet = facets.column_lineage_facet(orders_graph(), PIPELINE_TARGET, Naming())
    assert facet is not None
    payload = dict_of(facet)

    transformation = payload["fields"]["amount_cents"]["inputFields"][0]["transformations"][0]
    assert transformation["subtype"] == "AGGREGATION"
    assert transformation["description"] == "sum (evidence: plan)"


def test_a_pass_through_column_is_an_identity() -> None:
    """No transform label means the column went through unchanged."""
    facet = facets.column_lineage_facet(orders_graph(), PIPELINE_TARGET, Naming())
    assert facet is not None
    payload = dict_of(facet)

    transformation = payload["fields"]["country"]["inputFields"][0]["transformations"][0]
    assert transformation["subtype"] == "IDENTITY"
    assert transformation["description"] == "evidence: sql_qualified"


def test_name_matched_edges_are_withheld_by_default() -> None:
    """A catalog arrow has no "probably" in it.

    `status` is connected to its upstream by nothing but a shared column name.
    Core already refuses to gate a CI run on that evidence, so publishing it as
    lineage somebody else will read as verified is worse than leaving a gap.
    """
    facet = facets.column_lineage_facet(orders_graph(), PIPELINE_TARGET, Naming())
    assert facet is not None
    payload = dict_of(facet)

    assert "status" not in payload["fields"]
    assert set(payload["fields"]) == {"day", "amount_cents", "country"}


def test_name_matched_edges_can_be_published_on_request() -> None:
    """A team that wants a complete-looking graph can have one, knowingly."""
    facet = facets.column_lineage_facet(
        orders_graph(),
        PIPELINE_TARGET,
        Naming(),
        include_name_match=True,
    )
    assert facet is not None
    payload = dict_of(facet)

    transformation = payload["fields"]["status"]["inputFields"][0]["transformations"][0]
    assert transformation["description"] == "evidence: name_match"


def test_withheld_edges_are_counted_rather_than_hidden() -> None:
    """The gap is visible in the node facet, so nobody has to guess at it."""
    graph = orders_graph()

    assert facets.omitted_edges(graph, PIPELINE_TARGET, include_name_match=False) == 1
    assert facets.omitted_edges(graph, PIPELINE_TARGET, include_name_match=True) == 0


def test_a_node_with_no_lineage_gets_no_facet() -> None:
    """An empty `columnLineage` facet reads as "we looked and found nothing"."""
    graph = opaque_graph()

    assert facets.column_lineage_facet(graph, PIPELINE_TARGET, Naming()) is None
    assert facets.column_lineage_facet(graph, ORDERS_URI, Naming()) is None


def test_input_fields_are_ordered_deterministically() -> None:
    """Two runs over one repository must produce identical bytes."""
    graph = orders_graph()
    graph.add_edge(
        Edge(
            source=ColumnRef(CUSTOMERS_URI, "customer_id"),
            target=ColumnRef(PIPELINE_TARGET, "day"),
            evidence=Evidence.PLAN,
        )
    )

    first = dict_of(facets.column_lineage_facet(graph, PIPELINE_TARGET, Naming()))
    second = dict_of(facets.column_lineage_facet(graph, PIPELINE_TARGET, Naming()))

    # Sorted by upstream node then column, so `lake://stg_orders` comes before
    # `s3://warehouse/...` however the edges were added.
    assert first == second
    assert [field["field"] for field in first["fields"]["day"]["inputFields"]] == [
        "created_at",
        "customer_id",
    ]


# --- The custom facets -------------------------------------------------------


def test_coverage_facet_says_nothing_was_executed() -> None:
    """The one claim a catalog must not get wrong about these events."""
    payload = facets.coverage_facet(coverage())

    assert payload["executed"] is False
    assert payload["confidence"] == "partial"
    assert payload["verified"] is False
    assert payload["reasons"] == ["status has no lineage evidence"]


def test_coverage_facet_reports_verified_only_when_it_is() -> None:
    """`verified` is EXACT and nothing else: not "mostly", not "probably"."""
    exact = facets.coverage_facet(coverage(Confidence.EXACT))
    unknown = facets.coverage_facet(coverage(Confidence.UNKNOWN))

    assert (exact["confidence"], exact["verified"]) == ("exact", True)
    assert (unknown["confidence"], unknown["verified"]) == ("unknown", False)


def test_coverage_ratios_are_rounded_so_the_payload_is_stable() -> None:
    """A float that renders differently per platform is a spurious diff."""
    payload = facets.coverage_facet(Coverage(columns_total=3, columns_typed=1, columns_traced=1))

    assert payload["typedRatio"] == 0.3333
    assert payload["tracedRatio"] == 0.3333


def test_node_facet_reports_an_opaque_node_as_opaque() -> None:
    """A leaf with no upstream and a leaf whose upstream was lost differ."""
    graph = opaque_graph()
    payload = facets.node_facet(graph.nodes[PIPELINE_TARGET], untraced=0, withheld=0)

    assert payload["opaque"] is True
    assert payload["kind"] == "python"
    assert payload["engine"] == "polars"
    assert payload["origin"] == "pipelines/revenue.py:9"


def test_node_facet_counts_what_the_catalog_will_not_see() -> None:
    """Untraced columns and withheld edges, both named in the schema."""
    graph = orders_graph()
    built: dict[str, Any] = dict(
        facets.dataset_facets(
            graph,
            graph.nodes[PIPELINE_TARGET],
            coverage(),
            Naming(),
            include_name_match=False,
        )
    )
    payload = built[facets.NODE_FACET]

    assert payload["columns"] == 4
    assert payload["untracedColumns"] == 1
    assert payload["withheldEdges"] == 1


def test_job_facets_state_the_integration_and_that_nothing_ran() -> None:
    """What a human sees first in a catalog UI."""
    payload = {key: dict_of(value) for key, value in facets.job_facets(coverage()).items()}

    assert payload["jobType"]["integration"] == "ROWLESS"
    assert payload["jobType"]["jobType"] == "DRY_RUN"
    assert "no rows were read or written" in payload["documentation"]["description"]
    assert payload[facets.COVERAGE_FACET]["executed"] is False


# --- The custom facets against their published schemas ----------------------


def spec_properties(filename: str, definition: str) -> tuple[set[str], set[str]]:
    """Property names and required names from a committed facet schema."""
    schema = json.loads((SPEC_DIR / filename).read_text(encoding="utf-8"))
    body = schema["$defs"][definition]["allOf"][1]
    return set(body["properties"]), set(body["required"])


def test_coverage_facet_matches_its_published_schema() -> None:
    """Code and schema, key for key, in both directions."""
    properties, required = spec_properties("RowlessCoverageFacet.json", "RowlessCoverageFacet")
    payload = facets.coverage_facet(coverage())
    emitted = set(payload) - {"_producer", "_schemaURL"}

    assert emitted <= properties, f"undocumented key(s): {emitted - properties}"
    assert required <= emitted, f"schema requires key(s) nobody emits: {required - emitted}"
    assert payload["_schemaURL"] == facets.COVERAGE_SCHEMA_URL


def test_node_facet_matches_its_published_schema() -> None:
    """Same check for the other invented facet."""
    properties, required = spec_properties("RowlessNodeFacet.json", "RowlessNodeFacet")
    node = Node(id="x", kind=NodeKind.SOURCE, schema=Schema.empty())
    payload = facets.node_facet(node, untraced=0, withheld=0)
    emitted = set(payload) - {"_producer", "_schemaURL"}

    assert emitted <= properties, f"undocumented key(s): {emitted - properties}"
    assert required <= emitted, f"schema requires key(s) nobody emits: {required - emitted}"
    assert payload["_schemaURL"] == facets.NODE_SCHEMA_URL


def test_every_node_kind_is_documented_in_the_node_schema() -> None:
    """A kind added to core must be added to the published enum."""
    schema = json.loads((SPEC_DIR / "RowlessNodeFacet.json").read_text(encoding="utf-8"))
    documented = set(schema["$defs"]["RowlessNodeFacet"]["allOf"][1]["properties"]["kind"]["enum"])

    assert documented == {kind.value for kind in NodeKind}


def test_custom_facets_survive_the_graph_being_empty() -> None:
    """`rowless emit` on a repository with no pipelines must not raise."""
    payload = facets.coverage_facet(Coverage())

    assert payload["typedRatio"] == 1.0
    assert payload["columnsTotal"] == 0
    assert isinstance(LineageGraph(), LineageGraph)
