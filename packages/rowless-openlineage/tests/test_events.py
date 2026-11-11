"""Event shapes: static lineage, one event per node, nothing that looks like a run."""

from __future__ import annotations

import json
from typing import Any

from openlineage.client.event_v2 import DatasetEvent, JobEvent
from openlineage_graphs import (
    CUSTOMERS_URI,
    ORDERS_URI,
    PIPELINE_TARGET,
    SINK_URI,
    coverage,
    opaque_graph,
    orders_graph,
)
from rowless.ir.confidence import Coverage
from rowless.lineage.graph import LineageGraph
from rowless_openlineage import facets
from rowless_openlineage.events import build_events, event_time, to_dicts
from rowless_openlineage.naming import Naming

AT = "2026-11-09T12:00:00+00:00"


def built(graph: LineageGraph | None = None, **kwargs: object) -> list[Any]:
    return build_events(
        graph if graph is not None else orders_graph(),
        coverage(),
        naming=Naming(),
        at=AT,
        **kwargs,  # type: ignore[arg-type]
    )


def payloads(graph: LineageGraph | None = None, **kwargs: object) -> list[dict[str, Any]]:
    return to_dicts(built(graph, **kwargs))


def test_nothing_is_a_run_event() -> None:
    """The one mistake that would corrupt a catalog.

    A `RunEvent` with a START and a COMPLETE would be recorded as a production
    job that ran and succeeded. Nothing ran: the pipeline was executed against
    empty frames to resolve a schema. So the emission is static lineage, and
    this test is here to fail loudly if anybody ever adds a run to it.
    """
    events = built()

    assert all(isinstance(event, DatasetEvent | JobEvent) for event in events)
    for payload in to_dicts(events):
        assert "run" not in payload
        assert "eventType" not in payload
        assert json.dumps(payload).count("COMPLETE") == 0
        assert json.dumps(payload).count("START") == 0


def test_every_node_becomes_exactly_one_event() -> None:
    """A node left out would make the graph look complete when it is not."""
    graph = orders_graph()
    events = payloads(graph)

    assert len(events) == len(graph.nodes)
    named = {_named(event) for event in events}
    assert len(named) == len(graph.nodes)


def test_sources_are_dataset_events_and_code_is_a_job_event() -> None:
    """The mapping in the module docstring, in order: sources, jobs, sinks."""
    events = built()

    assert [type(event).__name__ for event in events] == [
        "DatasetEvent",
        "DatasetEvent",
        "JobEvent",
        "DatasetEvent",
    ]
    assert _dataset_of(to_dicts(events)[0]) == ("lake", "stg_orders")
    assert _job_of(to_dicts(events)[2]) == ("rowless", PIPELINE_TARGET)
    assert _dataset_of(to_dicts(events)[3]) == ("s3://marts", "revenue")


def test_the_job_reads_the_datasets_its_columns_came_from() -> None:
    """Dataset-level lineage is what the pipeline declared, deduplicated."""
    job = to_dicts(built())[2]

    assert [(found["namespace"], found["name"]) for found in job["inputs"]] == [
        ("lake", "stg_orders"),
        ("s3://warehouse", "customers/part-0.parquet"),
    ]


def test_an_input_dataset_carries_no_facets() -> None:
    """Each schema is published once, by the dataset's own event.

    Sending it twice invites a catalog to hold two versions of one schema, one
    of which is already stale.
    """
    job = to_dicts(built())[2]

    assert all(not found.get("facets") for found in job["inputs"])


def test_the_job_output_carries_the_schema_and_the_lineage() -> None:
    """The facets a catalog draws the graph from."""
    job = to_dicts(built())[2]
    output = job["outputs"][0]

    assert (output["namespace"], output["name"]) == ("rowless", PIPELINE_TARGET)
    assert set(output["facets"]) == {
        facets.SCHEMA_FACET,
        facets.COLUMN_LINEAGE_FACET,
        facets.COVERAGE_FACET,
        facets.NODE_FACET,
    }
    assert set(output["facets"][facets.COLUMN_LINEAGE_FACET]["fields"]) == {
        "day",
        "amount_cents",
        "country",
    }


def test_the_sink_is_lineage_in_its_own_right() -> None:
    """`source -> pipeline -> table`, with a facet on each hop."""
    sink = to_dicts(built())[3]
    lineage = sink["dataset"]["facets"][facets.COLUMN_LINEAGE_FACET]

    assert lineage["fields"]["amount_cents"]["inputFields"] == [
        {
            "namespace": "rowless",
            "name": PIPELINE_TARGET,
            "field": "amount_cents",
            "transformations": [
                {"type": "DIRECT", "subtype": "IDENTITY", "description": "evidence: plan"}
            ],
        }
    ]


def test_every_event_carries_the_coverage_facet() -> None:
    """A schema that arrives without one will be read as verified.

    A catalog may ingest one of these events and drop the rest, so the score
    cannot live only on the job.
    """
    for payload in payloads():
        holder = payload.get("dataset") or payload.get("job")
        assert holder is not None
        assert holder["facets"][facets.COVERAGE_FACET]["executed"] is False
        assert holder["facets"][facets.COVERAGE_FACET]["confidence"] == "partial"


def test_an_opaque_pipeline_is_still_published() -> None:
    """A missing node would make a lost trail look like the end of the graph."""
    events = payloads(opaque_graph())
    job = events[1]
    output = job["outputs"][0]

    assert output["facets"][facets.NODE_FACET]["opaque"] is True
    assert facets.COLUMN_LINEAGE_FACET not in output["facets"]
    assert job["inputs"] == []


def test_an_empty_graph_emits_nothing_rather_than_failing() -> None:
    """`rowless emit` on a repository with no pipelines is a no-op."""
    assert build_events(LineageGraph(), Coverage(), naming=Naming(), at=AT) == []


def test_two_emissions_of_one_graph_are_byte_identical() -> None:
    """Otherwise `--dry-run` cannot be diffed and a catalog versions noise."""
    first = json.dumps(payloads(), sort_keys=True)
    second = json.dumps(payloads(orders_graph()), sort_keys=True)

    assert first == second


def test_one_timestamp_for_the_whole_emission() -> None:
    """Half a graph stamped a second later is a graph in two versions."""
    stamps = {payload["eventTime"] for payload in payloads()}

    assert stamps == {AT}


def test_the_generated_event_time_is_offset_aware() -> None:
    """The spec wants RFC 3339; a naive timestamp is a different instant."""
    generated = event_time()

    assert generated.endswith("+00:00")


def test_the_payloads_are_json_serialisable() -> None:
    """What the transport does, done here, so a facet that cannot be sent fails."""
    assert json.loads(json.dumps(payloads()))


def test_name_match_flows_through_to_the_events() -> None:
    """The flag is honoured end to end, not only in the facet builder."""
    without = to_dicts(built())[2]["outputs"][0]["facets"]
    included = to_dicts(built(include_name_match=True))[2]["outputs"][0]["facets"]

    assert "status" not in without[facets.COLUMN_LINEAGE_FACET]["fields"]
    assert "status" in included[facets.COLUMN_LINEAGE_FACET]["fields"]
    assert without[facets.NODE_FACET]["withheldEdges"] == 1
    assert included[facets.NODE_FACET]["withheldEdges"] == 0


def test_namespaces_are_consistent_across_one_emission() -> None:
    """A job in one namespace and its output in another is two graphs."""
    events = payloads()
    namespaces = {_named(event)[0] for event in events}

    assert namespaces == {"lake", "s3://warehouse", "rowless", "s3://marts"}


def test_sources_keep_their_own_identity_in_the_graph() -> None:
    """Both sources are published, each under the name its scheme implies."""
    events = payloads()

    assert _dataset_of(events[0]) == ("lake", "stg_orders")
    assert _dataset_of(events[1]) == ("s3://warehouse", "customers/part-0.parquet")
    assert ORDERS_URI not in json.dumps(events[0]["dataset"]["facets"][facets.SCHEMA_FACET])
    assert CUSTOMERS_URI not in json.dumps(events[1]["dataset"]["facets"][facets.SCHEMA_FACET])
    assert SINK_URI.rstrip("/").endswith("revenue")


def _named(payload: dict[str, Any]) -> tuple[str, str]:
    """The dataset or job an event is about, whichever it carries."""
    found = _dataset_of(payload) or _job_of(payload)
    assert found is not None, f"event names neither a dataset nor a job: {payload}"
    return found


def _dataset_of(payload: dict[str, Any]) -> tuple[str, str] | None:
    dataset = payload.get("dataset")
    if dataset is None:
        return None
    return (dataset["namespace"], dataset["name"])


def _job_of(payload: dict[str, Any]) -> tuple[str, str] | None:
    job = payload.get("job")
    if job is None:
        return None
    return (job["namespace"], job["name"])
