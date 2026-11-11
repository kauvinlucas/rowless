"""Emitter tests: assert on event payloads, never on a live transport."""

from __future__ import annotations

import json
from typing import Any

import pytest
from openlineage_graphs import PIPELINE_TARGET, coverage, opaque_graph, orders_graph
from rowless.adapters.registry import load, registered_names
from rowless.ir.confidence import Confidence
from rowless_openlineage import facets, transport
from rowless_openlineage.emitter import ENV_INCLUDE_NAME_MATCH, OpenLineageEmitter
from rowless_openlineage.errors import EmitFailed, TransportNotConfigured
from rowless_openlineage.transport import ENV_URL

AT = "2026-11-09T12:00:00+00:00"


class FakeClient:
    """A client that records what it was handed, and can be told to fail."""

    def __init__(self, fail_after: int | None = None) -> None:
        self.events: list[Any] = []
        self.closed = False
        self._fail_after = fail_after

    def emit(self, event: Any) -> None:
        if self._fail_after is not None and len(self.events) >= self._fail_after:
            raise ConnectionError("marquez is down")
        self.events.append(event)

    def close(self) -> bool:
        self.closed = True
        return True


def emitter(**kwargs: Any) -> OpenLineageEmitter:
    kwargs.setdefault("env", {})
    kwargs.setdefault("at", AT)
    return OpenLineageEmitter(**kwargs)


# --- Availability ------------------------------------------------------------


def test_the_emitter_is_registered_as_a_plugin() -> None:
    """`rowless emit` finds it through the entry point, not an import."""
    assert "openlineage" in registered_names("rowless.emitters")
    assert isinstance(load("rowless.emitters", "openlineage"), OpenLineageEmitter)


def test_unavailable_without_a_transport() -> None:
    """Installed and unconfigured is the common case, and it is not usable."""
    found = emitter()

    assert found.available() is False
    assert ENV_URL in found.unavailable_reason()


def test_available_with_a_transport() -> None:
    found = emitter(env={ENV_URL: "http://marquez:5000"})

    assert found.available() is True


def test_an_injected_client_is_its_own_authorisation() -> None:
    """An embedder with a client has already decided where events go."""
    found = emitter(client=FakeClient())

    assert found.available() is True


def test_configuration_is_read_per_call_not_per_instance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The registry caches one instance for the life of the process.

    A long-lived host - the language server, a test session - must not be
    pinned to the environment as it stood when the plugin first loaded.
    """
    found = OpenLineageEmitter()
    monkeypatch.delenv(ENV_URL, raising=False)
    assert found.available() is False

    monkeypatch.setenv(ENV_URL, "http://marquez:5000")
    assert found.available() is True


# --- Emission ----------------------------------------------------------------


def test_emit_hands_every_event_to_the_client() -> None:
    """One event per node, and an injected client is left open.

    It belongs to the caller, who may be emitting from elsewhere in the same
    process; closing somebody else's transport is not this method's business.
    """
    client = FakeClient()
    graph = orders_graph()

    emitter(client=client).emit(graph, coverage())

    assert len(client.events) == len(graph.nodes)
    assert client.closed is False


def test_emit_closes_a_client_it_opened(monkeypatch: pytest.MonkeyPatch) -> None:
    """Closing matters: the async and Kafka transports buffer.

    Skipping it drops whatever is still in the buffer when the process exits,
    which would make `rowless emit` report success for events nobody received.
    """
    client = FakeClient()
    monkeypatch.setattr(transport, "client", lambda env=None: client)

    OpenLineageEmitter(env={ENV_URL: "http://marquez:5000"}, at=AT).emit(
        orders_graph(),
        coverage(),
    )

    assert client.closed is True


def test_emit_refuses_when_nothing_is_configured() -> None:
    """No destination is an error, not a thousand lines of JSON on stdout."""
    with pytest.raises(TransportNotConfigured):
        emitter().emit(orders_graph(), coverage())


def test_a_failed_transport_reports_how_much_got_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A half-published graph is the outcome somebody has to reason about."""
    client = FakeClient(fail_after=2)
    monkeypatch.setattr(transport, "client", lambda env=None: client)

    with pytest.raises(EmitFailed) as caught:
        OpenLineageEmitter(env={ENV_URL: "http://marquez:5000"}, at=AT).emit(
            orders_graph(),
            coverage(),
        )

    assert "marquez is down" in str(caught.value)
    assert "2 of 4" in str(caught.value)
    assert client.closed is True, "a failed emission still flushes what it sent"


def test_what_emit_sends_is_what_dry_run_prints() -> None:
    """A dry run that showed a different shape would be worse than none."""
    from openlineage.client.serde import Serde

    client = FakeClient()
    graph = orders_graph()
    found = emitter(client=client)

    found.emit(graph, coverage())
    printed = found.to_events(graph, coverage())

    assert [Serde.to_dict(event) for event in client.events] == printed


def test_to_events_is_json_and_carries_the_coverage_facet() -> None:
    """What `rowless emit --dry-run` puts on stdout."""
    payload = emitter().to_events(orders_graph(), coverage())

    assert json.loads(json.dumps(payload))
    for event in payload:
        holder: Any = event.get("dataset") or event.get("job")
        assert holder["facets"][facets.COVERAGE_FACET]["executed"] is False


def test_to_events_needs_no_transport() -> None:
    """Reviewing an emission must not require a catalog to review it against."""
    assert emitter().available() is False
    assert emitter().to_events(orders_graph(), coverage())


def test_the_name_match_flag_is_read_from_the_environment() -> None:
    """The one knob with a real cost, so it is explicit and documented."""
    quiet = emitter().to_events(orders_graph(), coverage())
    loud = emitter(env={ENV_INCLUDE_NAME_MATCH: "1"}).to_events(orders_graph(), coverage())

    assert "status" not in _lineage_of(quiet)
    assert "status" in _lineage_of(loud)


# --- Individual facets -------------------------------------------------------


def test_column_lineage_facet_for_one_node() -> None:
    """The accessor the skeleton promised, for tooling that wants one facet."""
    facet: dict[str, Any] = emitter().column_lineage_facet(orders_graph(), PIPELINE_TARGET)

    assert set(facet["fields"]) == {"day", "amount_cents", "country"}
    assert facet["_producer"].startswith("https://github.com/kauvinlucas/rowless")


def test_column_lineage_facet_is_empty_for_an_opaque_node() -> None:
    """An empty facet would claim rowless looked and found nothing."""
    assert emitter().column_lineage_facet(opaque_graph(), PIPELINE_TARGET) == {}


def test_coverage_facet_accessor_matches_what_the_events_carry() -> None:
    """One implementation, so the accessor cannot drift from the emission."""
    found = emitter()
    direct = found.coverage_facet(coverage(Confidence.EXACT))
    first: dict[str, Any] = found.to_events(orders_graph(), coverage(Confidence.EXACT))[0]
    in_event = first["dataset"]["facets"]

    assert direct == in_event[facets.COVERAGE_FACET]
    assert direct["verified"] is True


def _lineage_of(payload: list[dict[str, Any]]) -> set[str]:
    """Columns with published lineage on the pipeline's output dataset."""
    for event in payload:
        for output in event.get("outputs", ()):
            lineage = output["facets"].get(facets.COLUMN_LINEAGE_FACET)
            if lineage is not None:
                return set(lineage["fields"])
    return set()
