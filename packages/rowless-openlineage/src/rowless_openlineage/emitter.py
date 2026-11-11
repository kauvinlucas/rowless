"""Emit the rowless column graph as OpenLineage events.

Column-level lineage is already a standard facet, and Marquez, DataHub, Atlan
and OpenMetadata all consume it. So rowless does not need a UI: it needs to put
its graph where the catalogs already look.

One thing to get right, because getting it wrong would corrupt a catalog: these
are not run events. Nothing executed, no rows moved, and the events must not be
mistaken for a production run. They are emitted as static lineage with a
rowless-specific producer URI and a job namespace of its own, and the coverage
score rides along in a custom facet so a catalog can tell complete lineage from
partial.

This module is the thin part. The decisions are next door and each has its
reasoning written down where it is implemented:

* `rowless_openlineage.naming` - which dataset in the catalog a rowless node
  *is*, which decides whether this lands on a team's existing graph or beside
  it.
* `rowless_openlineage.facets` - the two standard facets, the two rowless had
  to invent, and why weak evidence is withheld from a catalog.
* `rowless_openlineage.events` - static lineage rather than run events, and
  what becomes a job as opposed to a dataset.
* `rowless_openlineage.transport` - refusing to emit when nothing is
  configured, instead of printing a thousand events to stdout and exiting 0.

## Configuration

Everything is an environment variable, because `rowless emit` builds this
through the plugin registry, which takes no arguments, and because every other
OpenLineage integration is configured the same way.

| variable | effect |
| --- | --- |
| `OPENLINEAGE_URL`, `OPENLINEAGE_CONFIG`, `OPENLINEAGE__*` | the destination |
| `OPENLINEAGE_NAMESPACE` | job namespace; default `rowless` |
| `ROWLESS_OPENLINEAGE_NAMESPACES` | per-scheme dataset namespace overrides |
| `ROWLESS_OPENLINEAGE_INCLUDE_NAME_MATCH` | publish name-matched edges too |

Every one of them is re-read on each call rather than captured in the
constructor: the registry caches one instance per process, and a long-lived
host (the LSP, a test suite) must not be stuck with the environment as it stood
when the plugin was first loaded.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, ClassVar

from rowless.ir.confidence import Coverage
from rowless.lineage.graph import LineageGraph

from rowless_openlineage import events as events_module
from rowless_openlineage import facets
from rowless_openlineage.errors import EmitFailed
from rowless_openlineage.naming import Naming
from rowless_openlineage.transport import describe

#: Publish `NAME_MATCH` edges as well as the trusted ones. Off by default; see
#: `facets.column_lineage_facet` for what that buys and what it costs.
ENV_INCLUDE_NAME_MATCH = "ROWLESS_OPENLINEAGE_INCLUDE_NAME_MATCH"

#: Spellings of yes, so a user who writes `true` and a user who writes `1` both
#: get what they meant.
_TRUTHY: frozenset[str] = frozenset({"1", "true", "yes", "on"})


class OpenLineageEmitter:
    """Convert a `LineageGraph` into OpenLineage events and transport them."""

    name: ClassVar[str] = "openlineage"
    requires: ClassVar[str] = "openlineage-python>=1.38"

    def __init__(
        self,
        *,
        env: Mapping[str, str] | None = None,
        client: Any | None = None,
        at: str | None = None,
    ) -> None:
        """Build an emitter.

        Args:
            env: Environment to read configuration from. Defaults to the real
                one; tests and embedders pass their own instead of mutating
                `os.environ`.
            client: A pre-built OpenLineage client. Bypasses the transport
                check, which is what an embedder with its own client wants and
                what makes `emit` assertable in a test without a server.
            at: Fixed event time for every event in an emission. Left unset in
                production; set by `rowless emit --dry-run` and by tests, which
                need byte-stable output.
        """
        self._env = env
        self._client = client
        self._at = at

    # --- Plugin protocol -----------------------------------------------------

    def available(self) -> bool:
        """True when openlineage-python imports and a transport is configured.

        Both halves matter. An installed client with nowhere to send events
        would otherwise fall back to printing them, which a CI job reads as
        success.
        """
        if self._client is not None:
            return True
        return describe(self._env).configured

    def unavailable_reason(self) -> str:
        """One line explaining what is missing. Read by `rowless doctor`.

        An optional hook rather than part of the plugin protocol: core falls
        back to the `requires` string when a plugin does not offer one, and a
        missing transport is not something a PEP 508 requirement can describe.
        """
        return describe(self._env).reason

    # --- Emission ------------------------------------------------------------

    def emit(self, graph: LineageGraph, coverage: Coverage) -> None:
        """Emit the whole graph as static lineage.

        A client this emitter opened is closed afterwards, because the async
        and Kafka transports buffer and skipping that drops events on exit. An
        *injected* client is left open: it belongs to the caller, who may be
        emitting from somewhere else in the same process, and closing somebody
        else's transport is not this method's business.

        Raises:
            TransportNotConfigured: when there is no destination.
            EmitFailed: when the transport refuses, naming how many events had
                already been sent. A half-published graph is the one outcome a
                caller has to know about, because the catalog is then holding
                part of a revision.
        """
        client = self._client if self._client is not None else self._transport_client()
        owned = self._client is None
        payload = self.events(graph, coverage)
        sent = 0
        try:
            for event in payload:
                client.emit(event)
                sent += 1
        except Exception as exc:
            raise EmitFailed(
                f"OpenLineage transport failed: {exc}",
                sent=sent,
                total=len(payload),
            ) from exc
        finally:
            if owned:
                self._close(client)

    def events(self, graph: LineageGraph, coverage: Coverage) -> list[Any]:
        """The typed event objects, before serialisation.

        Public because an embedder with its own client wants the objects, not
        the dicts, and because `emit` and `to_events` must not be able to
        disagree about what an emission contains.
        """
        return events_module.build_events(
            graph,
            coverage,
            naming=Naming.from_env(self._env),
            at=self._at,
            include_name_match=self._include_name_match(),
        )

    def to_events(self, graph: LineageGraph, coverage: Coverage) -> list[dict[str, object]]:
        """Build the event payloads without transporting them.

        Separate from `emit` so the events can be asserted on in tests and
        dumped to a file by `rowless emit --dry-run`.
        """
        return list(events_module.to_dicts(self.events(graph, coverage)))

    # --- Individual facets ---------------------------------------------------

    def column_lineage_facet(self, graph: LineageGraph, node_id: str) -> dict[str, object]:
        """Build the `columnLineage` facet for one node.

        Returns an empty dict for a node with no publishable lineage, which is
        how a node is left without the facet rather than given an empty one: a
        `columnLineage` facet with no fields in it reads, in a catalog, as "we
        looked and there is nothing", and for an opaque node that is false.
        """
        from openlineage.client.serde import Serde

        facet = facets.column_lineage_facet(
            graph,
            node_id,
            Naming.from_env(self._env),
            include_name_match=self._include_name_match(),
        )
        if facet is None:
            return {}
        return dict(Serde.to_dict(facet))

    def coverage_facet(self, coverage: Coverage) -> dict[str, object]:
        """A custom facet carrying the coverage score.

        Without it a catalog cannot distinguish lineage rowless verified from
        lineage it partially guessed, and would present both as equally solid.
        """
        return dict(facets.coverage_facet(coverage))

    # --- Internals -----------------------------------------------------------

    def _transport_client(self) -> Any:
        """The configured client, or a refusal that names the fix."""
        from rowless_openlineage.transport import client

        return client(self._env)

    def _include_name_match(self) -> bool:
        """Whether weakly-evidenced edges are published, read per call."""
        environ = os.environ if self._env is None else self._env
        return environ.get(ENV_INCLUDE_NAME_MATCH, "").strip().lower() in _TRUTHY

    @staticmethod
    def _close(client: Any) -> None:
        """Flush a client this emitter opened, tolerating one that cannot close.

        `close` is not part of any protocol rowless controls, so a client
        without one is fine, and a client whose close fails must not turn a
        successful emission into a failure.
        """
        closer = getattr(client, "close", None)
        if closer is None:
            return
        try:
            closer()
        except Exception:  # pragma: no cover - transport-specific shutdown
            return
