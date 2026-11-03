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
"""

from __future__ import annotations

from typing import ClassVar

from rowless.ir.confidence import Coverage
from rowless.lineage.graph import LineageGraph


class OpenLineageEmitter:
    """Convert a `LineageGraph` into OpenLineage events and transport them."""

    name: ClassVar[str] = "openlineage"
    requires: ClassVar[str] = "openlineage-python>=1.20"

    def available(self) -> bool:
        """True when openlineage-python imports and a transport is configured."""
        raise NotImplementedError("TODO(openlineage)")

    def emit(self, graph: LineageGraph, coverage: Coverage) -> None:
        """Emit the whole graph as static lineage."""
        raise NotImplementedError("TODO(openlineage)")

    def to_events(self, graph: LineageGraph, coverage: Coverage) -> list[dict[str, object]]:
        """Build the event payloads without transporting them.

        Separate from `emit` so the events can be asserted on in tests and
        dumped to a file by `rowless emit --dry-run`.
        """
        raise NotImplementedError("TODO(openlineage)")

    def column_lineage_facet(self, graph: LineageGraph, node_id: str) -> dict[str, object]:
        """Build the `columnLineage` facet for one node."""
        raise NotImplementedError("TODO(openlineage)")

    def coverage_facet(self, coverage: Coverage) -> dict[str, object]:
        """A custom facet carrying the coverage score.

        Without it a catalog cannot distinguish lineage rowless verified from
        lineage it partially guessed, and would present both as equally solid.
        """
        raise NotImplementedError("TODO(openlineage)")
