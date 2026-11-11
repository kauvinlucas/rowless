"""OpenLineage emitter for rowless.

rowless does not need a lineage UI: column-level lineage is already an
OpenLineage facet, and Marquez, DataHub, Atlan and OpenMetadata all read it. So
this package puts the cross-language column graph where the catalogs already
look, as **static lineage** - no run events, because nothing ran - with a
coverage facet attached so a catalog can tell what rowless verified from what
it only partly saw.

`OpenLineageEmitter` is registered under the `rowless.emitters` entry-point
group and is reached through `rowless emit`. The rest is exported for embedders
that have their own client or their own naming rules, and for tests that assert
on payloads instead of standing up a server.
"""

from rowless_openlineage.emitter import OpenLineageEmitter
from rowless_openlineage.errors import EmitFailed, TransportNotConfigured
from rowless_openlineage.events import build_events, to_dicts
from rowless_openlineage.facets import (
    column_lineage_facet,
    coverage_facet,
    dataset_facets,
    job_facets,
    node_facet,
    producer,
    schema_facet,
)
from rowless_openlineage.naming import DatasetName, Naming
from rowless_openlineage.transport import TransportState, describe

__all__ = [
    "DatasetName",
    "EmitFailed",
    "Naming",
    "OpenLineageEmitter",
    "TransportNotConfigured",
    "TransportState",
    "build_events",
    "column_lineage_facet",
    "coverage_facet",
    "dataset_facets",
    "describe",
    "job_facets",
    "node_facet",
    "producer",
    "schema_facet",
    "to_dicts",
]
