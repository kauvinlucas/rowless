# rowless-openlineage

Emit the rowless column graph as OpenLineage events.

## Why

Column-level lineage is already an OpenLineage facet, and Marquez, DataHub,
Atlan and OpenMetadata all read it. rowless does not need its own UI; it needs
to put its graph where the catalogs already look.

## The one rule

These are static lineage events, not run events. Nothing executed. They carry a
rowless producer URI, their own job namespace, and a coverage facet, so a
catalog can tell verified lineage from partial lineage and nobody mistakes a dry
run for a production job.

## Status

Skeleton. Targeted for the months 4-6 CI milestone.

## Owner

Kauvin Lucas ([@kauvinlucas](https://github.com/kauvinlucas)).
