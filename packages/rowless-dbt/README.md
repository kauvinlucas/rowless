# rowless-dbt

dbt integration: manifest schemas, model discovery, and the Python-to-SQL link.

## The trick

Read `target/manifest.json` with the standard library and nothing else. No
dbt-core dependency and no warehouse credentials, so this works in the CI job
where it matters. Models whose columns are not declared get resolved by running
their compiled SQL through `rowless-sql`.

## Why this package carries the proposal

The dbt column-lineage tools list Python models as unsupported, and the Python
lineage tools stop at the SQL boundary. This package is the join: a dbt model
and a Polars function become nodes in the same column graph, so one rename
reports both.

## Status

Skeleton. Manifest versions v11 and v12 are the first target.

## Owner

Kauvin Lucas ([@kauvinlucas](https://github.com/kauvinlucas)).
