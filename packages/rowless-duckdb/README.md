# rowless-duckdb

DuckDB adapter for rowless: empty tables plus DESCRIBE.

## The trick

Empty tables with the declared schema, then `DESCRIBE`. The DuckDB
binder answers, so the coercion rules are its own. Also the bridge that puts SQL
strings and Python frames into one lineage graph.

## Status

Skeleton. Every method raises `NotImplementedError`. The conformance suite in
`rowless.testing.conformance` defines what finished means.

## Owner

Kauvin Lucas ([@kauvinlucas](https://github.com/kauvinlucas)).

## Development

```bash
uv sync --package rowless-duckdb
pytest packages/rowless-duckdb
```
