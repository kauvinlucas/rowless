# rowless-polars

Polars adapter for rowless: resolve LazyFrame schemas without executing.

## The trick

`LazyFrame.collect_schema()` resolves the query plan using the type
rules of Polars itself and executes nothing. The resolved plan is also walkable,
so this is the one adapter that gets exact schemas and full column lineage from
the same call.

## Status

Skeleton. Every method raises `NotImplementedError`. The conformance suite in
`rowless.testing.conformance` defines what finished means.

## Owner

Kauvin Lucas ([@kauvinlucas](https://github.com/kauvinlucas)).

## Development

```bash
uv sync --package rowless-polars
pytest packages/rowless-polars
```
