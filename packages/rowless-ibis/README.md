# rowless-ibis

Ibis adapter for rowless: unbound tables and expression schemas.

## The trick

`ibis.table(schema)` is already unbound and `expr.schema()` resolves
with no backend, so this adapter is mostly a dtype mapping. Bonus capability:
resolve one expression against several backends and report where their type
systems disagree.

## Status

Skeleton. Every method raises `NotImplementedError`. The conformance suite in
`rowless.testing.conformance` defines what finished means.

## Owner

Kauvin Lucas ([@kauvinlucas](https://github.com/kauvinlucas)).

## Development

```bash
uv sync --package rowless-ibis
pytest packages/rowless-ibis
```
