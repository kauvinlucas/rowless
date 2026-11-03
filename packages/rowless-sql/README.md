# rowless-sql

SQL adapter for rowless: sqlglot qualification and type annotation.

## The trick

`sqlglot` qualifies identifiers and annotates types against the input
schemas, so SQL resolves with no database and no credentials. Lower confidence
than a real binder, and it reports that: unannotated functions become ANY
columns with a note, never a guess.

## Status

Skeleton. Every method raises `NotImplementedError`. The conformance suite in
`rowless.testing.conformance` defines what finished means.

## Owner

Kauvin Lucas ([@kauvinlucas](https://github.com/kauvinlucas)).

## Development

```bash
uv sync --package rowless-sql
pytest packages/rowless-sql
```
