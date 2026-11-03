# rowless-pyspark

PySpark adapter for rowless: the analyzed plan, via Spark Connect.

## The trick

`df.schema` triggers Catalyst analysis without launching a job, so the
schema is the one Spark itself would produce, implicit casts included. Spark
Connect keeps it fast: the startup cost of a classic local session alone would
break the linter-speed promise.

## Status

Skeleton. Every method raises `NotImplementedError`. The conformance suite in
`rowless.testing.conformance` defines what finished means.

## Owner

Kauvin Lucas ([@kauvinlucas](https://github.com/kauvinlucas)).

## Development

```bash
uv sync --package rowless-pyspark
pytest packages/rowless-pyspark
```
