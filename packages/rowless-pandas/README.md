# rowless-pandas

pandas adapter for rowless: empty typed frames, with the dtype drift fixed.

## The trick

Zero-row frames with dtypes declared up front. The work is in the
fixup table: the places where the pandas dtype for an empty frame differs from
the one it would produce with rows, each fixup documented and reported rather
than applied silently.

## Status

Skeleton. Every method raises `NotImplementedError`. The conformance suite in
`rowless.testing.conformance` defines what finished means.

## Owner

Kauvin Lucas ([@kauvinlucas](https://github.com/kauvinlucas)).

## Development

```bash
uv sync --package rowless-pandas
pytest packages/rowless-pandas
```
