# Examples

Each example is a small, self-contained repository that rowless can be pointed
at. They double as integration tests: `nox -s examples` runs `rowless check`
over every one and asserts on the findings, so the examples cannot rot.

| Example | Shows |
| --- | --- |
| `polars_demo` | The spike: Polars plus Parquet sources, one real finding |
| `sql_demo` | SQL models resolved with sqlglot, no database |
| `mixed_demo` | The point of the project: a rename in SQL breaking Python |

## Success metric

The proposal measures success by running rowless against 3 to 5 public pipeline
repositories and counting real errors it would have caught. Those runs belong in
`benchmarks/`, not here: these examples are small and deliberately broken, which
proves the mechanism works but proves nothing about usefulness.
