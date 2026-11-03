# Roadmap

From the proposal, with the work broken out per package. Dates are relative to
project start, not calendar dates.

## Weeks 1-4: spike

Prove the core trick, end to end, on one engine.

- [x] `rowless.ir` - dtypes, schemas, confidence
- [x] `rowless.runtime` - dry run, worker, sandbox
- [ ] `rowless-polars` - the reference adapter
- [x] `rowless.sources.parquet`
- [x] `rowless check` with terminal output
- [ ] a demo on one real open-source pipeline

Everything in core is done and tested with no engine installed: the dry-run
runtime drives an adapter through the contract in
`rowless.testing.conformance`, and the core test suite exercises it against a
small in-process adapter. The remaining work in this phase is the Polars
adapter, which is what makes the demo possible.

Exit criterion: rowless finds a real broken column reference in somebody else
repository, not in an example we wrote.

## Months 2-3: v0.1 on PyPI

Breadth across engines, plus the lowest-friction adoption path.

- [ ] `rowless-pandas` - including the dtype drift fixup table
- [ ] `rowless-duckdb` - and with it the Python-to-SQL seam
- [ ] `rowless-sql` - credential-free SQL via sqlglot
- [ ] `rowless-pytest` - contract tests in an existing suite
- [x] `rowless.discovery` - plain functions and Dagster
- [x] adapter conformance suite complete enough to onboard an owner

## Months 4-6: CI

Breaking-change detection, which is where the tool earns a place in a pipeline.

- [x] `rowless.lineage` - the cross-language column graph
- [x] `rowless diff` - base graph in a git worktree, cached
- [ ] `rowless-dbt` - manifest schemas and model discovery
- [ ] the GitHub Action - comment, annotations, gate
- [ ] `rowless-openlineage`
- [ ] `rowless-ibis`

## After that

- [ ] `rowless-pyspark` - via Spark Connect
- [ ] `rowless-iceberg` - exact renames from field IDs
- [ ] `rowless-lsp` - diagnostics, completion, inline schema hints
- [ ] Airflow and dlt discovery
- [ ] a talk on the bugs it caught

## Success metric

Run rowless on 3 to 5 public pipeline repositories and count the real errors it
finds that would otherwise have failed at runtime. Recorded in `benchmarks/`,
with false positives counted just as prominently: a finding count with no
precision figure next to it is marketing, not a measurement.

## Not scheduled

Named so nobody has to ask twice: static schema inference from Python source, a
web UI, a schema registry, and runtime enforcement. See the end of
`architecture.md` for why each is out.
