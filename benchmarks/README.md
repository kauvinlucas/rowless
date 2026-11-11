# Benchmarks

Two things are measured here, and only one of them is about speed.

## Findings on real repositories

The success metric from the proposal: run rowless against 3 to 5 public pipeline
repositories and count the real errors it finds that would otherwise have failed
at runtime. That number is the pitch for adoption and the talk.

Each run records: repository and commit, pipelines discovered, coverage score,
findings, and for each finding a judgement of whether it is real. False
positives are recorded with the same prominence as true ones; a count of
findings with no precision figure next to it is marketing, not a measurement.

Candidate repositories (TODO: pick and justify each):
- [ ] a dbt project with Python models
- [ ] a Dagster project with Polars assets
- [ ] an Airflow repository with pandas tasks
- [ ] a dlt pipeline collection
- [ ] a PySpark repository

## Speed

"At linter speed" is a promise, so it gets measured: time per dry run by engine,
cold versus warm worker, and the full-repository check time for each benchmark
repository. The budget assertion lives in `tests/test_examples.py` so a
regression fails CI.
