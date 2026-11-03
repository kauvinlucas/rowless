# rowless GitHub Action

```yaml
- uses: kauvinlucas/rowless@v0
  with:
    command: diff
    engines: polars duckdb dbt
    fail-on: breaking
```

## What it does

On a pull request, builds the column graph for the base ref and for the head,
diffs them, and posts the cross-language blast radius:

```
BREAKING  stg_orders.amount -> amount_cents
  |- pipelines/revenue.py::daily_revenue    (Python, Polars)
  |- marts/finance_kpis.sql                 (SQL, DuckDB)
```

Plus inline annotations on the changed lines, and optional SARIF for code
scanning.

## Design notes

**Composite, not Docker.** The dry run must use the engine versions the
repository pins. A Docker image with its own Polars would report schema
differences that do not exist in the project.

**Credential-free by default.** With `rowless-dbt` the manifest is read from the
repo, and warehouse schemas come from a committed snapshot. The action needs no
secrets unless a source is configured to require them.

**Two checkouts.** `diff` needs the base ref, which the action gets with
`git worktree` rather than a second `actions/checkout`, so the base graph can be
cached by commit sha across pull requests.

**One comment, updated.** The comment is edited in place on every push rather
than appended, because a long pull request otherwise accumulates a wall of
stale blast radii.

## Implementation checklist

- [ ] install uv, then `rowless` plus the requested adapters
- [ ] run with `--output json` and `--output github` in one pass
- [ ] set the outputs from the JSON report
- [ ] find-or-create the pull request comment via `gh api`
- [ ] upload SARIF when `sarif` is set
- [ ] cache the base graph under `.rowless/` keyed by base sha
