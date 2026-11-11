# mixed_demo

The example that makes the case for the project: one column renamed in a SQL
model, with the breakage reported in Python.

TODO(examples):
- a dbt-style project with `target/manifest.json` committed
- `models/stg_orders.sql` renaming `amount` to `amount_cents`
- `pipelines/revenue.py` still reading `amount` through Polars
- `marts/finance_kpis.sql` also reading `amount`
- expected `rowless diff main` output asserted in the test suite, matching the
  tree in the proposal
