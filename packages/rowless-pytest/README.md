# rowless-pytest

Schema contract tests for pytest, at roughly unit-test cost.

```python
def test_revenue_schema(rowless_check):
    rowless_check(daily_revenue, {"day": "date", "amount": "int64"})
```

## Why start here

It asks a team for nothing but a test file: no CI stage, no new command, no
credentials. A dry run moves no data, so covering every pipeline in a repo is
affordable.

An unverified result fails the test. A contract test that silently stops
testing is worse than no test.

## Status

Skeleton. Targeted for v0.1 alongside pandas, DuckDB and SQL.

## Owner

Kauvin Lucas ([@kauvinlucas](https://github.com/kauvinlucas)).
