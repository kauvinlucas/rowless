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

There is no trick, and no second opinion either. The comparison is
`rowless.api.compare_schemas`, the same function behind `rowless.assert_schema`,
and the inputs resolve through the `rowless.toml` the CLI reads. A test that
passed here while `rowless check` failed would be two tools wearing one name.

What the plugin adds is the part only pytest can do: the engine a test needs
decided at collection time, one source cache shared across the suite, and the
coverage the suite actually earned printed next to its passes.

## Install

```bash
pip install rowless-pytest rowless-polars    # plus whichever adapters you use
```

It registers as a pytest entry point, so the fixtures exist as soon as it is
installed. There is nothing to add to `conftest.py`.

## The rule

**An unverified result fails the test.** A contract test that quietly stops
testing - because the pipeline grew an `if df.is_empty()` and the dry run took
a branch production would not - is worse than no test at all, so a result whose
confidence is `UNKNOWN` fails:

```
rowless did not verify pipelines.revenue:daily_revenue, so this test proves nothing:
  reason: polars.DataFrame.is_empty depends on there being rows at revenue.py:14 and got
  False; production would take a different path, so this schema is not verified
  fix the row-dependent code, or pass allow_unverified=True to assert on the schema anyway
  RWL201 unverified: row-dependent call polars.DataFrame.is_empty in pipelines.revenue:daily_revenue (revenue.py:14)
```

`allow_unverified=True` is the opt-out, per assertion, and the session summary
counts how many times it was used. A column rowless typed as ANY fails the same
way even against a matching declaration: the dtype algebra lets ANY match
anything, which is right for inference and wrong for a test meant to prove
something.

## Fixtures

| fixture | what it gives you |
| --- | --- |
| `rowless_check` | run a pipeline on rowless inputs and assert its output schema |
| `rowless_frame` | a frame with a full schema and no rows, of the engine's own type |
| `rowless_source` | the frame a source URI resolves to, or `.schema(uri)` for the schema alone |

```python
rowless_check(target, expected, *, inputs=None, exact=True, allow_unverified=False, engine=None)
```

| argument | what it takes |
| --- | --- |
| `target` | the pipeline function |
| `expected` | a `Schema`, or `{"day": "date", "amount": "int64"}` |
| `inputs` | `{param: Schema \| {"col": "dtype"} \| "lake://table"}` |
| `exact` | `False` allows output columns the test did not name |
| `allow_unverified` | accept a result rowless could not verify |
| `engine` | adapter name, where nothing else says |

With no `inputs`, the pipeline's own declaration is used, so a pipeline says
where its inputs come from once:

```python
@rowless.pipeline(inputs={"orders": "lake://stg_orders"})
def daily_revenue(orders): ...


def test_revenue(rowless_check):
    rowless_check(daily_revenue, {"day": "date", "amount": "int64"})
```

`rowless_check` returns the `DryRunResult`, so a test that wants more than
dtypes has it - column lineage, the writes the pipeline attempted, how long the
dry run took.

The other two fixtures are for the functions that are not pipelines:

```python
def test_normalise(rowless_frame):
    out = normalise(rowless_frame({"amount": "int64"}, name="orders"))
    assert rowless.schema_of(out)["amount"].dtype == float_(64)


def test_the_source_still_has_the_column(rowless_source):
    assert "amount" in rowless_source.schema("lake://stg_orders")
```

`rowless_source.schema(uri)` materializes nothing, so a test about an upstream
table - is the column this repository reads still there - runs with no engine
installed at all. Calling the fixture instead gives you the frame, which does
need one.

A frame from `rowless_frame` is a real `pl.LazyFrame` or `pd.DataFrame`, not a
proxy, and it is built outside a dry-run session - so row-dependent code called
with one is **not** reported. Anything shaped like a pipeline belongs in
`rowless_check`.

## The marker

```python
@pytest.mark.rowless(engine="polars")
def test_revenue(rowless_check): ...
```

`engine` does two things: it names the adapter the test's fixtures use, and it
**skips** the test when that engine is not installed.

The skip reason is the registry's own, so it names the engine and the extra:

```
rowless: adapter 'polars' is installed but its engine is not; install polars>=1.0
```

Skipped, not failed: a contributor with only pandas installed should be able to
run the whole suite. Put the marker on a module or a class and it covers every
test under it. A misspelled keyword is a usage error rather than a quiet
no-op, because the failure mode it otherwise has is the bad one - the test
stops being gated and starts erroring on the machine that lacks the engine.

## Which engine runs

In order, first one wins:

1. the `engine=` argument at the call site
2. the pipeline's own `@rowless.pipeline(engine=...)`
3. the test's `@pytest.mark.rowless(engine=...)`
4. `--rowless-engine`
5. a single engine in `engines` in `rowless.toml`, or a single installed one

Two of those look out of order and are not. The pipeline's declaration beats
the test's marker because it is a fact about the code under test rather than
about the test. And a marker beats the command-line flag because the flag is
how a suite is pointed at an engine it names nowhere, while a marker is a
statement about one test, whose body is written against that engine's API.

Where nothing names an engine, `rowless_check` lets the runtime infer it from
the pipeline's annotations - the same inference `rowless check` makes.
`rowless_frame` has no pipeline to read, so it takes the only installed engine
or asks:

```
more than one engine is installed (duckdb, pandas, polars), so rowless will not
choose for you; name one with @pytest.mark.rowless(engine=...), the engine=
argument, or --rowless-engine
```

## Coverage, and the gate

Passes alone overstate a contract suite: forty green tests over a schema
rowless typed half of is a different state of the world from forty over one it
typed all of. So the summary prints both.

```
---------------------------------- rowless -----------------------------------
rowless: 23 contract test(s), 211 column(s), 96% typed, 184ms in dry runs
```

```bash
pytest --rowless-min-coverage=0.9
```

Below the floor, the run fails even though every assertion passed - which is
the point: `exact=False` on a column rowless typed is a true statement about
that column and says nothing about the other one.

```
FAILED RWL204 typed coverage 50% is below the floor of 90%, over 1 rowless contract test(s)
```

The floor defaults to `min_coverage` in `rowless.toml`, so CI and the editor
and this plugin ratchet together. A suite that ran no rowless tests is never
failed by a floor: nothing was asserted, so there is nothing to score.

## Options

```
--rowless-engine NAME           adapter for tests that do not name one
--rowless-min-coverage RATIO    fail the run below this typed coverage (0 to 1)
```

## What it reads from `rowless.toml`

The `[sources]` rewrites, `[engine.<name>]` options, `min_coverage`, and
`cache_dir` for the schema cache. Config is discovered at the pytest rootdir
and read when a fixture first asks for it, not at start-up: this package is
installed in an environment rather than chosen by a repository, and a broken
`[tool.rowless]` table in some unrelated project must not fail a suite that
never mentioned rowless. A broken one in *your* project errors the tests that
needed it, naming the file and the key.

## Speed

Dry runs here go through `rowless.runtime.dryrun.dry_run_callable`, in the
pytest process, not through the sandboxed worker subprocess the CLI uses. The
code is already imported - the test imported it - and an interpreter start per
assertion would cost more than the dry run does, which would spend the one
property this plugin is sold on. The sandbox exists to protect a user from code
rowless *discovered* in a repository; a test suite is code they ran on purpose.

Source schemas are cached on disk by core's `SourceCache`, shared with the CLI,
so a suite of forty contract tests reads a Parquet footer or a dbt manifest
once, and reads nothing at all on the second run.

## What it does not do

* **find your pipelines.** This plugin asserts on the ones a test names;
  `rowless check` is what walks a repository and reports every one of them.
  A contract test is the pipeline you care about, not the coverage sweep.
* **sandbox the dry run**, as above.
* **check a SQL model.** A dbt model's target is a node id rather than an
  importable function, and the runtime has no path yet for a pipeline that is a
  compiled query - the same gap `rowless-dbt` and `rowless-lsp` document.
* **aggregate across `pytest-xdist` workers.** The tally and the gate are
  per-process, like `pytest-cov` without its own aggregation step. Anything
  stricter than a smoke floor belongs in `rowless check`.
* **diff two revisions.** That is `rowless diff`, and it needs two trees.
* **apply the code-level policy.** `error_codes`, `ignore_codes` and
  `fail_on_unverified` decide which findings gate a `rowless check` run; a
  contract test asserts one schema and fails on the three things above. So
  turning `fail_on_unverified` off for CI does not quietly disarm a suite of
  contract tests - only `allow_unverified=True`, written in the test, does.

## Versions

The floor is `pytest>=8.0`, and unlike the engine adapters' floors it is not an
API boundary: everything this plugin uses exists in 7.0, where `pytest.Stash`
landed. It is a supported-line choice rather than a hard requirement. CI runs
the suite against `pytest==8.0.0` as well as the current release, because a
floor nobody tests is a guess.

## Status

Implemented. There is no adapter conformance suite to pass here - this package
is a front end rather than an engine adapter - so what stands in for it is the
pytester suite: every rule above is pinned by a real pytest run, asserting on
the outcome, the message and the exit code, against a fake adapter and no
engine at all.

## Owner

Kauvin Lucas ([@kauvinlucas](https://github.com/kauvinlucas)).

## Development

```bash
uv sync --package rowless-pytest
pytest packages/rowless-pytest
```

No engine is installed in that environment, which is what keeps the suite
testing the plugin: the dry runs go against a fake adapter in
`tests/rowless_fake_pytest.py`, and `tests/plugin_fixtures.py` hides the real
engine entry points so a contributor's workspace full of adapters answers the
same way the CI cell does.

`tests/test_plugin.py` drives pytest inside pytest with the `pytester` fixture,
because what matters about an assertion is not that it raises - it is that a
test fails, with a message somebody can act on, and that the run's exit code
says so. `tests/test_session.py` pins the pieces those runs are made of, which
are ordinary functions and much cheaper to test directly.
