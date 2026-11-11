# rowless-pyspark

PySpark adapter for rowless: the analyzed plan, via Spark Connect.

## The trick

`df.schema` triggers Catalyst analysis without launching a job, so the schema
is the one Spark itself would produce, implicit casts included. Spark Connect
keeps it fast: the startup cost of a classic local session alone would break
the linter-speed promise.

## Status

Implemented. The conformance suite in `rowless.testing.conformance` passes, and
`rowless_pyspark.cases` carries 24 operation cases with the per-engine
expectations attached - the catalogue of where a pipeline's output schema
changes when it moves between Spark and the other engines.

What it does **not** do: trace the columns of a SQL string (Spark reports no
column lineage for a query, so `run_sql` resolves the schema and reports the
columns untraced - route a dbt model through `rowless-sql` or `rowless-ibis`
for that), and probe a UDF. Spark cannot infer a UDF return type at all, so
there is no probe to offer; see below.

## Configuration

```toml
# rowless.toml
[engine.pyspark]
remote = "sc://spark.internal:15002"   # Spark Connect: the mode to prefer
# mode = "local"                       # or start a JVM here instead
# master = "local[1]"                  # local mode only
# validate_each_call = true            # analyze after every builder call
```

`SPARK_REMOTE` is honoured too, and `ROWLESS_SPARK_REMOTE` takes precedence
over it so a repository can point its *checks* at a scratch cluster without
moving where its pipelines run. An explicit `mode = "local"` beats both: an
endpoint exported by somebody's shell profile is not consent to send a dry run
to it.

With no endpoint and no usable JVM, the adapter reports itself inactive and
`rowless doctor` says which of the two to fix. That is deliberate - pyspark
installs happily with nothing to talk to, and an adapter that claimed to work
in that state would turn every Spark pipeline in a repository into a
traceback.

## The two modes

| | Connect | Local |
| --- | --- | --- |
| client | pure Python | a JVM child process |
| first analysis | one round trip | tens of seconds of startup |
| sandbox needs | the configured host | loopback, and a subprocess |
| Java | none here | 17+ for Spark 4, 8+ for Spark 3.5 |

Either way the session is built **once per process** and reused by every dry
run in a check, because starting one is the expensive part. The session is also
built before the pipeline module is imported, which means a repository whose
module does `spark = SparkSession.builder.getOrCreate()` at import time gets
*this* session rather than starting a second one. Real pipelines run
unmodified.

## The part worth reading before you point this at a cluster

Every other adapter in rowless runs an engine inside the dry-run process. This
one talks to a cluster that probably also serves production, over a socket the
sandbox was told to allow - so the audit hook cannot tell `spark.sql("INSERT
OVERWRITE TABLE orders ...")` from the analysis round trips it is deliberately
permitting. `rowless_pyspark.interception` is what can:

* **a statement that is not a query is not sent.** `spark.sql` is passed
  through for `SELECT`, `WITH`, `VALUES`, `SHOW`, `DESCRIBE` and `EXPLAIN`.
  Anything else - `INSERT`, `CREATE`, `DROP`, `MERGE`, `UPDATE`, `DELETE`,
  `TRUNCATE`, `GRANT` - is recorded as an output of the pipeline, and the empty
  frame Spark returns for a command is returned in its place.
* **every writer is a recorder.** `DataFrameWriter.save`, `saveAsTable`,
  `insertInto`, the V2 writer's `append` and `overwrite`,
  `DataStreamWriter.start` and `Catalog.createTable` write nothing and report
  where they would have written.
* **the egress is exactly one host.** The adapter tells the sandbox the
  endpoint it was configured with, and nothing else in the process can open a
  socket. Local mode asks for loopback and for permission to start a JVM, and
  says so.

The conformance suite asserts the first one against a table that exists nowhere
(`statement_that_is_not_a_query_is_not_sent`): if the guard ever stops working,
Spark fails to analyze the INSERT instead of quietly returning no columns.

## What it does to PySpark

Wrappers, installed once per process and inert outside a dry run:

* **row dependence.** `collect`, `count`, `first`, `head`, `take`, `isEmpty`,
  `toPandas`, `toLocalIterator`, `foreach` and `show` answer from zero rows, so
  each one downgrades the result to unverified and names the line.
* **analysis at the line that broke it.** Under Connect a plan is not analyzed
  until something asks, so `df.select("nope")` would otherwise fail at the end
  of the dry run and point the finding at the wrong line. Each builder analyzes
  its result - one round trip, no job - which is what makes the finding land on
  the user's own call. `validate_each_call = false` turns it off where the round
  trips are not worth it, at the cost of findings that point at the end of the
  run and of lineage, which needs the resolved names.
* **column lineage**, recorded per builder call from the expressions the call
  was given and the resolved output names. `select` and `agg` produce one
  column per expression in order, so the attribution is positional and exact; a
  `*` that expanded inside Spark leaves the extra columns untraced rather than
  guessed at.
* **UDFs.** `udf(f)` with no `returnType` is not inference - Spark types the
  column `StringType` by default - so the run is degraded and the function is
  named. Passing `returnType=` makes it go away, and is worth having anyway.

## Where Spark disagrees with the other engines

Each line is a place where moving a pipeline between engines changes its output
schema. `rowless_pyspark.cases` asserts every one of them.

| Operation | Spark | Elsewhere |
| --- | --- | --- |
| `groupBy().agg(sum("amount"))` | `sum(amount)`, `bigint` | DuckDB widens to 128-bit; Polars keeps the name `amount`; ibis calls it `Sum(amount)` |
| `sum` of `decimal(18,2)` | `decimal(28,2)` | Polars keeps `decimal(18,2)` |
| `amount / 2` | `double` | same in Polars, DuckDB, ibis |
| join on a condition | keeps both keys, renames nothing | Polars coalesces and suffixes `_right`; pandas `_x`/`_y`; DuckDB keeps both |
| join on a shared name | coalesced, and the key comes **first** | Polars and DuckDB keep its position; ibis does not coalesce an outer join |
| `union` | **positional** | pandas `concat` and Polars `concat` align on names |
| `size(tags)` | `int32`, named `size(tags)` | Polars: `uint32`, original name |
| two output columns of one name | allowed | Polars refuses; rowless reports it either way |
| nullability | tracked and propagated | only DuckDB and ibis model it at all |
| `TIMESTAMP` | an instant in the *session* timezone | Polars and DuckDB carry the zone on the dtype |
| UDF with no return type | silently `string` | Polars calls the function to find out |
| `collect()` | costs confidence | the Polars adapter exempts it, because a Polars pipeline ends in one |

The `union` row is the expensive one. It produces no error: two frames whose
columns are in different orders are unioned into the left-hand names with the
right-hand values, and if the dtypes happen to agree nothing anywhere
complains. `unionByName` is the one that behaves like the other engines.

## Dtypes it refuses rather than approximating

* **unsigned integers.** Spark has none. A signed type of the same width would
  lose the top half of the range.
* **millisecond and nanosecond timestamps.** Spark timestamps are
  microseconds; either lowering would change the resolution silently.
* **a named timezone that is not the session's.** `TIMESTAMP` carries no zone
  of its own, so a column declared `datetime[us, Europe/London]` against a UTC
  session would move every value by an hour. The adapter pins the sessions it
  creates to UTC, and reads the zone back from a session it adopted instead of
  assuming one.
* **`enum{...}`.** No closed variant set exists in Spark, and a plain string
  loses exactly the thing the enum declares.
* **`time[...]` on PySpark older than 4.1**, which has no `TIME` type at all.
  The absence is environment-dependent, the way the pandas adapter's depends on
  pyarrow.

In the other direction, `YearMonthIntervalType` and `CalendarIntervalType`
become `UNKNOWN` with the reason attached - a month is not a fixed number of
microseconds - and so does `VariantType`.

## Versions

Floor `pyspark>=3.5`, and a tested floor: CI runs the suite at 3.5 as well as
at the current release. The differences the adapter handles rather than
pretends away:

* on 3.5 the Connect `DataFrame` is a separate class, and from Spark 4 it
  subclasses the classic one. Both are patched, and only where each spells the
  method itself, so an inherited method is never wrapped twice;
* `getCondition()` on an exception was `getErrorClass()` before Spark 4, and
  the error translation reads whichever exists;
* a classic session analyzes **eagerly**, so `df.select("nope")` raises inside
  the builder call; under Connect the same plan resolves and fails when the
  schema is read. Both are translated, so the finding is the same either way;
* `TIME` arrived in 4.1 and is declared absent before that;
* Spark 4 defaults to ANSI mode, which changes what *fails* rather than what
  resolves. The conformance cases are written to be the same answer under both.

One version note is a real constraint rather than a difference: **PySpark
3.5's Connect client cannot be imported on Python 3.12 or newer**, because it
reaches pandas through `distutils`. This package requires Python 3.12, so on
the floor release Connect mode is unavailable and local mode is the only one -
and that is reported as a `rowless doctor` line naming the import failure,
rather than as a crash when the session is built. Connect mode on a modern
Python wants `pyspark>=4`.

## Owner

Kauvin Lucas ([@kauvinlucas](https://github.com/kauvinlucas)).

## Development

```bash
uv sync --package rowless-pyspark
pytest packages/rowless-pyspark
```

Half the suite needs no Spark at all: the dtype table, the error translation,
the lineage rules, the statement guard and the session selection are pure
Python and run anywhere. The conformance suite needs a Spark, and skips with
the reason when there is neither an endpoint nor a usable JVM - so
`JAVA_HOME=/path/to/jdk17 pytest packages/rowless-pyspark` is the local way to
run all of it.
