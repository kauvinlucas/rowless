# rowless-pandas

pandas adapter for rowless: empty typed frames, with the dtype drift fixed.

## The trick

Zero-row frames with dtypes declared up front, handed to real pandas code.
pandas has no lazy plan to resolve, so there is nothing else to do: the dtypes
it computes while the pipeline builds a frame *are* the answer, and `describe`
reads them back off `df.dtypes`.

That works better than the folklore suggests. On pandas 2.2 and 3.x,
projections, `merge`, `groupby().agg("sum")`, `concat`, arithmetic and the
`.str` and `.dt` accessors all compute the same dtypes from zero rows as they
do from data. What does not is a short, specific list, and the work of this
adapter is that list: `rowless_pandas.drift` holds every entry, with what
pandas does on zero rows, what it would have done with rows, and what the
adapter does about it.

## The drift table

Two kinds of entry, and the difference is the whole point.

**Corrected.** The populated answer is derivable, so the result is cast to it
and the correction is recorded as a note. Every one of these has the same
cause: the operation takes a function, and pandas has no row to call it with,
so the output keeps the *input* dtype.

| What | Zero rows | With rows | Correction |
| --- | --- | --- | --- |
| `Series.apply`, `.map`, `.transform` | the input dtype | the function's return dtype | infer the return dtype, cast |
| `DataFrame.apply`, `.map` | `float64` | the function's return dtype | the same |
| `GroupBy.agg`, `.transform` with a callable | the input dtype | the callable's return dtype | the same, per column |

The inference is `rowless.udf.infer`: an explicit dtype, then a return
annotation, then - only when `probe_udfs` is on - one synthetic call. A
`Series.map(dict)` needs none of it, since the values are right there and their
join is the answer.

**Degraded.** The empty answer is wrong and the right one is not derivable, so
the run is reported `Confidence.UNKNOWN` with the reason. A correction nobody
could derive would be a guess, and a guessed schema reported as a pass is the
one failure mode this project will not have.

| What | Why it cannot be corrected |
| --- | --- |
| `GroupBy.apply` | the output shape depends on how many groups there are, and zero rows form none |
| `pivot`, `pivot_table`, `unstack`, `crosstab`, `get_dummies`, `T` | the column *names* are data |
| `where`, `mask` with no `other` | a numpy column widens only if something is masked out, which is a property of the data: two possible dtypes, no way to choose |
| `idxmax`, `idxmin`, `argmax`, `argmin` | they raise on zero rows. Reported as row dependence, not as a broken pipeline |

`rowless explain` shows which entries fired on a run, and
`PandasAdapter.dtype_drift_fixups()` lists the table. Correcting silently was
never an option: a user who sees `float64` in a notebook and `int64` in a
finding has to be able to find out why in one command.

## Pick the dtypes that cannot drift

The adapter materializes pyarrow-backed dtypes when pyarrow is installed, and
pandas' masked extension dtypes (`Int64`, `boolean`, `string`) otherwise. Both
hold NA without changing dtype, which removes most of the drift before it
happens - a plain `int64` column becomes `float64` the first time a `where`
masks something out, and an `Int64` column does not.

Without pyarrow there is a real cost, and it is reported rather than hidden:
pandas has no decimal, date, time, binary or null dtype at all. All five would
be `object` columns, which carry no type, so `from_ir` refuses them and
`unsupported_dtypes()` declares them absent. Install pyarrow, or set
`arrow_dtypes = true`, and all five map exactly.

```toml
# rowless.toml
[engine.pandas]
arrow_dtypes = true   # default: use pyarrow when it is installed
probe_udfs = false    # default: never call a UDF with invented values
```

## Where pandas disagrees with the other engines

Each line is a place where moving a pipeline between engines changes its output
schema. The conformance cases in `rowless_pandas.cases` assert every one.

* **`groupby("status").agg(...)` puts the key in the index, not in a column.**
  The schema is `{amount}`, where Polars and DuckDB both produce
  `{status, amount}`. The biggest difference between pandas and everything else
  rowless talks to, and `as_index=False` is the fix. The adapter names the
  index levels in a note rather than inventing a column that is not there.
* **A `merge` suffixes collisions `_x` and `_y`.** Polars suffixes `_right`;
  DuckDB keeps both columns under the same name.
* **`merge(left_on=, right_on=)` keeps both key columns**, where Polars
  coalesces the right one away.
* **Duplicate column labels are legal in pandas.** They are not describable in
  a name-addressed schema, so `describe` reports a finding.
* **A `MultiIndex` column axis has no flat names.** `agg(["sum", "mean"])`
  produces one, and the adapter reports the schema as unresolved rather than
  inventing `amount_sum`. A named aggregation avoids it.
* **`.dt.date` is an `object` column** without pyarrow, so it has no dtype to
  report at all. It is `date32` with pyarrow, and `date` in Polars either way.
* **Nullability is a property of the dtype here**, so `describe` reports it: a
  numpy `int64` column cannot hold NA, a masked one can. Polars does not model
  nullability at all.
* **A named timezone survives.** `datetime64[ns, Europe/London]` round-trips,
  where DuckDB has to refuse it.
* **Second resolution exists.** `datetime64[s]` is a real pandas dtype, where
  Polars refuses it.

## Row dependence costs pandas more than it costs Polars

Everything that reads a value reads it from zero rows, so `df.empty`,
`len(df)`, `s.max()` and `iterrows()` all answer from nothing and the result is
reported unverified. That is harsher here than on a lazy engine, and the
asymmetry is real rather than a limitation of the adapter:
`df["amount"].max()` reads data, while the Polars spelling of the same
aggregate, `pl.col("amount").max()`, is an expression that resolves to a dtype
without touching any. An engine whose natural spelling of an aggregate reads
values is an engine where more pipelines come out unverified.

What keeps that from swallowing every pipeline is the rule in
`rowless_pandas.interception`: pandas calls its own public API constantly -
`merge` reads `len(self)`, nearly everything touches `empty` - and only calls
made from *outside* pandas, numpy and pyarrow are reported. So a value read
pandas performs on the pipeline's behalf is invisible here, as is one inside a
callback pandas invokes: `agg(lambda s: s.max())` reads values by definition,
and reporting it would mark every aggregation unverified. The drift table
covers those by correcting or degrading the dtype the callback produced, and
the static pass in `rowless.analysis` flags the row-dependent names in the
source whether or not the dry run reached them.

## What the empty frame hides

The honest other half. `astype` coercions that would fail on data succeed on
zero rows: `df["status"].astype("int64")` raises in production and returns an
empty `int64` column here. The same goes for an aggregation that would
overflow, and for a `merge` whose keys would duplicate rows. rowless checks
schemas, not values.

## Not implemented

**Column lineage.** pandas has no expression objects:
`df.assign(total=lambda d: d.a + d.b)` hides its inputs in a closure, and
`df["a"] + df["b"]` produces a Series that remembers nothing. Recording
lineage for the name-preserving operations alone would report computed columns
as untraced while implying the rest was complete, so pandas columns are
reported untraced and `rowless diff` has no blast radius for them. Attributing
a computed column needs a way to follow a Series through an arithmetic
expression, which is the next thing worth doing here.

**Nested types.** `rowless_pandas.dtypes` maps lists, structs and maps exactly
in both directions, but only through `ArrowDtype`. pyarrow is not a dependency
of this distribution, so the capability flag stays off: one that half the
installs cannot honour is worse than one that is missing, because coverage
scores are computed from those flags.

## Status

Implemented. The conformance suite in `rowless.testing.conformance` passes,
twice - once per dtype table - and every correction in the drift table is
asserted against what pandas actually does on three rows, so an entry that
stops being true fails the suite rather than quietly reporting the wrong dtype.

## Owner

Kauvin Lucas ([@kauvinlucas](https://github.com/kauvinlucas)).

## Development

```bash
uv sync --package rowless-pandas
pytest packages/rowless-pandas

# the arrow dtype table as well as the masked one
uv run --package rowless-pandas --with pyarrow pytest packages/rowless-pandas
```
