# Roadmap

From the proposal, with the work broken out per package. Dates are relative to
project start, not calendar dates.

## Weeks 1-4: spike

Prove the core trick, end to end, on one engine.

- [x] `rowless.ir` - dtypes, schemas, confidence
- [x] `rowless.runtime` - dry run, worker, sandbox
- [x] `rowless-polars` - the reference adapter
- [x] `rowless.sources.parquet`
- [x] `rowless check` with terminal output
- [ ] a demo on one real open-source pipeline

Everything in core is done and tested with no engine installed: the dry-run
runtime drives an adapter through the contract in
`rowless.testing.conformance`, and the core test suite exercises it against a
small in-process adapter. The Polars adapter now passes that same contract,
which makes `examples/polars_demo` a real end-to-end check rather than a
report about a missing engine; `tests/test_examples.py` asserts the planted
finding, its line and the absence of any other. The remaining work in this
phase is the demo on somebody else repository.

Exit criterion: rowless finds a real broken column reference in somebody else
repository, not in an example we wrote.

## Months 2-3: v0.1 on PyPI

Breadth across engines, plus the lowest-friction adoption path.

- [x] `rowless-pandas` - including the dtype drift fixup table
- [x] `rowless-duckdb` - and with it the Python-to-SQL seam
- [x] `rowless-sql` - credential-free SQL via sqlglot
- [ ] `rowless-pytest` - contract tests in an existing suite
- [x] `rowless.discovery` - plain functions and Dagster
- [x] adapter conformance suite complete enough to onboard an owner

`rowless-pandas` is implemented, which is the adapter the proposal expected to
be hardest and the one most repositories need. It needed no change in core,
and the two things worth recording about it are both surprises.

The first is that the drift is smaller than the proposal assumed, and the table
is more useful for it. On pandas 2.2 and 3.x, projections, `merge`,
`groupby().agg("sum")`, `concat`, arithmetic and the `.str` and `.dt`
accessors all compute the same dtypes from zero rows as from data. What drifts
is a short list with one dominant cause: an operation that takes a function,
which pandas has no row to call. So the correction is the UDF inference that
already existed in `rowless.udf.infer`, cast onto the empty result and recorded
as a note - and the adapter's own suite runs every correction twice, once on a
rowless frame and once on three real rows, so an entry that stops matching
pandas fails rather than quietly reporting the wrong dtype. Four entries cannot
be corrected at all, and they are the interesting half: `GroupBy.apply`,
anything that builds column names out of values (`pivot`, `unstack`,
`get_dummies`), `where` without an `other` - where the populated dtype depends
on whether anything was actually masked, so there are two possible answers and
no way to choose - and `idxmax`, which raises rather than returning something
useless. Those degrade the run instead, which is the ADR 0003 position applied
to a dtype rather than to a schema.

The second is the rule that makes patching an eager engine viable. pandas calls
its own public API constantly - `merge` reads `len(self)`, nearly everything
touches `empty` - and `rowless.runtime.rowdep.user_frame` cannot sort that out,
because it only walks out of *rowless* frames and a pandas-internal call has a
pandas frame directly above the wrapper. So every wrapper in the pandas adapter
is gated on the nearest non-rowless frame not being inside pandas, numpy or
pyarrow. Without it, every pandas pipeline would be reported unverified before
it got past its first operation; with it, a realistic merge-assign-groupby
pipeline comes back `EXACT`. The cost is that a value read pandas performs on
the pipeline's behalf is invisible to the dynamic layer, which the static pass
in `rowless.analysis` is there to cover.

What it does not do is column lineage, and the reason is worth stating because
it is not laziness. pandas has no expression objects: `df["a"] + df["b"]`
produces a Series that remembers nothing about where it came from, and
`assign(total=lambda d: ...)` hides its inputs in a closure. Lineage for the
name-preserving operations alone would report computed columns as untraced
while implying the rest was complete, so the capability is declared off and
pandas columns get no blast radius in `rowless diff`. Following a Series
through an arithmetic expression is the next thing worth doing in this package.

`rowless-sql` is implemented, which completes the engine adapters and is the
only one with no engine behind it: `sqlglot.optimizer.qualify` resolves a query's
identifiers against the registered input schemas and `annotate_types` types the
projection, so a model resolves with a parser and nothing else. That is what
makes it the adapter for a CI job with no warehouse access, and `rowless-dbt`
prefers it over DuckDB for exactly that reason.

Three things about it are worth recording.

The first is that the honest version of this adapter has to *refuse* more than
it reports. sqlglot answers questions it has no evidence for: it names an
unaliased projection `_col_0`, where DuckDB would call that column
`sum(amount)`, Spark `_col0` and Postgres `sum`; it resolves a qualified
column against a table nobody registered a schema for; it reports a union's
types as the left-hand side's, so a `VARCHAR` unioned with a `BIGINT` reads as
a string; and on the oldest supported release it reports an ambiguous column
with the same message it uses for a column that exists nowhere. Each of those
is a finding here instead - unnamed, unregistered, unresolved, ambiguous - and
the set operation is combined through the IR's own dtype lattice rather than
taken from one side. An answer no warehouse would agree with is worse than no
answer, which is ADR 0003 applied to a resolver rather than to a dry run.

The second is that it wins the one thing the DuckDB adapter gives up. DuckDB
refuses to attribute a projection that sits behind a CTE or a subquery, because
matching names across a scope it cannot see through is how a tool invents a
rename that never happened. sqlglot's scopes *are* the data structure, so
lineage resolves through both, and a dbt model built out of four stacked CTEs
gets a real blast radius. One qualification per pipeline pays for it: the
resolved expression is handed to `sqlglot.lineage` as a prebuilt scope rather
than letting it re-qualify the query once per column.

The third is a deduplication. The sqlglot-to-IR type mapping was written in
`rowless-dbt` first, to read the warehouse type strings a manifest carries,
with a note saying it should move the day this adapter landed. It has:
`rowless_sql.dtypes` owns both directions now and `rowless_dbt.dtypes` parses
and delegates, with a test asserting the two agree so the copy cannot come
back.

## Months 4-6: CI

Breaking-change detection, which is where the tool earns a place in a pipeline.

- [x] `rowless.lineage` - the cross-language column graph
- [x] `rowless diff` - base graph in a git worktree, cached
- [x] `rowless-dbt` - manifest schemas and model discovery
- [ ] the GitHub Action - comment, annotations, gate
- [x] `rowless-openlineage`
- [x] `rowless-ibis`

`rowless-dbt` reads v11 and v12 manifests, resolves `dbt://` URIs from
contracts, the catalog or compiled SQL, and discovers models as pipelines, so a
rename in a dbt contract already breaks the Python pipeline that reads it. What
a discovered dbt model still cannot do is run: resolving one needs an engine
implementing `SupportsSql` - `rowless-sql`, `rowless-duckdb` and `rowless-ibis`
all do now - and a runtime path for a target that is not an importable
function. That path is the one remaining gap in the plan, and it is in core
rather than in a package: until it lands, models are reported as unchecked
rather than passed, and the finding now says rowless could not reach the
pipeline instead of blaming it for raising.

`rowless-openlineage` is implemented, which is the answer to "where does the
graph go". Nowhere new: column-level lineage is already an OpenLineage facet
that Marquez, DataHub, Atlan and OpenMetadata read, so the emitter publishes
the cross-language graph into the catalog a team already has rather than into a
UI this project would then have to maintain.

Two decisions in it are worth recording, because both cost something and both
are the reason this is safe to point at a shared catalog.

The first is that nothing emitted is a run event. OpenLineage's usual shape is
a `RunEvent` with a START, a COMPLETE and a run id; rowless sends static
lineage - `DatasetEvent` and `JobEvent`, no run - because nothing executed, and
a catalog that recorded a dry run as a production job would show a successful
run at a time when all that happened was that somebody saved a file. The
coverage score rides on every event in a custom facet carrying
`executed: false`, so the claim is in the payload rather than in a README, and
a catalog can tell lineage rowless verified from lineage it partly inferred
instead of drawing both with the same confident arrow.

The second is that name-matched edges are withheld. rowless labels every edge
with its evidence, and `NAME_MATCH` means two columns share a name and nothing
saw one feed the other - evidence core already refuses to gate a CI run on. A
catalog arrow has no "probably" in it, so publishing one would turn a guess
into verified lineage in the hands of somebody who was not there. The withheld
count travels in the node facet, so the graph looks incomplete where it is
incomplete, which is the trade this project keeps making on purpose.

What it does not solve is dataset identity for a warehouse table. The naming
conventions want the datasource in the namespace - `postgres://host:5432` - and
a rowless node id does not carry the host, because the DSN is a credential kept
in an environment variable. The same applies to a bare filesystem path, where
the convention wants an absolute path and rowless has the repo-relative URI the
pipeline was written with. Both are configuration
(`ROWLESS_OPENLINEAGE_NAMESPACES`) rather than inference, and that is the
honest shape of it: a namespace nobody could derive is a namespace a team has
to state.

Two small core changes came with it, both generic rather than
OpenLineage-shaped: `rowless emit --dry-run`, which prints the payloads and
transports nothing so a first emission can be reviewed before it reaches a
shared catalog, and an optional `unavailable_reason()` hook on a plugin.
`requires` can only name a package, and the usual reason an emitter cannot run
is that its destination is not configured, which is not something a PEP 508
requirement can describe.

`rowless-ibis` is implemented, which brings the credential-free SQL path
forward from the phase above by a different route than `rowless-sql` was going
to take: `ibis.parse_sql` resolves a compiled model against the input schemas
with sqlglot parsing and ibis typing, so no warehouse and no DuckDB is needed
to resolve one. It also adds the portability finding nothing else in the
project can produce - the same expression lowered through each backend's own
ibis type mapper, reporting where DuckDB, BigQuery, Snowflake and SQLite
disagree about the output schema.

## After that

- [x] `rowless-pyspark` - via Spark Connect
- [x] `rowless-iceberg` - exact renames from field IDs
- [x] `rowless-lsp` - diagnostics, completion, inline schema hints
- [ ] Airflow and dlt discovery
- [ ] a talk on the bugs it caught

`rowless-pyspark` is implemented, which closes the last engine the proposal
promised and is the first adapter whose engine is not in the process. `df.schema`
runs Catalyst analysis and launches no job, so the schema is Spark's own answer;
Spark Connect makes that affordable, because analysis happens on a server that
is already running and a dry run costs a round trip rather than a JVM startup.
Local mode stays available for people without a cluster, with the session built
once per process and reused.

What it changed in the shape of the project is the sandbox. Every other adapter
runs its engine here, so denying egress outright was free; this one needs a
socket, which is what `Capabilities.remote` has always promised and nothing
wired. An adapter that declares it is now asked - through an optional
`egress()` hook, before the sandbox is installed - exactly which hosts to open,
and whether reaching the engine means starting a process, which a local JVM
does. That is the whole core change: one dataclass, one hook, one policy
helper, and nothing that is specific to Spark.

The consequence is the part worth stating plainly. Once the endpoint is
allowlisted, the audit hook cannot tell a pipeline's
`spark.sql("INSERT OVERWRITE TABLE orders ...")` from the adapter's own
analysis traffic, so the adapter's interception stops being about better
findings and becomes the thing standing between a dry run and a production
table: a statement that is not a query is recorded and not sent, and every
writer entry point is a recorder. A conformance case aims that guard at a table
that exists nowhere, so a guard that stopped working fails loudly instead of
quietly passing.

The disagreement catalogue it adds is the largest of any adapter so far, and
one entry is worse than the rest: `union` is positional in Spark where pandas
and Polars align on names, so a ported pipeline can union two frames into the
right column names with the wrong values and raise nothing at all. That one is
in `rowless_pyspark.cases` with the others - a decimal sum that gains ten
digits of precision, a `TIMESTAMP` whose zone is the session's rather than the
column's, a join that keeps both keys, and a UDF whose return type Spark
defaults to string rather than inferring.

`rowless-iceberg` is implemented, pulled forward from this phase because it
needed nothing that was not already there. Iceberg keeps a column's field ID
across a rename, so a resolved schema can fill in the `derived_from` slot that
`rowless.ir.diff` already trusts above its own heuristics - which makes this the
first source whose renames are recorded facts rather than pairings, and it cost
no change in core to get there. The two things it adds beyond that: a schema
resolves at a snapshot, branch or tag, so the base side of a diff needs no git
checkout, and `renames_since` answers the question exactly, with a dropped
column unable to appear as a rename because it shares no field ID with
anything.

What it does not do is veto. `derived_from` is positive evidence, so core's
position-and-dtype rule still runs underneath it and can still mis-pair a drop
with an unrelated addition; teaching the differ to prefer an authoritative
answer over its own guesses is a core change, and it is the next thing worth
doing here.

`rowless-lsp` is implemented, which closes the proposal's first clause: the
findings arrive in the editor and not only in CI. It is a front end rather than
an adapter - every finding comes out of `rowless.check.run`, so the editor
cannot disagree with the pipeline about what is broken - and what it adds is
the state that makes the same answer usable interactively: one config per
workspace, one index per buffer, debounced re-checks with superseded work
cancelled, and the last check's schemas kept so completion, hover and inlay
hints answer without running anything.

The constraint it is built around is that a dry run imports the module, so it
can only describe the file on disk. A full check therefore happens on save,
and what runs between saves executes nothing: column names read out of the
buffer, compared against the schema the last check resolved, plus the static
pass from `rowless.analysis` - which gained an `analyse_source` entry point so
it can run on a buffer that was never written to disk, the one core change
this needed. That live check is a lookup rather than inference, and it is
deliberately quiet wherever it could be wrong: a function with a computed
column name says nothing at all, and a buffer that does not parse reports
nothing, because a finding placed by stale line numbers accuses code that is
fine.

What is still missing is the warm worker pool, and the reason is worth
recording: a reused worker has the user's module already imported, so a
re-check needs a reload, and reloading a module whose dependencies also changed
is a correctness problem rather than a performance one. Each dry run is one
subprocess until that question has an answer. The other gap is shared with
`rowless-dbt`: a SQL model still cannot be resolved in the editor, because
nothing yet runs a target that is not an importable function.

## Success metric

Run rowless on 3 to 5 public pipeline repositories and count the real errors it
finds that would otherwise have failed at runtime. Recorded in `benchmarks/`,
with false positives counted just as prominently: a finding count with no
precision figure next to it is marketing, not a measurement.

## Not scheduled

Named so nobody has to ask twice: static schema inference from Python source, a
web UI, a schema registry, and runtime enforcement. See the end of
`architecture.md` for why each is out.
