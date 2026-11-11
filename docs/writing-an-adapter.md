# Writing an adapter

The self-contained way to contribute. Pick the engine you use daily: you will
know its type quirks better than the core team does.

## Setup

```bash
uv sync --package rowless-sql
uv run pytest packages/rowless-sql    # all failing, by design
```

The conformance suite is the specification. Work through its failures.

`rowless-polars` is the worked example: it passes the suite, and the four
required methods, the dtype table, the engine-quirk catalogue and the
interception it needs are each in a module of their own with a docstring saying
why. Read it before starting, and copy its shape rather than its contents -
your engine disagrees with it in ways that are the whole point of your adapter.

`rowless-duckdb` is the second one, and worth reading if your engine is a
database rather than a dataframe library: it shows the SQL seam
(`SupportsSql`), what to do when the engine writes through its own filesystem
instead of Python's, and how to declare a dtype the engine does not have.

`rowless-pandas` is the one to read if your engine is **eager**, because it is
the only one where the rowless frame can answer a question differently than a
populated frame would. Three things in it generalise:

* the drift table in `rowless_pandas.drift`, which separates a dtype the
  adapter can *derive* (so it corrects it, and says so in a note) from one it
  cannot (so it degrades the run), rather than having either buried in a
  wrapper;
* the interception rule, which only reports a row-dependent call when the
  nearest non-rowless stack frame is outside the engine. An eager engine calls
  its own public API constantly, and an adapter that reported those calls would
  mark every pipeline unverified before it got past its first operation;
* an `unsupported_dtypes()` that depends on the environment. pandas has no
  decimal, date, time, binary or null dtype without pyarrow, and declaring them
  absent on an install where they work would be as dishonest as approximating
  them on one where they do not.

`rowless-ibis` is the shortest. Read it if your engine already
separates building a query from running it, because it shows how little an
adapter has to do when it does: lineage is a walk of the expression tree rather
than anything recorded, the SQL seam is the engine's own parser, and the
interception exists only to produce better findings - there is no side effect
to stop, because an unbound expression cannot execute. It is also where to look
for a capability that is not in the contract at all: `backend_disagreements`
reports where two backends would type the same expression differently, which is
a finding only that engine can produce. An adapter is free to add methods of
its own; what the capability flags gate is what *core* will call.

`rowless-sql` is the one to read if your engine is a **resolver rather than a
runtime**: there is nothing to execute, nothing to intercept and no side effect
to stop, so the whole adapter is the four methods plus a catalogue of input
schemas. Two things in it generalise:

* **an answer the engine will give you is not always an answer worth
  reporting.** sqlglot names an unaliased projection `_col_0`, resolves a
  qualified column against a table nobody registered, and types a union as its
  left-hand side. Each of those is refused here and reported as a finding,
  because a name or a dtype no engine downstream would agree with is worse than
  an honest "rowless does not know". If your engine answers questions it has no
  evidence for, the adapter is where that stops;
* **a capability the others cannot offer is worth saying out loud.** Lineage
  through a CTE or a subquery resolves here and is deliberately refused by the
  DuckDB adapter, so the README and the module docstring say which one to use
  when lineage through a stack of CTEs matters.

`rowless-pyspark` is the one to read if your engine is **somewhere else**: a
cluster, a warehouse, anything reached over a socket. Four things in it
generalise:

* `remote=True` plus an `egress()` hook is how an adapter gets a network at
  all. The sandbox denies egress outright, and what it opens is exactly the
  hosts that hook names - see below;
* **interception stops mattering only for findings and starts mattering for
  safety.** Once the endpoint is allowlisted, the audit hook cannot tell a
  pipeline's `INSERT OVERWRITE` from the adapter's own analysis round trips.
  The statement guard in `rowless_pyspark.interception` is the layer that can,
  and a conformance case points it at a table that exists nowhere so that a
  guard which stopped working fails loudly;
* `available()` is about more than an import. pyspark installs happily with no
  cluster configured and no Java new enough to start one, so the adapter
  answers for whether it could actually resolve a schema and explains the
  blocker through `unavailable_reason()`;
* a dtype mapping may depend on **session state** rather than only on the
  engine. A Spark `TIMESTAMP` is an instant rendered in the session's
  timezone, so the IR dtype it maps to is a function of the session - which is
  why the adapter pins the sessions it creates and reads the zone back from one
  it adopted.

## The four required methods

```python
class MyAdapter:
    name = "myengine"
    requires = "myengine>=1.0"
    capabilities = Capabilities(lazy=True, nested_types=True)

    def available(self) -> bool: ...  # engine importable, right version
    def handles(self, obj) -> bool: ...  # is this one of my frame types
    def materialize(self, schema, ctx): ...  # IR schema -> real empty frame
    def describe(self, obj) -> Relation: ...  # engine object -> IR schema
    def to_ir(self, native) -> DType: ...  # engine dtype -> IR
    def from_ir(self, dtype): ...  # IR -> engine dtype
```

Rules that are not negotiable:

**Never import the engine at module scope.** `available()` must work for a user
who has none of the engines your adapter targets. Import inside the methods.

**`materialize` returns a real engine object.** Not a proxy, not a mock. User
code will call engine methods on it, and the premise of the project is that real
code works unmodified.

**`describe` must not execute.** The conformance suite asserts this against the
sandbox side-effect log, not by inspection.

**Translate engine exceptions.** A missing column should surface as
`ColumnNotFound` with the column name and a near-miss suggestion, not as the
engine traceback.

**Never substitute a near-enough dtype.** `from_ir` raises `UnsupportedDType`;
`to_ir` returns `Kind.UNKNOWN`. A silently approximated dtype is the kind of bug
that destroys trust in the whole tool.

## Declare the dtypes your engine does not have

The conformance corpus asks every adapter to round-trip the same dtypes, and
some engines genuinely cannot. DuckDB has no nanosecond timestamp carrying a
named zone; lowering `datetime[ns, Europe/London]` to the nearest thing it does
have would shift every value by hours and change the resolution. So the adapter
says so:

```python
def unsupported_dtypes(self):
    """Corpus dtypes this engine has no equivalent for."""
    return (datetime(TimeUnit.NANOSECOND, "Europe/London"),)
```

Those dtypes drop out of the round trip, and the suite instead requires
`from_ir` to raise `UnsupportedDType` for each one. They drop out of the fixture
schemas too: a column whose dtype your engine does not have cannot survive
`materialize` and `describe` by definition, so the suite leaves it out of those
cases rather than asking you to approximate it. The rest of the fixture is still
materialized. The declaration is a claim the suite holds you to, not an
exemption - and if you find yourself wanting to
add an entry for a dtype your engine *does* have in a slightly different shape,
that is a mapping to write rather than an absence to declare.

The same rule applies in the other direction. Refuse a dtype whose round trip
would not return what it was given, even when the engine could store the value:
DuckDB `TIME` is microseconds, so `time[ms]` would come back as `time[us]` and
be reported downstream as a dtype change nobody made.

## Supply your operation cases

The universal half of the conformance suite is written once in core: dtype
round trips, `materialize` producing a real frame, `describe` round-tripping it
without executing, UNKNOWN being refused rather than guessed, and your
capability flags matching what you actually implement.

The operations battery cannot live in core. `df.select("a")` is not the same
call in Polars, pandas and DuckDB, and writing one abstract vocabulary for all
of them would be the proxy approach this project rejects. So an adapter
supplies its own cases:

```python
from rowless.ir.confidence import Confidence
from rowless.testing import fixtures
from rowless.testing.conformance import ConformanceCase


def conformance_cases(self):
    orders = {"orders": fixtures.ORDERS}
    return (
        ConformanceCase(
            name="group_by_sum",
            pipeline=lambda orders: orders.group_by("status").agg(pl.col("amount").sum()),
            inputs=orders,
            expect=Schema.from_mapping({"status": string(), "amount": int_(64)}),
            reason="Polars keeps the aggregated column's name; DuckDB renames it",
        ),
        ConformanceCase(
            name="missing_column",
            pipeline=lambda orders: orders.select("amount_cents"),
            inputs=orders,
            raises=ColumnNotFound,
        ),
        ConformanceCase(
            name="row_dependent",
            pipeline=lambda orders: orders if orders.is_empty() else orders,
            inputs=orders,
            expect_confidence=Confidence.UNKNOWN,
        ),
    )
```

Each case runs through the same runtime the CLI uses, so what is checked is the
schema a user would get. An adapter with no cases fails
`test_declares_conformance_cases` rather than passing quietly, and `reason` is
where the engine disagreements get written down.

## Optional capabilities

Implement what your engine can actually support, and declare only that:

| Protocol | Flag | What it buys |
| --- | --- | --- |
| `SupportsColumnLineage` | `column_lineage` | blast radius in `rowless diff` |
| `SupportsSql` | `sql` | SQL strings in the same graph as Python |
| `SupportsUdfProbe` | `udf_probe` | typed UDF outputs instead of ANY |
| `SupportsEmptyFrameFixups` | - | documented dtype drift corrections |
| `NeedsEgress` | `remote` | the sandbox opens your endpoint and nothing else |

Three optional hooks are read by name rather than through a protocol, because
an adapter that does not need them should not have to say so:
`conformance_cases()` supplies the operations battery,
`unsupported_dtypes()` declares the dtypes above, and `unavailable_reason()`
explains an adapter that cannot run for a reason `requires` cannot describe -
no endpoint configured, no credentials, no JVM. `rowless doctor` and the
conformance suite's skip message both prefer it over `requires`.

A capability that is advertised but broken is worse than one that is missing,
because coverage scores are computed from these flags.

## If your engine is on a server

An adapter that declares `remote=True` is asked what to permit before the
sandbox is installed:

```python
def egress(self, options=None) -> Egress:
    """Exactly what this adapter needs, and nothing wider."""
    return Egress(
        hosts=(host_from(options),),
        subprocesses=False,     # True only if reaching the engine starts one
        reason="analysis happens on the server",
    )
```

Three things follow from this, and they are the price of the capability:

* **name the host, not the network.** `Egress(hosts=("warehouse.internal",))`
  opens one name. There is no form of this hook that opens everything, on
  purpose.
* **the engine has to be named ahead of the import.** An engine inferred from a
  pipeline's annotations is not known until the module has been imported, which
  happens inside the sandbox - so a remote adapter wants its engine stated in
  the annotations or in `rowless.toml`, and a connection attempted without one
  is blocked and reported rather than quietly allowed.
* **`describe` is not asserted side-effect-free for you.** The conformance
  suite skips `test_describe_does_not_execute` for a remote adapter, because
  its describe legitimately opens a socket. Everything that *reaches* the
  engine is then your adapter's responsibility to account for: if a pipeline
  can send a statement that changes something, your interception is the only
  thing standing between a dry run and a production table.

## Document the disagreements

The most valuable thing you will write is not code. When you discover that your
engine produces a different schema than the others for the same operation, write
it down in the adapter module docstring and add a conformance expectation
override. Those overrides are the catalogue of places where moving a pipeline
between engines changes its output, and nobody else has that catalogue.

## Register it

```toml
[project.entry-points."rowless.engines"]
myengine = "rowless_myengine.adapter:MyAdapter"
```

Out-of-tree adapters work identically: depend on `rowless`, register the entry
point, import the conformance suite from `rowless.testing`. Nothing about being
in this repository is required, and a third-party adapter is a perfectly good
outcome.

## Checklist

- [ ] conformance suite passes
- [ ] `conformance_cases()` covers select, rename, filter, group-by, joins,
      unions, casts and at least one row-dependent pipeline - or, for an engine
      where reading a value is impossible rather than merely unwise, one case
      that comes back `UNKNOWN` for the reason that engine actually loses
      confidence (`rowless-sql` asserts a set operation with no common dtype)
- [ ] dtype round trip covers every type the engine has
- [ ] dtypes the engine lacks declared in `unsupported_dtypes()`
- [ ] engine-version floor in `requires`, and tested at that floor
- [ ] engine quirks documented in the module docstring
- [ ] capabilities declared accurately
- [ ] a remote engine: `egress()` names its hosts, and every call that could
      change something on the other end is intercepted
- [ ] added to the CI adapter matrix and the nightly engine job
