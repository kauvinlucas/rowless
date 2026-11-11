# rowless-openlineage

Emit the rowless column graph as OpenLineage events.

## Why

Column-level lineage is already an OpenLineage facet, and Marquez, DataHub,
Atlan and OpenMetadata all read it. rowless does not need its own UI; it needs
to put its graph where the catalogs already look.

```bash
export OPENLINEAGE_URL=http://marquez:5000
rowless emit
```

```
emitted lineage for 7 node(s) via openlineage
```

## The one rule

**These are static lineage events, not run events.** Nothing executed. The
pipeline was run against frames with a schema and no rows, which is the whole
premise of the project, and a catalog that recorded that as a production job
would show a successful run at a time when all that happened was that somebody
saved a file.

So: `DatasetEvent` and `JobEvent`, never `RunEvent`. No run id, no START, no
COMPLETE. A rowless producer URI on every event and facet. A job namespace of
its own. And a coverage facet carrying `executed: false`, so the claim is made
in the payload and not only in this README.

[`test_events.py`](tests/test_events.py) asserts all of that, including that
the strings `START` and `COMPLETE` appear nowhere in an emission, because this
is the one property whose loss would damage somebody else's system rather than
produce a wrong answer in ours.

## What becomes what

| rowless node | event | carries |
| --- | --- | --- |
| a source | `DatasetEvent` | `schema`, `rowless_coverage`, `rowless_node` |
| a pipeline, SQL model or dbt model | `JobEvent` | the job, its inputs, and its output dataset with `columnLineage` |
| a recovered write target | `DatasetEvent` | `schema` and the `columnLineage` of the write |

Every non-source node is therefore both a job and a dataset. That is not a
workaround: `columnLineage` lives on a *dataset*, and a pipeline whose result
is passed on in memory still has to be addressable as the thing downstream
columns came from. Where a pipeline does write a table, the sink is its own
dataset and gets the facet too, so the chain in the catalog reads
`source -> pipeline -> table` with lineage on each hop.

Each node produces exactly one event, and an input dataset inside a `JobEvent`
carries no facets: its own event already published them, and sending a schema
twice invites a catalog to hold two versions of it, one already stale.

## Naming, which decides whether this is useful at all

A catalog merges lineage by dataset identity and nothing else. If
`s3://warehouse/stg_orders` does not come out as the namespace and name that
the Spark, Airflow and dbt integrations give it, rowless publishes a second
disconnected copy of the warehouse and the whole emission is decoration. So the
mapping follows the [OpenLineage naming
conventions](https://openlineage.io/docs/spec/naming):

| rowless node id | namespace | name |
| --- | --- | --- |
| `s3://bucket/key` | `s3://bucket` | `key` |
| `gs://bucket/key` | `gs://bucket` | `key` |
| `file:///srv/data/orders` | `file` | `/srv/data/orders` |
| `data/orders.parquet` | `file` | `data/orders.parquet` |
| `iceberg://catalog.db.table` | `iceberg://catalog` | `db.table` |
| `warehouse://db.schema.table` | `warehouse` | `db.schema.table` |
| `dbt://model.shop.stg_orders` | `dbt` | `model.shop.stg_orders` |
| `lake://stg_orders` | `lake` | `stg_orders` |
| `pipelines.revenue:daily_revenue` | the job namespace | the target, verbatim |

A node id with a scheme is the same dataset wherever it appears, including when
it is a dbt model rowless *ran* rather than one it read. That is the
Python-to-SQL seam arriving in the catalog as one graph instead of two.

Three rows are compromises rather than conventions, and all three are places
where rowless does not hold what the convention wants:

* **Azure.** The canonical namespace is
  `abfss://container@account.dfs.core.windows.net` and an `az://` URI carries
  no account. A name that looks canonical and points nowhere is worse than a
  partial one, so the container alone is emitted.
* **A warehouse table.** The convention wants `postgres://host:5432`. rowless
  keeps the DSN in an environment variable rather than in config, because it is
  a credential, so a node id has no host in it.
* **A bare path.** The convention wants an absolute path. rowless node ids are
  the URI the pipeline was written with, which is usually repo-relative - and a
  relative name is the same on every machine while an absolute one is not.

All three are why the override exists. It is the one piece of knowledge that
cannot be derived from the repository, so it is configuration:

```bash
export ROWLESS_OPENLINEAGE_NAMESPACES="warehouse=postgres://db.internal:5432,lake=s3://warehouse"
```

## What is withheld, and why that is the point

rowless labels every edge with the evidence behind it. `PLAN` means the engine
resolved it, `SQL_QUALIFIED` means sqlglot did, `MANIFEST` means dbt declared
it - and `NAME_MATCH` means two columns share a name and nothing saw one feed
the other. Core already refuses to gate a CI run on a name match.

A catalog arrow has no "probably" in it. Published, a guess becomes verified
lineage in the hands of somebody who was not there, so name-matched edges are
**not emitted** by default. The gap is not silent: `rowless_node` carries
`withheldEdges` and `untracedColumns`, so a reader can see that rowless knew
more than it published and why.

```bash
export ROWLESS_OPENLINEAGE_INCLUDE_NAME_MATCH=1   # a complete-looking graph, knowingly
```

The per-edge evidence travels in the facet's `transformationDescription` -
`sum (evidence: plan)` - because the spec has nowhere else to put it, and
because a plan-walked derivation and a declared one should not arrive
indistinguishable.

## The two custom facets

Standard facets carry the lineage; these two carry what a catalog needs in
order not to overstate it. Both have published JSON schemas, under
[`docs/spec/`](../../docs/spec/), and the tests check code and schema against
each other key by key.

**`rowless_coverage`**, on every event - not just the job's, because a catalog
may ingest one dataset event and drop the rest, and a schema that arrives
without a score beside it will be read as verified:

```json
{
  "executed": false,
  "confidence": "partial",
  "verified": false,
  "columnsTotal": 14,
  "columnsTyped": 14,
  "columnsTraced": 11,
  "typedRatio": 1.0,
  "tracedRatio": 0.7857,
  "reasons": ["status has no lineage evidence"]
}
```

**`rowless_node`**, on each dataset: the node's kind, the engine that resolved
it, where it is defined, how much lineage was withheld - and `opaque`, which is
the field that matters. A node rowless could not see inside is published
anyway, because a missing node makes a lost trail look like the end of the
graph. The flag is how a catalog tells one from the other.

## Reviewing before publishing

```bash
rowless emit --dry-run        # prints the payloads, sends nothing
```

Needs no transport configured, and the output is byte-stable: nodes sorted,
edges sorted inside each facet, one timestamp for the whole emission. Two runs
over an unchanged repository produce identical bytes, so the output is
diffable in review and a catalog does not record a version per invocation.

What `--dry-run` prints is built by the same code path and serialised with the
client's own `Serde`, so it is what `emit` would have sent.

## Configuration

| variable | effect |
| --- | --- |
| `OPENLINEAGE_URL` | HTTP destination, read by the client |
| `OPENLINEAGE_CONFIG`, `./openlineage.yml`, `~/.openlineage/openlineage.yml` | YAML config |
| `OPENLINEAGE__TRANSPORT__*` | structured config: Kafka, file, composite |
| `OPENLINEAGE_DISABLED=true` | emit nothing, and say so rather than pretend |
| `OPENLINEAGE_NAMESPACE` | job namespace; default `rowless` |
| `ROWLESS_OPENLINEAGE_NAMESPACES` | per-scheme dataset namespace overrides |
| `ROWLESS_OPENLINEAGE_INCLUDE_NAME_MATCH` | publish name-matched edges too |

Everything is an environment variable because `rowless emit` builds this
through the plugin registry, which takes no arguments, and because every other
OpenLineage integration is configured the same way. Each one is re-read per
call rather than captured at construction, so a long-lived host - the language
server, a test session - is not pinned to the environment as it stood when the
plugin first loaded.

### The refusal

The openlineage client never fails to produce a transport: configured with
nothing, it logs a warning and prints events to stdout. A CI job meant to
publish to Marquez would then pass while publishing nothing, and nobody would
notice for a month. So `rowless emit` refuses:

```
rowless: emitter 'openlineage' is not usable here: no OpenLineage transport is
configured: set OPENLINEAGE_URL to your OpenLineage endpoint (e.g.
http://localhost:5000), or write an openlineage.yml in the repository root
```

A console transport asked for *explicitly* is a configured transport and is
accepted. What is refused is the fallback nobody chose.

## Using it from code

```python
from rowless.check import run
from rowless.config import Config
from rowless_openlineage import OpenLineageEmitter

outcome = run(Config.discover())
emitter = OpenLineageEmitter(client=my_client)  # your client, your rules
emitter.emit(outcome.graph, outcome.report.coverage)

events = emitter.events(outcome.graph, outcome.report.coverage)  # typed objects
payloads = emitter.to_events(outcome.graph, outcome.report.coverage)  # dicts
```

Passing a client bypasses the transport check - an embedder with a client has
already decided where events go - and leaves it open afterwards, since it
belongs to the caller. A client rowless opened is closed, because the async and
Kafka transports buffer and skipping that drops events on exit.

A transport failure raises `EmitFailed` naming how many events got through,
because a half-published graph is the one outcome a caller has to reason about.

## The version floor

`openlineage-python>=1.38`, and it is measured rather than guessed: 1.38 is the
first release that exports `StaticDataset` from `openlineage.client.event_v2`,
which is the type a static dataset event needs. Releases from 1.30 do have the
class, but only under `openlineage.client.generated.base` - generated code, not
API, and importing from there would tie this package to the client's build
layout. The `adapter-floor` CI job runs this suite against 1.38.0 exactly.

## Status

Implemented and tested. The suite asserts on payloads and never on a live
transport: a test that needed Marquez running would be skipped in CI, and a
skipped test on the one path that could corrupt a catalog is not a test.

## Owner

Kauvin Lucas ([@kauvinlucas](https://github.com/kauvinlucas)).
