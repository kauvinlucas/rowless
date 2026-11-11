# rowless-iceberg

Iceberg schema source, via PyIceberg catalog metadata.

## The trick

Read the table metadata, never the data files. Iceberg keeps stable field IDs
across renames and retains schema history, so this is the one source where a
rename is a recorded fact rather than a heuristic, and a diff can run against a
historical snapshot without a git checkout.

## Why that is worth a package

Everywhere else in rowless a rename is a guess. `rowless.ir.diff` pairs a
removed column with an added one by lineage evidence, then by position and
dtype, then by there being exactly one candidate on each side - and that last
rule is weak enough to be switchable. The guessing is good, and it is still
guessing.

Iceberg assigns every column a field ID when it is created and keeps it through
renames, retypes and reorders. So `amount` becoming `amount_cents` is a fact in
the table metadata, and this source puts it in the `derived_from` slot that core
already trusts above its own heuristics:

```python
# pipelines/revenue.py
orders = rowless.source("iceberg://prod.sales.orders")


def daily_revenue(orders: pl.LazyFrame) -> pl.LazyFrame:
    return orders.select("order_id", "amount")
```

```
$ rowless check
pipelines/revenue.py:9  ColumnNotFound 'amount', did you mean 'amount_cents'?  RWL101

$ rowless diff main
BREAKING  prod.sales.orders.amount -> amount_cents
```

The second line is the one no other source can produce. It says `->` because
the format recorded the rename, not because two columns happened to line up.

## Where it beats the heuristic, and where it ties

Worth being precise about, because "exact renames" oversells it if the
heuristic would have got there anyway:

| change | `rowless diff` alone | with field IDs |
| --- | --- | --- |
| rename, same type and position | rename | rename |
| rename plus a type promotion | removal plus addition | rename |
| rename plus a reorder | removal plus addition | rename |
| several renames in one commit | one guess at best | all of them |

So the field IDs confirm the easy cases and answer the hard ones.
`tests/test_iceberg_source.py` asserts both ends of that on one table: the
rename core finds on its own, and the one next to a type promotion that only
the metadata can resolve.

What `derived_from` does **not** do is veto. It is positive evidence, so core's
heuristics still run underneath it, and a dropped column sitting where a new
one appeared is still reported as a rename nobody made. Refusing that needs the
field IDs on both sides of the comparison rather than evidence on one, which is
what `renames_since` is for:

```python
source.renames_since(ref, snapshot_id)  # {"amount": "amount_cents"}
```

A column that was dropped shares no field ID with anything in the new schema,
so it cannot appear there at all. Both behaviours are pinned by tests, the
limitation included.

## Status

Implemented: the `iceberg://` schema source, the type mapping, schema history
and field-ID rename resolution. `IcebergSource` also exposes `schema_at`,
`renames_since` and `field_ids` beyond the `SchemaSource` protocol, for callers
that want column identity rather than a schema.

One Iceberg change this source cannot report: **a nested field's nullability**.
The IR carries `nullable` on a top-level field only, so a struct subfield going
from required to optional is invisible. That is a gap in the shared IR rather
than in this package, and it is listed here rather than worked around - a
guessed answer would be worse than a known blind spot.

## URIs

```
iceberg://prod.sales.orders                      catalog prod, table sales.orders
iceberg://sales.orders                           the default catalog
iceberg://sales.orders?catalog=prod              the whole path is the table's
iceberg://sales.orders?snapshot=7788990011223344 the schema that snapshot ran
iceberg://sales.orders?tag=release-2024-11       the schema behind a tag
iceberg://orders?metadata=.../v7.metadata.json   no catalog at all
```

The first segment is the catalog name whenever a path has three or more of
them, because Iceberg namespaces nest arbitrarily and no URI can be
unambiguous about it. `?catalog=` overrides that, and
`ROWLESS_ICEBERG_CATALOG` sets a default for a repository that would otherwise
repeat it in every URI. When a lookup fails, the message says how rowless split
the path, because reading a two-level namespace as a catalog name is the one
mistake that otherwise produces a `NoSuchTableError` for a table the user can
see.

`snapshot`, `branch` and `tag` are mutually exclusive, and naming two is
refused rather than resolved by precedence.

## Credentials

Catalog configuration is PyIceberg's, out of `.pyiceberg.yaml` or `PYICEBERG_*`
variables, and is never read from `rowless.toml`: a REST catalog URI and its
token are credentials.

`?metadata=` skips the catalog entirely. An Iceberg table is self-describing,
so a metadata file is the whole schema rather than a subset of it, and a CI job
with a committed or artifact-fetched metadata file resolves the table with no
catalog, no credentials and no engine installed. That is the same property the
committed catalog snapshot gives core's warehouse source, and it is what makes
the check runnable in the job that most needs it.

## Types

The Iceberg spec defines twelve primitives and three nested types, every writer
agrees on them, and PyIceberg prints them with the spec's own spelling - so the
mapping is a lookup table on that spelling rather than a tree of `isinstance`
checks. A primitive added in a later spec version therefore lands as UNKNOWN
with its spec name in the reason, instead of failing to import.

The judgement calls, all asserted in `tests/test_iceberg_dtypes.py` against a
metadata file holding every type in the spec:

| Iceberg | IR | why |
| --- | --- | --- |
| `int`, `long` | `int32`, `int64` | the spec has two widths and no unsigned type |
| `time` | `time[us]` | one time type, microsecond resolution |
| `timestamp` | `datetime[us]` | naive, and that is the distinction that breaks pipelines |
| `timestamptz` | `datetime[us, UTC]` | a UTC instant by definition, not a normalisation |
| `fixed[L]` | `binary` | agrees with the Parquet footer; no promotion can change L |
| `uuid`, `geometry`, `geography` | ANY | no IR equivalent, with the spelling in the reason |
| `unknown` (v3) | `null` | a column whose every value is null, which is what IR NULL is |
| anything else | UNKNOWN | with the spelling, so it poisons rather than passes |

`uuid` is the call worth arguing about. `string` would be worse than useless - a
`string` to `uuid` retype would then read as no change at all. UNKNOWN would be
worse in the other direction: most Iceberg tables have a UUID key, and marking
every one of them UNVERIFIED forever is how a CI gate gets switched off. ANY is
the escape hatch that lowers the coverage score and says why.

## Versions

PyIceberg 0.12 and up. The floor is not the oldest release that works but the
oldest one whose type parser accepts the v3 geospatial primitives: 0.9 through
0.11 read the other fixtures fine and reject a `geometry` column outright,
which would leave the type corpus partly tested rather than tested. The
`adapter-floor` CI job runs this suite pinned at `pyiceberg==0.12.0`, so the
claim is checked rather than declared.

## Owner

Kauvin Lucas ([@kauvinlucas](https://github.com/kauvinlucas)).

## Development

```bash
uv sync --package rowless-iceberg
pytest packages/rowless-iceberg
```

The test fixtures are hand-written metadata files rather than tables built by
PyIceberg's writer; `tests/fixtures/README.md` says why, and documents the
field-ID history the rename tests assert against.
