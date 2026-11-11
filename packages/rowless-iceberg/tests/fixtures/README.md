# Iceberg metadata fixtures

Hand-written table metadata, committed rather than generated. Writing them with
PyIceberg would need a catalog, a warehouse directory and a data file per
commit, and the tests would then be asserting that PyIceberg can round-trip its
own writer. What they need to assert is that rowless reads the *format*, so the
format is what is checked in.

No manifests, no manifest lists and no data files: this source reads table
metadata and nothing else, so a fixture that carried them would be describing
reads that never happen.

| file | what it is for |
| --- | --- |
| `orders_v2.metadata.json` | the rename story: three retained schemas, three snapshots, a branch and a tag |
| `types_v3.metadata.json` | every primitive and nested type in the spec, including the v3 additions |
| `legacy_v1.metadata.json` | a v1 table whose snapshot records no `schema-id` |

## `orders_v2.metadata.json`

The table the proposal's terminal block describes, with the history that makes a
rename a fact:

| field id | schema 0 | schema 1 | schema 2 |
| --- | --- | --- | --- |
| 1 | `order_id` long | `order_id` long | `order_id` long |
| 2 | `amount` int | `amount_cents` long | `amount_cents` long |
| 3 | `status` | `status` | `order_status` |
| 4 | `customer_email` | `customer_email` | *dropped* |
| 5 | | | `created_at` |
| 6 | | | `tags` |

So field 2 is renamed once, field 3 is renamed once two steps later, field 4 is
a genuine drop that must not be reported as a rename, and fields 5 and 6 are
additions with no history to carry. Each schema has a snapshot written against
it, so every row of that table is reachable through `?snapshot=`, and
`release-2024-11` tags the middle one.

Field 2 is also **promoted from `int` to `long` in the same commit as its
rename**, which is what separates this fixture from one core could handle on its
own. `rowless.ir.diff` pairs a removal with an addition by position *and* dtype,
so a rename that keeps its type is a rename it finds unaided - `status` to
`order_status` is one, and the tests assert that core finds it with the field
IDs stripped out. A rename that retypes at the same time defeats that rule, and
its weaker fallback needs exactly one unpaired column on each side where this
table has two and three. Iceberg is the only source that can answer it, which
is the whole claim this package makes. Both halves are asserted, so a fixture
edited to keep the dtype would quietly delete the point of the test.

An Iceberg-legal promotion on purpose: the format permits `int` to `long` and
forbids, say, `decimal` to `long`, so a fixture using the latter would be
testing against a table that cannot exist.

## `legacy_v1.metadata.json`

`schema-id` is optional on a snapshot, and a table written by an old enough
writer does not record which schema a snapshot ran against. The fixture exists
so the refusal is tested rather than assumed: rowless reports that the schema is
not recoverable instead of substituting the current one.

## Adding one

Commit the fixture before the code that reads it, and keep it minimal: a
fixture with a field nobody asserts on is a field nobody notices breaking.
