# Writing an adapter

The self-contained way to contribute. Pick the engine you use daily: you will
know its type quirks better than the core team does.

## Setup

```bash
uv sync --package rowless-polars
uv run pytest packages/rowless-polars     # all failing, by design
```

The conformance suite is the specification. Work through its failures.

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

A capability that is advertised but broken is worse than one that is missing,
because coverage scores are computed from these flags.

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
      unions, casts and at least one row-dependent pipeline
- [ ] dtype round trip covers every type the engine has
- [ ] engine-version floor in `requires`, and tested at that floor
- [ ] engine quirks documented in the module docstring
- [ ] capabilities declared accurately
- [ ] added to the CI adapter matrix and the nightly engine job
