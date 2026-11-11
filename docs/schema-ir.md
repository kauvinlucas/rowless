# The schema IR

The one vocabulary every part of rowless shares. Adapters translate their engine
into it; sources parse metadata into it; the CLI renders it.

## Dtypes

A `DType` is a `Kind` plus parameters: width, time unit, timezone, decimal
precision and scale, and nested element types. The parameters are modelled
explicitly because each one is a real source of pipeline breakage. A tool that
collapses `datetime[us, UTC]` and `datetime[ns]` into "timestamp" cannot report
the bug where they differ.

Three cases deserve care:

**`UNKNOWN`** means the adapter could not determine the dtype. It does not match
even itself: two dtypes nobody could resolve are not evidence that a pipeline is
sound. This is the mechanism that keeps rowless from reporting false passes.
It carries a `reason` string, which `rowless explain` prints and which is
excluded from comparison: two unresolved dtypes are equally unusable whatever
the reason says.

**`ANY`** is the gradual-typing escape hatch, like `Any` in mypy. It matches
anything, and it lowers the coverage score. UDFs with no inferable return type
and dynamically named columns produce it.

**`NULL`** is a column known to be all-null with no further type information,
which pandas produces readily. Distinct from both UNKNOWN and ANY: rowless knows
exactly what this is, and it is a problem the user should see.

## Schemas

`Schema` is an ordered, name-addressed tuple of `Field`s. Immutable, so a dry run
keeps its input schema to diff against the output. A `Field` carries `nullable`,
an `Origin` (file, line, which makes diagnostics clickable), and `derived_from`,
the input columns it was computed from, which is the lineage substrate.

Adapters return `Relation`, not bare `Schema`: a schema plus which engine
produced it, plus notes. The interesting part of a dry run is often what the
engine could not tell us, and `Relation` is where that lives.

## Comparison

Two operations, different purposes:

- `matches` is structural equality, with ANY matching anything. Used for
  identity checks and conformance round trips.
- `accepts` is one-directional: int32 flows into an int64 slot, not the reverse.
  Used when checking a computed schema against a declared contract.

`unify` is the lattice join, for `concat`, set operations and branch merges. It
falls back to UNKNOWN rather than inventing a coercion the engine would have
refused.

## Why not reuse Arrow

Arrow was the obvious candidate and gets most of the way there. It was not
adopted as the IR for two reasons:

1. It would make pyarrow a hard dependency of core, and core deliberately has no
   engine or format dependency at all.
2. There is nowhere in Arrow to put UNKNOWN and ANY, and those two are the
   project honesty mechanism rather than an edge case.

The Arrow bridge is first class, though: `rowless.sources.parquet.arrow_to_ir`
is shared by every adapter with an Arrow path, so the mapping is written once.

## Round trips

Every adapter must satisfy `to_ir(from_ir(dtype)) == dtype` for every dtype its
engine supports. The conformance suite enforces it. This is where losing a
timezone or a decimal scale gets caught, instead of surfacing months later as a
mysterious production diff.
