# 0002: A rowless dtype IR rather than Arrow

Status: accepted
Date: project start

## Context

Every engine in scope has an Arrow bridge, and Arrow has a well-specified type
system. Reusing it as the IR was the obvious move.

## Decision

A small rowless dtype IR, with Arrow as a first-class bridge rather than the
vocabulary.

## Why

**Core must have no format dependency.** Using Arrow as the IR makes pyarrow a
hard dependency of core. Today core depends on typer, rich, pydantic and pyyaml,
and nothing that knows what a dataframe is.

**UNKNOWN and ANY have no home in Arrow.** They are not edge cases; they are the
mechanism that stops rowless reporting false passes. Encoding them as extension
types or as a side table would make the central honesty mechanism the least
natural thing in the type system.

**Engine-specific types that Arrow flattens.** Polars `Enum`, DuckDB `ENUM` and
pandas `Categorical` carry information rowless wants in diffs.

## Costs

A dtype mapping per adapter, rather than one Arrow mapping. Mitigated by
`arrow_to_ir`, shared by every adapter with an Arrow path, so in practice most
adapters write a thin layer over it.
