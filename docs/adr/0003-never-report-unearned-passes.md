# 0003: Unknown is never reported as safe

Status: accepted
Date: project start

## Context

rowless cannot always resolve a schema. Control flow depends on row counts, UDFs
are opaque, column names get computed, sqlglot does not annotate every function.
A tool in that position can guess, or it can abstain.

## Decision

Abstain, visibly, every time. Specifically:

- `UNKNOWN` does not match itself, so comparisons involving it never pass.
- Row-dependent control flow downgrades the whole pipeline result to UNKNOWN.
- A diff change that cannot be classified is UNVERIFIED and fails the gate.
- Every downgrade carries a reason, and `rowless explain` prints them.
- Coverage ratios are reported alongside every result.

## Why

The product being sold is a green check in CI. If that check is sometimes green
because rowless lost track of a column, then it means nothing, and a team that
discovers this once will never trust it again. False negatives are not a quality
issue here; they are an existential one.

The inverse risk, being noisy, is real but recoverable: line-level suppression,
a coverage floor that starts at zero, and codes that do not gate by default.

## Consequences

rowless will report "I do not know" more often than a tool willing to guess, and
will look worse in a shallow comparison. The coverage score exists so that this
is legible rather than embarrassing: a user can see exactly how much was
verified and decide.

A bug where rowless reports a pass it has not earned is the highest-priority bug
class in the project, with its own issue template.
