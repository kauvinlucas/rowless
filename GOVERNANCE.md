# Governance

Light, and shaped to the structure of the work: a small core and independent
adapters.

## Roles

**Core team** (2-3 people). The schema IR, the runtime, the sandbox, the CLI,
and the adapter contract. Decides anything that changes the contract, since that
affects every adapter at once.

**Adapter owners** (one or two per engine). Own one package: review rights,
merge rights, and the call on engine-specific behaviour. An owner decides how
their engine maps into the IR; they do not decide what the IR is.

**Integration owners.** Same deal for the pytest plugin, the GitHub Action, the
OpenLineage emitter and the language server.

## Current owners

Everything: Kauvin Lucas ([@kauvinlucas](https://github.com/kauvinlucas)) -
core and the adapter contract, all eight engine adapters, the pytest plugin, the
OpenLineage emitter, the language server and the GitHub Action.

`.github/CODEOWNERS` is the enforced version of that, one line per package.
Those lines are not redundant: one person owning the lot is a pre-v1.0 state,
not the intended end state, and the point of the split is that each engine ends
up with someone who uses it daily. Handing one over is a one-line change there
and a line here.

## Becoming an owner

Claim an adapter by opening an issue, then land something: a dtype mapping, a
conformance fix, a real finding on a real repository. Ownership follows
contribution rather than preceding it. An adapter that already has an owner can
still change hands, and the owner above will hand over any engine someone else
is better placed to maintain.

## Decisions

Most things: open a pull request. Changes to the adapter contract, the dtype IR,
or the confidence model need an ADR in `docs/adr/` and sign-off from two core
team members, because every adapter pays for them. While core has a single
owner the ADR is still required - the written reasoning is the part that
outlives the decision - and the second signature starts applying the day there
is a second core owner.

Disagreement between an adapter owner and core about engine semantics is settled
by what the engine actually does. Write the conformance test; the engine
arbitrates.

## Releases

All packages share a version and release together until v1.0. One compatibility
story is easier to explain than twelve, and the adapter contract is going to move
before v1.0.
