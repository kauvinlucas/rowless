# Contributing to rowless

```bash
git clone https://github.com/kauvinlucas/rowless
cd rowless
uv sync --all-packages --all-extras
uv run nox -s lint types tests
```

The workspace is one repository with many distributions. `uv sync --package
rowless-polars` gives you just core plus the Polars adapter, which is usually
what you want while working on one adapter.

## Where to start

**An adapter.** The self-contained workstream, and the reason the project is
split this way. Pick the engine you already use daily; you will know its type
quirks better than the core team does. Run the conformance suite first and work
through the failures:

```bash
uv sync --package rowless-polars
uv run pytest packages/rowless-polars
```

**Core.** The schema IR, the runtime, the sandbox, the CLI. Smaller surface,
more design discussion. Read `docs/architecture.md` first.

**Integrations.** The pytest plugin, the GitHub Action, the OpenLineage emitter,
the language server. Each is independent of the others.

## The one rule

rowless must never report a pass it has not earned. Everything else is
negotiable; this is not.

When you cannot determine a schema, say so: return UNKNOWN, attach a reason,
lower the coverage score. A tool that is confidently wrong about one pipeline
loses the user for every pipeline, and the whole premise of checking in CI
depends on people believing the green result.

In practice:

- Never substitute a near-enough dtype. Raise `UnsupportedDType` or return
  `Kind.UNKNOWN`.
- Never report confidence for a path a dry run did not take.
- Never let an engine exception escape as a traceback where it could be a
  diagnostic with a file and a line.
- A capability declared in `Capabilities` must work. Coverage scores are
  computed from those flags.

## Writing an adapter

See `docs/writing-an-adapter.md`. In short: four required methods, optional
capability protocols, one entry point, and the conformance suite as the
specification.

## Style

Ruff and mypy strict, both enforced in CI. Engine libraries are the exception to
strict typing, which is why they are confined to adapters.

Comments earn their place by explaining *why*. The engine quirks are the
valuable documentation in this project: when you find out that pandas changes a
dtype on an empty frame, write down where and why, because the next person will
otherwise rediscover it.

## Governance

See `GOVERNANCE.md`. Adapter owners have review rights over their package and
can merge there.
