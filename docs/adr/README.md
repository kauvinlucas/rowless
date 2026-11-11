# Architecture decision records

Changes to the adapter contract, the dtype IR or the confidence model get an ADR
and sign-off from two core team members, because every adapter pays for them.

Everything else is just a pull request.

| ADR | Decision |
| --- | --- |
| [0001](0001-adapters-as-separate-distributions.md) | One distribution per adapter |
| [0002](0002-own-schema-ir-not-arrow.md) | A rowless dtype IR instead of Arrow |
| [0003](0003-never-report-unearned-passes.md) | Unknown is never reported as safe |
