# Confidence and coverage

## The failure mode this exists to prevent

A checker that reports a pass it has not earned is worse than no checker. People
stop reading its output, then they stop running it, and the CI gate becomes
decoration. Every design decision below trades reported certainty for honesty.

## Confidence

Three levels, taken as a minimum when results combine:

- `EXACT` - the engine resolved every column itself. No rowless guesswork.
- `PARTIAL` - some columns are ANY or untraced. Reported as partial.
- `UNKNOWN` - nothing usable. A failed dry run, or row-dependent control flow.

## Coverage

Two ratios, reported per pipeline and in aggregate:

- **typed** - columns with a known, non-ANY dtype
- **traced** - columns with lineage, which gates blast-radius claims

`min_coverage` in config sets a floor, defaulting to 0 so a first run reports
rather than blocks. Teams ratchet it up, the way mypy strictness gets ratcheted.

## What lowers it

| Cause | Effect |
| --- | --- |
| `if df.is_empty()`, `.item()`, iteration | UNKNOWN for the whole pipeline |
| UDF with no inferable return type | ANY column |
| Column name built at runtime | ANY column, lineage gap |
| sqlglot function with no annotated type | ANY column |
| Adapter cannot report lineage | untraced columns |
| Unresolvable source | pipeline skipped, reported |
| Opaque node in a diff path | impact marked possibly larger |

Every one of these carries a reason string. `rowless explain` prints them, which
is how a user finds out *why* rowless is unsure rather than just that it is.

## Unverified is not compatible

In a diff, a change rowless could not classify is `UNVERIFIED` and fails the
gate by default (`fail_on_unverified`). Treating it as compatible would mean the
tool reports all-clear exactly when it has lost the trail, which is the one
outcome worth engineering against.

## Suppression

Because the first run on a real repository will find things nobody is fixing
this week:

```python
amount = df["amount"]  # rowless: ignore[RWL101]
```

Line-level, code-specific, and visible in review. A tool with no way to say "not
now" gets deleted rather than configured.
