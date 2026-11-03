## What this changes

<!-- One or two sentences. -->

## Checklist

- [ ] `nox -s lint types` passes
- [ ] Tests cover the change
- [ ] For an adapter change: `nox -s conformance` passes
- [ ] For a new capability: declared in `Capabilities` and actually implemented

## For adapter changes

- [ ] Engine versions tested:
- [ ] Any engine-specific schema behaviour documented in the adapter module
      docstring, especially where this engine disagrees with the others

## Honesty check

rowless is only useful if it never reports a pass it has not earned.

- [ ] No path in this change reports EXACT confidence for a schema that was
      partly guessed
- [ ] Every new failure mode produces a diagnostic, not a silent fallback
