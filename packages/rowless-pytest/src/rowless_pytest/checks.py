"""`rowless_check`: the assertion the whole plugin exists for.

One call does four things: resolve the pipeline's inputs, dry-run it, compare
what the engine computed against the declared contract, and fail with a
message a reader can act on without opening rowless.

The comparison itself is `rowless.api.compare_schemas`, the same function
behind `rowless.assert_schema`. A test must not be able to pass here and fail
there, so the only thing this module adds is framing: which engine, which
inputs, and what the failure looks like.

Three ways to fail, in the order they are checked, because each one makes the
next question meaningless:

1. nothing resolved - the pipeline raised, or an input would not resolve;
2. the result is unverified - a dry run that lost confidence proves nothing
   about the schema, so it fails unless the test opted in;
3. the schema does not match the contract.

Rule 2 is the one that looks unfriendly and is not. A contract test whose
pipeline grew a row-dependent branch stops testing anything, and the
alternative to failing it is a green suite that says a column is fine when
rowless never saw it.

What is deliberately *not* read here is the code-level policy in
`rowless.toml`: `error_codes`, `ignore_codes` and `fail_on_unverified` decide
which findings gate a `rowless check` run, and a contract test is a different
question - it asserts one schema, and it fails on the three things above and
nothing else. Every diagnostic still travels in the failure message, because
the finding is usually the explanation. Turning off `fail_on_unverified` for
CI therefore does not quietly disarm a suite of contract tests; only
`allow_unverified=True`, written in the test, does that.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from rowless.api import compare_schemas, marker_of
from rowless.ir.confidence import Confidence
from rowless.ir.diff import ChangeKind, diff_schemas
from rowless.ir.schema import Schema
from rowless.runtime.dryrun import DryRunResult

from rowless_pytest import engines
from rowless_pytest.session import RowlessSession


@dataclass(frozen=True, slots=True)
class SchemaCheck:
    """What the `rowless_check` fixture hands a test.

    Carries the session and the engine the test's marker declared, so the
    signature a test sees stays short:

        def test_revenue(rowless_check):
            rowless_check(daily_revenue, {"day": "date", "amount": "int64"})
    """

    session: RowlessSession
    engine: str | None = None
    """The engine `@pytest.mark.rowless(engine=...)` named, if any."""

    def __call__(
        self,
        target: Callable[..., Any],
        expected: Schema | Mapping[str, str],
        *,
        inputs: Mapping[str, Schema | Mapping[str, str] | str] | None = None,
        exact: bool = True,
        allow_unverified: bool = False,
        engine: str | None = None,
    ) -> DryRunResult:
        """Assert that `target` resolves to `expected`, and return the result.

        `expected` is a `Schema` or the `{"day": "date"}` shorthand.

        `inputs` maps each parameter to a schema, a shorthand mapping or a
        source URI; leave it out and the `@rowless.pipeline(inputs=...)`
        declaration is used, which is how one pipeline's inputs stay written
        down once.

        `exact=False` allows output columns the test did not name, which is
        what makes this usable on a wide mart.

        `allow_unverified=True` accepts a result rowless could not verify. It
        is deliberately a per-assertion opt-in with a loud name, and the
        session summary counts how many times it was used.

        The `DryRunResult` comes back so a test can go further - assert on
        lineage, on the recorded writes, or on how long it took.
        """
        wanted = self.session.schema(expected)
        marker = marker_of(target)
        chosen = engines.choose(
            requested=engine,
            pipeline=marker.engine if marker is not None else None,
            declared=self.engine,
            flag=self.session.engine,
            configured=self.session.config.engines,
        )
        resolved = self.session.inputs(target, inputs)
        result = self.session.dry_run(target, resolved, engine=chosen)

        output = result.output
        lost = result.coverage.confidence is Confidence.UNKNOWN
        self.session.record(
            result,
            allowed_unverified=allow_unverified and lost and output is not None,
        )

        if output is None:
            raise AssertionError(unresolved_message(result))
        if lost and not allow_unverified:
            raise AssertionError(unverified_message(result))

        problems = compare_schemas(wanted, output.schema, exact=exact)
        if problems:
            raise AssertionError(mismatch_message(result, wanted, problems, exact=exact))
        return result


def unresolved_message(result: DryRunResult) -> str:
    """Nothing came back: the pipeline raised, or an input would not resolve."""
    reasons = "; ".join(result.coverage.reasons) or "no schema was produced"
    return _message(
        f"rowless could not resolve {result.target}, so the contract was not tested",
        [f"reason: {reasons}"],
        result,
    )


def unverified_message(result: DryRunResult) -> str:
    """A schema came back that rowless does not stand behind."""
    reasons = "; ".join(result.coverage.reasons) or "confidence was lost"
    return _message(
        f"rowless did not verify {result.target}, so this test proves nothing",
        [
            f"reason: {reasons}",
            "fix the row-dependent code, or pass allow_unverified=True to assert on the "
            "schema anyway",
        ],
        result,
    )


def mismatch_message(
    result: DryRunResult,
    wanted: Schema,
    problems: Sequence[str],
    *,
    exact: bool,
) -> str:
    """The engine resolved a schema, and it is not the one the test declared."""
    actual = result.output.schema if result.output is not None else Schema.empty()
    lines = [*problems, *_renames(wanted, actual)]
    lines.append(f"expected: {_columns(wanted)}")
    lines.append(f"resolved: {_columns(actual)}")
    if exact and all(_is_exactness_problem(problem) for problem in problems):
        lines.append("pass exact=False to check only the columns this test names")
    return _message(f"{result.target} does not match the expected schema", lines, result)


def _renames(wanted: Schema, actual: Schema) -> list[str]:
    """Rename hints: a declared column that came back under another name.

    The same pairing `rowless diff` does, for the case this plugin is sold on:
    a column the test asks for by its old name. "missing day, unexpected
    day_utc" is two facts; "day -> day_utc" is the answer.
    """
    diff = diff_schemas(wanted, actual)
    return [
        f"{change.column} -> {change.renamed_to}: the resolved schema has it under a new name"
        for change in diff.changes
        if change.kind is ChangeKind.COLUMN_RENAMED
    ]


def _is_exactness_problem(problem: str) -> bool:
    """True when a problem would go away under `exact=False`.

    String matching on `compare_schemas` output, which is not pretty. The
    alternative is for core to return structured problems, and a test author
    reading a failure needs the sentence either way - so the hint is built
    from the sentence rather than the comparison growing a second shape for
    one caller.
    """
    return problem.startswith(("unexpected column", "column order differs"))


def _message(headline: str, lines: Sequence[str], result: DryRunResult) -> str:
    """Headline, then the detail and every diagnostic, indented under it."""
    body = [*lines, *_diagnostics(result)]
    return headline + ":\n" + "\n".join(f"  {line}" for line in body)


def _diagnostics(result: DryRunResult) -> list[str]:
    """The run's findings, rendered the way the terminal renderer does.

    Included in every failure, because the finding usually is the explanation:
    a contract test that fails on a missing column wants to say which upstream
    file renamed it, and the diagnostic is where that lives.
    """
    lines: list[str] = []
    for diagnostic in result.diagnostics:
        where = f" ({diagnostic.origin})" if diagnostic.origin is not None else ""
        lines.append(f"{diagnostic.code.value} {diagnostic.level}: {diagnostic.message}{where}")
        if diagnostic.detail:
            lines.extend(f"  {line}" for line in diagnostic.detail.splitlines())
        if diagnostic.suggestion:
            lines.append(f"  {diagnostic.suggestion}")
        for origin in diagnostic.related:
            lines.append(f"  upstream {origin}")
    return lines


def _columns(schema: Schema) -> str:
    """A schema on one line: `day date, amount int64`."""
    return ", ".join(f"{field.name} {field.dtype}" for field in schema.fields) or "(no columns)"
