"""Static AST analysis: the half of the job a dry run cannot do.

A dry run sees one path through the code. The things it cannot see:

* branches not taken, which may contain the row-dependent calls from
  `rowless.runtime.rowdep`
* column names built by string formatting, so lineage is partial by nature
* pipelines whose module fails to import at all

This is deliberately *not* the schema inference strategy: the proposal's whole
argument is that parsing Python source to guess schemas breaks on real code.
The AST pass only flags places where a dry-run result should be trusted less,
which is a much weaker claim and therefore a reliable one.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from rowless.runtime.rowdep import ROW_DEPENDENT_APIS

#: Accessors that return a scalar on a frame but build an expression on a
#: column reference. `pl.col("amount").sum()` resolves to a dtype without
#: touching data; `df["amount"].sum()` does not. Flagging the first would put
#: a finding on every correct Polars pipeline, so the receiver decides.
_EXPRESSION_SAFE_APIS = frozenset(
    {
        "max",
        "min",
        "sum",
        "mean",
        "n_unique",
        "unique",
        "value_counts",
        "first",
        "head",
    }
)

#: Calls that produce a column expression rather than a frame. A chain rooted
#: in one of these is lazy all the way up.
_EXPRESSION_ROOTS = frozenset({"col", "lit", "element", "nth", "exclude", "all", "len"})

#: Methods that take column names, where a non-literal argument means rowless
#: cannot know which column was read.
_COLUMN_SELECTORS = frozenset(
    {
        "col",
        "select",
        "drop",
        "rename",
        "with_columns",
        "group_by",
        "groupby",
        "agg",
        "sort",
        "alias",
    }
)


@dataclass(frozen=True, slots=True)
class StaticFinding:
    """Something in the source that limits confidence in a dry-run result."""

    kind: str
    """row_dependent_call, dynamic_column, unreached_branch, import_cycle."""

    file: str
    line: int
    detail: str
    reachable_in_dry_run: bool = False
    """False means the dry run never executed this line, so the finding is
    about a path production might take and rowless did not."""


def analyse_module(path: str) -> Sequence[StaticFinding]:
    """Walk one module's AST for confidence-limiting patterns."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return (_unparseable(path),)
    return analyse_source(text, path)


def analyse_source(text: str, path: str = "<buffer>") -> Sequence[StaticFinding]:
    """The same walk over source text that was never written to disk.

    What the language server calls on an unsaved buffer. A dry run needs a
    file - it imports the module - but these findings are about the source
    itself, so they are the ones an editor can report between saves.
    """
    try:
        tree = ast.parse(text, filename=path)
    except (SyntaxError, ValueError):
        # ValueError as well as SyntaxError: a source with a null byte in it
        # raises the former, and an editor holding a half-decoded buffer is
        # exactly where that turns up.
        return (_unparseable(path),)
    findings = [*_row_dependent_calls(tree, path), *_dynamic_columns(tree, path)]
    return _deduped(findings)


def _unparseable(path: str) -> StaticFinding:
    """The finding for a module no dry run can reach."""
    return StaticFinding(
        kind="import_cycle",
        file=path,
        line=1,
        detail="module does not parse, so no dry run can reach it",
    )


def dynamic_column_names(path: str) -> Sequence[StaticFinding]:
    """Flag column names that are not literals.

    `pl.col(f"amount_{currency}")` cannot be resolved statically or
    dynamically, and the honest answer is an ANY column plus a note, not a
    guess that happens to be right in the dry run's single iteration.
    """
    tree = _parse(path)
    return () if tree is None else _deduped(_dynamic_columns(tree, path))


def _deduped(findings: Iterable[StaticFinding]) -> tuple[StaticFinding, ...]:
    """One finding per kind per line, in source order.

    A nested expression can trip the same check twice (`select(pl.col(name))`
    is one problem, not two), and a doubled finding reads as a tool that does
    not understand the code.
    """
    best: dict[tuple[int, str], StaticFinding] = {}
    for finding in findings:
        best.setdefault((finding.line, finding.kind), finding)
    return tuple(sorted(best.values(), key=lambda finding: (finding.line, finding.kind)))


def branch_coverage(path: str, executed_lines: frozenset[int]) -> Sequence[StaticFinding]:
    """Compare executed lines against branches in the source.

    Feeds the coverage score: a pipeline whose every branch ran on zero rows
    is better verified than one where half the code was skipped, and the user
    deserves to know which they have.
    """
    tree = _parse(path)
    if tree is None:
        return ()
    findings: list[StaticFinding] = []
    for node in ast.walk(tree):
        for label, body in _branches(node):
            lines = {statement.lineno for statement in body}
            if not lines or lines & executed_lines:
                continue
            findings.append(
                StaticFinding(
                    kind="unreached_branch",
                    file=path,
                    line=min(lines),
                    detail=f"{label} never ran on zero rows",
                    reachable_in_dry_run=False,
                )
            )
    return tuple(sorted(findings, key=lambda finding: finding.line))


def _branches(node: ast.AST) -> Iterator[tuple[str, list[ast.stmt]]]:
    """The conditional bodies of one node, labelled for the diagnostic."""
    if isinstance(node, ast.If):
        yield "if branch", node.body
        if node.orelse:
            yield "else branch", node.orelse
    elif isinstance(node, ast.For | ast.While):
        yield "loop body", node.body
    elif isinstance(node, ast.Try):
        for handler in node.handlers:
            yield "except handler", handler.body
    elif isinstance(node, ast.Match):
        for case in node.cases:
            yield "match case", case.body


def _parse(path: str) -> ast.Module | None:
    """Parse a file, returning None when it cannot be read or parsed."""
    try:
        return ast.parse(Path(path).read_text(encoding="utf-8"), filename=path)
    except (OSError, SyntaxError, ValueError):
        return None


def _row_dependent_calls(tree: ast.Module, path: str) -> Iterator[StaticFinding]:
    """Calls whose value depends on there being rows.

    The same list the runtime watches, so a user who reads one diagnostic can
    look up every case rowless knows about in one place.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            accessor = node.func.attr
            if accessor not in ROW_DEPENDENT_APIS:
                continue
            if accessor in _EXPRESSION_SAFE_APIS and _is_expression_chain(node):
                continue
            yield StaticFinding(
                kind="row_dependent_call",
                file=path,
                line=node.lineno,
                detail=f".{accessor}() depends on there being rows",
            )
        elif isinstance(node, ast.Compare) and _is_length_check(node):
            yield StaticFinding(
                kind="row_dependent_call",
                file=path,
                line=node.lineno,
                detail="comparing a row count takes a different branch on real data",
            )


def _is_expression_chain(node: ast.Call) -> bool:
    """True when a call chain is rooted in a column expression, not a frame."""
    current: ast.expr = node
    while isinstance(current, ast.Call):
        if _selector_name(current) in _EXPRESSION_ROOTS:
            return True
        func = current.func
        if not isinstance(func, ast.Attribute):
            return False
        current = func.value
    return False


def _is_length_check(node: ast.Compare) -> bool:
    """True for `len(df) == 0` and friends, which branch on row count."""
    left = node.left
    return (
        isinstance(left, ast.Call)
        and isinstance(left.func, ast.Name)
        and left.func.id == "len"
        and any(isinstance(comparator, ast.Constant) for comparator in node.comparators)
    )


def _dynamic_columns(tree: ast.Module, path: str) -> Iterator[StaticFinding]:
    """Column references rowless cannot resolve to a name."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and _is_dynamic(node.slice):
            yield StaticFinding(
                kind="dynamic_column",
                file=path,
                line=node.lineno,
                detail="column name is computed, so lineage for it is a gap",
            )
        elif isinstance(node, ast.Call) and _is_dynamic_selector(node):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else "selector"
            yield StaticFinding(
                kind="dynamic_column",
                file=path,
                line=node.lineno,
                detail=f"{name}() is given a computed column name",
            )


def _is_dynamic_selector(node: ast.Call) -> bool:
    """True when a column-taking call is passed something not a literal."""
    if _selector_name(node) not in _COLUMN_SELECTORS:
        return False
    return any(_is_dynamic(argument) for argument in node.args)


def _is_dynamic(node: ast.expr) -> bool:
    """True when an expression is not a literal column name or list of them.

    A variable holding a constant would be resolvable in principle, and
    deliberately is not resolved here: constant propagation across a module is
    exactly the kind of source analysis that breaks on real code.

    The walk only follows positions where a *column name* can appear. It does
    not treat `pl` in `pl.col("a")` as a computed name, and it does not treat
    an expression chain as dynamic just because it contains a call: those two
    false positives would fire on every well-written Polars pipeline, and a
    checker that cries wolf on correct code gets switched off.
    """
    if isinstance(node, ast.Constant):
        return False
    if isinstance(node, ast.JoinedStr | ast.BinOp | ast.Name | ast.Starred):
        return True
    if isinstance(node, ast.List | ast.Tuple | ast.Set):
        return any(_is_dynamic(element) for element in node.elts)
    if isinstance(node, ast.Dict):
        keys = [key for key in node.keys if key is not None]
        return any(_is_dynamic(key) for key in keys) or any(
            _is_dynamic(value) for value in node.values
        )
    if isinstance(node, ast.Call):
        return _call_is_dynamic(node)
    return False


def _call_is_dynamic(node: ast.Call) -> bool:
    """True when a call builds a column reference out of something computed."""
    if _selector_name(node) in _COLUMN_SELECTORS and any(
        _is_dynamic(argument) for argument in node.args
    ):
        return True
    receiver = node.func.value if isinstance(node.func, ast.Attribute) else None
    # Follow a method chain (`pl.col(name).sum()`) but not its root, which is
    # a module or a frame variable rather than a column name.
    return isinstance(receiver, ast.Call) and _call_is_dynamic(receiver)


def _selector_name(node: ast.Call) -> str:
    """The called name, whether it was reached as a method or a function."""
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    return getattr(func, "id", "")
