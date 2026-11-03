"""Breaking-change analysis: `rowless diff <ref>`.

Build the graph on both revisions, diff the schemas, then walk downstream from
every change. Two revisions means two checkouts, which is why the graph is
serialisable: the base revision is built once in a git worktree and cached.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from rowless.errors import RowlessError
from rowless.ir.diff import Change, SchemaDiff, Severity, diff_schemas, worst
from rowless.ir.schema import Schema
from rowless.lineage.graph import ColumnRef, LineageGraph


@dataclass(frozen=True, slots=True)
class Impact:
    """One change plus everything it reaches."""

    change: Change
    relation: str
    affected: tuple[ColumnRef, ...] = ()

    unverified_paths: tuple[str, ...] = ()
    """Paths that pass through an opaque node, so the true radius may be
    larger. Reported explicitly: a blast radius that looks small because
    rowless lost the trail is the one thing that would make this tool
    dangerous to trust."""


@dataclass(frozen=True, slots=True)
class BlastRadius:
    """The full `rowless diff` result."""

    base_ref: str
    head_ref: str
    impacts: tuple[Impact, ...] = ()
    diffs: tuple[SchemaDiff, ...] = ()

    @property
    def severity(self) -> Severity:
        """The worst severity across all impacts."""
        return worst(impact.change.severity for impact in self.impacts)

    @property
    def is_breaking(self) -> bool:
        """True when the gate should fail."""
        return any(diff.is_breaking for diff in self.diffs)


def compare(
    base: LineageGraph,
    head: LineageGraph,
    *,
    base_ref: str,
    head_ref: str,
    detect_renames: bool = True,
) -> BlastRadius:
    """Diff two graphs and attach downstream impact to each change.

    Impact is read off the **base** graph. The consumers that break are the
    ones that were reading the column before the change; in the head graph
    those edges are already gone, which would make a breaking change look
    like it affects nothing.
    """
    diffs: list[SchemaDiff] = []
    impacts: list[Impact] = []
    for node_id in sorted(set(base.nodes) & set(head.nodes)):
        before = base.nodes[node_id].schema
        after = head.nodes[node_id].schema
        diff = diff_schemas(before, after, detect_renames=detect_renames)
        if not diff.changes:
            continue
        diffs.append(SchemaDiff(relation=node_id, changes=diff.changes))
        for change in diff.changes:
            if change.severity is Severity.COMPATIBLE:
                # Additive changes have no blast radius worth walking: nothing
                # downstream could have been reading a column that is new.
                impacts.append(Impact(change=change, relation=node_id))
                continue
            ref = ColumnRef(node_id, change.column)
            affected = tuple(base.downstream(ref))
            impacts.append(
                Impact(
                    change=Change(
                        kind=change.kind,
                        severity=change.severity,
                        column=change.column,
                        before=change.before,
                        after=change.after,
                        renamed_to=change.renamed_to,
                        affects=tuple(_describe(base, found) for found in affected),
                    ),
                    relation=node_id,
                    affected=affected,
                    unverified_paths=base.paths_through_opaque(ref),
                )
            )

    for node_id in sorted(set(base.nodes) - set(head.nodes)):
        # A relation that no longer exists is every bit as breaking as a
        # column that does not, and is easy to miss when only shared nodes
        # are compared.
        removed = diff_schemas(base.nodes[node_id].schema, Schema.empty())
        diffs.append(SchemaDiff(relation=node_id, changes=removed.changes))
        impacts.extend(Impact(change=change, relation=node_id) for change in removed.changes)

    return BlastRadius(
        base_ref=base_ref,
        head_ref=head_ref,
        impacts=tuple(impacts),
        diffs=tuple(diffs),
    )


def _describe(graph: LineageGraph, ref: ColumnRef) -> str:
    """One blast-radius line: the consumer, its language and its engine.

    Matches the proposal's tree: a reader needs to know whether the thing that
    breaks is Python or SQL before they know who to ask.
    """
    node = graph.nodes.get(ref.node)
    if node is None:
        return str(ref)
    engine = f", {node.engine}" if node.engine else ""
    return f"{ref} ({node.kind.value}{engine})"


def build_base_graph(
    ref: str,
    *,
    cache_dir: str = ".rowless",
    builder: Callable[[Path], LineageGraph] | None = None,
) -> LineageGraph:
    """Build (or load) the graph for a git ref.

    Uses `git worktree add` into a temp directory rather than stashing, so the
    user's working tree is never touched. Cached by commit sha, since the base
    graph is identical across every PR run on the same base.
    """
    sha = resolve_sha(ref)
    cached = Path(cache_dir, "graphs", f"{sha}.json")
    if cached.is_file():
        try:
            return LineageGraph.from_json(cached.read_text(encoding="utf-8"))
        except (ValueError, KeyError):
            # A graph written by an older rowless is a cache miss, not a
            # failure: rebuilding costs one checkout.
            pass

    build = builder or _default_builder
    with _worktree(sha) as tree:
        graph = build(tree)
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(graph.to_json(), encoding="utf-8")
    return graph


def resolve_sha(ref: str) -> str:
    """The commit sha a ref points at, so the cache key is exact."""
    completed = _git("rev-parse", ref)
    return completed.strip()


@contextmanager
def _worktree(sha: str) -> Iterator[Path]:
    """A detached worktree at `sha`, removed afterwards."""
    with TemporaryDirectory(prefix="rowless-base-") as directory:
        tree = Path(directory, "tree")
        _git("worktree", "add", "--detach", "--quiet", str(tree), sha)
        try:
            yield tree
        finally:
            # `--force` because the dry run may have left a cache directory
            # behind, and a leaked worktree registration confuses the next run
            # more than a forced removal does.
            _git("worktree", "remove", "--force", str(tree))


def _git(*args: str) -> str:
    """Run git, raising `RowlessError` with stderr when it fails."""
    completed = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RowlessError(f"git {' '.join(args)} failed: {detail}")
    return completed.stdout.strip()


def _default_builder(tree: Path) -> LineageGraph:
    """Build a graph for a checkout, by running the normal check over it.

    Imported late because the check layer reaches back into lineage, and the
    base graph is the only caller that needs both halves at once.
    """
    from rowless.check import build_graph
    from rowless.config import Config

    config = Config.discover(str(tree))
    return build_graph(config)
