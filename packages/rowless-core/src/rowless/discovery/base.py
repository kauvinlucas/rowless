"""Finding the pipelines. Hard problem 04.

`rowless check` with no arguments has to work, or nobody adopts it. That means
locating entry points in somebody else's repo: Dagster assets, Airflow tasks,
dlt resources, or plain annotated functions.

Finders must not execute pipeline bodies, only import enough to read
decorators and annotations. Importing an orchestrator module can itself have
side effects, so discovery runs in the same sandboxed worker as a dry run.

Registered under the `rowless.discovery` entry-point group.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Protocol, runtime_checkable

from rowless.adapters.registry import load, registered_names
from rowless.errors import PluginLoadError

#: Directories no repository keeps its pipelines in, pruned before a finder
#: opens anything. Without this, `rowless check` in a project with a virtual
#: environment in it spends its time parsing other people's libraries, which
#: is both slow and a source of findings nobody can act on.
PRUNED_DIRECTORIES: frozenset[str] = frozenset(
    {
        ".bzr",
        ".direnv",
        ".eggs",
        ".git",
        ".hg",
        ".mypy_cache",
        ".nox",
        ".pytest_cache",
        ".ruff_cache",
        ".svn",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "site-packages",
        "venv",
    }
)


def python_files(roots: Sequence[str]) -> Iterator[Path]:
    """Every `.py` file under `roots`, pruned and de-duplicated.

    Shared by every finder, so "where rowless looks" is one answer rather than
    four, and a file named twice (a root inside another root) is parsed once.
    """
    seen: set[Path] = set()
    for root in roots:
        base = Path(root)
        if base.is_file():
            candidates: Iterable[Path] = [base]
        else:
            candidates = _walk(base)
        for candidate in candidates:
            if candidate.suffix != ".py":
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            yield candidate


def _walk(directory: Path) -> Iterator[Path]:
    """Recurse into `directory`, skipping the pruned names entirely."""
    try:
        entries = sorted(directory.iterdir())
    except OSError:
        return
    for entry in entries:
        if entry.is_dir():
            if entry.name in PRUNED_DIRECTORIES:
                continue
            yield from _walk(entry)
        elif entry.is_file():
            yield entry


@dataclass(frozen=True, slots=True)
class Pipeline:
    """One discovered entry point."""

    target: str
    """Importable reference: `pipelines.revenue:daily_revenue`."""

    inputs: tuple[tuple[str, str | None], ...] = ()
    """Parameter name paired with a source URI where the framework declares
    one (a Dagster asset key, a dlt resource). `None` means rowless has to be
    told, and the diagnostic says so instead of skipping the pipeline."""

    engine: str | None = None
    """Adapter inferred from annotations, when they are present."""

    framework: str = "plain"
    file: str | None = None
    line: int | None = None

    tags: tuple[str, ...] = ()
    """Framework metadata, used by `--select` filters."""

    @property
    def name(self) -> str:
        """The function name alone, for `--select` globs and terminal output."""
        return self.target.rpartition(":")[2] or self.target

    @property
    def unbound(self) -> tuple[str, ...]:
        """Parameters with no source URI. Reported, never silently skipped."""
        return tuple(name for name, uri in self.inputs if uri is None)


@runtime_checkable
class PipelineFinder(Protocol):
    """Locates pipelines in a repository."""

    name: ClassVar[str]
    requires: ClassVar[str]

    def available(self) -> bool:
        """True when the framework is importable, or its config files exist."""
        ...

    def find(self, roots: Sequence[str]) -> Iterable[Pipeline]:
        """Yield entry points under `roots`, without running any of them."""
        ...


def discover(
    roots: Sequence[str],
    *,
    finders: Sequence[str] | None = None,
    exclude: Sequence[str] = (),
) -> list[Pipeline]:
    """Run every available finder and merge the results.

    Deduplicates by target, preferring the framework-aware finder over the
    plain one, since the framework knows the input bindings.

    A finder that raises is skipped rather than fatal: one broken framework
    integration must not hide the pipelines every other finder located.
    """
    wanted = set(finders) if finders else None
    found: dict[str, Pipeline] = {}
    for name in sorted(registered_names("rowless.discovery")):
        if wanted is not None and name not in wanted:
            continue
        try:
            finder = load("rowless.discovery", name)
        except PluginLoadError:
            continue
        try:
            if not finder.available():
                continue
            discovered = list(finder.find(roots))
        except Exception:  # a finder's framework can fail in any way it likes
            continue
        for pipeline in discovered:
            if _is_excluded(pipeline, exclude):
                continue
            existing = found.get(pipeline.target)
            if existing is None or _prefer(pipeline, existing):
                found[pipeline.target] = pipeline
    return sorted(found.values(), key=lambda pipeline: (pipeline.file or "", pipeline.line or 0))


def _is_excluded(pipeline: Pipeline, exclude: Sequence[str]) -> bool:
    """True when a discovered pipeline matches one of the exclude globs."""
    if pipeline.file is None or not exclude:
        return False
    text = Path(pipeline.file).as_posix()
    for pattern in exclude:
        cleaned = pattern.lstrip("/")
        anywhere = f"*/{cleaned.removeprefix('**/')}"
        if fnmatch.fnmatch(text, cleaned) or fnmatch.fnmatch(text, anywhere):
            return True
    return False


def _prefer(candidate: Pipeline, existing: Pipeline) -> bool:
    """True when `candidate` is the better record of the same pipeline.

    Framework knowledge wins, because an asset key is a real input binding and
    an annotation is only a type. Between two equals, the one with fewer
    unbound inputs wins.
    """
    if (candidate.framework != "plain") != (existing.framework != "plain"):
        return candidate.framework != "plain"
    return len(candidate.unbound) < len(existing.unbound)


def module_name_for(path: Path | str) -> str:
    """The importable module name for a file.

    Walks up while `__init__.py` is present, so `pipelines/revenue.py` in a
    package becomes `pipelines.revenue` and a loose script becomes `revenue`.
    The worker puts the resulting root on `sys.path`, which is what lets
    rowless import a repository that was never installed.
    """
    file = Path(path)
    parts = [file.stem]
    directory = file.parent
    while (directory / "__init__.py").is_file():
        parts.append(directory.name)
        directory = directory.parent
    return ".".join(reversed(parts))


def import_root_for(path: Path | str) -> Path:
    """The directory to put on `sys.path` so `module_name_for(path)` imports."""
    file = Path(path).resolve()
    directory = file.parent
    while (directory / "__init__.py").is_file():
        directory = directory.parent
    return directory
