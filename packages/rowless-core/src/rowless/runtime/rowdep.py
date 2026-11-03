"""Detecting logic that depends on there being rows. Hard problem 02.

Zero rows is a legitimate input, but it is not a *representative* one:

    if df.is_empty():          # takes the other branch on real data
        return fallback()
    threshold = df["x"].max()  # None here, a number in production
    for row in df.iter_rows(): # loop body never runs

Any of these means the schema rowless computed is the schema of a code path
that production would not take. The result must be marked
`Confidence.UNKNOWN`, with the line that caused it, and never reported as a
pass. Guessing is worse than abstaining: a tool that is confidently wrong about
a branch is a tool nobody can trust in CI.

Detection is two-sided, because neither side is sufficient alone:

* **Dynamic**, here: adapters route row-count-sensitive accessors through
  `observe`, which records the call and its stack frame. Catches real calls
  including ones inside library code, but only on the path taken.
* **Static**, in `rowless.analysis`: an AST pass over the target's module
  flags the same accessors on paths the dry run did not reach.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from types import FrameType

#: Package prefixes whose frames are rowless's own plumbing. A diagnostic has
#: to point at the user's line, not at the adapter that noticed.
_INTERNAL_PREFIXES: tuple[str, ...] = ("rowless", "rowless_")

#: Directory of the rowless package, used to recognise its frames by file as
#: well as by module name: an adapter calling in from a re-exported alias
#: still has a rowless file on the stack.
_PACKAGE_ROOT = str(Path(__file__).resolve().parent.parent)


@dataclass(frozen=True, slots=True)
class RowDependency:
    """One row-count-sensitive operation that was executed."""

    api: str
    """Qualified accessor: `polars.DataFrame.is_empty`, `pandas.Series.max`."""

    file: str | None = None
    line: int | None = None

    returned: str | None = None
    """What rowless handed back (`None`, `0`, `False`), so the diagnostic can
    explain which branch was taken and why."""

    @property
    def explanation(self) -> str:
        """The sentence shown under the diagnostic."""
        where = f" at {self.file}:{self.line}" if self.file else ""
        got = f" and got {self.returned}" if self.returned is not None else ""
        return (
            f"{self.api} depends on there being rows{where}{got}; "
            "production would take a different path, so this schema is not verified"
        )


@dataclass(slots=True)
class RowDependencyLog:
    """Everything observed during one dry run."""

    observed: list[RowDependency] = field(default_factory=list)

    @property
    def taints_result(self) -> bool:
        """True when the schema must be downgraded to UNKNOWN."""
        return bool(self.observed)

    @property
    def reasons(self) -> tuple[str, ...]:
        """One explanation per distinct accessor, for `rowless explain`."""
        return tuple(dict.fromkeys(entry.explanation for entry in self.observed))


#: Accessors whose value depends on row count. Adapters extend this with their
#: own; keeping the canonical list in core means a user reading one diagnostic
#: can look up every case rowless knows about.
ROW_DEPENDENT_APIS: tuple[str, ...] = (
    "is_empty",
    "item",
    "to_list",
    "iter_rows",
    "itertuples",
    "iterrows",
    "first",
    "head",
    "max",
    "min",
    "sum",
    "mean",
    "n_unique",
    "unique",
    "value_counts",
    "collect",
    "compute",
    "fetchall",
    "toPandas",
)

#: The log the current dry run is recording into. A context variable rather
#: than a module global: the pytest plugin runs dry runs in the test process,
#: and one test's observations must not leak into the next.
_ACTIVE: ContextVar[RowDependencyLog | None] = ContextVar("rowless_rowdep", default=None)


@contextmanager
def recording() -> Iterator[RowDependencyLog]:
    """Collect row dependencies observed inside the block."""
    log = RowDependencyLog()
    token = _ACTIVE.set(log)
    try:
        yield log
    finally:
        _ACTIVE.reset(token)


def observe(api: str, returned: object = None) -> None:
    """Called by adapters when user code touches a row-dependent accessor.

    Deliberately a plain function, not a decorator: adapters call it from
    inside their own wrappers, where they know which engine API was hit.

    A no-op outside a dry run, so an adapter can call it unconditionally
    without checking whether rowless is driving.
    """
    log = _ACTIVE.get()
    if log is None:
        return
    file, line = user_frame()
    log.observed.append(
        RowDependency(
            api=api,
            file=file,
            line=line,
            returned=None if returned is None else repr(returned),
        )
    )


def active_log() -> RowDependencyLog | None:
    """The log currently recording, if any. For adapters that batch up calls."""
    return _ACTIVE.get()


def user_frame_from_traceback(tb: object) -> tuple[str | None, int | None]:
    """The deepest frame in a traceback that is neither rowless nor an adapter.

    A diagnostic about a pipeline has to point at the pipeline. The engine
    raises inside its own code and rowless re-raises inside its own, so the
    line worth quoting is the deepest one in between.
    """
    found: tuple[str | None, int | None] = (None, None)
    current = tb
    while current is not None:
        frame = getattr(current, "tb_frame", None)
        if frame is None:
            break
        name = frame.f_globals.get("__name__", "")
        filename = frame.f_code.co_filename
        internal = (
            name.startswith(_INTERNAL_PREFIXES)
            or filename.startswith(_PACKAGE_ROOT)
            or filename.startswith("<")
            or "importlib" in filename
        )
        if not internal:
            found = (filename, getattr(current, "tb_lineno", None))
        current = getattr(current, "tb_next", None)
    return found


def user_frame() -> tuple[str | None, int | None]:
    """The nearest stack frame that is neither rowless nor an adapter.

    Walks out of rowless's own frames so a diagnostic points at the line a
    user can fix. Falls back to the innermost frame rather than reporting
    nothing: a location in the wrong file is still better than a finding with
    no location at all, and it is visibly wrong rather than silently missing.
    """
    frame: FrameType | None = sys._getframe(1)
    innermost: tuple[str | None, int | None] = (None, None)
    while frame is not None:
        name = frame.f_globals.get("__name__", "")
        filename = frame.f_code.co_filename
        if innermost == (None, None):
            innermost = (filename, frame.f_lineno)
        internal = name.startswith(_INTERNAL_PREFIXES) or filename.startswith(_PACKAGE_ROOT)
        if not internal:
            return filename, frame.f_lineno
        frame = frame.f_back
    return innermost
