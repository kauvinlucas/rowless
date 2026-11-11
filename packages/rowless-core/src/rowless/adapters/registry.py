"""Plugin discovery via entry points.

Four registries, same mechanism, so a third-party package extends any of them
without touching core:

    rowless.engines     EngineAdapter    one per dataframe or SQL engine
    rowless.sources     SchemaSource     where input schemas come from
    rowless.discovery   PipelineFinder   how pipelines are located in a repo
    rowless.emitters    lineage emitters OpenLineage and anything like it
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
from importlib.metadata import EntryPoint, entry_points
from typing import Any, Literal, cast

from rowless.adapters.base import EngineAdapter
from rowless.errors import (
    AdapterNotFound,
    AdapterUnavailable,
    AmbiguousAdapter,
    PluginLoadError,
)

Group = Literal[
    "rowless.engines",
    "rowless.sources",
    "rowless.discovery",
    "rowless.emitters",
]


@dataclass(frozen=True, slots=True)
class PluginInfo:
    """A registered plugin, loaded or not.

    Entry points are listed without importing them, so `rowless adapters` can
    show an engine as installed-but-inactive without paying the import cost or
    crashing on a broken plugin.
    """

    name: str
    group: Group
    distribution: str | None
    loaded: bool = False
    error: str | None = None
    """Import or availability failure, shown verbatim by `rowless doctor`."""


#: Instantiated plugins, keyed by (group, name). Plugins are stateless handles
#: around an engine library, and importing polars twice is not free.
_INSTANCES: dict[tuple[str, str], Any] = {}

#: Import failures, remembered so a broken plugin is reported once per run
#: rather than retried on every lookup.
_FAILURES: dict[tuple[str, str], str] = {}


def registered_names(group: Group) -> set[str]:
    """Every plugin name in `group`: entry points plus in-process registrations."""
    names = {point.name for point in _entry_points(group)}
    return names | {plugin for (owner, plugin) in _INSTANCES if owner == group}


def _entry_points(group: Group) -> list[EntryPoint]:
    """Entry points in `group`, sorted by name for stable output."""
    return sorted(entry_points(group=group), key=lambda point: point.name)


def iter_plugins(group: Group) -> Iterator[PluginInfo]:
    """List registered plugins in `group` without importing them."""
    for point in _entry_points(group):
        key = (group, point.name)
        yield PluginInfo(
            name=point.name,
            group=group,
            distribution=point.dist.name if point.dist is not None else None,
            loaded=key in _INSTANCES,
            error=_FAILURES.get(key),
        )


def probe_plugins(group: Group) -> Iterator[PluginInfo]:
    """List plugins in `group`, importing each one to find out if it works.

    What `rowless doctor` and `rowless adapters` call: the whole point is to
    report a broken or inactive plugin as a line of output, so every failure
    here is captured rather than raised.
    """
    for point in _entry_points(group):
        distribution = point.dist.name if point.dist is not None else None
        try:
            plugin = load(group, point.name)
        except PluginLoadError as exc:
            yield PluginInfo(point.name, group, distribution, loaded=False, error=str(exc))
            continue
        available = _available(plugin)
        error = None if available else _why_unavailable(plugin)
        yield PluginInfo(point.name, group, distribution, loaded=available, error=error)


def load(group: Group, name: str) -> Any:
    """Import and instantiate one plugin, raising `PluginLoadError` on failure."""
    key = (group, name)
    if key in _INSTANCES:
        return _INSTANCES[key]
    if key in _FAILURES:
        raise PluginLoadError(f"{group}:{name} failed to load earlier: {_FAILURES[key]}")

    for point in _entry_points(group):
        if point.name != name:
            continue
        try:
            factory = point.load()
            plugin = factory() if isinstance(factory, type) else factory
        except Exception as exc:  # a plugin may fail in any way it likes
            _FAILURES[key] = f"{type(exc).__name__}: {exc}"
            raise PluginLoadError(f"{group}:{name} failed to load: {exc}") from exc
        _INSTANCES[key] = plugin
        return plugin
    raise PluginLoadError(f"{group}:{name} is not registered")


def engines(*, available_only: bool = True) -> dict[str, EngineAdapter]:
    """All engine adapters, keyed by name.

    With `available_only`, adapters whose engine library is missing are
    omitted, which is the normal case: most users have two or three engines
    installed, not nine.

    A plugin that fails to import is skipped rather than fatal. One broken
    adapter must not stop a check that does not need it; `rowless doctor` is
    where the failure is shown.
    """
    found: dict[str, EngineAdapter] = {}
    for name in registered_names("rowless.engines"):
        try:
            adapter = load("rowless.engines", name)
        except PluginLoadError:
            continue
        if available_only and not _available(adapter):
            continue
        found[name] = adapter
    return found


def engine(name: str) -> EngineAdapter:
    """One adapter by name, raising `AdapterNotFound` with install advice."""
    registered = registered_names("rowless.engines")
    if name not in registered:
        known = ", ".join(sorted(registered)) or "none"
        raise AdapterNotFound(f"no adapter named {name!r}; registered adapters: {known}")
    adapter = load("rowless.engines", name)
    if not _available(adapter):
        requires = getattr(adapter, "requires", "")
        hint = f"; install {requires}" if requires else ""
        raise AdapterUnavailable(f"adapter {name!r} is installed but its engine is not{hint}")
    return cast("EngineAdapter", adapter)


def adapter_for(obj: object) -> EngineAdapter:
    """Route an engine object to its adapter via `handles`.

    Raises `AdapterNotFound` when nothing claims it, and `AmbiguousAdapter`
    when two do, which means an adapter's `handles` is too loose.
    """
    claimed = []
    for name, adapter in engines().items():
        try:
            handles = adapter.handles(obj)
        except Exception:  # a broken `handles` is still only one adapter's problem
            continue
        if handles:
            claimed.append((name, adapter))
    if len(claimed) == 1:
        return claimed[0][1]
    if not claimed:
        installed = ", ".join(sorted(engines())) or "none"
        raise AdapterNotFound(
            f"no installed adapter handles {type(obj).__module__}.{type(obj).__qualname__}; "
            f"active adapters: {installed}"
        )
    names = ", ".join(sorted(name for name, _ in claimed))
    raise AmbiguousAdapter(
        f"adapters {names} all claim {type(obj).__qualname__}; one of their `handles` is too loose"
    )


def register(group: Group, name: str, plugin: Any) -> None:
    """Install a plugin for this process only, bypassing entry points.

    For tests and for embedding rowless in another tool. Not a supported way to
    ship an adapter: a real one registers an entry point so that `rowless
    doctor` can see it.
    """
    _INSTANCES[(group, name)] = plugin
    _FAILURES.pop((group, name), None)


def clear_cache() -> None:
    """Drop memoised lookups. Tests install adapters mid-session."""
    _INSTANCES.clear()
    _FAILURES.clear()


def plugin_info(group: Group, name: str) -> PluginInfo:
    """One plugin's status, without importing anything new."""
    for info in iter_plugins(group):
        if info.name == name:
            return info
    return replace(
        PluginInfo(name=name, group=group, distribution=None),
        error="not registered",
    )


def _why_unavailable(plugin: object) -> str:
    """One line for `rowless doctor` saying why a plugin cannot run.

    `requires` names a package, which covers an adapter whose engine is
    missing and nothing else. A plugin whose problem is configuration - an
    emitter with no destination, a source with no credentials - can say so by
    offering `unavailable_reason()`; this prefers that answer when it exists,
    and treats a plugin that raises while explaining itself as one that did
    not.
    """
    explain = getattr(plugin, "unavailable_reason", None)
    if explain is not None:
        try:
            reason = str(explain())
        except Exception:  # pragma: no cover - third-party code
            reason = ""
        if reason:
            return reason
    requires = getattr(plugin, "requires", "")
    return f"requires {requires or 'an engine library'}"


def _available(plugin: object) -> bool:
    """Call a plugin's `available()`, treating a raise as unavailable.

    The protocol says `available` must not raise. Adapters are third-party
    code, so this does not take that on trust.
    """
    checker = getattr(plugin, "available", None)
    if checker is None:
        return True
    try:
        return bool(checker())
    except Exception:  # an adapter that cannot answer is not available
        return False
