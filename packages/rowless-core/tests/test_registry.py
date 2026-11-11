"""Plugin discovery, with no engines installed."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pytest
from rowless.adapters.registry import (
    PluginInfo,
    adapter_for,
    clear_cache,
    engine,
    engines,
    iter_plugins,
    load,
    probe_plugins,
    register,
)
from rowless.errors import AdapterNotFound, AdapterUnavailable, PluginLoadError


@pytest.fixture(autouse=True)
def _clean_registry() -> None:
    """Each test starts from a cold registry; lookups are memoised."""
    clear_cache()


def test_lists_plugins_without_importing() -> None:
    """Entry points are listed without importing their modules.

    Checked in a fresh interpreter, because by the time this test runs in a
    full suite half the plugin modules are legitimately imported already, and
    unloading them would hand other tests a different module object than the
    one they are holding.
    """
    program = """
import json, sys
from rowless.adapters.registry import iter_plugins

names = sorted(info.name for info in iter_plugins("rowless.sources"))
loaded = [info.name for info in iter_plugins("rowless.sources") if info.loaded]
print(json.dumps({
    "names": names,
    "loaded": loaded,
    "imported": "rowless.sources.literal" in sys.modules,
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(program)],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    observed = json.loads(completed.stdout)

    assert set(observed["names"]) >= {"json", "parquet"}
    assert observed["loaded"] == []
    assert not observed["imported"], "listing must not import a plugin's module"
    assert all(isinstance(info, PluginInfo) for info in iter_plugins("rowless.sources"))


def test_unavailable_adapter_is_reported_not_raised() -> None:
    """A missing engine library yields an inactive adapter plus a reason."""

    class Absent:
        name = "absent"
        requires = "someengine>=1.0"

        def available(self) -> bool:
            return False

    register("rowless.engines", "absent", Absent())

    assert "absent" not in engines(), "an inactive adapter is not handed to the runtime"
    assert "absent" in engines(available_only=False)
    with pytest.raises(AdapterUnavailable, match=r"someengine>=1\.0"):
        engine("absent")


def test_broken_plugin_does_not_break_the_registry() -> None:
    """One plugin that fails to import leaves the others usable."""

    class Exploding:
        name = "exploding"
        requires = ""

        def available(self) -> bool:
            raise RuntimeError("engine import blew up")

        def handles(self, obj: object) -> bool:
            raise RuntimeError("engine import blew up")

    class Working:
        name = "working"
        requires = ""

        def available(self) -> bool:
            return True

        def handles(self, obj: object) -> bool:
            return isinstance(obj, str)

    register("rowless.engines", "exploding", Exploding())
    register("rowless.engines", "working", Working())

    active = engines()

    assert "working" in active
    assert "exploding" not in active, "an adapter that cannot answer is not available"
    assert adapter_for("a string").name == "working"


def test_probe_reports_why_a_plugin_is_inactive() -> None:
    """`rowless doctor` needs the reason, not just the status."""
    statuses = {info.name: info for info in probe_plugins("rowless.sources")}

    assert statuses["json"].loaded
    assert statuses["pandera"].error, "an inactive source says what it needs"


def test_a_plugin_can_explain_itself_better_than_requires_can(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A requirement names a package; the usual blocker is configuration.

    An emitter with nowhere to send events, or a source with no credentials, is
    installed and still unusable. `requires` cannot say that, so a plugin may
    offer `unavailable_reason()` and `rowless doctor` prefers it.
    """
    from rowless.adapters.registry import _why_unavailable

    class Configurable:
        requires = "nothing"

        def unavailable_reason(self) -> str:
            return "no transport configured: set OPENLINEAGE_URL"

    class Mute:
        requires = "polars>=1.0"

    class Rude:
        requires = "polars>=1.0"

        def unavailable_reason(self) -> str:
            raise RuntimeError("no")

    assert _why_unavailable(Configurable()) == "no transport configured: set OPENLINEAGE_URL"
    assert _why_unavailable(Mute()) == "requires polars>=1.0"
    assert _why_unavailable(Rude()) == "requires polars>=1.0"


def test_adapter_for_unknown_object_raises() -> None:
    """An object nothing claims raises AdapterNotFound, naming the installed set."""
    with pytest.raises(AdapterNotFound) as caught:
        adapter_for(object())

    assert "active adapters" in str(caught.value)


def test_unregistered_plugin_name_fails_clearly() -> None:
    """A typo in an engine name should not look like a missing install."""
    with pytest.raises(AdapterNotFound, match="registered adapters"):
        engine("polrs")
    with pytest.raises(PluginLoadError, match="not registered"):
        load("rowless.sources", "nonexistent")
