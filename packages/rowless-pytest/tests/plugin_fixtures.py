"""Fixtures both test modules need, registered as a plugin rather than a conftest.

Two files named `conftest` in this workspace are one module name to
`mypy packages`, and rowless-core already owns the top of that namespace;
rowless-dbt owns the `tests.conftest` spelling that a package-shaped test
directory would produce. So the shared fixtures live under a name nothing else
uses, and each test module pulls them in with

    pytest_plugins = ["plugin_fixtures"]

which registers the fixtures properly instead of importing their names, so
nothing here shadows a test's own parameter.

Both fixtures hide the real engine entry points, for the reason core's
`conftest` gives: this package's CI cell installs no engine at all, a
contributor's workspace installs nine, and a suite whose answers depend on
which is testing the virtualenv rather than the plugin.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from rowless_fake_pytest import FakeAdapter, install


@pytest.fixture
def fake_engine(no_engines: None) -> Iterator[FakeAdapter]:
    """The fake adapters, registered for one test, and the only ones installed.

    `fake` is usable and `fake_absent` is not, which is the pair the skip
    story needs. With nothing else registered, a test can leave the engine
    unnamed and still get a deterministic answer.
    """
    _ = no_engines
    from rowless.adapters import registry

    yield install()
    registry.clear_cache()


@pytest.fixture
def no_engines(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """An environment with no engine adapter registered at all.

    What a first install looks like, and the only way to test the message
    that tells somebody to go and install one.
    """
    from rowless.adapters import registry

    original = registry._entry_points

    def in_process_engines_only(group: str) -> list[Any]:
        return [] if group == "rowless.engines" else list(original(group))  # type: ignore[arg-type]

    monkeypatch.setattr(registry, "_entry_points", in_process_engines_only)
    registry.clear_cache()
    yield
    registry.clear_cache()
