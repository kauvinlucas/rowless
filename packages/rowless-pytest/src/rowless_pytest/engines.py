"""Which engine a test runs on, and whether it is installed.

Two questions, answered here rather than inside the fixtures because the
collection hook needs the second one before any fixture exists: a test that
needs Polars, on a machine that has only pandas, is skipped - and a skip has
to be decided while items are being collected.

Choosing is deliberately unclever. rowless refuses to guess a schema, and an
engine is a bigger guess than a dtype: running a pandas pipeline through the
Polars adapter does not fail with "wrong engine", it fails with something
confusing about `select`. So the order is

1. the `engine=` argument at the call site,
2. the pipeline's own `@rowless.pipeline(engine=...)`,
3. the test's `@pytest.mark.rowless(engine=...)`,
4. `--rowless-engine`,
5. a single configured engine, or a single installed one,

and where more than one remains possible, the fixture raises and names them
rather than picking.

Two of those look out of order and are not. The pipeline's declaration beats
the test's marker because it is a fact about the code under test rather than
about the test - a pipeline that says it is Polars is Polars, whoever is
asserting on it. And a marker beats the command-line flag because the flag is
how a suite is pointed at an engine it names nowhere, while a marker is a
statement about *this* test, whose body is written against that engine's API.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
from rowless.adapters import registry
from rowless.errors import AdapterNotFound, AdapterUnavailable, PluginLoadError

#: The marker this plugin registers: `@pytest.mark.rowless(engine="polars")`.
MARKER = "rowless"

MARKER_HELP = (
    "rowless(engine='polars'): a rowless contract test. `engine` names the "
    "adapter the test needs, which is used for the rowless fixtures and "
    "skips the test when that engine is not installed."
)

#: Keyword arguments the marker accepts. A typo here would silently stop the
#: test being gated, so an unknown one is an error rather than ignored.
MARKER_KWARGS = frozenset({"engine"})


def declared_engine(node: pytest.Item) -> str | None:
    """The engine `@pytest.mark.rowless(engine=...)` names, if any.

    `get_closest_marker` rather than `own_markers`, so a marker on the class
    or module covers every test under it - which is how a file of Polars
    contract tests is written.
    """
    marker = node.get_closest_marker(MARKER)
    if marker is None:
        return None
    engine = marker.kwargs.get("engine")
    if engine is None and marker.args:
        engine = marker.args[0]
    return str(engine) if engine else None


def marker_problem(node: pytest.Item) -> str | None:
    """A complaint about how the marker was written, or None when it is fine."""
    marker = node.get_closest_marker(MARKER)
    if marker is None:
        return None
    unknown = sorted(set(marker.kwargs) - MARKER_KWARGS)
    if unknown:
        accepted = ", ".join(sorted(MARKER_KWARGS))
        return (
            f"@pytest.mark.rowless got unknown argument(s) {', '.join(unknown)}; takes {accepted}"
        )
    if len(marker.args) > 1:
        return "@pytest.mark.rowless takes one engine name, not several"
    return None


def unavailable(name: str) -> str | None:
    """Why engine `name` cannot be used here, or None when it can.

    The answer the collection hook skips on. It imports the adapter, because
    "installed" and "usable" are different questions - `rowless-pyspark` is a
    pure-Python wheel that is useless without a JVM, and the registry is the
    only thing that knows the difference.
    """
    try:
        registry.engine(name)
    except (AdapterNotFound, AdapterUnavailable, PluginLoadError) as exc:
        return str(exc)
    return None


def installed() -> tuple[str, ...]:
    """Every engine adapter that is importable and whose engine is present."""
    return tuple(sorted(registry.engines()))


def adapter_for(name: str) -> Any:
    """The adapter object for one engine name."""
    return registry.engine(name)


def choose(
    *,
    requested: str | None = None,
    pipeline: str | None = None,
    declared: str | None = None,
    flag: str | None = None,
    configured: Sequence[str] = (),
) -> str | None:
    """The engine a test asked for, or None when nothing named one.

    `requested` is the call site, `pipeline` the `@rowless.pipeline` marker on
    the code under test, `declared` the `@pytest.mark.rowless` marker on the
    test, `flag` the command line, `configured` the `engines` key in
    `rowless.toml`.

    None is a real answer rather than a failure: `rowless_check` hands it to
    the runtime, which infers the adapter from the pipeline's annotations -
    the same inference `rowless check` makes, so a test agrees with CI about
    which engine a pipeline belongs to.
    """
    for candidate in (requested, pipeline, declared, flag):
        if candidate:
            return candidate
    if len(configured) == 1:
        return configured[0]
    return None


def require(chosen: str | None) -> str:
    """A concrete engine name for a fixture that has no pipeline to infer from.

    `rowless_frame` has to pick an adapter with nothing to go on but the
    environment, so a single installed engine is used and anything else is an
    error that lists the alternatives. Silently taking the first of four would
    make a passing test mean nothing in particular.
    """
    if chosen is not None:
        return chosen
    available = installed()
    if len(available) == 1:
        return available[0]
    if not available:
        raise AdapterNotFound(
            "no engine adapter is installed, so there is no engine to build a rowless frame "
            "with; install one, e.g. `pip install rowless-polars`"
        )
    raise AdapterNotFound(
        f"more than one engine is installed ({', '.join(available)}), so rowless will not "
        "choose for you; name one with @pytest.mark.rowless(engine=...), the engine= argument, "
        "or --rowless-engine"
    )
