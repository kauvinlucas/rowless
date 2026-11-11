"""Is there anywhere to send events, and refusing to pretend there is.

The openlineage client never fails to produce a transport: configured with
nothing at all, it logs a warning and returns a console transport, which prints
the events to stdout. For a library that is a kind choice. For `rowless emit`
it is the wrong one - a CI job that was supposed to publish lineage to Marquez
would print a thousand lines of JSON and exit 0, and nobody would notice for a
month. So this module answers the question the client does not: has somebody
actually configured a destination?

It mirrors the client's own discovery, in the client's own order, which is the
honest way to answer "would the client find a transport":

1. `OPENLINEAGE_DISABLED=true` - configured *off*, which counts as an answer
   and is reported as one rather than treated as missing config.
2. `OPENLINEAGE_URL` - the simple HTTP case.
3. `OPENLINEAGE__*` environment variables - the structured config case.
4. A YAML config file: `OPENLINEAGE_CONFIG`, then `./openlineage.yml`, then
   `~/.openlineage/openlineage.yml`.

A console transport asked for *explicitly* - `type: console` in a config file,
or `OPENLINEAGE__TRANSPORT__TYPE=console` - is a configured transport and is
accepted. What is refused is the fallback nobody chose.

Mirroring rather than calling has one cost worth recording: a future client
release could add a discovery path this module does not know about, and rowless
would report "not configured" for a setup that would in fact work. The
alternative is constructing a client inside `available()`, which both logs that
warning and has to be done during `rowless doctor` on a machine with no
OpenLineage at all. The decision is revisited if the client ever exposes its
config resolution as a function.
"""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rowless_openlineage.errors import TransportNotConfigured

#: Set to `true` to turn OpenLineage off. Honoured by every integration.
ENV_DISABLED = "OPENLINEAGE_DISABLED"

#: The one-variable HTTP setup: `OPENLINEAGE_URL=http://marquez:5000`.
ENV_URL = "OPENLINEAGE_URL"

#: Path to a YAML config file, overriding the searched locations.
ENV_CONFIG = "OPENLINEAGE_CONFIG"

#: Prefix for structured config in the environment, e.g.
#: `OPENLINEAGE__TRANSPORT__TYPE=kafka`.
ENV_PREFIX = "OPENLINEAGE__"

#: Searched, in order, when `OPENLINEAGE_CONFIG` is unset.
CONFIG_FILENAME = "openlineage.yml"

#: What to tell a user who has installed the package and configured nothing.
HOW_TO_CONFIGURE = (
    f"set {ENV_URL} to your OpenLineage endpoint (e.g. http://localhost:5000), "
    f"or write an {CONFIG_FILENAME} in the repository root"
)


@dataclass(frozen=True, slots=True)
class TransportState:
    """Whether events have a destination, and how that was decided."""

    configured: bool
    source: str | None
    """What settled it: an environment variable name, or a config file path."""

    reason: str
    """One line, shown by `rowless doctor` and by `rowless emit` when it refuses."""


def installed() -> bool:
    """True when openlineage-python is importable, without importing it.

    `find_spec` rather than an import: this is called while listing plugins,
    including by `rowless doctor` on a machine with no OpenLineage, and
    importing a client to find out whether it exists is a second of startup
    nobody asked for.
    """
    return importlib.util.find_spec("openlineage.client") is not None


def config_paths(env: Mapping[str, str] | None = None) -> tuple[Path, ...]:
    """Config files the client would look at, in its order."""
    environ = os.environ if env is None else env
    explicit = environ.get(ENV_CONFIG)
    paths: list[Path] = []
    if explicit:
        paths.append(Path(explicit))
    paths.append(Path.cwd() / CONFIG_FILENAME)
    paths.append(Path.home() / ".openlineage" / CONFIG_FILENAME)
    return tuple(paths)


def describe(env: Mapping[str, str] | None = None) -> TransportState:
    """Answer whether a transport is configured, and say what decided it."""
    environ = os.environ if env is None else env

    if environ.get(ENV_DISABLED, "").strip().lower() == "true":
        return TransportState(
            configured=False,
            source=ENV_DISABLED,
            reason=f"{ENV_DISABLED}=true, so nothing would be emitted",
        )
    if not installed():
        return TransportState(
            configured=False,
            source=None,
            reason="openlineage-python is not installed: pip install rowless-openlineage",
        )
    url = environ.get(ENV_URL)
    if url:
        return TransportState(configured=True, source=ENV_URL, reason=f"HTTP to {url}")
    structured = sorted(key for key in environ if key.startswith(ENV_PREFIX))
    if structured:
        return TransportState(
            configured=True,
            source=structured[0],
            reason=f"configured by {len(structured)} {ENV_PREFIX}* variable(s)",
        )
    for path in config_paths(environ):
        if _readable(path):
            return TransportState(configured=True, source=str(path), reason=f"config at {path}")
    return TransportState(
        configured=False,
        source=None,
        reason=f"no OpenLineage transport is configured: {HOW_TO_CONFIGURE}",
    )


def client(env: Mapping[str, str] | None = None) -> Any:
    """Build a client, refusing when nothing is configured.

    `env` decides whether rowless is willing to emit. It does not reach inside
    the client, which resolves its own transport from the process environment:
    an injected mapping is for the decision, and the two agree in production
    because there the mapping *is* `os.environ`.

    Raises:
        TransportNotConfigured: when `describe` says there is no destination.
            Refusing here rather than letting the client fall back to its
            console transport is this module's entire purpose.
    """
    state = describe(env)
    if not state.configured:
        raise TransportNotConfigured(state.reason)
    from openlineage.client import OpenLineageClient

    return OpenLineageClient()


def _readable(path: Path) -> bool:
    """True when the path is a file this process can stat.

    Every filesystem error is a no: this runs inside `available()`, which must
    not raise, and an unreadable config file is indistinguishable from an
    absent one as far as the answer is concerned.
    """
    try:
        return path.is_file()
    except OSError:  # pragma: no cover - permission or path-length failures
        return False
