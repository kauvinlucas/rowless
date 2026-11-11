"""Whether there is anywhere to send events.

The behaviour under test is a refusal. The openlineage client, configured with
nothing, prints events to stdout and reports success; a CI job that was meant
to publish to Marquez would then pass while publishing nothing, which is the
kind of silence this project is built to avoid.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from rowless_openlineage.errors import TransportNotConfigured
from rowless_openlineage.transport import (
    CONFIG_FILENAME,
    ENV_URL,
    client,
    config_paths,
    describe,
    installed,
)


def test_openlineage_is_installed_in_this_environment() -> None:
    """A guard: the rest of the suite would be testing nothing without it."""
    assert installed() is True


def test_nothing_configured_is_not_available() -> None:
    """The console fallback is not a destination."""
    state = describe({})

    assert state.configured is False
    assert ENV_URL in state.reason


def test_a_url_is_a_destination() -> None:
    state = describe({ENV_URL: "http://marquez:5000"})

    assert state.configured is True
    assert state.source == ENV_URL
    assert "http://marquez:5000" in state.reason


def test_disabled_beats_a_configured_url() -> None:
    """`OPENLINEAGE_DISABLED=true` is an answer, and it is reported as one.

    Emitting anyway would ignore an explicit instruction; emitting and claiming
    success would be worse. The reason says which variable decided it, because
    a user who set it months ago in a shell profile needs to be told.
    """
    state = describe({"OPENLINEAGE_DISABLED": "TRUE", ENV_URL: "http://marquez:5000"})

    assert state.configured is False
    assert state.source == "OPENLINEAGE_DISABLED"


def test_structured_environment_configuration_counts() -> None:
    """The transport does not have to be HTTP."""
    state = describe({"OPENLINEAGE__TRANSPORT__TYPE": "kafka"})

    assert state.configured is True
    assert state.source == "OPENLINEAGE__TRANSPORT__TYPE"


def test_an_explicit_console_transport_is_accepted() -> None:
    """What is refused is the fallback nobody chose, not console output."""
    state = describe({"OPENLINEAGE__TRANSPORT__TYPE": "console"})

    assert state.configured is True


def test_a_config_file_counts(tmp_path: Path) -> None:
    """The YAML path every other integration supports."""
    config = tmp_path / CONFIG_FILENAME
    config.write_text("transport:\n  type: console\n", encoding="utf-8")

    state = describe({"OPENLINEAGE_CONFIG": str(config)})

    assert state.configured is True
    assert state.source == str(config)


def test_a_config_path_that_does_not_exist_is_not_configuration(tmp_path: Path) -> None:
    """Pointing at a missing file is the mistake most worth catching early."""
    state = describe({"OPENLINEAGE_CONFIG": str(tmp_path / "absent.yml")})

    assert state.configured is False


def test_config_paths_are_searched_in_the_clients_order() -> None:
    """Mirroring the client means mirroring its precedence."""
    paths = config_paths({"OPENLINEAGE_CONFIG": "/etc/rowless/openlineage.yml"})

    assert paths[0] == Path("/etc/rowless/openlineage.yml")
    assert paths[1].name == CONFIG_FILENAME
    assert paths[2].parent.name == ".openlineage"


def test_client_refuses_rather_than_falling_back() -> None:
    """The refusal names the fix, because that is the whole content of it."""
    with pytest.raises(TransportNotConfigured) as caught:
        client({})

    assert ENV_URL in str(caught.value)


def test_client_builds_from_the_process_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real variables, because the client reads them itself.

    An injected mapping decides whether rowless is willing to emit; it cannot
    reach inside the client, which resolves its own transport from the process
    environment. A test that built a client from a fake mapping would be
    asserting against a console fallback and calling it HTTP.
    """
    monkeypatch.setenv(ENV_URL, "http://localhost:5000")

    built = client()

    assert built.transport.kind == "http"
