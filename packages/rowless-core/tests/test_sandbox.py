"""Sandbox behaviour. Run in a subprocess: audit hooks cannot be uninstalled."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from rowless.runtime.sandbox import (
    NETWORK_EVENTS,
    WRITE_EVENTS,
    Action,
    Policy,
    audit_events_denied_by,
    sandboxed,
)

#: Each case runs in its own interpreter. An audit hook is permanent, so a test
#: that installed one would change every test that ran after it.
_PROGRAM = """
import json, socket, sys
from pathlib import Path
from rowless.errors import SideEffectBlocked
from rowless.runtime.sandbox import Policy, sandboxed

scratch = sys.argv[1]
outside = sys.argv[2]
result = {}
policy = Policy.strict().allowing_writes_to(scratch)
with sandboxed(policy) as log:
    try:
        with open(Path(outside, "escaped.txt"), "w", encoding="utf-8") as handle:
            handle.write("nope")
        result["outside_write"] = "allowed"
    except SideEffectBlocked as exc:
        result["outside_write"] = "blocked"

    try:
        with open(Path(scratch, "scratch.txt"), "w", encoding="utf-8") as handle:
            handle.write("fine")
        result["scratch_write"] = "allowed"
    except SideEffectBlocked:
        result["scratch_write"] = "blocked"

    try:
        socket.create_connection(("example.invalid", 80), timeout=0.1)
        result["network"] = "allowed"
    except SideEffectBlocked:
        result["network"] = "blocked"
    except OSError:
        result["network"] = "errored"

    result["blocked_log"] = list(log.blocked)
    result["writes_log"] = list(log.writes)
print(json.dumps(result))
"""


def _run_sandboxed(scratch: Path, outside: Path) -> dict[str, object]:
    """Run the probe program in a fresh interpreter and return its findings."""
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_PROGRAM), str(scratch), str(outside)],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    observed: dict[str, object] = json.loads(completed.stdout)
    return observed


@pytest.fixture(scope="module")
def probed(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """One subprocess, several assertions: starting an interpreter is not free."""
    scratch = tmp_path_factory.mktemp("scratch")
    outside = tmp_path_factory.mktemp("outside")
    return _run_sandboxed(scratch, outside)


def test_blocks_network_connect(probed: dict[str, object]) -> None:
    """A socket connect inside the sandbox raises SideEffectBlocked."""
    assert probed["network"] == "blocked"


def test_blocks_file_write_outside_scratch(probed: dict[str, object]) -> None:
    """Opening a file for write outside the scratch directory is blocked."""
    assert probed["outside_write"] == "blocked"


def test_allows_write_inside_scratch(probed: dict[str, object]) -> None:
    """The scratch directory stays writable; adapters need it."""
    assert probed["scratch_write"] == "allowed"


def test_records_blocked_attempts(probed: dict[str, object]) -> None:
    """Blocked attempts land in the side-effect log, which doubles as output lineage."""
    blocked = probed["blocked_log"]
    assert isinstance(blocked, list)
    assert any("escaped.txt" in str(entry) for entry in blocked)
    assert any("example.invalid" in str(entry) for entry in blocked)

    written = probed["writes_log"]
    assert isinstance(written, list)
    assert any("scratch.txt" in str(entry) for entry in written)


def test_allowlisted_host_permitted_for_remote_adapter() -> None:
    """A Spark Connect endpoint is reachable when allowlisted, and nothing else is."""
    program = """
import json, socket, sys
from rowless.errors import SideEffectBlocked
from rowless.runtime.sandbox import Policy, sandboxed

result = {}
policy = Policy.strict().allowing_hosts("spark.internal")
with sandboxed(policy) as log:
    for host in ("spark.internal", "example.invalid"):
        try:
            socket.create_connection((host, 15002), timeout=0.05)
            result[host] = "allowed"
        except SideEffectBlocked:
            result[host] = "blocked"
        except OSError:
            result[host] = "attempted"
    result["log"] = list(log.network)
print(json.dumps(result))
"""
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(program)],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)

    assert result["spark.internal"] == "attempted", "the allowlisted host was not blocked"
    assert result["example.invalid"] == "blocked"
    assert any("spark.internal" in entry for entry in result["log"])


def test_denied_events_are_documented_data() -> None:
    """The docs state exactly what is blocked, so it has to be readable data."""
    denied = set(audit_events_denied_by(Policy.strict()))

    assert "open" in denied
    assert "socket.connect" in denied
    assert "subprocess.Popen" in denied
    assert set(WRITE_EVENTS) <= denied
    assert set(NETWORK_EVENTS) <= denied

    permissive = set(audit_events_denied_by(Policy.permissive()))
    assert not permissive, "permissive mode records instead of denying"
    assert Policy.permissive().filesystem_write is Action.RECORD


def test_nesting_a_sandbox_is_an_error() -> None:
    """The second policy would silently fail to narrow the first."""
    program = """
from rowless.errors import SideEffectBlocked
from rowless.runtime.sandbox import Policy, sandboxed

with sandboxed(Policy.strict()):
    try:
        with sandboxed(Policy.permissive()):
            print("nested")
    except SideEffectBlocked:
        print("refused")
"""
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(program)],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert completed.stdout.strip() == "refused", completed.stderr


def test_sandbox_is_usable_from_the_same_process_sequentially() -> None:
    """Sequential entry is fine: the hook is permanent, the session is not."""
    with sandboxed(Policy.permissive()) as first:
        first.writes.append("marker")
    with sandboxed(Policy.permissive()) as second:
        assert second.writes == [], "a new session starts with an empty log"
