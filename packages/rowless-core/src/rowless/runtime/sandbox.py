"""Side-effect containment. Hard problem 01.

A dry run executes code that was written to move data. It will try to open
sockets, write Parquet, drop tables and post to Slack. None of that may
happen, and "the user should pass a dry-run flag" is not a defence.

Layers, from cheapest to strongest:

1. `sys.addaudithook` denies `socket.connect`, `open` for write, `subprocess`
   and `os.remove`, outside an explicit allowlist. Audit hooks cannot be
   removed once installed, which is why they are the first line rather than a
   monkeypatch.
2. Library-boundary interception: an adapter's writer entry points
   (`write_parquet`, `to_sql`, `saveAsTable`) are replaced with recorders, so a
   blocked write is reported as "this pipeline writes here" rather than as a
   crash.
3. One subprocess per dry run, so a segfault, an `os._exit` or a C-level write
   in an engine takes down only that run.

Layer 1 is not a security boundary against hostile code: a determined
C extension gets around it. It is a boundary against *ordinary* pipeline code
doing exactly what it was written to do. Repos that run untrusted code should
put the whole thing in a container, and the docs say so.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

from rowless.errors import SideEffectBlocked


class Action(StrEnum):
    """What to do when sandboxed code attempts a side effect."""

    DENY = "deny"
    """Raise `SideEffectBlocked`. The default for writes and network calls."""

    RECORD = "record"
    """Allow, and log it as an output of the pipeline. Used for writers that
    an adapter has already replaced with a no-op recorder."""

    ALLOW = "allow"
    """Permit silently. Only for paths the sandbox itself needs."""


#: Audit events that mean "something is being written".
WRITE_EVENTS: tuple[str, ...] = (
    "open",
    "os.rename",
    "os.remove",
    "os.unlink",
    "os.link",
    "os.symlink",
    "os.truncate",
    "os.mkdir",
    "os.rmdir",
    "os.chmod",
    "os.chown",
    "shutil.copyfile",
    "shutil.copymode",
    "shutil.copystat",
    "shutil.move",
    "shutil.rmtree",
    "tempfile.mkstemp",
)

#: Audit events that mean "something is leaving the process".
#:
#: Name resolution is in here with the connects: looking a host up is already
#: an egress attempt, and blocking it turns "DNS failed after 20 seconds" into
#: a diagnostic that names the host the pipeline wanted.
NETWORK_EVENTS: tuple[str, ...] = (
    "socket.getaddrinfo",
    "socket.gethostbyname",
    "socket.connect",
    "socket.connect_ex",
    "socket.sendto",
    "socket.sendmsg",
    "socket.bind",
    "urllib.Request",
    "ftplib.connect",
    "smtplib.connect",
    "smtplib.send",
    "imaplib.open",
    "poplib.connect",
    "telnetlib.Telnet.open",
)

#: Audit events that mean "another process is starting".
SUBPROCESS_EVENTS: tuple[str, ...] = (
    "subprocess.Popen",
    "os.system",
    "os.exec",
    "os.posix_spawn",
    "os.spawn",
    "os.fork",
    "os.forkpty",
    "pty.spawn",
)

#: File suffixes and directory names the interpreter itself writes. Blocking
#: these would stop rowless importing the user's own modules, which is not the
#: side effect anybody is worried about.
_INTERPRETER_WRITES: tuple[str, ...] = (".pyc", ".pyo", "__pycache__")

#: Modes that mean a file is being opened for writing.
_WRITE_MODES = frozenset("wxa+")


@dataclass(frozen=True, slots=True)
class Policy:
    """The sandbox policy for one dry run."""

    network: Action = Action.DENY

    network_allowlist: tuple[str, ...] = ()
    """Hosts a remote adapter legitimately needs: a Spark Connect endpoint, a
    warehouse reading `information_schema`. Set per adapter, never globally."""

    filesystem_write: Action = Action.DENY

    write_allowlist: tuple[str, ...] = ()
    """Normally just the run's scratch directory."""

    subprocess_spawn: Action = Action.DENY
    env_read: Action = Action.ALLOW
    """Reads are allowed; secrets are a problem for exfiltration, which the
    network policy already covers."""

    @classmethod
    def strict(cls) -> Policy:
        """Deny everything outside the scratch directory. The CLI default."""
        return cls()

    @classmethod
    def permissive(cls) -> Policy:
        """Record instead of denying. For debugging a pipeline that will not run.

        Still does not *perform* anything an adapter has intercepted; it only
        stops the audit hook raising, so a user can see the full list of what
        their pipeline would do instead of only the first thing.
        """
        return cls(
            network=Action.RECORD,
            filesystem_write=Action.RECORD,
            subprocess_spawn=Action.RECORD,
        )

    def allowing_writes_to(self, *paths: str) -> Policy:
        """This policy plus write access to `paths`. Used for the scratch dir."""
        resolved = tuple(str(Path(path).resolve()) for path in paths)
        return replace(self, write_allowlist=self.write_allowlist + resolved)

    def allowing_hosts(self, *hosts: str) -> Policy:
        """This policy plus egress to `hosts`, for a remote adapter."""
        return replace(self, network_allowlist=self.network_allowlist + tuple(hosts))


@dataclass(slots=True)
class SideEffectLog:
    """Everything the sandbox intercepted, reported by `rowless explain`.

    Doubles as free output lineage: the writes a pipeline attempted are the
    tables it produces.
    """

    writes: list[str] = field(default_factory=list)
    network: list[str] = field(default_factory=list)
    subprocesses: list[str] = field(default_factory=list)
    blocked: list[str] = field(default_factory=list)


@dataclass(slots=True)
class _Session:
    """The policy and log the installed hook is currently enforcing."""

    policy: Policy
    log: SideEffectLog


#: Set while a sandbox is active. The hook reads this and does nothing when it
#: is None, which is how a permanent audit hook stops costing anything once
#: the dry run is over.
_SESSION: _Session | None = None

#: Audit hooks cannot be uninstalled, so one per process is installed on first
#: use and reused afterwards.
_HOOK_INSTALLED = False


@contextmanager
def sandboxed(policy: Policy) -> Iterator[SideEffectLog]:
    """Install the audit hook and library interceptors for the duration.

    Audit hooks are process-global and permanent, so this is only entered in a
    worker subprocess, never in the parent. Entering it twice in one process is
    an error rather than a no-op, because the second policy would silently
    fail to narrow the first.
    """
    global _SESSION, _HOOK_INSTALLED
    if _SESSION is not None:
        raise SideEffectBlocked(
            "a sandbox is already active in this process; nesting policies would silently "
            "widen the inner one"
        )
    log = SideEffectLog()
    _SESSION = _Session(policy=policy, log=log)
    if not _HOOK_INSTALLED:
        # Written bytecode is the interpreter's own side effect, and importing
        # the user's pipeline needs it. Turning it off is cheaper than
        # allowlisting every __pycache__ directory in the repository.
        sys.dont_write_bytecode = True
        sys.addaudithook(_hook)
        _HOOK_INSTALLED = True
    try:
        yield log
    finally:
        _SESSION = None


def audit_events_denied_by(policy: Policy) -> Sequence[str]:
    """The CPython audit event names this policy blocks.

    Spelled out as data rather than branching inside the hook, both because the
    hook runs on every single event and because the docs need to state exactly
    what is blocked.
    """
    denied: list[str] = []
    if policy.filesystem_write is Action.DENY:
        denied.extend(WRITE_EVENTS)
    if policy.network is Action.DENY:
        denied.extend(NETWORK_EVENTS)
    if policy.subprocess_spawn is Action.DENY:
        denied.extend(SUBPROCESS_EVENTS)
    return tuple(denied)


def _hook(event: str, args: tuple[Any, ...]) -> None:
    """The audit hook. Runs on every audited event in the process.

    Kept to a dictionary lookup in the common case: this is called for every
    import, every `open`, every compile, and a slow hook would make a dry run
    cost more than the query it replaces.
    """
    session = _SESSION
    if session is None:
        return
    handler = _HANDLERS.get(event)
    if handler is None:
        return
    handler(session, event, args)


def _on_write(session: _Session, event: str, args: tuple[Any, ...]) -> None:
    """A filesystem write: allowed inside the scratch directory, else denied."""
    path = _path_argument(event, args)
    if path is None:
        return
    if _is_interpreter_write(path):
        return
    if _is_allowlisted(path, session.policy.write_allowlist):
        session.log.writes.append(path)
        return
    _apply(session, session.policy.filesystem_write, f"write {path}", session.log.writes)


def _on_network(session: _Session, event: str, args: tuple[Any, ...]) -> None:
    """A network call: allowed only for an allowlisted host."""
    target = _network_target(args)
    host = target.partition(":")[0]
    if host and host in session.policy.network_allowlist:
        session.log.network.append(target)
        return
    _apply(session, session.policy.network, f"{event} {target}", session.log.network)


def _on_subprocess(session: _Session, event: str, args: tuple[Any, ...]) -> None:
    """Starting another process is never part of resolving a schema."""
    detail = f"{event} {args[0] if args else ''}"
    _apply(session, session.policy.subprocess_spawn, detail, session.log.subprocesses)


def _apply(session: _Session, action: Action, detail: str, recorder: list[str]) -> None:
    """Record or deny one attempt, according to the policy."""
    if action is Action.ALLOW:
        return
    if action is Action.RECORD:
        recorder.append(detail)
        return
    session.log.blocked.append(detail)
    raise SideEffectBlocked(
        f"blocked: {detail}. A dry run resolves schemas; it does not move data. "
        "Writes belong in the adapter's scratch directory."
    )


def _path_argument(event: str, args: tuple[Any, ...]) -> str | None:
    """The path a write event concerns, or None when it is not a write.

    `open` fires for reads too, and reads are fine: a pipeline reading its
    own config or a Parquet footer is the normal case.
    """
    if not args:
        return None
    if event == "open":
        mode = args[1] if len(args) > 1 else ""
        if not isinstance(mode, str) or not (_WRITE_MODES & set(mode)):
            return None
    raw = args[0]
    if isinstance(raw, int):
        # A file descriptor rather than a path: no name to report, and the
        # descriptor was obtained through an event that was already checked.
        return None
    try:
        return os.fspath(raw) if isinstance(raw, str | os.PathLike) else str(raw)
    except TypeError:
        return None


def _network_target(args: tuple[Any, ...]) -> str:
    """A readable host:port (or URL) for a network event."""
    for argument in args:
        if isinstance(argument, tuple) and argument:
            return ":".join(str(part) for part in argument[:2])
        if isinstance(argument, str) and argument:
            return argument
    return "unknown"


def _is_allowlisted(path: str, allowlist: tuple[str, ...]) -> bool:
    """True when `path` is inside one of the allowlisted directories."""
    if not allowlist:
        return False
    try:
        resolved = Path(path).resolve()
    except OSError:
        return False
    for allowed in allowlist:
        try:
            resolved.relative_to(Path(allowed).resolve())
        except ValueError:
            continue
        return True
    return False


def _is_interpreter_write(path: str) -> bool:
    """True for bytecode caches and the like, which Python writes on import."""
    return any(marker in path for marker in _INTERPRETER_WRITES)


_HANDLERS: dict[str, Any] = {
    **{event: _on_write for event in WRITE_EVENTS},
    **{event: _on_network for event in NETWORK_EVENTS},
    **{event: _on_subprocess for event in SUBPROCESS_EVENTS},
}
