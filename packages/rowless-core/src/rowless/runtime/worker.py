"""The subprocess that actually runs user code.

Invoked as `python -m rowless.runtime.worker`, reads a request on stdin, writes
a result on stdout, both as JSON. The parent process never imports the user's
pipeline or an engine library, which keeps a crashing engine from taking the
CLI with it.

Schemas cross the boundary as IR JSON, not pickles: a pickle of a Polars
LazyFrame would need the engine in the parent, and the whole point is that it
is not there.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class WorkerLimits:
    """Resource ceilings. A dry run that hits one of these is a bug worth reporting.

    "Linter speed" is the product promise, so the default timeout is low enough
    that an accidentally-executing query is caught rather than waited on.
    """

    timeout_s: float = 30.0
    max_rss_mb: int = 2048
    max_output_bytes: int = 8 * 1024 * 1024


def run_worker(argv: list[str] | None = None) -> int:
    """Worker entry point. Returns the process exit code."""
    _ = argv
    raw = sys.stdin.read()
    try:
        request = json.loads(raw)
    except json.JSONDecodeError as exc:
        _emit({"error": f"unreadable request: {exc}"})
        return 2

    # Everything below this line runs with the user's code in the process, so
    # the result stream is captured first: a `print` in a pipeline must not
    # end up in the middle of the JSON reply.
    real_stdout = sys.stdout
    sys.stdout = sys.stderr

    try:
        payload = _execute(request)
    except BaseException as exc:  # a worker never leaks a traceback to the parent
        payload = {"error": f"{type(exc).__qualname__}: {exc}"}
        sys.stdout = real_stdout
        _emit(payload)
        return 1
    sys.stdout = real_stdout
    _emit(payload)
    return 0


def _execute(request: dict[str, Any]) -> dict[str, Any]:
    """Resolve one dry run inside the sandbox, and return its JSON payload."""
    from rowless.ir.serde import schema_from_dict
    from rowless.runtime.dryrun import dry_run_target
    from rowless.runtime.sandbox import Policy, sandboxed

    for entry in reversed(request.get("sys_path", ())):
        if entry not in sys.path:
            sys.path.insert(0, entry)

    inputs = {name: schema_from_dict(schema) for name, schema in request.get("inputs", {}).items()}
    target = request["target"]
    engine = request.get("engine")
    # Whether UDF probing is allowed is configuration, and the adapter is what
    # acts on it, so it travels with the adapter's own options.
    options = {
        **(request.get("options") or {}),
        "probe_udfs": bool(request.get("probe_udfs", False)),
    }
    uris = request.get("uris") or {}

    with tempfile.TemporaryDirectory(prefix="rowless-worker-") as workdir:
        # The scratch directory and the system temp directory are writable;
        # everything else is not. Engines spill to temp, and so does the
        # standard library, while the writes worth blocking are the ones that
        # land in a warehouse or a data lake.
        policy = Policy.strict().allowing_writes_to(workdir, tempfile.gettempdir())
        policy = _with_adapter_egress(policy, engine, options)
        with sandboxed(policy) as side_effects:
            # The import happens inside the sandbox too: a pipeline module that
            # opens a connection at import time is the normal case, not an
            # exotic one.
            result = dry_run_target(target, inputs, engine=engine, options=options, uris=uris)
        payload: dict[str, Any] = json.loads(result.to_json())
        payload["side_effects"] = [*side_effects.writes, *side_effects.blocked]
        return payload


def _with_adapter_egress(policy: Any, engine: str | None, options: dict[str, Any]) -> Any:
    """Open the network a remote adapter needs, and nothing more.

    The sandbox denies egress outright, which is right for every adapter that
    runs its engine in this process and impossible for one that resolves
    schemas on a server. An adapter that declares `remote=True` is asked what
    it needs through the optional `egress()` hook (see
    `rowless.adapters.base.NeedsEgress`), and exactly that is permitted: the
    named hosts, plus a child process when reaching the engine means starting
    one.

    Only possible when the engine was named in the request. An engine inferred
    from the pipeline's annotations is not known until the module has been
    imported, which happens inside the sandbox - so a remote adapter wants its
    engine stated, and a connection attempt without one is blocked and reported
    rather than silently allowed. Anything that goes wrong while asking leaves
    the strict policy in place, because the safe failure here is a blocked
    connection and not an open one.
    """
    if not engine:
        return policy
    try:
        from rowless.adapters.registry import engine as get_engine

        adapter = get_engine(engine)
        if not getattr(adapter.capabilities, "remote", False):
            return policy
        hook = getattr(adapter, "egress", None)
        needs = hook(options) if callable(hook) else None
    except Exception:
        return policy
    if needs is None:
        return policy
    hosts = tuple(str(host) for host in getattr(needs, "hosts", ()) if host)
    if hosts:
        policy = policy.allowing_hosts(*hosts)
    if getattr(needs, "subprocesses", False):
        policy = policy.allowing_subprocesses()
    return policy


def spawn(request_json: str, limits: WorkerLimits) -> str:
    """Parent side: run one worker and return its raw result JSON.

    Enforces the limits, and turns a crash, a timeout or a non-JSON reply into
    a diagnostic rather than an exception. A segfault in an engine is a normal
    thing to encounter here and must not abort the whole check run.
    """
    target = _target_of(request_json)
    command = [sys.executable, "-X", "faulthandler", "-m", "rowless.runtime.worker"]
    try:
        completed = subprocess.run(
            command,
            input=request_json,
            capture_output=True,
            text=True,
            timeout=limits.timeout_s,
            check=False,
            env=_child_env(limits),
        )
    except subprocess.TimeoutExpired:
        return _failure_reply(
            target,
            "WORKER_TIMEOUT",
            f"dry run exceeded {limits.timeout_s:g}s, which usually means something executed",
        )
    except OSError as exc:
        return _failure_reply(target, "WORKER_CRASHED", f"cannot start worker: {exc}")

    reply = (completed.stdout or "").strip()
    if len(reply) > limits.max_output_bytes:
        return _failure_reply(target, "WORKER_CRASHED", "worker reply exceeded the output limit")
    if not reply:
        detail = _tail(completed.stderr) or f"worker exited with code {completed.returncode}"
        return _failure_reply(target, "WORKER_CRASHED", detail)

    try:
        parsed = json.loads(reply)
    except json.JSONDecodeError:
        detail = f"worker reply was not JSON: {_tail(reply)}"
        return _failure_reply(target, "WORKER_CRASHED", detail)
    if "error" in parsed:
        return _failure_reply(target, "WORKER_CRASHED", str(parsed["error"]))
    return reply


def _child_env(limits: WorkerLimits) -> dict[str, str]:
    """Environment for the worker.

    The memory ceiling travels as an environment variable because the only
    portable place to apply it is inside the child: `resource` is POSIX-only,
    and on Windows there is nothing equivalent to set from here.
    """
    env = dict(os.environ)
    env["ROWLESS_MAX_RSS_MB"] = str(limits.max_rss_mb)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _apply_memory_limit() -> None:
    """Cap the worker's address space where the platform allows it."""
    ceiling = os.environ.get("ROWLESS_MAX_RSS_MB")
    if not ceiling:
        return
    if os.name != "posix":
        # Windows has no rlimit. The timeout is the backstop there: a dry run
        # that starts allocating gigabytes is a dry run that started executing.
        return
    # Imported dynamically rather than with a plain import: `resource` exists
    # only on POSIX, and a static import of it is a type error on Windows,
    # where this line is never reached.
    resource = importlib.import_module("resource")
    limit = int(ceiling) * 1024 * 1024
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    if hard != resource.RLIM_INFINITY:
        limit = min(limit, hard)
    resource.setrlimit(resource.RLIMIT_AS, (limit, hard))


def _target_of(request_json: str) -> str:
    """The target named in a request, for a failure reply."""
    try:
        return str(json.loads(request_json).get("target", "<unknown>"))
    except json.JSONDecodeError:
        return "<unknown>"


def _failure_reply(target: str, code_name: str, detail: str) -> str:
    """A result payload describing a worker that did not come back.

    Shaped exactly like a successful reply so the parent has one parsing path:
    a crashed dry run is an ordinary finding, not a special case.
    """
    from rowless.diagnostics.codes import Code

    code = Code[code_name]
    return json.dumps(
        {
            "target": target,
            "output": None,
            "coverage": {
                "columns_total": 0,
                "columns_typed": 0,
                "columns_traced": 0,
                "confidence": "unknown",
                "reasons": [detail],
            },
            "duration_ms": 0.0,
            "side_effects": [],
            "diagnostics": [
                {
                    "code": code.value,
                    "level": "error",
                    "message": f"dry run of {target} did not complete",
                    "origin": {"kind": "python", "ref": target},
                    "related": [],
                    "detail": detail,
                    "suggestion": None,
                }
            ],
        }
    )


def _tail(text: str | None, limit: int = 800) -> str:
    """The last of a stream, which is where the traceback is."""
    if not text:
        return ""
    cleaned = text.strip()
    return cleaned if len(cleaned) <= limit else f"...{cleaned[-limit:]}"


def _emit(payload: dict[str, Any]) -> None:
    """Write the reply on the real stdout."""
    sys.stdout.write(json.dumps(payload))
    sys.stdout.flush()


if __name__ == "__main__":
    _apply_memory_limit()
    sys.path.insert(0, str(Path.cwd()))
    raise SystemExit(run_worker())
