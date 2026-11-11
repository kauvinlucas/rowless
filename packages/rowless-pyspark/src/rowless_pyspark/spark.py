"""Getting hold of a SparkSession, which is the hard part of this adapter.

Resolving the schema is easy: `df.schema` runs Catalyst analysis and launches
no job, so the answer is Spark's own, implicit casts included. Having a Spark
to ask is the problem, and it is the reason this adapter was last on the
roadmap.

## The two modes

**Connect** (`[engine.pyspark] remote = "sc://host:15002"`, or `SPARK_REMOTE`
in the environment) is the mode this adapter is designed around. The client is
pure Python, analysis happens on a server that is already running, and a dry
run costs one round trip per plan. It is also the only mode that keeps the
linter-speed promise, and the reason `remote=True` is declared in the
capabilities: the sandbox opens the configured endpoint and nothing else.

**Local** (`[engine.pyspark] mode = "local"`) is for people without a cluster.
It starts a JVM in a child process, which costs tens of seconds once per
process, so the session is cached for the lifetime of the worker and reused by
every dry run in a check rather than rebuilt per pipeline. Local mode needs
more from the sandbox than Connect does - a subprocess, and a loopback socket
for the driver - and `PySparkAdapter.egress` is what declares it.

## What this module configures, and why each one matters

* `spark.sql.session.timeZone = UTC`, because `TimestampType` has no zone of
  its own: it is an instant rendered in the *session's* timezone, so the IR
  dtype rowless reports for it is only honest if the session's zone is known.
  A session this adapter did not create is left alone and its zone read
  instead, which is why `session_timezone()` exists.
* `spark.sql.catalogImplementation = in-memory` in local mode, so a dry run
  does not leave a Derby metastore in the repository it is checking.
* `spark.sql.warehouse.dir` under the run's scratch directory, which is the
  only path the sandbox permits writes to.
* `spark.ui.enabled = false` and one shuffle partition: a dry run resolves
  schemas, so a web UI and 200 empty shuffle tasks are pure cost.

## Why the session is created before the pipeline is imported

The runtime calls `materialize` before it imports the target module, and this
is where the session is made. A Spark repository usually builds its session at
module scope (`spark = SparkSession.builder.getOrCreate()`), and because the
rowless session exists first, that call returns *this* session rather than
starting a second one. The pipeline runs unmodified and still lands on the
configured endpoint.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any

from rowless.errors import UnknownSchemaError

from rowless_pyspark import session as run

#: Environment variables that name a Connect endpoint. `SPARK_REMOTE` is
#: Spark's own; the rowless one exists so a repository can point its checks at
#: a scratch cluster without changing where its pipelines run.
REMOTE_ENV: tuple[str, ...] = ("ROWLESS_SPARK_REMOTE", "SPARK_REMOTE")

#: `[engine.pyspark]` keys that name a Connect endpoint, in order of
#: preference. Both spellings are accepted because Spark's own builder method
#: is `remote()` and every other adapter in rowless calls this an endpoint.
REMOTE_OPTIONS: tuple[str, ...] = ("remote", "endpoint")

#: The oldest PySpark this adapter supports.
MINIMUM_PYSPARK: tuple[int, int] = (3, 5)

#: Java major version each Spark line needs. Spark 4 dropped Java 8 and 11, so
#: an environment that satisfies `pyspark>=3.5` can still be unable to start a
#: local session - which is a configuration problem rather than a missing
#: library, and `unavailable_reason()` is how it gets reported.
JAVA_FLOOR: dict[int, int] = {3: 8, 4: 17}

#: Spark configuration this adapter sets on a session it creates. The reasons
#: are in the module docstring; `spark.sql.warehouse.dir` is added per run,
#: because it has to land in the run's scratch directory.
BASE_CONF: dict[str, str] = {
    "spark.sql.session.timeZone": "UTC",
    "spark.ui.enabled": "false",
    "spark.sql.shuffle.partitions": "1",
    "spark.sql.adaptive.enabled": "false",
}

#: Local-mode-only configuration. An in-memory catalog is what keeps a dry run
#: from writing a Derby metastore into the repository under check.
LOCAL_CONF: dict[str, str] = {
    "spark.sql.catalogImplementation": "in-memory",
    "spark.driver.host": "127.0.0.1",
    "spark.driver.bindAddress": "127.0.0.1",
}

#: Hosts local mode needs to reach: the driver talks to its own executors over
#: loopback even when there is only one JVM.
LOCAL_HOSTS: tuple[str, ...] = ("localhost", "127.0.0.1", "0.0.0.0", "::1")

#: `sc://host:15002/;token=...`: the host is what the sandbox allowlists.
_ENDPOINT = re.compile(r"^(?:sc://)?(?P<host>[^:/;?]+)")

#: `JAVA_VERSION="17.0.11"` out of a JDK release file.
_JAVA_VERSION = re.compile(r'JAVA_VERSION="?([0-9._]+)')

#: The session this process is using, and how it was obtained. Deliberately
#: *not* reset between runs: starting a session is the expensive part, and
#: reusing it is what makes a multi-pipeline check affordable.
_SESSION: Any = None
_MODE: str | None = None
_TIMEZONE: str = "UTC"

#: Where a local session spills and keeps its warehouse. Process-scoped, for
#: the reason `scratch_dir` explains.
_SCRATCH: Path | None = None


def mode(options: Mapping[str, Any] | None = None) -> str:
    """Which mode this run uses: `connect` when an endpoint is configured, else `local`.

    An explicit `[engine.pyspark] mode` wins, so a repository can insist on
    local mode even with `SPARK_REMOTE` set in the environment - an endpoint
    that happens to be exported by a shell profile is not consent to send a dry
    run to it.

    `options` is passed explicitly by the one caller that has the adapter's
    options and no open run: the worker, asking what egress to permit before
    it installs the sandbox. Everything else reads the run's own context.
    """
    configured = _option(options, "mode")
    if configured:
        return str(configured).strip().lower()
    return "connect" if endpoint(options) else "local"


def endpoint(options: Mapping[str, Any] | None = None) -> str | None:
    """The configured Connect endpoint, from options or the environment."""
    for key in REMOTE_OPTIONS:
        value = _option(options, key)
        if value:
            return str(value)
    for variable in REMOTE_ENV:
        value = os.environ.get(variable)
        if value:
            return value
    return None


def _option(options: Mapping[str, Any] | None, key: str, default: Any = None) -> Any:
    """One option, from an explicit mapping when given and from the run otherwise."""
    if options is not None:
        return options.get(key, default)
    return run.option(key, default)


def endpoint_host(url: str | None = None) -> str | None:
    """The host part of a Connect endpoint, which is what the sandbox allows."""
    target = url if url is not None else endpoint()
    if not target:
        return None
    found = _ENDPOINT.match(target.strip())
    return found.group("host") if found else None


def egress_hosts(options: Mapping[str, Any] | None = None) -> tuple[str, ...]:
    """Hosts this run legitimately needs to reach, by mode.

    Read by `PySparkAdapter.egress`, which the worker asks before it installs
    the sandbox. Connect gets exactly the configured endpoint; local mode gets
    loopback, because the driver and its executors talk over it even when there
    is only one JVM.
    """
    if mode(options) == "local":
        return LOCAL_HOSTS
    target = endpoint(options)
    host = endpoint_host(target) if target else None
    return (host,) if host else ()


def session() -> Any:
    """The SparkSession for this process, created on first use and then reused.

    Reuse is the point: a classic local session costs tens of seconds to start,
    and rebuilding one per pipeline would make a check slower than the queries
    it replaces. A Connect session is cheap to create and still worth reusing,
    because every analysis after the first lands on a warm server.
    """
    global _SESSION, _MODE, _TIMEZONE

    if _SESSION is not None and not _stopped(_SESSION):
        return _SESSION
    existing = _active_session()
    if existing is not None:
        # The pipeline, or a previous run, already has one. Adopting it is both
        # cheaper and less surprising than starting a second Spark in the same
        # process - and its timezone is read rather than overwritten, because
        # somebody else's session is not this adapter's to reconfigure.
        _SESSION = existing
        _MODE = "adopted"
        _TIMEZONE = _read_timezone(existing)
        return existing
    chosen = mode()
    _SESSION = _connect_session() if chosen == "connect" else _local_session()
    _MODE = chosen
    _TIMEZONE = _read_timezone(_SESSION)
    return _SESSION


def active_mode() -> str | None:
    """How the current session was obtained: `connect`, `local` or `adopted`."""
    return _MODE


def session_timezone() -> str:
    """The session timezone, which is the zone `TimestampType` values are in.

    `UTC` until a session has been built, which is also what this adapter
    configures on a session it creates. A session it adopted may be in another
    zone, and reporting `datetime[us, UTC]` for a frame Spark will render in
    `America/Denver` would be a dtype rowless invented.
    """
    return _TIMEZONE


def forget() -> None:
    """Drop the cached session without stopping it. For tests."""
    global _SESSION, _MODE, _TIMEZONE

    _SESSION = None
    _MODE = None
    _TIMEZONE = "UTC"


def forget_if_stopped() -> None:
    """Drop the cached session when it has been stopped, so the next run rebuilds.

    A pipeline is entitled to call `spark.stop()`, and a check runs many
    pipelines in one process. Noticing it here - rather than failing on the
    next `materialize` - is what keeps one pipeline's teardown from taking the
    rest of the run with it.
    """
    if _SESSION is not None and _stopped(_SESSION):
        forget()


def stop() -> None:
    """Stop the session this process started, if any. For tests and teardown."""
    current = _SESSION
    adopted = _MODE == "adopted"
    forget()
    if current is not None and not adopted:
        # A session that is already gone needs no stopping, and a client
        # mid-teardown is entitled to refuse the call.
        with suppress(Exception):
            current.stop()


# --- Building ----------------------------------------------------------------


def _connect_session() -> Any:
    """Build a Spark Connect session against the configured endpoint."""
    from pyspark.sql import SparkSession

    target = endpoint()
    if not target:
        raise UnknownSchemaError(
            "the pyspark adapter is in Connect mode and no endpoint is configured: set "
            "[engine.pyspark] remote in rowless.toml, export SPARK_REMOTE, or set "
            "[engine.pyspark] mode = local to start a local session instead"
        )
    builder = SparkSession.builder.remote(target)
    for key, value in BASE_CONF.items():
        builder = builder.config(key, value)
    return builder.getOrCreate()


def _local_session() -> Any:
    """Build a classic local session, pointed at a scratch directory of its own."""
    from pyspark.sql import SparkSession

    builder = SparkSession.builder.master(str(run.option("master", "local[1]")))
    builder = builder.appName(f"rowless-{run.run_id() or 'dryrun'}")
    for key, value in {**BASE_CONF, **LOCAL_CONF}.items():
        builder = builder.config(key, value)
    scratch = scratch_dir()
    builder = builder.config("spark.sql.warehouse.dir", (scratch / "warehouse").as_uri())
    builder = builder.config("spark.local.dir", str(scratch / "tmp"))
    return builder.getOrCreate()


def scratch_dir() -> Path:
    """A writable directory that lasts as long as the session does.

    Deliberately **not** `DryRunContext.workdir`. That directory belongs to one
    dry run and is deleted when it finishes, while the session is reused by
    every run in a check - so a session whose spill directory was the first
    run's workdir starts failing with `FileNotFoundError` on the second
    pipeline, which is exactly the bug this function exists to have fixed.
    The scratch directory is process-scoped because the session is.

    It lives under the system temp directory, which is the other path the
    worker's sandbox policy permits writes to, so nothing here needs the
    policy widened.
    """
    global _SCRATCH

    if _SCRATCH is None or not _SCRATCH.is_dir():
        import tempfile

        _SCRATCH = Path(tempfile.mkdtemp(prefix="rowless-spark-"))
    return _SCRATCH


def _active_session() -> Any:
    """A SparkSession somebody else already built in this process, or None."""
    try:
        from pyspark.sql import SparkSession
    except Exception:  # pragma: no cover - only without pyspark
        return None
    getter = getattr(SparkSession, "getActiveSession", None)
    candidate = None
    if callable(getter):
        try:
            candidate = getter()
        except Exception:  # pragma: no cover - a client mid-teardown
            candidate = None
    if candidate is None:
        candidate = getattr(SparkSession, "_instantiatedSession", None)
    if candidate is None or _stopped(candidate):
        return None
    return candidate


def _stopped(spark: Any) -> bool:
    """True when a session is no longer usable, so a new one has to be built."""
    stopped = getattr(spark, "is_stopped", None)
    if isinstance(stopped, bool):
        return stopped
    if not callable(stopped):
        return False
    try:
        return bool(stopped())
    except Exception:  # pragma: no cover - a client that cannot answer
        return True


def _read_timezone(spark: Any) -> str:
    """The session's configured timezone, defaulting to UTC when it will not say."""
    try:
        with run.own_work():
            found = spark.conf.get("spark.sql.session.timeZone")
    except Exception:
        return "UTC"
    return str(found) if found else "UTC"


# --- Availability ------------------------------------------------------------


def importable() -> bool:
    """True when pyspark imports at a supported version.

    Deliberately separate from `usable()`: a user with pyspark installed and no
    cluster configured has the library and still cannot run a dry run, and
    those are two different lines in `rowless doctor`.
    """
    version = pyspark_version()
    return version is not None and version >= MINIMUM_PYSPARK


def pyspark_version() -> tuple[int, int] | None:
    """The installed PySpark version, or None when it does not import."""
    try:
        import pyspark
    except Exception:
        # An engine that fails to import is simply not available. A
        # half-installed PySpark - the jars missing from the wheel, say -
        # raises on import.
        return None
    return parse_version(str(getattr(pyspark, "__version__", "0")))


def usable() -> bool:
    """True when this adapter could actually resolve a schema right now.

    Connect needs the client extras and an endpoint; local mode needs a Java
    new enough for the installed Spark. Reporting the adapter as available when
    neither holds would turn every Spark pipeline in a repository into a
    traceback instead of one clear line saying what to configure.
    """
    if not importable():
        return False
    if _session_in_process():
        return True
    if mode() == "connect":
        return bool(endpoint()) and connect_client_available()
    return java_is_new_enough()


def unavailable_reason() -> str:
    """One line for `rowless doctor` saying what is missing.

    The optional hook the registry prefers over `requires`, and this adapter is
    the case it exists for: the usual reason a Spark dry run cannot run is
    configuration, which a PEP 508 requirement cannot describe.
    """
    version = pyspark_version()
    if version is None:
        return "requires pyspark>=3.5"
    if version < MINIMUM_PYSPARK:
        spelled = ".".join(str(part) for part in version)
        return f"pyspark {spelled} is installed; this adapter needs >= 3.5"
    if mode() == "connect":
        if not endpoint():
            return (
                "pyspark is installed but no Spark Connect endpoint is configured: set "
                "[engine.pyspark] remote, export SPARK_REMOTE, or set "
                "[engine.pyspark] mode = local"
            )
        failure = connect_client_error()
        if failure:
            return (
                f"a Connect endpoint is configured and the client will not import ({failure}): "
                "install pyspark[connect], and note that the 3.5 client needs Python 3.11 or "
                "older, so Connect mode on a newer Python wants pyspark>=4"
            )
        return ""
    found = java_version()
    needed = required_java()
    if found is None:
        home = java_home()
        where = f" (looked in {home})" if home is not None else " and none was found on PATH"
        return (
            f"local mode needs Java {needed} and its version could not be determined{where}: set "
            "JAVA_HOME to a JDK, or configure a Spark Connect endpoint so that no local JVM is "
            "needed"
        )
    return (
        f"local mode needs Java {needed} for this Spark and found Java {found}: install a newer "
        "JDK, or configure a Spark Connect endpoint"
    )


def connect_client_available() -> bool:
    """True when the Spark Connect client can actually be imported."""
    return not connect_client_error()


def connect_client_error() -> str:
    """Why the Connect client cannot be imported, or the empty string.

    The client is imported rather than having its dependencies counted, because
    counting them misses the case this exists for: PySpark 3.5's Connect client
    imports pandas through `distutils`, which Python 3.12 removed, so on the
    declared floor the client fails to import on a Python this package
    supports. `find_spec("pandas")` says yes and the client still does not
    work. Importing it is one import and a decisive answer, and the message
    goes into the `rowless doctor` line so the fix is obvious: Connect mode
    wants pyspark>=4 here, or local mode.
    """
    try:
        import importlib

        importlib.import_module("pyspark.sql.connect.session")
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return ""


def required_java() -> int:
    """The Java major version the installed Spark needs for a local session."""
    version = pyspark_version()
    major = version[0] if version else 4
    return JAVA_FLOOR.get(major, 17)


def java_is_new_enough() -> bool:
    """True when the Java this process would start is new enough for local mode.

    Read from the `release` file in the JDK rather than by running
    `java -version`: a dry run happens inside a sandbox that denies starting
    processes, so a probe that spawned one would be a side effect in the middle
    of an availability check.

    A version that cannot be read counts as **not** new enough, and that is a
    deliberate choice rather than a limitation. The alternative is to claim the
    adapter is available and then have every pipeline in the repository fail
    with a JVM error, which is the worst of both outcomes: the usual reason the
    version cannot be read is a launcher shim on `PATH` rather than a JDK, and
    `JAVA_HOME` is one line of configuration that fixes it. `rowless doctor`
    says exactly that.
    """
    found = java_version()
    return found is not None and found >= required_java()


def java_version() -> int | None:
    """The major version of the Java that would be used, or None when unknown."""
    home = java_home()
    if home is None:
        return None
    return _major_from_release(home / "release")


def java_home() -> Path | None:
    """Where the JVM lives: `JAVA_HOME`, else the `java` on `PATH`."""
    import shutil

    configured = os.environ.get("JAVA_HOME")
    if configured:
        return Path(configured)
    found = shutil.which("java")
    if not found:
        return None
    # `<home>/bin/java` is the layout every distribution uses, so the home is
    # two levels up from the executable.
    return Path(found).resolve().parent.parent


def _major_from_release(path: Path) -> int | None:
    """Read `JAVA_VERSION` out of a JDK `release` file as a major version."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    found = _JAVA_VERSION.search(text)
    return _major(found.group(1)) if found else None


def _major(version: str) -> int | None:
    """Java's two numbering schemes: `1.8.0_451` is 8, `17.0.11` is 17."""
    parts = version.replace("_", ".").split(".")
    try:
        first = int(parts[0])
    except (ValueError, IndexError):
        return None
    if first != 1:
        return first
    try:
        return int(parts[1]) if len(parts) > 1 else None
    except ValueError:
        return None


def _session_in_process() -> bool:
    """True when a usable session already exists, whoever built it."""
    if _SESSION is not None and not _stopped(_SESSION):
        return True
    return _active_session() is not None


def parse_version(version: str) -> tuple[int, int] | None:
    """`(major, minor)` from a version string, ignoring any suffix."""
    parts: list[int] = []
    for component in version.split(".")[:2]:
        digits = ""
        for character in component:
            if not character.isdigit():
                break
            digits += character
        if not digits:
            return None
        parts.append(int(digits))
    while len(parts) < 2:
        parts.append(0)
    return (parts[0], parts[1])
