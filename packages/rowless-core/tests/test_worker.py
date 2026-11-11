"""The worker subprocess: the boundary that keeps a crashing engine contained.

No engine adapter can be registered in a child process, so these tests drive
the boundary itself: the JSON protocol, the limits, and what a crash, a
timeout and a chatty pipeline turn into. The dry run above the boundary is
covered in `test_check.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

from rowless.adapters.base import Capabilities, Egress
from rowless.adapters.registry import clear_cache, register
from rowless.diagnostics.codes import Code
from rowless.ir.confidence import Confidence, Coverage
from rowless.ir.schema import Relation
from rowless.runtime.dryrun import DryRunRequest, DryRunResult, dry_run
from rowless.runtime.sandbox import Action, Policy
from rowless.runtime.worker import WorkerLimits, _with_adapter_egress, spawn
from rowless.testing import fixtures


def _request(
    target: str,
    *,
    sys_path: tuple[str, ...] = (),
    engine: str | None = None,
) -> DryRunRequest:
    """A request for `target`, with the orders fixture as its only input."""
    return DryRunRequest(
        target=target,
        inputs={"orders": fixtures.ORDERS},
        engine=engine,
        uris={"orders": "lake://stg_orders"},
        sys_path=sys_path,
    )


def test_request_and_result_cross_as_ir_json() -> None:
    """Schemas travel as IR JSON, never as a pickle of an engine object."""
    payload = json.loads(_request("pipelines.revenue:daily_revenue").to_json())

    assert payload["inputs"]["orders"]["fields"][0]["dtype"] == "int64"
    assert payload["uris"] == {"orders": "lake://stg_orders"}

    result = DryRunResult(
        target="pipelines.revenue:daily_revenue",
        output=Relation(schema=fixtures.ORDERS, engine="fake", notes=("materialized empty",)),
        coverage=Coverage(columns_total=4, columns_typed=4, confidence=Confidence.EXACT),
    )
    restored = DryRunResult.from_json(result.to_json())

    assert restored.target == result.target
    assert restored.output is not None
    assert restored.output.schema == fixtures.ORDERS
    assert restored.output.notes == ("materialized empty",)
    assert restored.coverage.confidence is Confidence.EXACT


def test_a_missing_module_is_a_finding_not_a_crash() -> None:
    """A repository that does not import is reported per pipeline.

    Reported as unverified rather than as a pipeline failure: a target whose
    own module cannot be found is one rowless never reached, and that is also
    what a discovered SQL model looks like from here - a dbt model is named by
    its node id, which is not an importable reference at all.
    """
    result = dry_run(_request("no.such.module:pipeline"))

    assert result.output is None
    assert not result.ok
    codes = {diagnostic.code.value for diagnostic in result.diagnostics}
    assert codes <= {"RWL205", "RWL502"}, codes
    assert result.coverage.reasons


def test_a_pipeline_missing_a_dependency_is_still_the_pipeline_s_problem(
    tmp_path: Path,
) -> None:
    """The other half of the distinction above, and the one worth getting right.

    A module rowless *did* reach, which then failed on an import of its own, is
    a broken pipeline and is reported as one. Only a target whose own module
    cannot be found reads as unchecked, so a repository that forgot to install
    something is not quietly excused.
    """
    module = tmp_path / "needs_something.py"
    module.write_text(
        "\n".join(
            [
                "import a_package_nobody_installed",
                "",
                "",
                "def pipeline(orders):",
                "    return orders",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = dry_run(_request("needs_something:pipeline", sys_path=(str(tmp_path),)))

    codes = {diagnostic.code for diagnostic in result.diagnostics}
    assert Code.PIPELINE_RAISED in codes
    assert Code.UNRESOLVED_SOURCE not in codes


def test_a_pipeline_that_prints_does_not_corrupt_the_reply(tmp_path: Path) -> None:
    """User code writes to stdout all the time; the protocol has to survive it."""
    module = tmp_path / "chatty.py"
    module.write_text(
        "\n".join(
            [
                "print('loading configuration')",
                "",
                "",
                "def pipeline(orders):",
                "    return orders",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result = dry_run(_request("chatty:pipeline", sys_path=(str(tmp_path),)))

    # What matters here is that the reply parsed at all. Whether a schema came
    # back depends on which engines are installed - none in the core CI job,
    # the real ones in a full workspace - and either outcome is fine as long as
    # the worker came back with a readable answer rather than a crash.
    assert result.target == "chatty:pipeline"
    assert Code.WORKER_CRASHED not in {diagnostic.code for diagnostic in result.diagnostics}


def test_timeout_is_reported_as_a_timeout(tmp_path: Path) -> None:
    """A dry run that takes seconds has executed something, and says so."""
    module = tmp_path / "slow.py"
    module.write_text(
        "\n".join(
            [
                "import time",
                "",
                "time.sleep(30)",
                "",
                "",
                "def pipeline(orders):",
                "    return orders",
                "",
            ]
        ),
        encoding="utf-8",
    )
    request = _request("slow:pipeline", sys_path=(str(tmp_path),))

    reply = spawn(request.to_json(), WorkerLimits(timeout_s=1.0))
    result = DryRunResult.from_json(reply)

    assert [diagnostic.code.value for diagnostic in result.diagnostics] == ["RWL503"]
    assert "exceeded 1s" in (result.diagnostics[0].detail or "")


def test_a_worker_that_dies_becomes_a_diagnostic() -> None:
    """A segfaulting engine is a normal thing to meet, and must not abort the run."""
    reply = spawn("not json at all", WorkerLimits(timeout_s=30.0))
    result = DryRunResult.from_json(reply)

    assert [diagnostic.code.value for diagnostic in result.diagnostics] == ["RWL502"]
    assert result.coverage.confidence.name == "UNKNOWN"


def test_the_reply_shape_is_the_same_whether_it_worked_or_not() -> None:
    """One parsing path in the parent: a crash is an ordinary finding."""
    failure = json.loads(spawn("{}", WorkerLimits(timeout_s=30.0)))

    assert set(failure) >= {"target", "output", "coverage", "diagnostics", "side_effects"}
    assert failure["output"] is None


def test_worker_limits_have_linter_speed_defaults() -> None:
    """The timeout is the backstop against a query that actually ran."""
    limits = WorkerLimits()

    assert limits.timeout_s <= 30.0
    assert limits.max_rss_mb <= 4096
    assert limits.max_output_bytes >= 1024 * 1024


# --- Egress for a remote adapter ---------------------------------------------


class _Remote:
    """An adapter whose engine is on a server, like the PySpark one."""

    name = "remote"
    requires = "someengine>=1.0"
    capabilities = Capabilities(remote=True)

    def __init__(self, needs: Egress) -> None:
        self.needs = needs
        self.asked: list[object] = []

    def available(self) -> bool:
        return True

    def egress(self, options: object = None) -> Egress:
        self.asked.append(options)
        return self.needs


class _Local:
    """An adapter that runs its engine in this process, like every other one."""

    name = "local"
    requires = "someengine>=1.0"
    capabilities = Capabilities()

    def available(self) -> bool:
        return True

    def egress(self, options: object = None) -> Egress:
        raise AssertionError("a non-remote adapter must never be asked for egress")


def test_a_remote_adapter_opens_exactly_what_it_asked_for() -> None:
    """The sandbox denies the network; a remote adapter names the host it needs."""
    adapter = _Remote(Egress(hosts=("cluster.internal",), reason="analysis runs on the server"))
    register("rowless.engines", "remote", adapter)
    try:
        policy = _with_adapter_egress(Policy.strict(), "remote", {"remote": "sc://x"})
    finally:
        clear_cache()

    assert policy.network_allowlist == ("cluster.internal",)
    assert policy.network is Action.DENY, "everything else stays denied"
    assert policy.subprocess_spawn is Action.DENY
    assert adapter.asked == [{"remote": "sc://x"}], "the adapter is asked with its own options"


def test_an_adapter_that_needs_a_jvm_may_start_one() -> None:
    """A local Spark session is a child process, and the adapter has to declare it."""
    register("rowless.engines", "remote", _Remote(Egress(hosts=("127.0.0.1",), subprocesses=True)))
    try:
        policy = _with_adapter_egress(Policy.strict(), "remote", {})
    finally:
        clear_cache()

    assert policy.subprocess_spawn is Action.RECORD
    assert "127.0.0.1" in policy.network_allowlist


def test_an_in_process_adapter_is_never_asked() -> None:
    """`remote=False` means the strict policy is already right."""
    register("rowless.engines", "local", _Local())
    try:
        policy = _with_adapter_egress(Policy.strict(), "local", {})
    finally:
        clear_cache()

    assert policy == Policy.strict()


def test_an_unnamed_engine_keeps_the_strict_policy() -> None:
    """An engine inferred after the import cannot have been consulted in time.

    The honest outcome is a blocked connection and a finding, not egress
    granted to an adapter nobody identified.
    """
    assert _with_adapter_egress(Policy.strict(), None, {}) == Policy.strict()
    assert _with_adapter_egress(Policy.strict(), "nobody-registered-this", {}) == Policy.strict()
