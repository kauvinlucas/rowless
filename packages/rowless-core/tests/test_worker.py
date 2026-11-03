"""The worker subprocess: the boundary that keeps a crashing engine contained.

No engine adapter can be registered in a child process, so these tests drive
the boundary itself: the JSON protocol, the limits, and what a crash, a
timeout and a chatty pipeline turn into. The dry run above the boundary is
covered in `test_check.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

from rowless.ir.confidence import Confidence, Coverage
from rowless.ir.schema import Relation
from rowless.runtime.dryrun import DryRunRequest, DryRunResult, dry_run
from rowless.runtime.worker import WorkerLimits, spawn
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
    """A repository that does not import is reported per pipeline."""
    result = dry_run(_request("no.such.module:pipeline"))

    assert result.output is None
    assert not result.ok
    codes = {diagnostic.code.value for diagnostic in result.diagnostics}
    assert codes <= {"RWL108", "RWL502"}, codes
    assert result.coverage.reasons


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

    # No engine is installed in this environment, so the run cannot resolve a
    # schema. What matters here is that the reply parsed at all.
    assert result.target == "chatty:pipeline"
    assert result.diagnostics


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
