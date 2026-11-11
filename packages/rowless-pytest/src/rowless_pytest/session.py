"""State the rowless fixtures share across one pytest run.

Three things live here, and each of them is here because a per-test fixture
could not hold it:

* the `rowless.toml` config, so a test resolves `lake://stg_orders` to the
  same table `rowless check` does, through the same `[sources]` rewrites;
* the source cache, so a suite with forty contract tests reads a Parquet
  footer or a dbt manifest once rather than forty times, and reads nothing at
  all on the second run;
* the running tally, which is what `--rowless-min-coverage` gates on and what
  the terminal summary reports.

Everything is built on first use. This plugin ships as an entry point, so
pytest loads it in every suite in the environment whether or not that suite
has heard of rowless - and discovering config at startup would turn a broken
`[tool.rowless]` table in some unrelated repository into somebody's failing
test run. Nothing happens until a test asks for a fixture.
"""

from __future__ import annotations

import tempfile
import uuid
from collections.abc import Mapping
from typing import Any

import pytest
from rowless.adapters.base import DryRunContext
from rowless.api import ResolvedInputs, as_schema, resolve_inputs
from rowless.config import Config
from rowless.diagnostics.codes import Code
from rowless.ir.confidence import Confidence, Coverage
from rowless.ir.schema import Schema
from rowless.runtime.dryrun import DryRunResult, dry_run_callable
from rowless.sources.base import SourceCache
from rowless.sources.base import resolve as resolve_source

from rowless_pytest import engines


class RowlessSession:
    """Config, cache and tally for the rowless assertions in one pytest run."""

    def __init__(
        self,
        config: Config,
        *,
        engine: str | None = None,
        min_coverage: float | None = None,
    ) -> None:
        self.config = config

        self.engine = engine
        """What `--rowless-engine` said, if anything. It fills in where a test
        names no engine; a test's own marker still wins over it."""

        self.cache = SourceCache(str(config.cache_path))
        self.checks = 0
        self.coverage = Coverage(confidence=Confidence.EXACT)
        self.duration_ms = 0.0

        self.unverified: list[str] = []
        """Targets that passed only because the test asked for
        `allow_unverified=True`. Counted in the summary: a suite that opted out
        of the rule should have to look at that number."""

        self._min_coverage = min_coverage
        self._workdir: tempfile.TemporaryDirectory[str] | None = None

    # --- construction --------------------------------------------------------

    @classmethod
    def from_pytest(cls, config: pytest.Config) -> RowlessSession:
        """Build from pytest's config: rowless config at the rootdir, plus flags.

        A broken `rowless.toml` raises `ConfigError` out of here, which errors
        every test that asked for a rowless fixture and leaves the rest of the
        suite alone. Reading the file at start-up instead would make a config
        this plugin has no business reading - it is installed in an
        environment, not chosen by a repository - into a failure for a suite
        that never mentioned rowless.
        """
        discovered = Config.discover(str(config.rootpath))
        floor = config.getoption("rowless_min_coverage", default=None)
        return cls(
            discovered,
            engine=config.getoption("rowless_engine", default=None),
            min_coverage=None if floor is None else float(floor),
        )

    # --- inputs --------------------------------------------------------------

    def resolve(self, uri: str) -> Schema:
        """Resolve one source URI the way `rowless check` resolves it.

        Through the `[sources]` rewrite, the cache under `.rowless/`, and
        `root_dir` for relative paths, so a suite run from a subdirectory sees
        what CI sees.
        """
        return resolve_source(
            self.config.resolve_source(uri),
            cache=self.cache,
            base_dir=self.config.root_dir,
        )

    def inputs(
        self,
        target: Any,
        declared: Mapping[str, Schema | Mapping[str, str] | str] | None = None,
    ) -> ResolvedInputs:
        """Resolve a test's inputs, falling back to `@rowless.pipeline`."""
        return resolve_inputs(target, declared, resolver=self.resolve)

    def schema(self, spec: Schema | Mapping[str, str]) -> Schema:
        """A `Schema` from a schema or from the `{"day": "date"}` shorthand."""
        return as_schema(spec)

    # --- running -------------------------------------------------------------

    def dry_run(
        self,
        target: Any,
        inputs: ResolvedInputs,
        *,
        engine: str | None = None,
    ) -> DryRunResult:
        """Dry-run one pipeline in this process.

        `dry_run_callable`, not the sandboxed worker the CLI uses. The code is
        already imported - the test imported it - and a subprocess per
        assertion would cost more than the dry run itself does, which would
        spend the property this plugin exists for. The sandbox is there to
        protect a user from code rowless discovered in a repository; a test
        suite is code they ran on purpose.
        """
        return dry_run_callable(
            target,
            inputs.schemas,
            engine=engine,
            options=self.options_for(engine),
            uris=inputs.uris,
        )

    def frame(self, schema: Schema, *, engine: str | None = None) -> Any:
        """A rowless frame of the engine's own type: full schema, zero rows.

        Built outside a dry-run session, so `rowless.source()` does not
        resolve inside a function called with one of these, and a row-dependent
        branch is not reported. That is what `rowless_check` is for; this is
        for handing a frame straight to a function and asserting on what comes
        back.
        """
        name = engines.require(engine)
        adapter = engines.adapter_for(name)
        return adapter.materialize(schema, self.context(name))

    def context(self, engine: str) -> DryRunContext:
        """A dry-run context for a frame built outside a dry run.

        The workdir is a real directory because an adapter is entitled to use
        it - DuckDB opens a database in it - and it lives as long as the pytest
        session, since the frames handed out do too.
        """
        if self._workdir is None:
            self._workdir = tempfile.TemporaryDirectory(prefix="rowless-pytest-")
        return DryRunContext(
            run_id=uuid.uuid4().hex[:12],
            workdir=self._workdir.name,
            options=dict(self.options_for(engine)),
        )

    def options_for(self, engine: str | None) -> Mapping[str, Any]:
        """Adapter options from `[engine.<name>]` in `rowless.toml`."""
        return self.config.engine_options.get(engine or "", {})

    # --- tally ---------------------------------------------------------------

    def record(self, result: DryRunResult, *, allowed_unverified: bool = False) -> None:
        """Add one assertion's result to the session tally.

        Called for failures too. A coverage figure computed only over the
        assertions that passed would improve every time a contract broke,
        which is the trap `rowless.check` avoids when a pipeline will not
        resolve.
        """
        self.checks += 1
        self.duration_ms += result.duration_ms
        self.coverage = self.coverage.merge(result.coverage)
        if allowed_unverified:
            self.unverified.append(result.target)

    @property
    def floor(self) -> float:
        """The coverage floor in force: the flag if given, else `min_coverage`."""
        return self.config.min_coverage if self._min_coverage is None else self._min_coverage

    def shortfall(self) -> str | None:
        """Why the coverage gate failed, or None when it did not.

        No assertions means no gate. A suite that ran no rowless tests has
        proved nothing about coverage, but failing it over a floor it never
        got the chance to meet would only teach people to drop the flag.
        """
        if not self.checks or self.floor <= 0:
            return None
        ratio = self.coverage.typed_ratio
        if ratio >= self.floor:
            return None
        return (
            f"{Code.COVERAGE_BELOW_THRESHOLD.value} typed coverage {_percent(ratio)} is below "
            f"the floor of {_percent(self.floor)}, over {self.checks} rowless contract test(s)"
        )

    def summary(self) -> str:
        """The one line the terminal summary prints."""
        parts = [
            f"{self.checks} contract test(s)",
            f"{self.coverage.columns_total} column(s)",
            f"{_percent(self.coverage.typed_ratio)} typed",
            f"{self.duration_ms:.0f}ms in dry runs",
        ]
        if self.unverified:
            parts.append(f"{len(self.unverified)} unverified result(s) allowed")
        return "rowless: " + ", ".join(parts)

    def close(self) -> None:
        """Drop the scratch directory the frames were built against."""
        if self._workdir is not None:
            self._workdir.cleanup()
            self._workdir = None


class SessionHolder:
    """A `RowlessSession` built on first use, and only if a test wants one.

    The lazy half of the module docstring's argument, and also what lets the
    hooks ask whether any rowless test ran without answering it themselves.

    One holder per pytest process. Under `pytest-xdist` that is one per
    worker, so the coverage gate applies to each worker's share of the suite
    rather than to the whole of it - the same limitation `pytest-cov` has
    without its own aggregation step, and the reason anything stricter than a
    smoke floor belongs in `rowless check`.
    """

    def __init__(self, config: pytest.Config) -> None:
        self._config = config
        self._session: RowlessSession | None = None

    def get(self) -> RowlessSession:
        """The session, built now if this is the first fixture to ask for it."""
        if self._session is None:
            self._session = RowlessSession.from_pytest(self._config)
        return self._session

    @property
    def active(self) -> RowlessSession | None:
        """The session if one was ever built, else None."""
        return self._session

    def close(self) -> None:
        """Release what the session held, if anything was ever built."""
        if self._session is not None:
            self._session.close()


def _percent(ratio: float) -> str:
    """A coverage ratio as a percentage, the way the CLI prints it."""
    return f"{ratio:.0%}"
