"""The two fixtures that hand out frames rather than assertions.

Small objects rather than closures, for one reason: a frame needs an engine
and a schema does not. `rowless_source("lake://stg_orders").schema` is a
question about an upstream table - whether the column a repository reads is
still there - and answering it by materializing a frame would mean a test
about a dbt contract could not run in an environment with no adapter
installed.

So `rowless_source` resolves the schema first and materializes second, and
exposes both:

    def test_the_contract_still_holds(rowless_source):
        assert "amount" in rowless_source.schema("lake://stg_orders")

    def test_the_pipeline(rowless_source):
        assert daily_revenue(rowless_source("lake://stg_orders")) is not None
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from rowless.ir.schema import Schema

from rowless_pytest import engines
from rowless_pytest.session import RowlessSession


@dataclass(frozen=True, slots=True)
class FrameBuilder:
    """What the `rowless_frame` fixture hands a test."""

    session: RowlessSession
    engine: str | None = None
    """The engine `@pytest.mark.rowless(engine=...)` named, if any."""

    def __call__(
        self,
        columns: Schema | Mapping[str, str],
        *,
        name: str | None = None,
        engine: str | None = None,
    ) -> Any:
        """A rowless frame with these columns: the engine's own type, no rows.

        `columns` is a `Schema` or the `{"amount": "int64"}` shorthand, and
        `name` labels the relation, which is what a finding about it is
        reported against.
        """
        schema = self.session.schema(columns)
        if name is not None:
            schema = schema.relabel(name)
        return self.session.frame(schema, engine=engine_for(self.session, engine, self.engine))


@dataclass(frozen=True, slots=True)
class SourceResolver:
    """What the `rowless_source` fixture hands a test."""

    session: RowlessSession
    engine: str | None = None
    """The engine `@pytest.mark.rowless(engine=...)` named, if any."""

    def __call__(self, uri: str, *, engine: str | None = None) -> Any:
        """The frame this source URI resolves to."""
        return self.session.frame(
            self.schema(uri),
            engine=engine_for(self.session, engine, self.engine),
        )

    def schema(self, uri: str) -> Schema:
        """The schema alone, which needs no engine installed.

        Through the `[sources]` rewrites and the cache, like the call itself -
        a test that asks what a table looks like should get the answer
        `rowless check` would use.
        """
        return self.session.resolve(uri)


def engine_for(session: RowlessSession, requested: str | None, declared: str | None) -> str:
    """The engine a frame-building fixture should use.

    `engines.require` rather than `engines.choose` alone: there is no pipeline
    here to infer an adapter from, so an environment with four engines in it
    has to be told which one, and is told so by name.
    """
    return engines.require(
        engines.choose(
            requested=requested,
            declared=declared,
            flag=session.engine,
            configured=session.config.engines,
        )
    )
