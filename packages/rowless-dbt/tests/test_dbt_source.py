"""Resolving `dbt://` URIs: which evidence answers, and what happens when none does.

The compiled-SQL fallback is mostly exercised against a stub engine,
deliberately: `rowless-duckdb`, `rowless-ibis` and `rowless-sql` all implement
`SupportsSql` and resolve these queries for real, and a dbt test that depended
on one of their answers would stop testing dbt. What is under test here is the
dbt half of that seam - the strategy order, the upstream recursion, the
relation names the inputs are keyed by, and the reasons reported when nothing
can answer. The stub resolves a projection with sqlglot, which is enough to
tell whether this package handed it the right query against the right inputs.

One test does use the real thing, because `rowless-dbt` depends on
`rowless-sql` and so always has an engine that needs no warehouse: it resolves
a model's compiled SQL through the installed adapter, end to end, which is the
assertion that the two halves fit together. The rest of the engine half is
tested in `packages/rowless-sql`, `packages/rowless-duckdb` and
`packages/rowless-ibis`, against the same three-part relation names.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

import pytest
from rowless.adapters.base import Capabilities, DryRunContext
from rowless.adapters.registry import clear_cache, register
from rowless.errors import BadSourceRef, SchemaNotFound
from rowless.ir.dtypes import DType, Kind, TimeUnit, any_, date, datetime, decimal, int_, string
from rowless.ir.schema import Field, Relation, Schema
from rowless.sources.base import SourceRef
from rowless_dbt.manifest import Manifest
from rowless_dbt.source import DbtManifestSource

from tests.conftest import DbtProject, ProjectBuilder


class StubSqlEngine:
    """A minimal `SupportsSql` engine: resolves a projection, runs nothing.

    Not a mock. It parses the query it is given and answers out of the schemas
    it was handed, so a wrong query or a mis-keyed input shows up as a wrong
    answer rather than as a passing assertion.
    """

    name: ClassVar[str] = "sql"
    requires: ClassVar[str] = ""
    capabilities: ClassVar[Capabilities] = Capabilities(sql=True, lazy=True)

    def available(self) -> bool:
        return True

    def handles(self, obj: object) -> bool:
        return isinstance(obj, Schema)

    def materialize(self, schema: Schema, ctx: DryRunContext) -> Any:
        return schema

    def describe(self, obj: object) -> Relation:
        assert isinstance(obj, Schema)
        return Relation(schema=obj, engine=self.name)

    def to_ir(self, native: Any) -> DType:
        return native if isinstance(native, DType) else DType(Kind.UNKNOWN)

    def from_ir(self, dtype: DType) -> Any:
        return dtype

    def run_sql(self, sql: str, inputs: dict[str, object], ctx: DryRunContext) -> Relation:
        from sqlglot import exp, parse_one

        statement = parse_one(sql)
        assert isinstance(statement, exp.Select), sql
        upstream = self._upstream(statement, inputs)
        fields: list[Field] = []
        for projection in statement.selects:
            column = projection.unalias() if isinstance(projection, exp.Alias) else projection
            name = projection.alias_or_name
            if isinstance(column, exp.Column) and upstream is not None and column.name in upstream:
                fields.append(Field(name=name, dtype=upstream[column.name].dtype))
            else:
                fields.append(Field(name=name, dtype=any_()))
        return Relation(schema=Schema(fields=tuple(fields)), engine=self.name)

    def _upstream(self, statement: Any, inputs: dict[str, object]) -> Schema | None:
        """The input schema the query reads from, by whichever name it used."""
        from sqlglot import exp

        table = statement.find(exp.Table)
        if table is None:
            return None
        for spelling in (table.sql(), table.sql().replace('"', ""), table.name):
            found = inputs.get(spelling)
            if isinstance(found, Schema):
                return found
        return None


@pytest.fixture
def sql_engine() -> Iterator[StubSqlEngine]:
    """Register the stub as the `sql` engine for one test."""
    clear_cache()
    engine = StubSqlEngine()
    register("rowless.engines", engine.name, engine)
    yield engine
    clear_cache()


@pytest.fixture
def no_engines(monkeypatch: pytest.MonkeyPatch) -> None:
    """An environment with no engine that can resolve SQL."""
    from rowless.adapters import registry

    monkeypatch.setattr(registry, "engines", lambda **_: {})


def _ref(project: DbtProject, node: str, **options: str) -> SourceRef:
    """A `dbt://` ref pointing at one node of a built project."""
    query = "&".join(f"{key}={value}" for key, value in options.items())
    manifest = f"manifest={project.manifest_path}"
    return SourceRef.parse(f"dbt://{node}?{manifest}" + (f"&{query}" if query else ""))


# --- The strategies ----------------------------------------------------------


def test_a_contract_answers_first(project: DbtProject) -> None:
    """An enforced contract is the only complete column list dbt guarantees."""
    schema = DbtManifestSource().resolve(_ref(project, "model.shop.stg_orders"))

    assert tuple(schema) == ("order_id", "status", "amount_cents", "created_at")
    assert schema["amount_cents"].dtype == decimal(18, 2)
    assert schema["created_at"].dtype == datetime(TimeUnit.MICROSECOND, "UTC")
    assert schema.name == "stg_orders"


def test_the_catalog_answers_when_there_is_no_contract(project: DbtProject) -> None:
    """Real warehouse types, including the column schema.yml forgot."""
    schema = DbtManifestSource().resolve(_ref(project, "model.shop.daily_revenue"))

    assert tuple(schema) == ("day", "revenue", "orders_count")
    assert schema["day"].dtype == date()
    assert schema["orders_count"].dtype == int_(64)


def test_a_bare_model_name_resolves(project: DbtProject) -> None:
    """`dbt://stg_orders` is what people will actually write."""
    schema = DbtManifestSource().resolve(_ref(project, "stg_orders"))

    assert "amount_cents" in schema


def test_declared_columns_are_opt_in(project: DbtProject) -> None:
    """schema.yml columns resolve on request, and are incomplete on purpose.

    The two-column answer here is exactly why this is not the default: the
    model has three.
    """
    schema = DbtManifestSource().resolve(_ref(project, "daily_revenue", columns="declared"))

    assert tuple(schema) == ("day", "revenue")


def test_a_strategy_that_cannot_answer_is_refused_not_approximated(project: DbtProject) -> None:
    """Asking for a contract that is not enforced reports, never falls back."""
    with pytest.raises(SchemaNotFound, match="no enforced contract"):
        DbtManifestSource().resolve(_ref(project, "daily_revenue", columns="contract"))


def test_an_unknown_strategy_names_the_valid_ones(project: DbtProject) -> None:
    with pytest.raises(BadSourceRef, match="auto, contract, catalog, declared, sql"):
        DbtManifestSource().resolve(_ref(project, "stg_orders", columns="vibes"))


# --- The compiled-SQL fallback ----------------------------------------------


def test_compiled_sql_resolves_through_an_engine_that_can_run_it(
    project: DbtProject,
    sql_engine: StubSqlEngine,
) -> None:
    """The path that works on a project with no contracts and no catalog."""
    schema = DbtManifestSource().resolve(_ref(project, "model.shop.finance_kpis"))

    assert tuple(schema) == ("day", "revenue")
    # The dtypes came from the upstream model, which means the stub was handed
    # the right inputs under a name its query actually used.
    assert schema["day"].dtype == date()
    assert schema["revenue"].dtype == decimal(18, 2)


def test_compiled_sql_resolves_through_the_installed_sql_adapter(
    project: DbtProject,
) -> None:
    """The seam end to end: a real adapter, a real query, and no warehouse.

    No stub and no `sql_engine` fixture here. `rowless-dbt` depends on
    `rowless-sql`, so a dbt project always has an engine that can resolve a
    compiled query with nothing but a parser - which is the combination that
    makes an uncontracted model checkable in a CI job with no credentials.

    The model is `finance_kpis`, whose upstream resolves from the catalog, so
    the dtypes asserted here are the warehouse's own answers carried through a
    query sqlglot qualified.
    """
    clear_cache()
    schema = DbtManifestSource().resolve(_ref(project, "model.shop.finance_kpis", columns="sql"))

    assert tuple(schema) == ("day", "revenue")
    assert schema["day"].dtype == date()
    assert schema["revenue"].dtype == decimal(18, 2)
    # Lineage comes back attached, which is what puts a SQL model's columns in
    # the same graph as a Python pipeline's.
    assert schema["revenue"].derived_from == ("revenue",)


def test_the_sql_fallback_recurses_through_upstream_models(
    project: DbtProject,
    sql_engine: StubSqlEngine,
) -> None:
    """A chain with no documentation anywhere still resolves, one hop at a time."""
    schema = DbtManifestSource().resolve(_ref(project, "model.shop.finance_kpis", columns="sql"))

    assert tuple(schema) == ("day", "revenue")


def test_sql_resolution_reports_a_cycle_rather_than_recursing_forever(
    build_project: ProjectBuilder,
    sql_engine: StubSqlEngine,
) -> None:
    """A hand-edited manifest can contain what dbt would have refused to build.

    No catalog here on purpose: with one, the cycle never comes up, because the
    upstream model resolves from the warehouse types before anything has to
    look at its query.
    """
    import json

    project = build_project(catalog=None)
    payload = json.loads(project.manifest_path.read_text(encoding="utf-8"))
    payload["nodes"]["model.shop.daily_revenue"]["depends_on"]["nodes"] = [
        "model.shop.finance_kpis"
    ]
    project.manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SchemaNotFound, match="depends on itself"):
        DbtManifestSource().resolve(_ref(project, "model.shop.finance_kpis", columns="sql"))


def test_with_no_sql_engine_the_finding_says_what_to_install(
    project: DbtProject,
    no_engines: None,
) -> None:
    """An honest report beats a schema nobody could stand behind."""
    with pytest.raises(SchemaNotFound) as caught:
        DbtManifestSource().resolve(_ref(project, "model.shop.finance_kpis"))

    message = str(caught.value)
    assert "no installed engine can resolve SQL" in message
    assert "rowless-sql" in message
    # Every path that was tried is named, so the user can pick one.
    assert "no enforced contract" in message
    assert "catalog.json" in message


def test_an_uncompiled_model_says_to_compile_it(
    project: DbtProject,
    sql_engine: StubSqlEngine,
) -> None:
    """`dbt parse` writes a manifest with no compiled code in it."""
    import json

    payload = json.loads(project.manifest_path.read_text(encoding="utf-8"))
    payload["nodes"]["model.shop.finance_kpis"]["compiled_code"] = ""
    project.manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SchemaNotFound, match="dbt compile"):
        DbtManifestSource().resolve(_ref(project, "model.shop.finance_kpis"))


# --- Locating the manifest, caching and cache keys ---------------------------


def test_available_only_where_there_is_a_manifest(
    project: DbtProject,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(project.root.parent)
    assert DbtManifestSource().available() is False

    monkeypatch.chdir(project.root)
    assert DbtManifestSource().available() is True


def test_an_unknown_node_lists_the_models_that_do_exist(project: DbtProject) -> None:
    with pytest.raises(SchemaNotFound) as caught:
        DbtManifestSource().resolve(_ref(project, "model.shop.stg_ordrs"))

    assert "model.shop.stg_orders" in str(caught.value)


def test_with_no_manifest_the_message_says_how_to_point_at_one(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SchemaNotFound, match="DBT_PROJECT_DIR"):
        DbtManifestSource().resolve(SourceRef.parse("dbt://model.shop.stg_orders"))


def test_the_manifest_is_parsed_once_per_run(
    project: DbtProject,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 40MB manifest re-read per model input is the difference between a
    linter and a chore."""
    from rowless_dbt import manifest as manifest_module

    loads = 0
    original = manifest_module.Manifest.load

    def counted(path: str, **kwargs: Any) -> Manifest:
        nonlocal loads
        loads += 1
        return original(path, **kwargs)

    monkeypatch.setattr(manifest_module.Manifest, "load", counted)
    source = DbtManifestSource()
    source.resolve(_ref(project, "model.shop.stg_orders"))
    source.resolve(_ref(project, "model.shop.daily_revenue"))

    assert loads == 1


def test_the_fingerprint_follows_the_dbt_run_that_wrote_the_manifest(project: DbtProject) -> None:
    """A new `dbt build` invalidates the cached schemas; nothing else does."""
    import json

    source = DbtManifestSource()
    ref = _ref(project, "model.shop.stg_orders")
    before = source.fingerprint(ref)
    assert before
    assert source.fingerprint(ref) == before
    assert source.fingerprint(_ref(project, "stg_orders", columns="declared")) != before

    payload = json.loads(project.manifest_path.read_text(encoding="utf-8"))
    payload["metadata"]["invocation_id"] = "00000000-0000-0000-0000-000000000000"
    project.manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    assert source.fingerprint(ref) != before


def test_the_fingerprint_is_empty_when_there_is_nothing_to_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    assert DbtManifestSource().fingerprint(SourceRef.parse("dbt://model.shop.x")) == ""


# --- Through the registry ----------------------------------------------------


def test_the_entry_point_is_registered(
    project: DbtProject,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`dbt://` resolves through the ordinary source registry, uncached."""
    from rowless.sources.base import resolve

    monkeypatch.chdir(project.root)
    schema = resolve("dbt://model.shop.stg_orders")

    assert tuple(schema) == ("order_id", "status", "amount_cents", "created_at")
    assert schema["status"].dtype == string()
