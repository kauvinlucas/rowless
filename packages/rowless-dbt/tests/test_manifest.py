"""Manifest parsing, against committed fixtures per dbt version."""

from __future__ import annotations

from pathlib import Path

import pytest
from rowless.errors import ConfigError
from rowless.ir.dtypes import TimeUnit, date, datetime, decimal, int_, string
from rowless_dbt.manifest import (
    SUPPORTED_VERSIONS,
    Manifest,
    find_manifest,
    project_name_of,
)

from tests.conftest import DbtProject, ProjectBuilder

FIXTURES = Path(__file__).parent / "fixtures"


def test_loads_v12_manifest(project: DbtProject) -> None:
    """A v12 manifest parses, and model schemas come from contracts first."""
    manifest = project.manifest()

    assert manifest.version == "v12"
    assert manifest.project_name == "shop"
    assert manifest.adapter_type == "postgres"
    assert manifest.dialect == "postgres"
    assert manifest.invocation_id == "4b4f6d0e-9c21-4a7f-8f51-2c0d6b1a7e33"

    node = manifest.nodes["model.shop.stg_orders"]
    assert node.schema_source == "contract"
    assert node.has_contract is True
    assert tuple(node.schema) == ("order_id", "status", "amount_cents", "created_at")
    assert node.schema["order_id"].dtype == int_(64)
    assert node.schema["status"].dtype == string()
    assert node.schema["amount_cents"].dtype == decimal(18, 2)
    assert node.schema["created_at"].dtype == datetime(TimeUnit.MICROSECOND, "UTC")


def test_contract_columns_carry_nullability_and_where_they_came_from(project: DbtProject) -> None:
    """A not_null constraint and the schema.yml that declared it both survive."""
    node = project.manifest().nodes["model.shop.stg_orders"]

    assert node.schema["order_id"].nullable is False
    assert node.schema["status"].nullable is True
    assert node.schema["amount_cents"].description is not None

    origin = node.schema["order_id"].origin
    assert origin is not None
    assert origin.kind == "contract"
    assert origin.ref == "model.shop.stg_orders"
    assert origin.file is not None and origin.file.endswith("schema.yml")


def test_a_model_with_only_a_catalog_uses_it(project: DbtProject) -> None:
    """Without a contract, the warehouse types from catalog.json answer."""
    manifest = project.manifest()
    node = manifest.nodes["model.shop.daily_revenue"]

    assert node.has_contract is False
    assert node.schema_source == "catalog"
    # Three columns, where schema.yml documents only two: the catalog is the
    # one source that is complete.
    assert tuple(node.schema) == ("day", "revenue", "orders_count")
    assert node.schema["day"].dtype == date()
    assert node.schema["orders_count"].dtype == int_(64)


def test_a_model_with_neither_has_no_schema_but_keeps_its_sql(project: DbtProject) -> None:
    """An undocumented model is empty and honest about it."""
    node = project.manifest().nodes["model.shop.finance_kpis"]

    assert node.schema_source == "none"
    assert not node.schema.fields
    assert node.compiled_sql is not None and "daily_revenue" in node.compiled_sql


def test_declared_columns_are_kept_but_not_used_as_the_schema(project: DbtProject) -> None:
    """schema.yml columns are available on request, never by default.

    dbt does not require a declaration to be complete, and a schema that is
    quietly missing columns turns correct pipelines into false findings.
    """
    manifest = project.manifest()
    node = manifest.nodes["model.shop.daily_revenue"]

    assert [column.name for column in node.columns] == ["day", "revenue"]
    declared = manifest.declared_schema("model.shop.daily_revenue")
    assert declared is not None
    assert tuple(declared) == ("day", "revenue")
    # The catalog answered for the node itself, and it has the third column.
    assert "orders_count" in node.schema


def test_catalog_is_optional(build_project: ProjectBuilder) -> None:
    """A project that never ran `dbt docs generate` still parses."""
    project = build_project(catalog=None)
    manifest = project.manifest()

    assert manifest.catalog == {}
    assert manifest.catalog_schema("model.shop.daily_revenue") is None
    assert manifest.nodes["model.shop.daily_revenue"].schema_source == "none"
    # The contract model does not need the catalog.
    assert manifest.nodes["model.shop.stg_orders"].schema_source == "contract"


def test_sources_and_seeds_are_nodes_too(project: DbtProject) -> None:
    """A pipeline can read a dbt source or a seed, so both are parsed."""
    manifest = project.manifest()

    source_node = manifest.nodes["source.shop.raw.orders"]
    assert source_node.resource_type == "source"
    assert source_node.columns[0].name == "order_id"
    assert source_node.relation_name == '"shop"."raw"."orders"'
    # A source has no contract mechanism at all, so the catalog is the only
    # evidence that is not a hand-written declaration.
    assert source_node.schema_source == "catalog"
    assert tuple(source_node.schema) == ("order_id", "status", "amount_cents", "created_at")

    seed = manifest.nodes["seed.shop.country_codes"]
    assert seed.resource_type == "seed"
    assert seed.schema_source == "catalog"
    assert tuple(seed.schema) == ("code", "country")


def test_test_nodes_are_skipped(project: DbtProject) -> None:
    """A dbt test produces no relation, so it is not a node rowless carries."""
    manifest = project.manifest()

    assert not [unique_id for unique_id in manifest.nodes if unique_id.startswith("test.")]


def test_models_come_back_in_dependency_order(project: DbtProject) -> None:
    """Upstream before downstream, so one pass can resolve the chain."""
    ordered = [node.unique_id for node in project.manifest().models()]

    assert ordered.index("model.shop.stg_orders") < ordered.index("model.shop.daily_revenue")
    assert ordered.index("model.shop.daily_revenue") < ordered.index("model.shop.finance_kpis")
    assert "seed.shop.country_codes" not in ordered


def test_models_with_a_cycle_are_still_listed(project: DbtProject, tmp_path: Path) -> None:
    """A hand-edited manifest with a cycle must not lose models silently."""
    import json

    payload = json.loads(project.manifest_path.read_text(encoding="utf-8"))
    payload["nodes"]["model.shop.stg_orders"]["depends_on"]["nodes"] = ["model.shop.finance_kpis"]
    project.manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    listed = {node.unique_id for node in project.manifest().models()}
    assert "model.shop.stg_orders" in listed
    assert "model.shop.finance_kpis" in listed


def test_python_models_are_parsed_as_python(project: DbtProject) -> None:
    """A dbt Python model is a Python pipeline, and the reader says so."""
    node = project.manifest().nodes["model.shop.revenue_report"]

    assert node.language == "python"
    assert node.path is not None and node.path.endswith(".py")


def test_node_for_accepts_a_unique_id_or_a_bare_name(project: DbtProject) -> None:
    """Nobody wants to type the package name into every URI."""
    manifest = project.manifest()

    assert manifest.node_for("model.shop.stg_orders") is not None
    assert manifest.node_for("stg_orders") is manifest.node_for("model.shop.stg_orders")
    assert manifest.node_for("orders") is manifest.nodes["source.shop.raw.orders"]
    assert manifest.node_for("no_such_model") is None


def test_loads_v11_manifest_with_snowflake_types(build_project: ProjectBuilder) -> None:
    """v11 parses, and the Snowflake spellings resolve through the dialect."""
    manifest = build_project("manifest_v11.json", catalog=None, name="warehouse").manifest()

    assert manifest.version == "v11"
    assert manifest.adapter_type == "snowflake"
    assert manifest.dialect == "snowflake"

    schema = manifest.nodes["model.warehouse.dim_customers"].schema
    # Snowflake `number` has a dialect default of (38, 0), which sqlglot knows
    # and rowless must not re-invent.
    assert schema["customer_id"].dtype == decimal(38, 0)
    assert schema["customer_id"].nullable is False
    assert schema["country"].dtype == string()
    # Snowflake `float` is a 64-bit double, unlike the SQL standard spelling.
    assert schema["lifetime_value"].dtype.bits == 64
    assert schema["signed_up_at"].dtype == datetime(TimeUnit.MICROSECOND, "UTC")
    # `variant` holds anything, which is ANY: it lowers coverage rather than
    # failing a check or passing one it has not earned.
    assert schema["traits"].dtype.is_any


def test_every_supported_version_has_a_fixture() -> None:
    """A supported version with no fixture is a version nobody tested."""
    for version in SUPPORTED_VERSIONS:
        assert (FIXTURES / f"manifest_{version}.json").is_file(), version


def test_rejects_unknown_version(build_project: ProjectBuilder) -> None:
    """An unsupported schema version raises rather than half-parsing."""
    project = build_project("manifest_unsupported.json", catalog=None, name="legacy")

    with pytest.raises(ConfigError) as caught:
        project.manifest()
    message = str(caught.value)
    assert "v7" in message
    assert "v11, v12" in message


def test_an_unreadable_manifest_names_the_file(tmp_path: Path) -> None:
    """Every failure says which file it was reading."""
    with pytest.raises(ConfigError, match="cannot read dbt manifest"):
        Manifest.load(str(tmp_path / "nope.json"))

    broken = tmp_path / "manifest.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError, match="is not valid JSON"):
        Manifest.load(str(broken))


def test_find_manifest_walks_up_to_the_project_root(
    project: DbtProject,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Running rowless from a subdirectory finds the project it is inside."""
    deep = project.root / "models" / "marts"
    monkeypatch.chdir(deep)

    assert find_manifest() == str(project.manifest_path)


def test_find_manifest_honours_dbt_environment_variables(
    project: DbtProject,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A CI job that runs dbt has already said where the project is."""
    monkeypatch.chdir(project.root.parent)
    assert find_manifest() is None

    monkeypatch.setenv("DBT_PROJECT_DIR", str(project.root))
    assert find_manifest() == str(project.manifest_path)

    monkeypatch.delenv("DBT_PROJECT_DIR")
    monkeypatch.setenv("DBT_TARGET_PATH", str(project.root / "target"))
    assert find_manifest() == str(project.manifest_path)


def test_find_manifest_reads_a_configured_target_path(
    build_project: ProjectBuilder,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`target-path:` in dbt_project.yml moves the artifacts."""
    project = build_project(target_path="artifacts")
    monkeypatch.chdir(project.root)

    assert find_manifest() == str(project.manifest_path)


def test_find_manifest_takes_an_explicit_path(
    project: DbtProject,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit answer beats every guess, directory or file."""
    monkeypatch.chdir(project.root.parent)

    assert find_manifest(target=str(project.manifest_path)) == str(project.manifest_path)
    assert find_manifest(target=str(project.root / "target")) == str(project.manifest_path)
    assert find_manifest(target=str(project.root / "nowhere")) is None


def test_project_name_is_read_without_a_yaml_parser(project: DbtProject) -> None:
    """One shallow key, matched as text, costs nobody a YAML dependency."""
    assert project_name_of(str(project.root)) == "shop"
    assert project_name_of(str(project.root / "models")) == ""
