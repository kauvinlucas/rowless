"""Configuration loading, source rewrites and suppression."""

from __future__ import annotations

from pathlib import Path

import pytest
from rowless.config import Config, inline_ignores
from rowless.diagnostics.codes import Code
from rowless.diagnostics.models import Level
from rowless.errors import ConfigError


def test_zero_config_works(tmp_path: Path) -> None:
    """A repo with no config file is the supported case, not an error."""
    config = Config.discover(str(tmp_path))

    assert config.roots == (".",)
    assert config.fail_on_unverified is True
    assert config.min_coverage == 0.0


def test_loads_rowless_toml(tmp_path: Path) -> None:
    """`rowless.toml` is read as the config table itself."""
    (tmp_path / "rowless.toml").write_text(
        "\n".join(
            [
                'roots = ["pipelines"]',
                'engines = ["polars"]',
                "min_coverage = 0.8",
                "fail_on_unverified = false",
                "",
                "[sources]",
                '"lake://*" = "s3://warehouse/{name}/"',
                "",
                "[engine.pandas]",
                "arrow_dtypes = true",
            ]
        ),
        encoding="utf-8",
    )

    config = Config.discover(str(tmp_path))

    assert config.roots == ("pipelines",)
    assert config.engines == ("polars",)
    assert config.min_coverage == 0.8
    assert config.fail_on_unverified is False
    assert config.engine_options["pandas"]["arrow_dtypes"] is True
    assert config.root_dir == str(tmp_path.resolve())


def test_loads_pyproject_table_and_skips_unrelated_ones(tmp_path: Path) -> None:
    """A pyproject with no `[tool.rowless]` belongs to somebody else."""
    (tmp_path / "pyproject.toml").write_text("[tool.ruff]\nline-length = 100\n", encoding="utf-8")
    nested = tmp_path / "pipelines"
    nested.mkdir()

    assert Config.discover(str(nested)).roots == (".",)

    (tmp_path / "pyproject.toml").write_text(
        '[tool.rowless]\nroots = ["pipelines"]\n', encoding="utf-8"
    )
    assert Config.discover(str(nested)).roots == ("pipelines",)


def test_rejects_a_typo_rather_than_ignoring_it(tmp_path: Path) -> None:
    """A silently ignored key is a setting the user thinks is applied."""
    (tmp_path / "rowless.toml").write_text("min_covrage = 0.9\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="unknown key"):
        Config.discover(str(tmp_path))


def test_rejects_an_unknown_diagnostic_code(tmp_path: Path) -> None:
    """A typo in a CI allowlist silently stops gating on a real finding."""
    (tmp_path / "rowless.toml").write_text('error_codes = ["RWL999"]\n', encoding="utf-8")

    with pytest.raises(ConfigError, match="unknown diagnostic code"):
        Config.discover(str(tmp_path))


def test_rejects_a_coverage_floor_outside_zero_to_one(tmp_path: Path) -> None:
    """A floor of 80 would never be met, and would look like a rowless bug."""
    (tmp_path / "rowless.toml").write_text("min_coverage = 80\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="between 0 and 1"):
        Config.discover(str(tmp_path))


def test_source_rewrites_apply_the_first_match() -> None:
    """Pipeline code names a table; config says where the table lives."""
    config = Config(source_map={"lake://*": "json://schemas/{name}.json"})

    assert config.resolve_source("lake://stg_orders") == "json://schemas/stg_orders.json"
    assert config.resolve_source("file://data.parquet") == "file://data.parquet"


def test_level_for_applies_the_unverified_policy() -> None:
    """`fail_on_unverified` is the whole reason this lives in config."""
    strict = Config()
    lenient = Config(fail_on_unverified=False)

    assert strict.level_for(Code.UNVERIFIED_CHANGE) is Level.UNVERIFIED
    assert lenient.level_for(Code.UNVERIFIED_CHANGE) is Level.WARNING
    assert strict.level_for(Code.COLUMN_NOT_FOUND) is Level.ERROR
    assert strict.level_for(Code.WRITE_RECORDED) is Level.INFO
    assert strict.level_for(Code.ANY_COLUMN_IN_OUTPUT) is Level.WARNING


def test_ignored_codes_are_suppressed_but_internal_ones_are_not() -> None:
    """A user can turn a finding off; they cannot turn off "rowless broke"."""
    config = Config(ignore_codes=frozenset({Code.COLUMN_NOT_FOUND}))

    assert config.level_for(Code.COLUMN_NOT_FOUND) is None
    assert config.level_for(Code.WORKER_CRASHED) is Level.ERROR


def test_inline_ignores_are_line_and_code_specific(tmp_path: Path) -> None:
    """A tool with no way to say "not now" gets deleted instead of configured."""
    module = tmp_path / "revenue.py"
    module.write_text(
        "\n".join(
            [
                "amount = df['amount']  # rowless: ignore[RWL101]",
                "other = df['other']  # rowless: ignore[RWL101, RWL201]",
                "future = df['x']  # rowless: ignore[RWL999]",
                "plain = df['y']",
            ]
        ),
        encoding="utf-8",
    )

    found = inline_ignores(str(module))

    assert found[1] == [Code.COLUMN_NOT_FOUND]
    assert found[2] == [Code.COLUMN_NOT_FOUND, Code.ROW_DEPENDENT_CONTROL_FLOW]
    assert 3 not in found, "an unknown code is a newer rowless, not a user error"
    assert 4 not in found


def test_exclude_globs_keep_tests_out(tmp_path: Path) -> None:
    """The default excludes are what make a first run readable."""
    config = Config(root_dir=str(tmp_path))

    assert config.is_excluded(tmp_path / "tests" / "test_revenue.py")
    assert not config.is_excluded(tmp_path / "pipelines" / "revenue.py")


def test_cli_overrides_win_over_the_file(tmp_path: Path) -> None:
    """A flag is how a user answers "just this once"."""
    config = Config(engines=("polars",), min_coverage=0.5, root_dir=str(tmp_path))

    overridden = config.with_overrides(engines=("duckdb",), min_coverage=0.9)

    assert overridden.engines == ("duckdb",)
    assert overridden.min_coverage == 0.9
    assert config.engines == ("polars",), "the original is immutable"
