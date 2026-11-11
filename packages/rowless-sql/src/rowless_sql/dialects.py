"""Which dialect a query is read in, and where that answer comes from.

Dialects are configuration, not detection. Guessing wrong turns a correct query
into a parse error - Snowflake's `::` casts, BigQuery's backticks and T-SQL's
`TOP` are each a syntax error in generic SQL - and a checker that reports valid
SQL as broken is a checker nobody runs twice. So nothing here sniffs the query
text. The answer comes from one of three places, in order:

1. `[engine.sql] dialect` in `rowless.toml`, which is the explicit answer and
   always wins.
2. the dbt manifest, when `rowless-dbt` is installed and a project is findable.
   dbt knows which warehouse it compiled for, and a dbt project that has to
   repeat it in `rowless.toml` is configuration nobody will keep in sync. The
   mapping from a dbt adapter name to a sqlglot dialect lives in
   `rowless_dbt.dtypes`, next to the type strings that need the same answer.
3. generic SQL, which reads the large majority of models and is the honest
   default: no dialect named means no dialect assumed.

A name sqlglot does not know is a **configuration error**, reported as one.
Falling back to generic SQL after a typo would turn `dialect = "snowflak"` into
a pile of parse failures a hundred lines away from the setting that caused
them.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rowless.errors import ConfigError

#: The dbt answer, worked out at most once per process. Reading a manifest is
#: not free, and a run resolves many models against one project.
_FROM_DBT: str | None = None


def for_options(options: Mapping[str, Any]) -> str:
    """The sqlglot dialect for this run, or an empty string for generic SQL.

    Args:
        options: the adapter's options, which the runtime fills from
            `[engine.sql]` in `rowless.toml`.

    Raises:
        rowless.errors.ConfigError: `dialect` names something sqlglot has no
            dialect for. The message lists what it does have.
    """
    configured = str(options.get("dialect") or "").strip()
    if configured:
        if not is_known(configured):
            raise ConfigError(
                f"[engine.sql] dialect = {configured!r} is not a dialect sqlglot knows. "
                f"Available: {', '.join(known_dialects())}"
            )
        return configured
    return from_dbt()


def is_known(name: str) -> bool:
    """True when sqlglot has a dialect by this name."""
    return name.strip().lower() in set(known_dialects())


def known_dialects() -> tuple[str, ...]:
    """Every dialect name sqlglot accepts, sorted, for a message to list."""
    try:
        from sqlglot.dialects.dialect import Dialects
    except ImportError:  # pragma: no cover - only without sqlglot
        return ()
    return tuple(sorted(str(dialect.value) for dialect in Dialects if dialect.value))


def from_dbt() -> str:
    """The dialect of the dbt project in the working tree, or an empty string.

    Returns an empty string when `rowless-dbt` is not installed, when there is
    no project to find, or when the manifest names a warehouse sqlglot has no
    dialect for - all three of which mean "read it as generic SQL" rather than
    "fail".
    """
    global _FROM_DBT

    if _FROM_DBT is not None:
        return _FROM_DBT
    _FROM_DBT = _read_dbt_dialect()
    return _FROM_DBT


def forget_dbt() -> None:
    """Drop the cached dbt answer. For tests, which write manifests in place."""
    global _FROM_DBT

    _FROM_DBT = None


def _read_dbt_dialect() -> str:
    """Ask `rowless-dbt` what the project compiled for, without requiring it.

    Imported inside the function on purpose. `rowless-dbt` depends on this
    distribution, so importing it at module scope would be a cycle; and an
    install with no dbt in it should pay nothing for this integration.
    """
    try:
        from rowless_dbt.manifest import Manifest, find_manifest
    except ImportError:
        return ""
    try:
        path = find_manifest()
        if path is None:
            return ""
        return str(Manifest.load(path).dialect)
    except Exception:
        # A manifest that cannot be read is reported by `rowless-dbt` itself
        # when a pipeline needs it, with the version in the message. Working
        # out a dialect is not the place to fail a run.
        return ""
