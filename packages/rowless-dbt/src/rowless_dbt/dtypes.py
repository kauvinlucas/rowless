"""Warehouse type strings to the rowless IR, and dbt adapters to SQL dialects.

A dbt manifest declares column types the way the warehouse spells them:
`varchar(255)`, `numeric(18,2)`, `timestamp with time zone`, `number`,
`array<struct<a int64>>`. There is no single grammar for that - every adapter
has its own - so the parsing is handed to `sqlglot`, which already knows each
dialect and already resolves its defaults. Snowflake `number` comes back as
`DECIMAL(38, 0)` and Snowflake `float` as `DOUBLE`, which is exactly the
dialect knowledge this module must not re-implement.

**The mapping onto the IR lives in `rowless_sql.dtypes`**, which is where it
belongs now that the SQL adapter exists: that module is the one
sqlglot-to-IR mapping in the workspace, in both directions, and this one
delegates to it rather than keeping a second copy. The judgement calls it
makes - an aware timestamp is UTC, a precision is a time unit, a decimal with
no precision is UNKNOWN rather than a guess, a semi-structured type is ANY -
are written down there, next to the lowering that has to agree with them.
`rowless-dbt` depends on `rowless-sql` already, because resolving a model that
declares no columns means running its compiled SQL.

What is left here is the dbt-shaped half:

* **a manifest with no declared type is UNKNOWN, not a guess.** A dbt model
  may document a column without typing it, and a column with no declared type
  must not read as a verified one.
* **the dbt-adapter-to-sqlglot-dialect table**, which is dbt's vocabulary
  rather than SQL's: `sqlserver` is a dbt adapter name and `tsql` is the
  dialect that reads it. `rowless-sql` asks for this answer too, through
  `rowless_sql.dialects.from_dbt`, so a dbt project does not have to repeat
  its warehouse in `rowless.toml`.
"""

from __future__ import annotations

from typing import Any

from rowless.ir.dtypes import DType, unknown

# The canonical mapping, re-exported under the name this module used to define
# it with so a caller that imported it keeps working. One implementation, in
# the adapter that also has to lower the other way.
from rowless_sql.dtypes import parse as _parse_type
from rowless_sql.dtypes import to_ir as dtype_from_sqlglot

__all__ = ["DIALECT_ALIASES", "dtype_from_sql", "dtype_from_sqlglot", "sqlglot_dialect"]

#: dbt adapter name to sqlglot dialect, for the adapters whose names differ or
#: which share a dialect with another engine. An adapter not listed here is
#: assumed to share its name with its dialect, and checked against sqlglot
#: before being used, so a warehouse sqlglot has never heard of degrades to the
#: default dialect rather than to a parse error on every column.
DIALECT_ALIASES: dict[str, str] = {
    "sqlserver": "tsql",
    "mssql": "tsql",
    "synapse": "tsql",
    "glue": "spark",
    "emr": "spark",
    "singlestore": "mysql",
    "mindsdb": "mysql",
    "postgres_legacy": "postgres",
}


def sqlglot_dialect(adapter_type: str) -> str:
    """The sqlglot dialect for a dbt adapter, or an empty string for none.

    Dialects are configuration, not detection: guessing wrong turns a correct
    query into a parse error, so an adapter sqlglot does not know comes back
    empty and the default dialect is used. Empty is also what a manifest with
    no `adapter_type` gets.
    """
    name = (adapter_type or "").strip().lower()
    if not name:
        return ""
    candidate = DIALECT_ALIASES.get(name, name)
    return candidate if _is_known_dialect(candidate) else ""


def _is_known_dialect(name: str) -> bool:
    """True when sqlglot has a dialect by this name."""
    from rowless_sql.dialects import is_known

    return is_known(name)


def dtype_from_sql(spelling: str | None, *, dialect: str = "") -> DType:
    """Parse a warehouse type string into an IR dtype.

    Args:
        spelling: the type as the manifest or the catalog declares it, e.g.
            `numeric(18,2)`. An empty or missing value is UNKNOWN: a dbt model
            may document a column without typing it, and a column with no
            declared type must not read as a verified one.
        dialect: the sqlglot dialect for the dbt adapter in use, from
            `rowless_dbt.discovery.DbtModelDiscovery.dialect_for`. Without it
            the parse still works for the common spellings, but the dialect
            defaults (Snowflake `number`, BigQuery `float64`) are lost.

    Returns:
        The IR dtype, or UNKNOWN carrying the reason it could not be read.
    """
    text = (spelling or "").strip()
    if not text:
        return unknown("the dbt manifest declares no data_type for this column")
    parsed: Any = _parse_type(text, dialect)
    if parsed is None:
        return unknown(f"sqlglot could not parse the SQL type {text!r}")
    return dtype_from_sqlglot(parsed, spelling=text)
