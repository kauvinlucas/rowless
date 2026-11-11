"""Schemas from a dbt manifest: `dbt://model.shop.stg_orders`.

The highest-value source in the project, because it is what connects the two
halves of the gap the proposal describes. A Python pipeline reading
`dbt://model.shop.stg_orders` gets the column that a SQL model declares, so
renaming it in SQL breaks the Python check in the same PR.

## Where the manifest comes from

In order, so an explicit answer always beats a guess: a `?manifest=` option on
the URI, then `DBT_TARGET_PATH` or `DBT_PROJECT_DIR` (dbt's own variables,
which a CI job that runs dbt has already set), then the nearest
`dbt_project.yml` walking up from the working directory.

## Which columns answer

`?columns=` picks the evidence, and `auto` is the default:

| value | behaviour |
| --- | --- |
| `auto` | an enforced contract, then the catalog, then the compiled SQL |
| `contract` | only an enforced contract |
| `catalog` | only `target/catalog.json` |
| `declared` | the `schema.yml` columns, contract or not. Opt-in: see below |
| `sql` | only the compiled SQL, through an engine that can run it |

`declared` is not in `auto` on purpose. dbt lets a project document three
columns of a twelve-column model, and nothing in the manifest says whether a
column list is complete. Feeding a nine-column-short schema to a dry run turns
every pipeline that reads one of the missing columns into a false
`ColumnNotFound`, and false positives are how a checker gets switched off. A
project whose documentation *is* complete can say so with
`dbt://model.shop.stg_orders?columns=declared`, or by a `[sources]` rewrite
that adds it for every model.

When nothing can answer, this source raises `SchemaNotFound` naming the four
ways to fix it rather than returning a schema it cannot stand behind.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, ClassVar

from rowless.adapters.base import DryRunContext, SupportsSql
from rowless.errors import BadSourceRef, SchemaNotFound
from rowless.ir.schema import Schema
from rowless.sources.base import SourceRef

from rowless_dbt.manifest import Manifest, ManifestNode, find_manifest

#: Resolution strategies, in the order `auto` tries them.
AUTO_ORDER: tuple[str, ...] = ("contract", "catalog", "sql")

#: Everything `?columns=` accepts.
STRATEGIES: tuple[str, ...] = ("auto", "contract", "catalog", "declared", "sql")

#: Parsed manifests, keyed by path and invalidated by mtime. `rowless check`
#: resolves one URI per model input, and re-reading a 40MB manifest for each of
#: them is the difference between a linter and a chore.
_LOADED: dict[str, tuple[float, Manifest]] = {}


class DbtManifestSource:
    """Resolve `dbt://` URIs against `target/manifest.json`."""

    name: ClassVar[str] = "dbt"
    schemes: ClassVar[tuple[str, ...]] = ("dbt",)
    requires: ClassVar[str] = ""

    def available(self) -> bool:
        """True when a manifest is found at the configured path."""
        try:
            return find_manifest() is not None
        except Exception:
            # `available()` is called while listing sources, including by
            # `rowless doctor`, and must not raise whatever the filesystem
            # does.
            return False

    def resolve(self, ref: SourceRef) -> Schema:
        """Prefer a declared contract, then the catalog, then compiled SQL.

        Falling back to resolving the compiled SQL through the sql adapter is
        what makes this work on projects with no contracts and no catalog,
        which is most projects.
        """
        manifest = self._manifest(ref)
        node = manifest.node_for(ref.path)
        if node is None:
            raise SchemaNotFound(
                f"{ref.uri}: no node {ref.path!r} in {manifest.path}. "
                f"Known models: {_sample(manifest)}"
            )
        strategy = self._strategy(ref)
        reasons: list[str] = []
        schema = self._schema_for(manifest, node, strategy=strategy, reasons=reasons, seen=set())
        if schema is not None:
            return schema
        raise SchemaNotFound(
            f"{ref.uri}: {manifest.path} has no columns for {node.unique_id}. " + "; ".join(reasons)
        )

    def fingerprint(self, ref: SourceRef) -> str:
        """`metadata.invocation_id` plus the manifest mtime."""
        path = self._manifest_path(ref)
        if path is None:
            return ""
        try:
            stamp = Path(path).stat().st_mtime_ns
        except OSError:
            return ""
        manifest = self._manifest(ref)
        return f"{manifest.invocation_id}:{stamp}:{self._strategy(ref)}"

    # --- Resolution ----------------------------------------------------------

    def _schema_for(
        self,
        manifest: Manifest,
        node: ManifestNode,
        *,
        strategy: str,
        reasons: list[str],
        seen: set[str],
    ) -> Schema | None:
        """The schema for one node under one strategy, or None with a reason."""
        wanted = AUTO_ORDER if strategy == "auto" else (strategy,)
        for attempt in wanted:
            schema = self._attempt(manifest, node, attempt, reasons=reasons, seen=seen)
            if schema is not None and schema.fields:
                return schema if schema.name else schema.relabel(node.name)
        return None

    def _attempt(
        self,
        manifest: Manifest,
        node: ManifestNode,
        strategy: str,
        *,
        reasons: list[str],
        seen: set[str],
    ) -> Schema | None:
        """One strategy, with the reason recorded when it cannot answer."""
        match strategy:
            case "contract":
                if node.schema_source == "contract":
                    return node.schema
                reasons.append(f"{node.unique_id} has no enforced contract with column types")
                return None
            case "catalog":
                found = manifest.catalog_schema(node.unique_id)
                if found is not None:
                    return found
                reasons.append(
                    "target/catalog.json has no entry for it (run `dbt docs generate` to write one)"
                )
                return None
            case "declared":
                found = manifest.declared_schema(node.unique_id)
                if found is not None:
                    return found
                reasons.append("schema.yml declares no columns for it")
                return None
            case "sql":
                return self._from_sql(manifest, node, reasons=reasons, seen=seen)
            case _:
                raise BadSourceRef(
                    f"unknown columns={strategy!r}; use one of {', '.join(STRATEGIES)}"
                )

    def _from_sql(
        self,
        manifest: Manifest,
        node: ManifestNode,
        *,
        reasons: list[str],
        seen: set[str],
    ) -> Schema | None:
        """Resolve the node's compiled SQL against its upstream schemas.

        The path that works on a project with no contracts and no catalog,
        which is most of them. It needs an engine that implements
        `SupportsSql` - `rowless-sql` resolves it with sqlglot and no
        warehouse, `rowless-duckdb` with a real binder - and reports which
        piece is missing when there is none.
        """
        if node.unique_id in seen:
            reasons.append(f"{node.unique_id} depends on itself through {' -> '.join(seen)}")
            return None
        if not node.compiled_sql:
            reasons.append(f"{node.unique_id} is not compiled (run `dbt compile`)")
            return None
        adapter = _sql_engine()
        if adapter is None:
            reasons.append(
                "no installed engine can resolve SQL: install rowless-sql for the "
                "credential-free path, rowless-duckdb to use a real binder, or rowless-ibis to "
                "resolve it through ibis's own SQL parser with no database"
            )
            return None

        ctx = DryRunContext(run_id="dbt-source", workdir=tempfile.gettempdir())
        inputs: dict[str, Any] = {}
        for parent_id in node.depends_on:
            parent = manifest.nodes.get(parent_id)
            if parent is None:
                continue
            upstream = self._schema_for(
                manifest,
                parent,
                strategy="auto",
                reasons=reasons,
                seen={*seen, node.unique_id},
            )
            if upstream is None:
                reasons.append(f"upstream {parent_id} did not resolve")
                return None
            relation = adapter.materialize(upstream, ctx)
            for key in _relation_keys(parent):
                inputs[key] = relation

        try:
            resolved = adapter.run_sql(node.compiled_sql, inputs, ctx)
            return Schema(fields=resolved.schema.fields, name=node.name)
        except Exception as exc:
            # A SQL engine that cannot resolve this query is reported, not
            # raised: the caller turns the collected reasons into one
            # SchemaNotFound naming every path that was tried.
            reasons.append(f"{type(exc).__name__} resolving its compiled SQL: {exc}")
            return None

    # --- Manifest ------------------------------------------------------------

    def _manifest(self, ref: SourceRef) -> Manifest:
        """The parsed manifest for this ref, cached by path and mtime."""
        path = self._manifest_path(ref)
        if path is None:
            raise SchemaNotFound(
                f"{ref.uri}: no dbt manifest found. Point at one with "
                "dbt://<node>?manifest=target/manifest.json, set DBT_PROJECT_DIR, or run "
                "rowless from inside the dbt project"
            )
        try:
            stamp = Path(path).stat().st_mtime_ns / 1e9
        except OSError as exc:
            raise SchemaNotFound(f"{ref.uri}: cannot read {path}: {exc}") from exc
        cached = _LOADED.get(path)
        if cached is not None and cached[0] == stamp:
            return cached[1]
        manifest = Manifest.load(path)
        _LOADED[path] = (stamp, manifest)
        return manifest

    def _manifest_path(self, ref: SourceRef) -> str | None:
        """Where to read the manifest for this ref, if anywhere."""
        return find_manifest(target=_option(ref, "manifest"))

    def _strategy(self, ref: SourceRef) -> str:
        """The `?columns=` strategy, defaulting to `auto`."""
        wanted = (_option(ref, "columns") or "auto").strip().lower()
        if wanted not in STRATEGIES:
            raise BadSourceRef(
                f"{ref.uri}: unknown columns={wanted!r}; use one of {', '.join(STRATEGIES)}"
            )
        return wanted


def clear_cache() -> None:
    """Forget every parsed manifest. For tests, which rewrite them in place."""
    _LOADED.clear()


def _option(ref: SourceRef, key: str) -> str | None:
    """One URI option, or None."""
    value = ref.options.get(key)
    return str(value) if value else None


def _sql_engine() -> Any:
    """An installed engine that can resolve a SQL string, or None.

    Typed loosely on purpose: what is needed is an `EngineAdapter` that *also*
    implements `SupportsSql` - `materialize` to turn a schema into whatever the
    engine wants, `run_sql` to resolve the query - and the intersection of two
    protocols is not a type that can be written down here.

    `rowless-sql` first: it is the credential-free path, and a dbt check that
    needs a warehouse connection is a dbt check that will not run in CI. A
    real binder is the better answer where one is available, so DuckDB is
    second, and anything else that claims the capability is third - which in
    practice means `rowless-ibis`, whose seam parses the query with sqlglot and
    types it with ibis's rules. It needs no database either, and it can be
    pointed at the dialect dbt compiled for; what it does not have is a binder,
    so DuckDB stays ahead of it wherever DuckDB can read the query at all.
    """
    from rowless.adapters.registry import engines

    available = engines()
    for name in ("sql", "duckdb"):
        candidate = available.get(name)
        if _claims_sql(candidate):
            return candidate
    for candidate in available.values():
        if _claims_sql(candidate):
            return candidate
    return None


def _claims_sql(adapter: object) -> bool:
    """True when an adapter both declares and implements the SQL capability."""
    if adapter is None:
        return False
    capabilities = getattr(adapter, "capabilities", None)
    return bool(getattr(capabilities, "sql", False)) and isinstance(adapter, SupportsSql)


def _relation_keys(node: ManifestNode) -> tuple[str, ...]:
    """Every name compiled SQL might refer to this node by.

    dbt compiles `ref()` into the fully qualified relation, so the quoted
    `"db"."schema"."table"` is the key that matters; the unquoted form and the
    bare name are offered as well, since dialects quote differently and a
    `SupportsSql` engine is free to match on whichever it understands.
    """
    keys = [node.name]
    relation = node.relation_name
    if relation:
        keys.append(relation)
        keys.append(relation.replace('"', "").replace("`", ""))
    return tuple(dict.fromkeys(key for key in keys if key))


def _sample(manifest: Manifest, limit: int = 5) -> str:
    """A few model ids, so a typo in a URI is fixable from the message."""
    names = [node.unique_id for node in manifest.models()[:limit]]
    if not names:
        return "none (has dbt compiled this project?)"
    more = "" if len(manifest.models()) <= limit else ", ..."
    return ", ".join(names) + more
