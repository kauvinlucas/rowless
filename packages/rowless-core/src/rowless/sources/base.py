"""Where input schemas come from.

A dry run needs a schema for every input before it starts. Sources provide
them, keyed by URI, and are the one place rowless is allowed to touch the
outside world: they run in the parent process, before the sandbox exists, and
their results are cached so a `rowless check` loop is offline after the first
run.

Registered under the `rowless.sources` entry-point group.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable
from urllib.parse import parse_qsl

from rowless.adapters.registry import load, registered_names
from rowless.errors import BadSourceRef, PluginLoadError, SchemaNotFound
from rowless.ir.schema import Schema
from rowless.ir.serde import schema_from_dict, schema_to_dict


@dataclass(frozen=True, slots=True)
class SourceRef:
    """A parsed source URI: `lake://stg_orders`, `dbt://model.shop.stg_orders`."""

    scheme: str
    path: str
    options: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, uri: str) -> SourceRef:
        """Parse a URI, raising `BadSourceRef` on anything unusable.

        A bare filesystem path parses with an empty scheme, which is what the
        Parquet source claims: `rowless check` on a repo full of Parquet files
        should need no configuration at all. Whether a *registered* source
        handles the scheme is decided in `resolve`, which is the layer that
        knows what is installed.
        """
        text = uri.strip()
        if not text:
            raise BadSourceRef("empty source URI")
        scheme, separator, remainder = text.partition("://")
        if not separator:
            scheme, remainder = "", text
        if separator and not scheme:
            raise BadSourceRef(f"source URI {uri!r} has no scheme before '://'")
        path, _, query = remainder.partition("?")
        if not path:
            raise BadSourceRef(f"source URI {uri!r} names no table or path")
        return cls(scheme=scheme, path=path, options=dict(parse_qsl(query)))

    @property
    def uri(self) -> str:
        """The URI this ref came from, minus any options."""
        return f"{self.scheme}://{self.path}" if self.scheme else self.path

    @property
    def table_name(self) -> str:
        """The last path segment: what a schema gets named after."""
        cleaned = self.path.rstrip("/")
        for separator in ("/", "."):
            if separator in cleaned:
                cleaned = cleaned.rsplit(separator, 1)[-1]
                break
        return cleaned or self.path


@runtime_checkable
class SchemaSource(Protocol):
    """Resolves a `SourceRef` to a schema."""

    name: ClassVar[str]
    schemes: ClassVar[tuple[str, ...]]
    """URI schemes this source claims, e.g. `("file", "s3", "lake")`."""

    requires: ClassVar[str]
    """PEP 508 requirement for the extra this source needs, if any."""

    def available(self) -> bool:
        """True when the dependencies are importable and config is present."""
        ...

    def resolve(self, ref: SourceRef) -> Schema:
        """Fetch the schema. Raise `SchemaNotFound` rather than returning empty.

        Must read metadata only: a Parquet footer, a catalog entry, a manifest.
        A source that scans rows is a bug, not a slow path.
        """
        ...

    def fingerprint(self, ref: SourceRef) -> str:
        """Cheap change token (etag, mtime, manifest hash) for cache validation."""
        ...


@dataclass(frozen=True, slots=True)
class SourceCache:
    """On-disk schema cache under `.rowless/sources/`.

    Checked into nothing, but stable across runs, so CI and the editor agree
    about what a table looks like and `rowless check` stays fast offline.
    """

    root: str

    def get(self, ref: SourceRef, fingerprint: str) -> Schema | None:
        """Return the cached schema when the fingerprint still matches."""
        path = self._path(ref)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # A corrupt or missing cache entry is a cache miss, never an error:
            # the only cost is one more metadata read.
            return None
        if payload.get("fingerprint") != fingerprint:
            return None
        try:
            return schema_from_dict(payload["schema"])
        except (KeyError, ValueError):
            return None

    def put(self, ref: SourceRef, fingerprint: str, schema: Schema) -> None:
        """Store a resolved schema."""
        path = self._path(ref)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "uri": ref.uri,
            "fingerprint": fingerprint,
            "schema": schema_to_dict(schema),
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _path(self, ref: SourceRef) -> Path:
        """Cache file for a ref: hashed, because a URI is not a filename."""
        digest = hashlib.sha256(ref.uri.encode("utf-8")).hexdigest()[:16]
        return Path(self.root, "sources", f"{digest}.json")


def source_for(ref: SourceRef) -> SchemaSource:
    """The registered source that claims `ref`'s scheme.

    Raises `BadSourceRef` listing what is registered, because the alternative
    error a user gets ("unknown scheme") does not tell them whether they need
    an extra installed or a typo fixed.
    """
    claimed: list[tuple[str, SchemaSource]] = []
    unavailable: list[str] = []
    for name in sorted(registered_names("rowless.sources")):
        try:
            candidate = load("rowless.sources", name)
        except PluginLoadError:
            continue
        if ref.scheme not in getattr(candidate, "schemes", ()):
            continue
        if candidate.available():
            claimed.append((name, candidate))
        else:
            unavailable.append(f"{name} (needs {getattr(candidate, 'requires', '?')})")
    if claimed:
        return claimed[0][1]
    if unavailable:
        raise BadSourceRef(
            f"no usable source for {ref.uri!r}: " + ", ".join(unavailable) + " not installed"
        )
    raise BadSourceRef(f"no source handles scheme {ref.scheme or '(bare path)'!r}: {_schemes()}")


def resolve(
    uri: str,
    *,
    cache: SourceCache | None = None,
    base_dir: str | None = None,
) -> Schema:
    """Resolve one source URI through the registry, with caching.

    `base_dir` anchors relative paths. A URI written in `rowless.toml` is
    relative to that file, not to whichever directory the user happened to run
    `rowless` from, and a checker whose findings depend on the shell's working
    directory is not usable from an editor.
    """
    ref = _anchored(SourceRef.parse(uri), base_dir)
    source = source_for(ref)
    fingerprint = _fingerprint(source, ref) if cache is not None else ""
    if cache is not None and fingerprint:
        cached = cache.get(ref, fingerprint)
        if cached is not None:
            return cached
    schema = source.resolve(ref)
    if not schema.fields:
        raise SchemaNotFound(f"source {uri!r} resolved to a schema with no columns")
    named = schema if schema.name else schema.relabel(ref.table_name)
    if cache is not None and fingerprint:
        cache.put(ref, fingerprint, named)
    return named


def _anchored(ref: SourceRef, base_dir: str | None) -> SourceRef:
    """Rewrite a relative filesystem path to sit under `base_dir`.

    Only when the file is actually there: `warehouse://db.schema.orders` is not
    a path, and must not be mangled into one.
    """
    if base_dir is None or Path(ref.path).is_absolute():
        return ref
    candidate = Path(base_dir, ref.path)
    if not candidate.exists():
        return ref
    return SourceRef(scheme=ref.scheme, path=str(candidate), options=ref.options)


def _fingerprint(source: SchemaSource, ref: SourceRef) -> str:
    """Best-effort change token.

    A source that cannot produce one (a live catalog with no change counter)
    still resolves; it just does not get cached, which is the right trade: a
    stale schema would be reported as fact.
    """
    try:
        return source.fingerprint(ref)
    except Exception:
        return ""


def _schemes() -> str:
    """The schemes every registered source claims, for an error message."""
    offered: list[str] = []
    for name in sorted(registered_names("rowless.sources")):
        try:
            candidate = load("rowless.sources", name)
        except PluginLoadError:
            continue
        for scheme in getattr(candidate, "schemes", ()):
            offered.append(scheme or "(bare path)")
    return "registered schemes: " + (", ".join(sorted(set(offered))) or "none")
