"""Configuration: `rowless.toml`, or a `[tool.rowless]` table in `pyproject.toml`.

Zero-config is the target: a repo with Parquet sources and annotated functions
should need no file at all. Everything here has a default that works.

    [tool.rowless]
    roots = ["pipelines"]
    engines = ["polars", "duckdb"]
    min_coverage = 0.8
    fail_on_unverified = true

    [tool.rowless.sources]
    "lake://*" = "s3://warehouse/{name}/"

    [tool.rowless.engine.pandas]
    arrow_dtypes = true
"""

from __future__ import annotations

import fnmatch
import re
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from rowless.diagnostics.codes import DEFAULT_ERROR_CODES, UNVERIFIED_CODES, Code
from rowless.diagnostics.models import Level
from rowless.errors import ConfigError

#: File names searched for, in order, walking up from the starting directory.
CONFIG_FILENAMES: tuple[str, ...] = ("rowless.toml", "pyproject.toml")

#: `# rowless: ignore[RWL101,RWL201]`, the mypy-shaped suppression comment.
IGNORE_PATTERN = re.compile(r"#\s*rowless:\s*ignore\[([^\]]*)\]")

_KNOWN_KEYS = frozenset(
    {
        "roots",
        "exclude",
        "engines",
        "finders",
        "sources",
        "min_coverage",
        "fail_on_unverified",
        "error_codes",
        "ignore_codes",
        "probe_udfs",
        "engine",
        "cache_dir",
        "discovery",
    }
)


@dataclass(frozen=True, slots=True)
class Config:
    """Resolved configuration."""

    roots: tuple[str, ...] = (".",)
    """Where to look for pipelines."""

    exclude: tuple[str, ...] = ("**/tests/**", "**/.venv/**")

    engines: tuple[str, ...] = ()
    """Restrict to these adapters. Empty means every available one."""

    finders: tuple[str, ...] = ()
    """Restrict discovery backends. Empty means every available one."""

    source_map: Mapping[str, str] = field(default_factory=dict)
    """Glob-to-template rewrites, so pipeline code says `lake://stg_orders`
    and the physical location stays out of the source tree."""

    min_coverage: float = 0.0
    """Fail when typed coverage drops below this. Starts at 0 so a first run
    reports rather than blocks, and teams ratchet it up like a mypy strictness
    setting."""

    fail_on_unverified: bool = True
    """Treat unverified results as failures. On by default: the alternative is
    a green check that means nothing."""

    error_codes: frozenset[Code] = frozenset()
    """Override which codes fail the run."""

    ignore_codes: frozenset[Code] = frozenset()

    probe_udfs: bool = False
    """Allow calling UDFs with synthetic values. Off by default; it runs user
    code with invented input."""

    engine_options: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    """Per-adapter options from `[tool.rowless.engine.<name>]`."""

    cache_dir: str = ".rowless"

    root_dir: str = "."
    """Directory the config was found in. Every relative path resolves against
    it, so `rowless check` behaves the same from a subdirectory as from the
    repository root."""

    @classmethod
    def discover(cls, start: str = ".") -> Config:
        """Find and load config by walking up from `start`."""
        here = Path(start).resolve()
        if here.is_file():
            here = here.parent
        for directory in (here, *here.parents):
            for filename in CONFIG_FILENAMES:
                candidate = directory / filename
                if not candidate.is_file():
                    continue
                table = _read_table(str(candidate))
                if table is not None:
                    return cls._from_table(str(candidate), table)
        # No config anywhere up the tree is the supported zero-config case, not
        # an error: the defaults are chosen to work on a repo that has none.
        return cls(root_dir=str(here))

    @classmethod
    def load(cls, path: str) -> Config:
        """Load one file, raising `ConfigError` with the offending key."""
        table = _read_table(path)
        if table is None:
            raise ConfigError(f"{path}: no [tool.rowless] table")
        return cls._from_table(path, table)

    @classmethod
    def _from_table(cls, path: str, table: Mapping[str, Any]) -> Config:
        """Build a config from an already-parsed table."""
        unknown = set(table) - _KNOWN_KEYS
        if unknown:
            keys = ", ".join(sorted(unknown))
            raise ConfigError(f"{path}: unknown key(s): {keys}")

        return cls(
            roots=_strings(path, table, "roots", (".",)),
            exclude=_strings(path, table, "exclude", ("**/tests/**", "**/.venv/**")),
            engines=_strings(path, table, "engines", ()),
            finders=_strings(path, table, "finders", ()),
            source_map=dict(table.get("sources", {})),
            min_coverage=_ratio(path, table.get("min_coverage", 0.0)),
            fail_on_unverified=bool(table.get("fail_on_unverified", True)),
            error_codes=_codes(path, table.get("error_codes", ())),
            ignore_codes=_codes(path, table.get("ignore_codes", ())),
            probe_udfs=bool(table.get("probe_udfs", False)),
            engine_options=dict(table.get("engine", {})),
            cache_dir=str(table.get("cache_dir", ".rowless")),
            root_dir=str(Path(path).parent.resolve()),
        )

    def with_overrides(self, **overrides: Any) -> Config:
        """Apply CLI flags on top of file config."""
        supplied = {key: value for key, value in overrides.items() if value not in (None, (), [])}
        for key in ("engines", "roots", "finders", "exclude"):
            if key in supplied:
                supplied[key] = tuple(supplied[key])
        return replace(self, **supplied)

    def resolve_source(self, uri: str) -> str:
        """Apply `source_map` rewrites to a source URI.

        The first matching glob wins, and the template can use `{name}` (the
        URI without its scheme), `{scheme}` and `{uri}`. One rewrite only: a
        chain of rewrites is impossible to debug from an error message.
        """
        for pattern, template in self.source_map.items():
            if not fnmatch.fnmatchcase(uri, pattern):
                continue
            scheme, separator, name = uri.partition("://")
            if not separator:
                scheme, name = "", uri
            try:
                return template.format(name=name, scheme=scheme, uri=uri)
            except KeyError as exc:
                raise ConfigError(
                    f"source template {template!r} uses unknown placeholder {exc}"
                ) from exc
        return uri

    def level_for(self, code: Code) -> Level | None:
        """The level a code is reported at, or None when it is suppressed.

        This is where `fail_on_unverified` and the code overrides actually
        apply. Keeping the policy in one function means the renderers never
        have to know about configuration.
        """
        if code in self.ignore_codes:
            return None
        if code.value.startswith("RWL5"):
            # rowless broke. Always loud, never configurable away: the exit
            # code needs to distinguish this from a pipeline problem.
            return Level.ERROR
        if code in UNVERIFIED_CODES:
            return Level.UNVERIFIED if self.fail_on_unverified else Level.WARNING
        if code in (self.error_codes or DEFAULT_ERROR_CODES):
            return Level.ERROR
        if code is Code.WRITE_RECORDED:
            return Level.INFO
        return Level.WARNING

    @property
    def cache_path(self) -> Path:
        """Absolute cache directory, created on demand by whoever writes to it."""
        return Path(self.root_dir, self.cache_dir)

    def resolved_roots(self) -> tuple[Path, ...]:
        """Configured roots as absolute paths."""
        return tuple(Path(self.root_dir, root).resolve() for root in self.roots)

    def is_excluded(self, path: Path) -> bool:
        """True when `path` matches an exclude glob."""
        try:
            relative = path.resolve().relative_to(Path(self.root_dir).resolve())
        except ValueError:
            relative = path
        text = relative.as_posix()
        for pattern in self.exclude:
            # `**/tests/**` is the familiar spelling and the one people write,
            # but `fnmatch` has no notion of path segments: it needs something
            # before the slash. Matching the pattern with its leading `**/`
            # removed as well makes a top-level `tests/` directory match too.
            cleaned = pattern.lstrip("/")
            if fnmatch.fnmatch(text, cleaned) or fnmatch.fnmatch(text, cleaned.removeprefix("**/")):
                return True
        return False


def _read_table(path: str) -> Mapping[str, Any] | None:
    """Parse a config file, returning None when it holds no rowless settings.

    A `pyproject.toml` with no `[tool.rowless]` is somebody else's file, so
    `discover` keeps walking up the tree instead of stopping at it.
    """
    source = Path(path)
    try:
        raw = tomllib.loads(source.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    if source.name == "pyproject.toml":
        tool = raw.get("tool", {})
        table = tool.get("rowless") if isinstance(tool, Mapping) else None
        return table if isinstance(table, Mapping) else None
    return raw


def _strings(
    path: str,
    table: Mapping[str, Any],
    key: str,
    default: tuple[str, ...],
) -> tuple[str, ...]:
    """Read a list-of-strings key, naming the file and key when it is wrong."""
    value = table.get(key, default)
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ConfigError(f"{path}: {key} must be a list of strings")
    return tuple(str(item) for item in value)


def _ratio(path: str, value: Any) -> float:
    """Read a 0..1 ratio."""
    try:
        ratio = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{path}: min_coverage must be a number between 0 and 1") from exc
    if not 0.0 <= ratio <= 1.0:
        raise ConfigError(f"{path}: min_coverage must be between 0 and 1, got {ratio}")
    return ratio


def _codes(path: str, value: Any) -> frozenset[Code]:
    """Read a list of diagnostic codes, rejecting ones that do not exist.

    Strict here, unlike the inline comments: a typo in a CI allowlist silently
    stops gating on a real finding, which is the expensive kind of mistake.
    """
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ConfigError(f"{path}: expected a list of diagnostic codes, e.g. ['RWL101']")
    codes: set[Code] = set()
    for item in value:
        try:
            codes.add(Code(str(item)))
        except ValueError as exc:
            raise ConfigError(f"{path}: unknown diagnostic code {item!r}") from exc
    return frozenset(codes)


def inline_ignores(path: str) -> Mapping[int, Sequence[Code]]:
    """Parse `# rowless: ignore[RWL101]` comments out of a file.

    Line-level suppression, like mypy's, because the first run on a real repo
    finds things nobody is going to fix this week, and a tool with no way to
    say "not now" gets deleted instead of configured.

    Unknown codes are skipped rather than rejected: codes are append-only, so
    an unrecognised one is almost always a newer rowless than this one, and
    failing the run over it would punish the wrong person.
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"{path}: {exc}") from exc

    found: dict[int, list[Code]] = {}
    for number, line in enumerate(text.splitlines(), start=1):
        match = IGNORE_PATTERN.search(line)
        if match is None:
            continue
        codes: list[Code] = []
        for raw in match.group(1).split(","):
            try:
                codes.append(Code(raw.strip()))
            except ValueError:
                continue
        if codes:
            found[number] = codes
    return found
