"""Dagster assets and ops. Extra: `rowless[dagster]`.

Dagster is the best case for rowless: asset keys give the input bindings
for free, so `rowless check` needs no configuration at all. Reads the
`Definitions` object and the asset graph, and never materialises anything.

Reading is done from the AST, not by importing the definitions module.
Loading a `Definitions` object runs whatever the repository does at import
time (resource construction, secret lookups, a database connection), and
discovery has to work on a repository that cannot do those things from a
laptop. The cost is that computed asset keys are invisible, which is reported
as an unbound input rather than guessed at.
"""

from __future__ import annotations

import ast
import importlib.util
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import ClassVar

from rowless.discovery.base import Pipeline, module_name_for, python_files
from rowless.discovery.plain import import_aliases, pipeline_decorators

#: Scheme upstream asset keys are reported under. A user maps it onto real
#: storage with a `[sources]` rewrite: `"dagster://*" = "lake://{name}"`.
ASSET_SCHEME = "dagster"

#: Decorators that make a function an entry point worth checking.
ASSET_DECORATORS = frozenset(
    {
        "dagster.asset",
        "dagster.multi_asset",
        "dagster.op",
        "dagster.graph_asset",
        "dagster._core.definitions.decorators.asset_decorator.asset",
    }
)


class DagsterDiscovery:
    """Find `@asset` and `@op` functions, with upstream keys as input URIs."""

    name: ClassVar[str] = "dagster"
    requires: ClassVar[str] = "dagster>=1.7"

    def available(self) -> bool:
        """True when dagster imports and a `Definitions` object is locatable."""
        try:
            return importlib.util.find_spec("dagster") is not None
        except (ImportError, ValueError):
            return False

    def find(self, roots: Sequence[str]) -> Iterable[Pipeline]:
        """Walk the repository for asset definitions, importing nothing."""
        for path in python_files(roots):
            yield from self._in_file(path)

    def _in_file(self, path: Path) -> Iterable[Pipeline]:
        """Assets declared in one module."""
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError):
            return
        aliases = import_aliases(tree)
        module = module_name_for(path)
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            decorators = pipeline_decorators(node, aliases)
            if not decorators & ASSET_DECORATORS:
                continue
            inputs = tuple(
                (argument.arg, self._input_uri(argument.arg))
                for argument in node.args.args
                if argument.arg not in {"context", "self"}
            )
            yield Pipeline(
                target=f"{module}:{node.name}",
                inputs=inputs,
                framework="dagster",
                file=str(path),
                line=node.lineno,
                tags=("asset",),
            )

    def _input_uri(self, parameter: str) -> str:
        """A Dagster asset's parameter names are its upstream asset keys."""
        return self.resolve_asset_key(parameter)

    def resolve_asset_key(self, key: str) -> str:
        """Dagster asset key to a rowless source URI.

        The key itself is what rowless reports, under the `dagster://` scheme.
        IO manager metadata would be better - it often names the real table -
        but reading it means importing the definitions, which is exactly what
        this finder refuses to do. A `[sources]` rewrite is how a user points
        the keys at storage:

            [sources]
            "dagster://*" = "lake://{name}"
        """
        return f"{ASSET_SCHEME}://{key.replace('/', '.')}"
