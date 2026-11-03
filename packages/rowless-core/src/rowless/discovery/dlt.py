"""dlt sources and resources. Extra: `rowless[dlt]`.

dlt resources declare their table name and often a column hint set, which
makes them both a pipeline to check and a schema source worth trusting.

Read from the AST, for the same reason as the other framework finders: a dlt
module runs credential lookups at import time, and discovery has to work
without them.
"""

from __future__ import annotations

import ast
import importlib.util
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import ClassVar

from rowless.discovery.base import Pipeline, module_name_for, python_files
from rowless.discovery.plain import decorator_calls, import_aliases, keyword_constant

#: Scheme a resource's declared table is reported under.
RESOURCE_SCHEME = "dlt"

RESOURCE_DECORATORS = frozenset({"dlt.resource", "dlt.transformer", "dlt.source"})


class DltDiscovery:
    """Find `@dlt.resource` and `@dlt.transformer` functions."""

    name: ClassVar[str] = "dlt"
    requires: ClassVar[str] = "dlt>=0.5"

    def available(self) -> bool:
        """True when dlt imports."""
        try:
            return importlib.util.find_spec("dlt") is not None
        except (ImportError, ValueError):
            return False

    def find(self, roots: Sequence[str]) -> Iterable[Pipeline]:
        """Read the resource declarations, pairing each with its declared table."""
        for path in python_files(roots):
            yield from self._in_file(path)

    def _in_file(self, path: Path) -> Iterable[Pipeline]:
        """Resources declared in one module."""
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError):
            return
        aliases = import_aliases(tree)
        module = module_name_for(path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            decorated = [
                (name, call)
                for name, call in decorator_calls(node, aliases)
                if name in RESOURCE_DECORATORS
            ]
            if not decorated:
                continue
            kind, call = decorated[0]
            table = keyword_constant(call, "table_name") or node.name
            # A transformer's upstream is its first parameter; a resource has
            # no rowless-visible input, and reporting that honestly is what
            # keeps it out of the coverage numbers.
            inputs: tuple[tuple[str, str | None], ...] = ()
            if kind == "dlt.transformer" and node.args.args:
                upstream = node.args.args[0].arg
                inputs = ((upstream, f"{RESOURCE_SCHEME}://{upstream}"),)
            yield Pipeline(
                target=f"{module}:{node.name}",
                inputs=inputs,
                framework="dlt",
                file=str(path),
                line=node.lineno,
                tags=(kind.rpartition(".")[2], f"table={table}"),
            )
