"""Plain functions: annotated `def f(df: pl.LazyFrame) -> pl.LazyFrame`.

The zero-config finder, and the only one with no dependencies. Walks the
AST rather than importing, so a repo whose modules fail to import still
yields a pipeline list, and the import failure is reported per pipeline
when the dry run reaches it.

Also honours an explicit `@rowless.pipeline` marker, which is the escape
hatch for code whose annotations do not name a frame type.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import ClassVar

from rowless.discovery.base import Pipeline, module_name_for, python_files

#: Annotations that name an engine's frame type, as `module.Attribute`. Data,
#: not imports: this finder runs in a process with no engine installed, and an
#: adapter that adds a frame type adds a line here rather than a code path.
FRAME_ANNOTATIONS: dict[str, str] = {
    "polars.LazyFrame": "polars",
    "polars.DataFrame": "polars",
    "pandas.DataFrame": "pandas",
    "pandas.Series": "pandas",
    "duckdb.DuckDBPyRelation": "duckdb",
    "pyspark.sql.DataFrame": "pyspark",
    "pyspark.sql.dataframe.DataFrame": "pyspark",
    "pyspark.sql.connect.dataframe.DataFrame": "pyspark",
    "ibis.Table": "ibis",
    "ibis.expr.types.Table": "ibis",
    "ibis.expr.types.relations.Table": "ibis",
}


class PlainFunctionDiscovery:
    """Find annotated top-level functions and `@rowless.pipeline` markers."""

    name: ClassVar[str] = "plain"
    requires: ClassVar[str] = ""

    def available(self) -> bool:
        """Always true."""
        return True

    def find(self, roots: Sequence[str]) -> Iterable[Pipeline]:
        """Walk `*.py` under `roots`, matching annotations to engine frame types."""
        for path in python_files(roots):
            yield from _pipelines_in(path)


def _pipelines_in(path: Path) -> Iterator[Pipeline]:
    """Parse one module and yield the pipelines it declares."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError):
        # A module that will not even parse is reported when a dry run reaches
        # it; discovery's job is to get through the repository.
        return

    aliases = import_aliases(tree)
    declared_sources = _module_sources(tree)
    module = module_name_for(path)

    for node in tree.body:
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        marker = _pipeline_marker(node, aliases)
        engine = marker.engine or _engine_of(node, aliases)
        if not marker.found and engine is None:
            continue
        inputs = tuple(
            (argument.arg, marker.inputs.get(argument.arg, declared_sources.get(argument.arg)))
            for argument in node.args.args
        )
        yield Pipeline(
            target=f"{module}:{node.name}",
            inputs=inputs,
            engine=engine,
            framework="plain",
            file=str(path),
            line=node.lineno,
            tags=("decorated",) if marker.found else (),
        )


class _Marker:
    """What an `@rowless.pipeline` decorator declared, if there was one."""

    __slots__ = ("engine", "found", "inputs")

    def __init__(self) -> None:
        self.found = False
        self.engine: str | None = None
        self.inputs: dict[str, str] = {}


def _pipeline_marker(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    aliases: dict[str, str],
) -> _Marker:
    """Read `@rowless.pipeline(inputs=..., engine=...)` off a function."""
    marker = _Marker()
    for decorator in node.decorator_list:
        call = decorator if isinstance(decorator, ast.Call) else None
        reference = call.func if call is not None else decorator
        if dotted_name(reference, aliases) not in {"rowless.pipeline", "rowless.api.pipeline"}:
            continue
        marker.found = True
        if call is None:
            continue
        for keyword in call.keywords:
            if keyword.arg == "engine" and isinstance(keyword.value, ast.Constant):
                marker.engine = str(keyword.value.value)
            if keyword.arg == "inputs" and isinstance(keyword.value, ast.Dict):
                for key, value in zip(keyword.value.keys, keyword.value.values, strict=True):
                    if isinstance(key, ast.Constant) and isinstance(value, ast.Constant):
                        marker.inputs[str(key.value)] = str(value.value)
    return marker


def _engine_of(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    aliases: dict[str, str],
) -> str | None:
    """The engine a function's annotations name, if they name one at all.

    The return annotation decides, because a function that takes a Polars
    frame and returns a pandas one is a conversion and belongs to its output.
    Parameters are the fallback.
    """
    candidates = [node.returns, *(argument.annotation for argument in node.args.args)]
    for annotation in candidates:
        if annotation is None:
            continue
        engine = FRAME_ANNOTATIONS.get(dotted_name(annotation, aliases))
        if engine is not None:
            return engine
    return None


def _module_sources(tree: ast.Module) -> dict[str, str]:
    """Module-level `orders = rowless.source("lake://stg_orders")` bindings.

    This is what makes the proposal's example work with no configuration: the
    module names its inputs next to the code that uses them, and a parameter
    of the same name is bound to the same URI.
    """
    bound: dict[str, str] = {}
    aliases = import_aliases(tree)
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        if dotted_name(node.value.func, aliases) not in {"rowless.source", "rowless.api.source"}:
            continue
        if not node.value.args or not isinstance(node.value.args[0], ast.Constant):
            continue
        uri = str(node.value.args[0].value)
        for target in node.targets:
            if isinstance(target, ast.Name):
                bound[target.id] = uri
    return bound


def decorator_calls(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    aliases: dict[str, str],
) -> list[tuple[str, ast.Call | None]]:
    """A function's decorators as `(dotted name, call)` pairs.

    Shared with the framework finders: every one of them reads decorators off
    an AST rather than importing the module, so the resolution of
    `@dlt.resource` to `dlt.resource` belongs in one place.
    """
    found: list[tuple[str, ast.Call | None]] = []
    for decorator in node.decorator_list:
        call = decorator if isinstance(decorator, ast.Call) else None
        reference = call.func if call is not None else decorator
        found.append((dotted_name(reference, aliases), call))
    return found


def pipeline_decorators(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    aliases: dict[str, str],
) -> set[str]:
    """The dotted names of a function's decorators."""
    return {name for name, _ in decorator_calls(node, aliases)}


def keyword_constant(call: ast.Call | None, name: str) -> str | None:
    """A literal keyword argument from a decorator call, when it is a constant."""
    if call is None:
        return None
    for keyword in call.keywords:
        if keyword.arg == name and isinstance(keyword.value, ast.Constant):
            return str(keyword.value.value)
    return None


def import_aliases(tree: ast.Module) -> dict[str, str]:
    """Local name to fully qualified name, from the module's imports.

    `import polars as pl` makes `pl.LazyFrame` resolvable, and
    `from polars import LazyFrame` makes the bare name resolvable. Without
    this, the finder would have to guess whether `DataFrame` means pandas or
    Polars, and guessing routes a pipeline to the wrong adapter.
    """
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    aliases[alias.asname] = alias.name
                else:
                    # `import pyspark.sql` binds `pyspark`, not `pyspark.sql`.
                    top = alias.name.split(".")[0]
                    aliases[top] = top
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            for alias in node.names:
                aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return aliases


def dotted_name(node: ast.expr, aliases: dict[str, str]) -> str:
    """Render an annotation or decorator as a dotted name, resolving aliases."""
    parts: list[str] = []
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(aliases.get(current.id, current.id))
    elif isinstance(current, ast.Constant) and isinstance(current.value, str):
        # A string annotation: `def f(df: "pl.LazyFrame")`.
        return aliases.get(current.value, current.value)
    elif isinstance(current, ast.Subscript):
        return dotted_name(current.value, aliases)
    else:
        return ""
    return ".".join(reversed(parts))
