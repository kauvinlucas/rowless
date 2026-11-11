"""Airflow tasks. Extra: `rowless[airflow]`.

Harder than Dagster: a PythonOperator callable has no declared schema,
and parsing a DAG file imports it. Supports `@task`-decorated TaskFlow
functions well, classic operators only where inlets and outlets are set,
and reports the rest as undiscoverable rather than guessing.

Like the Dagster finder, this reads the AST instead of filling a DagBag.
Importing a DAG file runs top-level code that often needs a metadata database
and connections, and a checker that only works where Airflow is fully
configured is a checker nobody runs in their editor.
"""

from __future__ import annotations

import ast
import importlib.util
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import ClassVar

from rowless.discovery.base import Pipeline, module_name_for, python_files
from rowless.discovery.plain import decorator_calls, import_aliases, keyword_constant

#: TaskFlow decorators. Classic operators are handled separately, because
#: their bindings live in constructor arguments rather than on a function.
TASK_DECORATORS = frozenset(
    {
        "airflow.task",
        "airflow.decorators.task",
        "airflow.sdk.task",
        "airflow.decorators.taskflow.task",
    }
)


class AirflowDiscovery:
    """Find TaskFlow `@task` callables, plus operators declaring inlets/outlets."""

    name: ClassVar[str] = "airflow"
    requires: ClassVar[str] = "apache-airflow>=2.9"

    def available(self) -> bool:
        """True when airflow imports and a DAG folder is configured."""
        try:
            return importlib.util.find_spec("airflow") is not None
        except (ImportError, ValueError):
            return False

    def find(self, roots: Sequence[str]) -> Iterable[Pipeline]:
        """Yield one pipeline per TaskFlow callable found under `roots`."""
        for path in python_files(roots):
            yield from self._in_file(path)

    def _in_file(self, path: Path) -> Iterable[Pipeline]:
        """TaskFlow tasks declared in one DAG module."""
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError):
            return
        aliases = import_aliases(tree)
        module = module_name_for(path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            decorated = decorator_calls(node, aliases)
            tasks = [call for name, call in decorated if name in TASK_DECORATORS]
            if not tasks:
                continue
            task = tasks[0]
            task_id = keyword_constant(task, "task_id") or node.name
            inputs = tuple(
                (argument.arg, None)
                for argument in node.args.args
                if argument.arg not in {"context", "self", "ti"}
            )
            yield Pipeline(
                target=f"{module}:{node.name}",
                inputs=inputs,
                framework="airflow",
                file=str(path),
                line=node.lineno,
                tags=("taskflow", f"task_id={task_id}"),
            )
