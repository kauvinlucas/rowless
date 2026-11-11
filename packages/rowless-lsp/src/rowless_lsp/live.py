"""Findings available between saves, with no dry run.

A dry run imports the module, so it can only ever check what is on disk. That
makes `did_save` the honest moment for a full check - and it would make the
editor silent for the entire time a developer is actually typing, which is the
moment the proposal's first clause is about.

What is available without running anything is narrower and still worth having:
the buffer holds the column names, and the last resolved schema holds the
column set. Comparing those two is not inference, it is a lookup, and it
catches the headline finding - a column that does not exist - while it is
being typed.

The rules that keep it from crying wolf, each of which costs a real finding
and is worth the cost:

* a function with any computed column name says nothing at all, because a
  name rowless cannot see could be the one it is about to report missing
* a name created inside the function is never missing, whatever the inputs say
* a name in the last resolved *output* is never missing, which covers the
  columns a chain builds in ways this module does not model
* no resolved schema means no finding; there is nothing to compare against

The result is a preview of what `did_save` will say, never a finding the save
will contradict.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from rowless.analysis import analyse_source
from rowless.config import IGNORE_PATTERN, Config
from rowless.diagnostics.codes import Code
from rowless.diagnostics.models import Diagnostic, Level
from rowless.errors import ColumnNotFound
from rowless.ir.schema import Origin, Schema

from rowless_lsp.documents import ColumnUse, DocumentIndex, IndexedFunction, Use

#: Static finding kind to the code it is reported under. `import_cycle` is
#: deliberately absent: a buffer mid-edit does not parse most of the time, and
#: every editor already underlines a syntax error without rowless's help.
_STATIC_CODES: dict[str, tuple[Code, Level]] = {
    "row_dependent_call": (Code.ROW_DEPENDENT_CONTROL_FLOW, Level.UNVERIFIED),
    "dynamic_column": (Code.DYNAMIC_COLUMN_NAMES, Level.WARNING),
}


def live_diagnostics(
    index: DocumentIndex,
    text: str,
    *,
    schemas: Mapping[str, Mapping[str, Schema]],
    outputs: Mapping[str, Schema] | None = None,
    config: Config | None = None,
) -> tuple[Diagnostic, ...]:
    """Everything the buffer alone supports, as ordinary rowless diagnostics.

    `schemas` maps a function name to its resolved input schemas by parameter;
    `outputs` maps a function name to the schema its last dry run produced.
    Both come from the session's last successful check, so this is a cheap
    re-read of an answer rowless already computed rather than a new claim.
    """
    found = [
        *_static(text, index.path, config),
        *_columns(index, schemas=schemas, outputs=outputs or {}, config=config),
    ]
    suppressed = _suppressions(text)
    return tuple(diagnostic for diagnostic in found if not _is_suppressed(diagnostic, suppressed))


def _static(text: str, path: str, config: Config | None) -> list[Diagnostic]:
    """Core's own static pass over the buffer, as diagnostics."""
    diagnostics: list[Diagnostic] = []
    for finding in analyse_source(text, path):
        mapped = _STATIC_CODES.get(finding.kind)
        if mapped is None:
            continue
        code, default = mapped
        level = _level(code, default, config)
        if level is None:
            continue
        diagnostics.append(
            Diagnostic(
                code=code,
                level=level,
                message=finding.detail,
                origin=Origin(kind="python", ref=path, file=path, line=finding.line),
                detail=(
                    "found in the buffer, so it is reported whether or not the dry run "
                    "reached this line"
                ),
            )
        )
    return diagnostics


def _columns(
    index: DocumentIndex,
    *,
    schemas: Mapping[str, Mapping[str, Schema]],
    outputs: Mapping[str, Schema],
    config: Config | None,
) -> list[Diagnostic]:
    """Column reads in the buffer that no resolved schema can account for."""
    level = _level(Code.COLUMN_NOT_FOUND, Level.ERROR, config)
    if level is None:
        return []
    diagnostics: list[Diagnostic] = []
    for function in index.functions:
        inputs = schemas.get(function.name)
        if not function.checked or not inputs or function.dynamic_lines:
            continue
        safe = _accounted_for(function, outputs.get(function.name))
        for use in index.reads_in(function.name):
            missing = _missing(use, inputs, safe)
            if missing is None:
                continue
            diagnostics.append(
                Diagnostic(
                    code=Code.COLUMN_NOT_FOUND,
                    level=level,
                    message=str(missing),
                    origin=Origin(
                        kind="python",
                        ref=f"{index.path}:{function.name}",
                        file=index.path,
                        line=use.span.line,
                        column=use.span.start + 1,
                    ),
                    detail="from the schema rowless resolved for this input",
                    suggestion=(
                        f"did you mean {missing.suggestion!r}?" if missing.suggestion else None
                    ),
                )
            )
    return diagnostics


def _accounted_for(function: IndexedFunction, output: Schema | None) -> frozenset[str]:
    """Names that are never missing, whatever the input schemas hold.

    Columns the function creates, and columns the last dry run saw come out of
    it. The second one is the safety net for every chain shape this module
    does not model: if the engine itself produced the column, rowless has no
    business reporting a read of it.
    """
    produced = frozenset(output) if output is not None else frozenset()
    return function.created | produced


def _missing(
    use: ColumnUse,
    inputs: Mapping[str, Schema],
    safe: frozenset[str],
) -> ColumnNotFound | None:
    """The finding for one read, or None when the name is accounted for.

    Returns the exception rather than a message so the text matches what the
    dry run will print on save, down to the near-miss suggestion: two
    spellings of the same finding would read as two different findings.
    """
    if use.use is not Use.READ or use.name in safe:
        return None
    schema = inputs.get(use.frame) if use.frame is not None else None
    if schema is not None:
        if use.name in schema:
            return None
        return ColumnNotFound(
            use.name,
            available=tuple(schema),
            relation=schema.name or use.frame,
        )
    # No attributable frame: `pl.col("amount")` could mean any input in scope,
    # so the name only counts as missing when *no* input has it.
    available = {name for schema in inputs.values() for name in schema}
    if use.name in available:
        return None
    return ColumnNotFound(use.name, available=tuple(sorted(available)), relation=use.function)


def _level(code: Code, default: Level, config: Config | None) -> Level | None:
    """The level a code reports at, honouring configuration when there is any.

    Going through `Config.level_for` means `ignore_codes` and
    `fail_on_unverified` work in the editor exactly as they do in CI, instead
    of the editor having a second opinion about severity.
    """
    if config is None:
        return default
    return config.level_for(code)


def _suppressions(text: str) -> dict[int, frozenset[Code]]:
    """`# rowless: ignore[...]` comments in the buffer.

    Parsed from the buffer rather than the file, which is the whole point:
    adding the comment should silence the squiggle before the save, or nobody
    believes the comment works.
    """
    found: dict[int, frozenset[Code]] = {}
    for number, line in enumerate(text.splitlines(), start=1):
        match = IGNORE_PATTERN.search(line)
        if match is None:
            continue
        codes: set[Code] = set()
        for raw in match.group(1).split(","):
            try:
                codes.add(Code(raw.strip()))
            except ValueError:
                continue
        if codes:
            found[number] = frozenset(codes)
    return found


def _is_suppressed(diagnostic: Diagnostic, suppressed: Mapping[int, frozenset[Code]]) -> bool:
    """True when a comment on the finding's own line turned it off."""
    origin = diagnostic.origin
    if origin is None or origin.line is None:
        return False
    return diagnostic.code in suppressed.get(origin.line, frozenset())


def suppress(
    diagnostics: Sequence[Diagnostic],
    text: str,
) -> tuple[Diagnostic, ...]:
    """Apply the buffer's ignore comments to findings from somewhere else.

    Used on the dry run's own results while they are being re-published
    against an edited buffer: the comment was typed after the check ran, and
    waiting for the next save to honour it is the kind of lag that makes a
    feature feel broken.
    """
    found = _suppressions(text)
    return tuple(diagnostic for diagnostic in diagnostics if not _is_suppressed(diagnostic, found))
