# rowless-lsp

Language server: schema diagnostics, column completion and inline schema hints.

## The trick

There is no trick. Every finding comes from `rowless.check.run`, the same call
`rowless check` makes, so the editor cannot have a second opinion about what is
broken. What this package adds is the state that makes that answer usable
interactively, and one honest admission about when it is available.

## The one hard constraint

A dry run **imports the module**. That is the whole premise of the project: run
the real code on rowless frames and let the engine compute the schema. It also
means a check can only ever describe the file *on disk*, because an unsaved
buffer is not importable, and running the old file to report on the new one
would be worse than saying nothing.

So the server splits along that line rather than pretending it is not there:

| when | what runs | what you see |
| --- | --- | --- |
| on save | a real dry run, in a thread | every finding `rowless check` finds |
| while typing | nothing is executed | column names checked against the resolved schema, plus the static pass |
| hover, completion, hints | nothing is executed | the last check's schemas, read from memory |

The second row is the one worth explaining. It is not schema inference - the
proposal rules that out, and rightly. It is a *lookup*: the buffer holds the
column names, the last check holds the column set, and comparing the two
catches the headline finding while it is being typed.

```
pipelines/revenue.py:9  ColumnNotFound 'amount' in orders, did you mean 'amount_cents'?  RWL101
```

That line is byte-for-byte what the save will print, because it is built by
raising core's own `ColumnNotFound` against the resolved schema. Two spellings
of one finding read as two findings.

## What keeps it from crying wolf

Each of these costs a real finding, and each is worth the cost. A live finding
the next save contradicts would teach a user to stop believing the editor,
which is a worse outcome than silence.

* a function with **any computed column name** says nothing at all about
  columns - a name rowless cannot see could be the one it is about to report
  missing - and the computed name is still reported, as `RWL203`
* a name the function **creates** is never missing, whatever the inputs hold:
  `alias`, a `rename` value, a `with_columns` keyword
* a name in the **last resolved output** is never missing, which covers every
  chain shape the index does not model
* a **join key** is not attributed to the frame the chain started from, because
  `orders.join(users, left_on="customer_id")` names a column of `users` as often
  as one of `orders`
* **no resolved schema, no finding**: before the first check there is nothing
  to compare against
* a buffer that **does not parse** reports nothing, because a finding placed by
  stale line numbers lands on code that is fine

`tests/test_lsp_live.py` is half findings and half silences for that reason.

## Staleness

The findings from the last save are kept while you type, because withdrawing
them on the first keystroke would make the editor flicker. They are kept
honestly: each one remembers the text of the line it was reported against, and
on every republish that text is looked for again. Unchanged in place, the
finding stays; moved, it follows; gone, it is dropped.

Dropping is the half that matters. A diagnostic left on a line whose text has
changed is how a checker gets blamed for a bug it reported correctly three
edits ago.

## Features

| feature | what it answers |
| --- | --- |
| diagnostics | the findings, on save and between saves |
| completion | the column names that exist, with their dtypes, inside `pl.col("...")`, `df["..."]`, `select`, `rename`, a join key |
| hover | a column's dtype and upstream lineage, an input's whole schema, a pipeline's output and **how much of it was verified** |
| inlay hints | the resolved schema at the end of the chain, and which source feeds which parameter |
| code actions | change a near-miss column name to the one that exists, or write the `# rowless: ignore[...]` comment |

Two of those are about rowless not knowing something, which is deliberate.
Hovering a pipeline name prints the same confidence line `rowless explain`
does, reasons included, and an unresolved output is labelled `-> unverified`
rather than left blank: a missing hint reads as "no information", and this
project's whole argument is that "rowless does not know" *is* information.

Completion is the one feature with no room for cleverness. A list containing a
column that does not exist is not a convenience, it is a way of introducing the
exact bug the tool exists to catch - so the candidates come from a resolved
schema or there are none.

## Severity

`UNVERIFIED` findings are **warnings** here, not errors. The GitHub renderer
promotes them, because a CI annotation has nowhere else to say "this gates the
build"; an editor gates nothing, and an error squiggle for "rowless does not
know" is how a language server gets switched off. The 2xx code travels with the
diagnostic and is visible in every editor's problem list, which is where the
distinction belongs.

`ignore_codes`, `fail_on_unverified` and the `# rowless: ignore[...]` comments
all apply, through `Config.level_for` - one policy, not a second opinion from
the editor. An ignore comment is read from the **buffer**, so it takes effect
on the keystroke rather than on the next save. A comment that needs a save to
work is a comment nobody believes.

## Latency

The honest status: good enough to use, and not yet what it should be.

* one `Config` per workspace, discovered once rather than per keystroke
* one index per buffer, with the last index that parsed kept as a fallback so
  completion does not stop working mid-word
* every re-check debounced per document, superseded work **cancelled** rather
  than queued - a queue makes the editor fall further behind the faster you
  type
* a save re-checks only the file that was saved, not the workspace
* dry runs run in a thread, so a check cannot block the completion request a
  user is most likely to make while waiting for one
* full document sync, not incremental: `ast.parse` has no incremental mode, so
  applying a delta would be work done twice

What is **not** here is the warm worker pool. Each dry run is still one
subprocess, which costs an interpreter start and an engine import per pipeline.
The reason it is not here is not effort: a reused worker has the user's module
already imported, and re-importing a module whose dependencies also changed is
a correctness problem rather than a performance one. The request-response shape
of `rowless.runtime.worker` leaves room for it, and doing it properly means
deciding what a reload means first. Source schemas are cached on disk by
core's `SourceCache`, which is shared with the CLI, so the second check of a
file resolves no sources.

## Editors

Nothing in here is editor-specific; both clients need only a command.

**VS Code** - any generic LSP client extension, pointed at the binary:

```json
{
  "rowless.serverCommand": ["rowless-lsp"],
  "files.watcherExclude": { "**/.rowless/**": true }
}
```

**Neovim** (0.11 or later):

```lua
vim.lsp.config.rowless = {
  cmd = { 'rowless-lsp' },
  filetypes = { 'python' },
  root_markers = { 'rowless.toml', 'pyproject.toml', '.git' },
}
vim.lsp.enable('rowless')
```

Register a file watcher for `rowless.toml` and `pyproject.toml` if your client
does not already: the server re-reads configuration and re-checks when one
changes, and a cached config outliving its file is how settings stop taking
effect.

```
rowless-lsp              speak LSP over stdio, as an editor launches it
rowless-lsp --tcp        listen on a socket, for attaching a debugger
rowless-lsp --debounce   quiet period before a re-check, in seconds
```

The `rowless.recheck` command forces a full workspace check. It is the escape
hatch for the change this server cannot see: an upstream table was altered, so
nothing in the repository looks different and every resolved schema is quietly
out of date.

## What it does not do

* **check an unsaved buffer** by running it, for the reason at the top
* **resolve SQL models**, which needs a runtime path for a target that is not
  an importable function - the same gap `rowless-dbt` documents
* **reuse a warm worker**, as above
* **infer a schema from source**, which is not scheduled and not wanted

## Owner

Kauvin Lucas ([@kauvinlucas](https://github.com/kauvinlucas)).

## Development

```bash
uv sync --package rowless-lsp
pytest packages/rowless-lsp
```

The suite drives the handlers directly, with no editor and no client attached:
publishing is captured, so a handler still builds a real
`PublishDiagnosticsParams` and the test asserts on it. Fixtures live in
`tests/lsp_fixtures.py` rather than a `conftest.py`, and that file says why.
The dry runs go in-process against a fake adapter in
`tests/rowless_fake_lsp.py`, because an adapter registered in this process
cannot be seen by a worker subprocess - the subprocess boundary is core's to
test.
