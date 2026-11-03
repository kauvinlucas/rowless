# rowless-lsp

Language server: schema diagnostics, column completion and inline schema hints.

## Why it is the last milestone, not the first

The protocol is the easy part. Latency is the problem: a dry run per keystroke
is not viable, so this needs debouncing, in-memory source schemas, and a pool of
warm worker subprocesses. The runtime is designed around a request-response
worker loop partly to make this possible later.

## Editor support

VS Code and Neovim first, since both need only a client config pointing at the
`rowless-lsp` binary.

## Status

Skeleton. Post-v0.1 on the roadmap.

## Owner

Kauvin Lucas ([@kauvinlucas](https://github.com/kauvinlucas)).
