# rowless-all

Meta-distribution: core plus every adapter and integration.

```bash
pip install rowless-all            # everything but PySpark and the LSP
pip install "rowless-all[spark,lsp]"
```

Most people want `rowless` plus the adapters they use:

```bash
pip install rowless rowless-polars rowless-dbt
```

This package exists so that core never depends on its own plugins, which would
make the dependency graph circular.

## Owner

Kauvin Lucas ([@kauvinlucas](https://github.com/kauvinlucas)).
