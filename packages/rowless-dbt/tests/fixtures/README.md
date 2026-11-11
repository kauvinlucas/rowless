# dbt artifact fixtures

Real `target/manifest.json` files, trimmed. A dbt manifest for a twelve-model
project is several megabytes and 90% of it is checksums, rendered configs and
macro definitions that rowless never reads. What is kept here is faithful in
*shape* - the same sections, the same nesting, the same key names and the same
value types dbt writes - so the reader is exercised against the structure it
will meet, without committing a megabyte per dbt version.

| file | what it is for |
| --- | --- |
| `manifest_v12.json` | dbt 1.8 on Postgres: an enforced contract, a model with only a catalog, a model with nothing but compiled SQL, a Python model, a seed, a source, and a test node that must be skipped |
| `catalog_v12.json` | the `dbt docs generate` artifact for the same project, covering one model and the seed |
| `manifest_v11.json` | dbt 1.7 on Snowflake: `number`, `varchar(16777216)`, `float`, `timestamp_tz` and `variant`, which is where the dialect-aware type parsing earns its keep |
| `manifest_unsupported.json` | a v7 manifest, to prove the reader refuses a format it has not been tested against instead of half-parsing it |

Adding a dbt version means committing a fixture for it and extending
`SUPPORTED_VERSIONS`, in that order. `tests/conftest.py` builds a throwaway dbt
project around any of these, so nothing here needs a `dbt_project.yml` or a
`models/` tree of its own.
