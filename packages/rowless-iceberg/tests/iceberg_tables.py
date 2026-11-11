"""Throwaway Iceberg tables built around the committed metadata fixtures.

A helper module rather than a `conftest.py`: every package in this workspace has
its own `tests/` directory, and two files called `conftest` are one module name
to mypy while two `tests` packages are one module name to pytest. `rowless-core`
solves it the same way with `rowless_fake.py`, so this follows that rather than
inventing a third answer.

What the helpers do is copy a fixture into `tmp_path` and hand back the URI that
reads it. Copying matters: `rowless_iceberg.catalog` caches a loaded table by
path and mtime, and a test that rewrote a committed fixture in place would leak
into the next one.
"""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from pathlib import Path

from rowless_iceberg import catalog

FIXTURES = Path(__file__).parent / "fixtures"

#: The fixture most tests use: three retained schemas with two renames and a
#: drop between them. See `fixtures/README.md` for the field-ID table.
ORDERS = "orders_v2.metadata.json"

#: Every primitive and nested type in the Iceberg spec, v3 included.
TYPES = "types_v3.metadata.json"

#: A v1 table whose snapshot records no schema id.
LEGACY = "legacy_v1.metadata.json"

#: Snapshot ids in `ORDERS`, oldest first, one per retained schema.
ORDERS_SNAPSHOTS: tuple[int, int, int] = (
    1122334455667788,
    3344556677889900,
    7788990011223344,
)


def metadata_path(tmp_path: Path, fixture: str = ORDERS) -> Path:
    """Copy a fixture into `tmp_path` and return where it landed.

    The name is kept, because PyIceberg refuses a metadata location that does
    not end in `.metadata.json` rather than reading it as one.
    """
    catalog.clear_cache()
    destination = tmp_path / fixture
    shutil.copyfile(FIXTURES / fixture, destination)
    return destination


def uri(
    tmp_path: Path,
    fixture: str = ORDERS,
    *,
    table: str = "sales.orders",
    options: Mapping[str, object] | None = None,
) -> str:
    """A `iceberg://` URI reading a fixture through the metadata route.

    The extra options arrive as a mapping rather than as `**kwargs` so that a
    caller forwarding its own options does not have to prove to the type
    checker that none of them is called `fixture` or `table`.

    Args:
        tmp_path: pytest's per-test directory.
        fixture: which committed metadata file to copy in.
        table: the identifier the URI names. Only a label on this route, but a
            real one: it is what the resolved schema is named after.
        options: extra URI options, e.g. `{"snapshot": 1122334455667788}`.

    Returns:
        The URI, with `?metadata=` pointing at the copy in `tmp_path`.
    """
    path = metadata_path(tmp_path, fixture)
    query = {
        "metadata": path.as_posix(),
        **{key: str(value) for key, value in (options or {}).items()},
    }
    joined = "&".join(f"{key}={value}" for key, value in query.items())
    return f"iceberg://{table}?{joined}"
