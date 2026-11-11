"""pytest plugin for rowless: schema contract tests at about unit-test cost.

Installed as a pytest entry point, so the fixtures exist as soon as the
distribution is in the environment - there is nothing to add to `conftest.py`:

    def test_revenue_schema(rowless_check):
        rowless_check(daily_revenue, {"day": "date", "amount": "int64"})

The assertion is the one behind `rowless.assert_schema`, and the config is the
`rowless.toml` the CLI reads, so a contract test and `rowless check` cannot
disagree about a column. What the plugin adds is the part only pytest can do:
the engine a test needs decided at collection time, the source cache shared
across a suite, and the coverage the suite actually earned reported next to
its passes.
"""

from rowless_pytest import plugin
from rowless_pytest.checks import SchemaCheck
from rowless_pytest.engines import MARKER
from rowless_pytest.frames import FrameBuilder, SourceResolver
from rowless_pytest.session import RowlessSession, SessionHolder

__all__ = [
    "MARKER",
    "FrameBuilder",
    "RowlessSession",
    "SchemaCheck",
    "SessionHolder",
    "SourceResolver",
    "plugin",
]
