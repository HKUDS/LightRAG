"""``NEO4J_KEEP_ALIVE`` / ``[neo4j] keep_alive`` follow the shared boolean rules.

The env-over-``config.ini`` lookup is passed through ``parse_bool_setting``,
so aliases such as ``t`` are accepted and a typo warns and keeps the default
(``True``). Runs without a live Neo4j: the driver factory is patched.
"""

import logging

import pytest

from lightrag.kg import neo4j_impl
from lightrag.kg.neo4j_impl import Neo4JStorage
from lightrag.kg.shared_storage import initialize_share_data


pytestmark = pytest.mark.offline


class _FakeResult:
    async def data(self):
        return []

    async def consume(self):
        return None


class _FakeSession:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def run(self, query, **params):
        return _FakeResult()


class _FakeDriver:
    def session(self, **kwargs):
        return _FakeSession()


@pytest.mark.parametrize(
    ("env_value", "config_value", "expected", "warn_keys"),
    [
        pytest.param("t", "off", True, set(), id="env-true-alias"),
        pytest.param("off", "true", False, set(), id="env-false-spelling"),
        pytest.param(
            "ture",
            "off",
            True,
            {"NEO4J_KEEP_ALIVE"},
            id="env-typo-defaults-and-warns",
        ),
        pytest.param(None, "off", False, set(), id="config-false-spelling"),
    ],
)
async def test_keep_alive_setting_is_parsed_for_driver(
    monkeypatch,
    lightrag_log_records: list[logging.LogRecord],
    env_value,
    config_value,
    expected,
    warn_keys,
):
    if env_value is None:
        monkeypatch.delenv("NEO4J_KEEP_ALIVE", raising=False)
    else:
        monkeypatch.setenv("NEO4J_KEEP_ALIVE", env_value)

    monkeypatch.setattr(
        neo4j_impl.config,
        "get",
        lambda _section, key, fallback=None: (
            config_value if key == "keep_alive" else fallback
        ),
    )
    driver = _FakeDriver()
    driver_kwargs = {}

    def fake_driver(_uri, **kwargs):
        driver_kwargs.update(kwargs)
        return driver

    monkeypatch.setattr(neo4j_impl.AsyncGraphDatabase, "driver", fake_driver)
    initialize_share_data()

    await Neo4JStorage(
        namespace="test", global_config={}, embedding_func=None, workspace="test"
    ).initialize()

    assert driver_kwargs["keep_alive"] is expected
    warned = {
        record.args[0]
        for record in lightrag_log_records
        if "is not a recognized boolean" in record.msg
    }
    assert warned == warn_keys
