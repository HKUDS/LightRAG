"""A full-text index name another workspace already owns must be refused.

``_get_fulltext_index_name`` is lossy ("foo-bar" and "foo_bar" share one
index name), so ``_create_fulltext_index`` compares the existing index's
``labelsOrTypes`` with this workspace's label before it skips (ONLINE) or
drops and recreates (otherwise) that index. Runs without a live Neo4j.
"""

import logging

import pytest

from lightrag.kg import neo4j_impl
from lightrag.kg.neo4j_impl import Neo4JStorage, WorkspaceIndexCollisionError
from lightrag.kg.shared_storage import initialize_share_data


pytestmark = pytest.mark.offline


class _FakeResult:
    def __init__(self, rows):
        self._rows = list(rows)

    async def data(self):
        return self._rows

    async def consume(self):
        return None


class _FakeSession:
    def __init__(self, calls, show_rows):
        self._calls = calls
        self._show_rows = show_rows

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def run(self, query, **params):
        self._calls.append(query)
        if query.strip().startswith("SHOW FULLTEXT INDEXES"):
            return _FakeResult(self._show_rows)
        return _FakeResult([])


class _FakeDriver:
    def __init__(self, show_rows):
        self.calls = []
        self._show_rows = show_rows

    def session(self, **kwargs):
        return _FakeSession(self.calls, self._show_rows)


def _make_storage(workspace: str) -> Neo4JStorage:
    return Neo4JStorage(
        namespace="test",
        global_config={},
        embedding_func=None,
        workspace=workspace,
    )


def _sent(calls: list[str]) -> tuple[bool, bool]:
    """Return (sent a DROP INDEX, sent a CREATE FULLTEXT INDEX)."""
    return (
        any(query.strip().startswith("DROP INDEX") for query in calls),
        any("CREATE FULLTEXT INDEX" in query for query in calls),
    )


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
    driver = _FakeDriver([])
    driver_kwargs = {}

    def fake_driver(_uri, **kwargs):
        driver_kwargs.update(kwargs)
        return driver

    monkeypatch.setattr(neo4j_impl.AsyncGraphDatabase, "driver", fake_driver)
    initialize_share_data()

    await _make_storage("test").initialize()

    assert driver_kwargs["keep_alive"] is expected
    warned = {
        record.args[0]
        for record in lightrag_log_records
        if "is not a recognized boolean" in record.msg
    }
    assert warned == warn_keys


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (None, (False, True)),  # no index yet: create it
        ("ONLINE", (False, False)),  # own index online: leave it alone
        ("POPULATING", (True, True)),  # own index not online: recreate it
    ],
)
async def test_own_or_missing_index_is_handled_as_before(state, expected):
    storage = _make_storage("foo-bar")
    label = storage._get_workspace_label()
    rows = []
    if state is not None:
        rows.append(
            {
                "name": storage._get_fulltext_index_name(label),
                "state": state,
                "labelsOrTypes": [storage._get_raw_workspace_label()],
            }
        )
    driver = _FakeDriver(rows)

    await storage._create_fulltext_index(driver, "neo4j", label)

    assert _sent(driver.calls) == expected


@pytest.mark.parametrize("state", ["ONLINE", "POPULATING"])
async def test_index_of_another_workspace_is_refused_untouched(state):
    owner = _make_storage("foo-bar")
    claimant = _make_storage("foo_bar")
    claimant_label = claimant._get_workspace_label()
    index_name = claimant._get_fulltext_index_name(claimant_label)
    assert owner._get_fulltext_index_name(owner._get_workspace_label()) == index_name
    driver = _FakeDriver(
        [
            {
                "name": index_name,
                "state": state,
                "labelsOrTypes": [owner._get_raw_workspace_label()],
            }
        ]
    )

    # Raised from inside the function's own catch-all ``except Exception``,
    # so this also pins that the refusal is not swallowed there.
    with pytest.raises(WorkspaceIndexCollisionError, match="foo-bar"):
        await claimant._create_fulltext_index(driver, "neo4j", claimant_label)

    # A non-ONLINE index would otherwise be dropped and recreated.
    assert _sent(driver.calls) == (False, False)
