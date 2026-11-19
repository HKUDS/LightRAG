"""Full-text index name collisions across workspaces must be refused, not
silently resolved by reusing or recreating someone else's index.

``_get_fulltext_index_name`` derives the index name from a lossy
normalization of the workspace label (``_normalize_index_suffix`` folds
every run of non-alphanumeric characters to a single underscore), so two
distinct workspace labels -- e.g. "foo-bar" and "foo_bar" -- can resolve to
the same index name. ``_create_fulltext_index`` must compare the existing
index's actual labels (``labelsOrTypes`` from ``SHOW FULLTEXT INDEXES``)
against this workspace's own label BEFORE deciding what to do with it:

* a matching label is the common case and proceeds unchanged (skip if
  ONLINE, recreate otherwise);
* a mismatched label must raise ``WorkspaceIndexCollisionError`` and send
  no ``DROP INDEX`` -- including when the colliding index is not ONLINE,
  which is exactly the case the unconditional "recreate" branch would
  otherwise drop and take over.

These tests run without a live Neo4j instance.
"""

import pytest

from lightrag.kg.neo4j_impl import Neo4JStorage, WorkspaceIndexCollisionError


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


def _sent_drop(calls: list[str]) -> bool:
    return any(query.strip().startswith("DROP INDEX") for query in calls)


def _sent_create(calls: list[str]) -> bool:
    return any("CREATE FULLTEXT INDEX" in query for query in calls)


async def test_matching_label_online_index_is_left_alone():
    storage = _make_storage("foo-bar")
    label = storage._get_workspace_label()
    index_name = storage._get_fulltext_index_name(label)
    driver = _FakeDriver(
        [
            {
                "name": index_name,
                "state": "ONLINE",
                "labelsOrTypes": [storage._get_raw_workspace_label()],
            }
        ]
    )

    await storage._create_fulltext_index(driver, "neo4j", label)

    assert not _sent_drop(driver.calls)
    assert not _sent_create(driver.calls)


async def test_matching_label_non_online_index_is_recreated():
    storage = _make_storage("foo-bar")
    label = storage._get_workspace_label()
    index_name = storage._get_fulltext_index_name(label)
    driver = _FakeDriver(
        [
            {
                "name": index_name,
                "state": "POPULATING",
                "labelsOrTypes": [storage._get_raw_workspace_label()],
            }
        ]
    )

    await storage._create_fulltext_index(driver, "neo4j", label)

    assert _sent_drop(driver.calls)
    assert _sent_create(driver.calls)


async def test_mismatched_label_online_index_raises_without_recreating():
    # "foo-bar" and "foo_bar" both normalize to the same index name.
    owner = _make_storage("foo-bar")
    claimant = _make_storage("foo_bar")
    claimant_label = claimant._get_workspace_label()
    index_name = claimant._get_fulltext_index_name(claimant_label)
    assert owner._get_fulltext_index_name(owner._get_workspace_label()) == index_name

    driver = _FakeDriver(
        [
            {
                "name": index_name,
                "state": "ONLINE",
                "labelsOrTypes": [owner._get_raw_workspace_label()],
            }
        ]
    )

    with pytest.raises(WorkspaceIndexCollisionError):
        await claimant._create_fulltext_index(driver, "neo4j", claimant_label)

    assert not _sent_drop(driver.calls)
    assert not _sent_create(driver.calls)


async def test_mismatched_label_non_online_index_raises_before_dropping():
    """The critical case: without the check, a non-ONLINE index is
    unconditionally dropped and recreated, taking over another workspace's
    index instead of merely misrouting searches to it.
    """
    owner = _make_storage("foo-bar")
    claimant = _make_storage("foo_bar")
    claimant_label = claimant._get_workspace_label()
    index_name = claimant._get_fulltext_index_name(claimant_label)

    driver = _FakeDriver(
        [
            {
                "name": index_name,
                "state": "POPULATING",
                "labelsOrTypes": [owner._get_raw_workspace_label()],
            }
        ]
    )

    with pytest.raises(WorkspaceIndexCollisionError):
        await claimant._create_fulltext_index(driver, "neo4j", claimant_label)

    assert not _sent_drop(driver.calls)
    assert not _sent_create(driver.calls)


async def test_no_existing_index_creates_a_new_one():
    storage = _make_storage("foo-bar")
    label = storage._get_workspace_label()
    driver = _FakeDriver([])

    await storage._create_fulltext_index(driver, "neo4j", label)

    assert not _sent_drop(driver.calls)
    assert _sent_create(driver.calls)
