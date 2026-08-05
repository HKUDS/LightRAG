"""Read-path contract Neo4JStorage shares with the other graph backends.

``get_node_edges`` must return None for a node that does not exist and ``[]``
for one that exists with no relations. Answering ``[]`` for both makes a
deleted entity indistinguishable from an isolated one, which is the whole point
of the ``list | None`` return type in BaseGraphStorage.
"""

import pytest

from lightrag.kg.neo4j_impl import Neo4JStorage


pytestmark = pytest.mark.offline


class _FakeNode(dict):
    """Neo4j node stand-in: property access via .get(), truthy when populated."""


class _FakeResult:
    def __init__(self, records):
        self._records = list(records)
        self.consumed = False

    def __aiter__(self):
        self._iter = iter(self._records)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration

    async def consume(self):
        self.consumed = True
        return None


class _FakeSession:
    def __init__(self, result, calls):
        self._result = result
        self._calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def run(self, query, **params):
        self._calls.append((query, params))
        return self._result


class _FakeDriver:
    def __init__(self, result, calls):
        self._result = result
        self._calls = calls

    def session(self, **kwargs):
        return _FakeSession(self._result, self._calls)


def _make_storage(records):
    calls = []
    storage = Neo4JStorage(
        namespace="chunk_entity_relation",
        global_config={"max_graph_nodes": 1000},
        embedding_func=None,
        workspace="test",
    )
    storage._driver = _FakeDriver(_FakeResult(records), calls)
    storage._DATABASE = "neo4j"
    return storage, calls


@pytest.mark.asyncio
async def test_get_node_edges_returns_none_for_missing_node():
    """Zero rows means the anchor MATCH never bound n: no such node."""
    storage, _ = _make_storage([])

    assert await storage.get_node_edges("NoSuchEntity") is None


@pytest.mark.asyncio
async def test_get_node_edges_returns_empty_list_for_isolated_node():
    """An existing node with no relations still yields one row from the
    OPTIONAL MATCH, with a NULL connected node."""
    storage, _ = _make_storage(
        [{"n": _FakeNode(entity_id="Lonely"), "r": None, "connected": None}]
    )

    assert await storage.get_node_edges("Lonely") == []


@pytest.mark.asyncio
async def test_get_node_edges_returns_connected_pairs():
    storage, _ = _make_storage(
        [
            {
                "n": _FakeNode(entity_id="Alpha"),
                "r": object(),
                "connected": _FakeNode(entity_id="Beta"),
            },
            {
                "n": _FakeNode(entity_id="Alpha"),
                "r": object(),
                "connected": _FakeNode(entity_id="Gamma"),
            },
        ]
    )

    assert await storage.get_node_edges("Alpha") == [
        ("Alpha", "Beta"),
        ("Alpha", "Gamma"),
    ]
