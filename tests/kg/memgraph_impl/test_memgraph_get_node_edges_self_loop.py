"""get_node_edges must list a self-loop exactly once.

``OPTIONAL MATCH (n)-[r]-(connected)`` is undirected with ``n`` anchored to
the node under inspection, so an ordinary neighbor binds to exactly one row.
A self-loop is the exception: both endpoints of ``r`` equal ``n``'s value,
so the pattern matches it from "both sides" and the server returns two rows
for the one relationship (``startNode(r)`` is ``n`` in both). Without a
client-side guard, the duplicate survives into the returned edge list,
which breaks the ``BaseGraphStorage.get_node_edges`` contract that a
self-loop "appears ONCE -- listed, never hidden" (see ``lightrag/base.py``)
and double-processes it wherever this method feeds entity deletion, rename,
merge or purge.
"""

import pytest

from lightrag.kg.memgraph_impl import MemgraphStorage


pytestmark = pytest.mark.offline


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
    def __init__(self, result):
        self._result = result

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def run(self, query, parameters=None, **kwargs):
        return self._result


class _FakeDriver:
    def __init__(self, result):
        self._result = result

    def session(self, **kwargs):
        return _FakeSession(self._result)


def _make_storage(records):
    storage = MemgraphStorage(
        namespace="chunk_entity_relation",
        global_config={"max_graph_nodes": 1000},
        embedding_func=None,
        workspace="test",
    )
    storage._driver = _FakeDriver(_FakeResult(records))
    storage._DATABASE = "memgraph"
    return storage


def _record(node_entity_id, connected_entity_id, start_entity_id):
    return {
        "node_entity_id": node_entity_id,
        "connected_entity_id": connected_entity_id,
        "start_entity_id": start_entity_id,
    }


@pytest.mark.asyncio
async def test_self_loop_is_listed_once():
    # The undirected anchor pattern matches a self-loop on "A" from both
    # sides, so the server reports the one relationship as two identical
    # rows -- startNode(r) is "A" in both, since both endpoints are "A".
    records = [
        _record("A", "A", "A"),
        _record("A", "A", "A"),
    ]
    storage = _make_storage(records)

    edges = await storage.get_node_edges("A")

    assert edges == [("A", "A")]


@pytest.mark.asyncio
async def test_ordinary_neighbor_is_unaffected():
    records = [_record("A", "B", "A")]
    storage = _make_storage(records)

    edges = await storage.get_node_edges("A")

    assert edges == [("A", "B")]


@pytest.mark.asyncio
async def test_self_loop_plus_distinct_neighbor_both_kept():
    records = [
        _record("A", "A", "A"),
        _record("A", "A", "A"),
        _record("A", "B", "A"),
    ]
    storage = _make_storage(records)

    edges = await storage.get_node_edges("A")

    assert sorted(edges) == [("A", "A"), ("A", "B")]
