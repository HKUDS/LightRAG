"""get_all_edges / iter_edges must return one row per stored relationship.

``MATCH (a)-[r]-(b)`` is undirected with both endpoints free, so Neo4j's
pattern matcher yields one row per orientation of every relationship --
{a:X, b:Y} and {a:Y, b:X}. Every edge is created through an undirected MERGE
(see upsert_edge), which still gives the relationship exactly one physical
direction in storage, so the fix matches it directed (``-[r]->``) instead:
each relationship is then returned exactly once, with no DISTINCT and no
client-side dedup set to keep in memory -- required for iter_edges, whose
batch/yield contract must stay O(batch_size), not O(graph size).
"""

import pytest

from lightrag.kg.neo4j_impl import Neo4JStorage


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


def _record(source, target, properties):
    """A directed match returns one row per relationship, in its stored
    orientation -- no rel_id needed since there is nothing left to dedupe."""
    return {"source": source, "target": target, "properties": dict(properties)}


@pytest.mark.asyncio
async def test_get_all_edges_returns_one_row_per_relationship():
    records = [_record("Alpha", "Beta", {"weight": 1.0})]
    storage, calls = _make_storage(records)

    edges = await storage.get_all_edges()

    assert len(edges) == 1
    assert edges[0]["source"] == "Alpha"
    assert edges[0]["target"] == "Beta"
    query, _ = calls[0]
    assert "-[r]->" in query
    assert "DISTINCT" not in query
    assert "id(r)" not in query


@pytest.mark.asyncio
async def test_get_all_edges_keeps_distinct_relationships():
    records = [
        _record("Alpha", "Beta", {"weight": 1.0}),
        _record("Alpha", "Gamma", {"weight": 2.0}),
    ]
    storage, _ = _make_storage(records)

    edges = await storage.get_all_edges()

    pairs = {(e["source"], e["target"]) for e in edges}
    assert len(edges) == 2
    assert pairs == {("Alpha", "Beta"), ("Alpha", "Gamma")}


@pytest.mark.asyncio
async def test_iter_edges_returns_one_row_per_relationship():
    records = [_record("Alpha", "Beta", {"weight": 1.0})]
    storage, calls = _make_storage(records)

    batches = [batch async for batch in storage.iter_edges(batch_size=10)]
    edges = [edge for batch in batches for edge in batch]

    assert len(edges) == 1
    assert edges[0]["source"] == "Alpha"
    assert edges[0]["target"] == "Beta"
    query, _ = calls[0]
    assert "-[r]->" in query
    assert "DISTINCT" not in query
    assert "id(r)" not in query
