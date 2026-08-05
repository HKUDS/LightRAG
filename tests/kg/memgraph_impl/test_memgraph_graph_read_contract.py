"""Read-path contracts MemgraphStorage shares with the other graph backends.

Two invariants:

1. ``get_node_edges`` returns None for a node that does not exist and ``[]``
   for one that exists with no relations. Answering ``[]`` for both makes a
   deleted entity indistinguishable from an isolated one.
2. A query failure in the label helpers propagates. Returning ``[]`` there
   reports a dead database as "this graph has no entities" -- and
   ``/graph/label/popular`` already turns an exception into a 500, so the
   swallow was the only thing standing between the user and an accurate error.
"""

import pytest

from lightrag.kg.memgraph_impl import MemgraphStorage


pytestmark = pytest.mark.offline


class _FakeResult:
    """Async-iterable result, optionally raising when iterated."""

    def __init__(self, records, error=None):
        self._records = list(records)
        self._error = error
        self.consumed = False

    def __aiter__(self):
        self._iter = iter(self._records)
        return self

    async def __anext__(self):
        if self._error is not None:
            raise self._error
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration

    async def consume(self):
        self.consumed = True
        return None


class _FakeSession:
    def __init__(self, result, run_error=None):
        self._result = result
        self._run_error = run_error

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def run(self, query, parameters=None, **kwargs):
        if self._run_error is not None:
            raise self._run_error
        return self._result


class _FakeDriver:
    def __init__(self, result, run_error=None):
        self._result = result
        self._run_error = run_error

    def session(self, **kwargs):
        return _FakeSession(self._result, self._run_error)


def _make_storage(records=(), run_error=None, iter_error=None):
    storage = MemgraphStorage(
        namespace="chunk_entity_relation",
        global_config={"max_graph_nodes": 1000},
        embedding_func=None,
        workspace="test",
    )
    storage._driver = _FakeDriver(_FakeResult(records, iter_error), run_error)
    storage._DATABASE = "memgraph"
    return storage


# ---------------------------------------------------------------------------
# get_node_edges: absent node vs isolated node
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_node_edges_returns_none_for_missing_node():
    """Zero rows means the anchor MATCH never bound n."""
    storage = _make_storage(records=[])

    assert await storage.get_node_edges("NoSuchEntity") is None


@pytest.mark.asyncio
async def test_get_node_edges_returns_empty_list_for_isolated_node():
    """An existing node with no relations still yields one row from the
    OPTIONAL MATCH, carrying a NULL connected_entity_id."""
    storage = _make_storage(
        records=[
            {
                "node_entity_id": "Lonely",
                "connected_entity_id": None,
                "start_entity_id": None,
            }
        ]
    )

    assert await storage.get_node_edges("Lonely") == []


@pytest.mark.asyncio
async def test_get_node_edges_preserves_edge_direction():
    storage = _make_storage(
        records=[
            {
                "node_entity_id": "Alpha",
                "connected_entity_id": "Beta",
                "start_entity_id": "Alpha",
            },
            {
                "node_entity_id": "Alpha",
                "connected_entity_id": "Gamma",
                "start_entity_id": "Gamma",
            },
        ]
    )

    assert await storage.get_node_edges("Alpha") == [
        ("Alpha", "Beta"),
        ("Gamma", "Alpha"),
    ]


# ---------------------------------------------------------------------------
# Label helpers: a query failure is not "no labels"
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_popular_labels_raises_on_query_error():
    storage = _make_storage(run_error=RuntimeError("memgraph is down"))

    with pytest.raises(RuntimeError, match="memgraph is down"):
        await storage.get_popular_labels(limit=10)


@pytest.mark.asyncio
async def test_get_popular_labels_raises_on_iteration_error():
    """A mid-stream failure loses part of the ranking — that partial result
    must not be returned as if it were the whole graph."""
    storage = _make_storage(iter_error=RuntimeError("connection reset"))

    with pytest.raises(RuntimeError, match="connection reset"):
        await storage.get_popular_labels(limit=10)


@pytest.mark.asyncio
async def test_search_labels_raises_on_query_error():
    storage = _make_storage(run_error=RuntimeError("memgraph is down"))

    with pytest.raises(RuntimeError, match="memgraph is down"):
        await storage.search_labels("alpha")


@pytest.mark.asyncio
async def test_search_labels_still_short_circuits_on_blank_query():
    """An empty query is a real "nothing to match", not an error."""
    storage = _make_storage(run_error=RuntimeError("must not be reached"))

    assert await storage.search_labels("   ") == []
