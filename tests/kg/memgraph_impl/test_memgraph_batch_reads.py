"""``get_nodes_batch`` / ``get_edges_batch`` are one UNWIND query each.

Without these overrides ``MemgraphStorage`` inherited the serial base
fallback, one round trip per id, which ``ainsert_custom_kg`` runs with every
touched entity lock held. Each batch read must answer exactly what the scalar
``get_node`` / ``get_edge`` answer for the same id.
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


class _FakeSession:
    def __init__(self, driver):
        self._driver = driver

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def run(self, query, parameters=None, **kwargs):
        self._driver.runs.append((query, kwargs))
        if self._driver.run_error is not None:
            raise self._driver.run_error
        return self._driver.result


class _FakeDriver:
    def __init__(self, records=(), run_error=None):
        self.result = _FakeResult(records)
        self.run_error = run_error
        self.runs: list[tuple[str, dict]] = []

    def session(self, **kwargs):
        return _FakeSession(self)


def _storage(driver: _FakeDriver) -> MemgraphStorage:
    storage = MemgraphStorage(
        namespace="chunk_entity_relation",
        global_config={"max_graph_nodes": 1000},
        embedding_func=None,
        workspace="test",
    )
    storage._driver = driver
    storage._DATABASE = "memgraph"
    return storage


@pytest.mark.asyncio
async def test_get_nodes_batch_is_one_query_with_get_node_results():
    label = "test"
    driver = _FakeDriver(
        [
            {
                "entity_id": "A",
                "n": {"entity_id": "A", "labels": [label, "concept"]},
            },
            # A duplicate keeps the first node, as get_node does.
            {"entity_id": "A", "n": {"entity_id": "A", "description": "dup"}},
            {"entity_id": "B", "n": {"entity_id": "B"}},
        ]
    )
    storage = _storage(driver)
    assert storage._get_workspace_label() == label

    nodes = await storage.get_nodes_batch(["A", "B", "Missing", "A"])

    assert len(driver.runs) == 1
    query, params = driver.runs[0]
    assert "UNWIND $ids" in query and f"`{label}`" in query
    assert params == {"ids": ["A", "B", "Missing"]}
    assert nodes == {
        "A": {"entity_id": "A", "labels": ["concept"]},
        "B": {"entity_id": "B"},
    }
    assert driver.result.consumed


@pytest.mark.asyncio
async def test_get_edges_batch_is_one_query_keyed_as_requested():
    driver = _FakeDriver(
        [
            {
                "src": "B",
                "tgt": "A",
                "edge_properties": {
                    "weight": 2.0,
                    "source_id": "c1",
                    "description": "d",
                    "keywords": "k",
                },
            },
            {"src": "B", "tgt": "A", "edge_properties": {"weight": 9.0}},
            # A missing core property gets get_edge's default.
            {"src": "A", "tgt": "C", "edge_properties": {"source_id": "c2"}},
        ]
    )
    storage = _storage(driver)

    edges = await storage.get_edges_batch(
        [{"src": "B", "tgt": "A"}, {"src": "A", "tgt": "C"}, {"src": "X", "tgt": "Y"}]
    )

    assert len(driver.runs) == 1
    query, params = driver.runs[0]
    assert "UNWIND $pairs" in query and "-[r]-" in query
    assert params == {
        "pairs": [
            {"src": "B", "tgt": "A"},
            {"src": "A", "tgt": "C"},
            {"src": "X", "tgt": "Y"},
        ]
    }
    assert edges == {
        ("B", "A"): {
            "weight": 2.0,
            "source_id": "c1",
            "description": "d",
            "keywords": "k",
        },
        ("A", "C"): {
            "weight": 1.0,
            "source_id": "c2",
            "description": None,
            "keywords": None,
        },
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "arg"),
    [("get_nodes_batch", ["A"]), ("get_edges_batch", [{"src": "A", "tgt": "B"}])],
)
async def test_batch_read_failure_propagates(method, arg):
    driver = _FakeDriver(run_error=RuntimeError("memgraph down"))
    with pytest.raises(RuntimeError, match="memgraph down"):
        await getattr(_storage(driver), method)(arg)
    assert len(driver.runs) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["get_nodes_batch", "get_edges_batch"])
async def test_empty_batch_issues_no_query(method):
    driver = _FakeDriver()
    assert await getattr(_storage(driver), method)([]) == {}
    assert driver.runs == []
