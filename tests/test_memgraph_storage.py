import pytest

from lightrag.kg.memgraph_impl import MemgraphStorage


pytestmark = pytest.mark.offline


class _FakeNode(dict):
    def __init__(self, node_id: int, entity_id: str, **properties):
        super().__init__(entity_id=entity_id, **properties)
        self.id = node_id


class _FakeResult:
    def __init__(self, record):
        self._record = record

    async def single(self):
        return self._record

    async def consume(self):
        return None


class _FakeSession:
    def __init__(self, record, calls):
        self._record = record
        self._calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def run(self, query, parameters=None, **kwargs):
        if parameters is None:
            parameters = kwargs
        self._calls.append((query, parameters))
        return _FakeResult(self._record)


class _FakeDriver:
    def __init__(self, record, calls):
        self._record = record
        self._calls = calls

    def session(self, **kwargs):
        return _FakeSession(self._record, self._calls)


def _make_storage(record):
    calls = []
    storage = MemgraphStorage(
        namespace="chunk_entity_relation",
        global_config={"max_graph_nodes": 1000},
        embedding_func=None,
        workspace="test",
    )
    storage._driver = _FakeDriver(record, calls)
    storage._DATABASE = "memgraph"
    return storage, calls


@pytest.mark.asyncio
async def test_get_knowledge_graph_preserves_isolated_start_node():
    start_node = _FakeNode(1, "Start", description="isolated")
    storage, calls = _make_storage(
        {
            "node_info": [{"node": start_node}],
            "relationships": [],
            "is_truncated": False,
        }
    )

    result = await storage.get_knowledge_graph("Start", max_depth=0, max_nodes=1)

    assert len(result.nodes) == 1
    assert result.nodes[0].labels == ["Start"]
    assert result.nodes[0].properties["entity_id"] == "Start"
    assert result.edges == []

    assert len(calls) == 1
    query, params = calls[0]
    assert "OPTIONAL MATCH path = (start)-[*BFS 0..0]-(end:`test`)" in query
    assert "THEN [start] + other_nodes" in query
    assert "OPTIONAL MATCH (n)-[r]-(m)" in query
    assert "[rel IN relationships WHERE rel IS NOT NULL] AS relationships" in query
    assert params["entity_id"] == "Start"
    assert params["max_nodes"] == 1
    assert params["max_other_nodes"] == 0


@pytest.mark.asyncio
async def test_get_knowledge_graph_reserves_capacity_for_start_node_when_truncating():
    start_node = _FakeNode(1, "Start")
    storage, calls = _make_storage(
        {
            "node_info": [{"node": start_node}],
            "relationships": [],
            "is_truncated": True,
        }
    )

    result = await storage.get_knowledge_graph("Start", max_depth=2, max_nodes=2)

    assert result.is_truncated is True
    assert len(calls) == 1
    query, params = calls[0]
    assert "other_nodes[0..$max_other_nodes]" in query
    assert "1 + size(other_nodes) > $max_nodes AS is_truncated" in query
    assert params["max_other_nodes"] == 1
