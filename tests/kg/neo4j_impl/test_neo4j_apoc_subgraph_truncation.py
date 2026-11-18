"""``get_knowledge_graph`` truncates an oversized APOC subgraph from one walk.

When the subgraph around a start node holds more than ``max_nodes`` nodes, the
first ``max_nodes`` nodes of the unlimited ``apoc.path.subgraphAll`` walk are
kept, with only the relationships between them, and the walk is not repeated
with APOC's ``limit`` key (why: the comment in ``get_knowledge_graph``).

These tests run without a live Neo4j instance.
"""

import pytest

from lightrag.kg.neo4j_impl import Neo4JStorage

pytestmark = pytest.mark.offline


class _Node(dict):
    def __init__(self, node_id):
        super().__init__(entity_id=f"e{node_id}")
        self.id = node_id


class _Rel(dict):
    def __init__(self, rel_id, start, end):
        super().__init__(weight=1.0)
        self.id = rel_id
        self.type = "DIRECTED"
        self.start_node = start
        self.end_node = end


class _FakeResult:
    def __init__(self, record):
        self._record = record

    async def single(self):
        return self._record

    async def consume(self):
        return None


class _FakeSession:
    """Answers every query with one subgraphAll walk; records the queries."""

    def __init__(self, record):
        self._record = record
        self.queries = []

    async def run(self, query, params=None):
        self.queries.append(query)
        return _FakeResult(self._record)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeDriver:
    def __init__(self, session):
        self._session = session

    def session(self, **kwargs):
        return self._session


def _walk():
    """Six nodes in traversal order from n0, and the relationships among them."""
    n = [_Node(i) for i in range(6)]
    rels = [
        _Rel(100, n[0], n[1]),
        _Rel(101, n[2], n[1]),
        _Rel(102, n[0], n[3]),
        _Rel(103, n[3], n[4]),
        _Rel(104, n[4], n[5]),
        _Rel(105, n[2], n[5]),
    ]
    return {
        "node_info": [{"node": node} for node in n],
        "relationships": rels,
        "total_nodes": len(n),
    }


async def _get_graph(max_nodes):
    storage = Neo4JStorage(
        namespace="test",
        global_config={"max_graph_nodes": 1000},
        embedding_func=None,
        workspace="ws",
    )
    session = _FakeSession(_walk())
    storage._driver = _FakeDriver(session)
    storage._DATABASE = None
    graph = await storage.get_knowledge_graph("e0", max_depth=3, max_nodes=max_nodes)
    return graph, session.queries


@pytest.mark.asyncio
async def test_oversized_subgraph_is_truncated_from_the_single_walk():
    graph, queries = await _get_graph(max_nodes=3)

    assert graph.is_truncated is True
    assert [node.id for node in graph.nodes] == ["0", "1", "2"]
    # Only relationships with both ends kept: n0-n1 and n2-n1.
    assert sorted(edge.id for edge in graph.edges) == ["100", "101"]
    assert all({edge.source, edge.target} <= {"0", "1", "2"} for edge in graph.edges)
    # One walk, and never APOC's `limit` key.
    subgraph_queries = [q for q in queries if "apoc.path.subgraphAll" in q]
    assert len(subgraph_queries) == 1
    assert "limit:" not in subgraph_queries[0]


@pytest.mark.asyncio
async def test_subgraph_within_max_nodes_is_returned_whole():
    graph, queries = await _get_graph(max_nodes=6)

    assert graph.is_truncated is False
    assert [node.id for node in graph.nodes] == ["0", "1", "2", "3", "4", "5"]
    assert len(graph.edges) == 6
    assert len([q for q in queries if "apoc.path.subgraphAll" in q]) == 1
