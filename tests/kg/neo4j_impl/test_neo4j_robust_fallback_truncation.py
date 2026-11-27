"""Regression tests for ``Neo4JStorage._robust_fallback`` truncation reporting.

This is the pure-Cypher BFS fallback used when the APOC plugin is unavailable.
It used to set ``is_truncated = True`` unconditionally the instant the node
cap was reached, without checking whether the queue actually still held an
unvisited, in-depth-limit candidate -- so a graph whose full node count
exactly equals ``max_nodes`` was falsely reported as truncated.

These tests run without a live Neo4j instance, mirroring the fake
driver/session style in ``test_workspace_label_injection.py``.
"""

import pytest
from neo4j.exceptions import ClientError

from lightrag.kg.neo4j_impl import Neo4JStorage


pytestmark = pytest.mark.offline


class _FakeNode(dict):
    """Minimal stand-in for a neo4j Node: dict-like plus a `._properties`."""

    def __init__(self, entity_id: str):
        props = {"entity_id": entity_id}
        super().__init__(props)
        self._properties = props


class _FakeRel(dict):
    """Minimal stand-in for a neo4j Relationship: dict-like plus `.type`."""

    def __init__(self, rel_type: str = "RELATED"):
        super().__init__({"weight": 1.0})
        self.type = rel_type


class _FakeResult:
    def __init__(self, records: list):
        self._records = records

    async def single(self):
        return self._records[0] if self._records else None

    async def fetch(self, n: int):
        return self._records[:n]

    async def consume(self):
        return None


class _FakeSession:
    """Simulates missing APOC and serves plain-Cypher node and edge queries."""

    def __init__(self, node_docs: dict, edges_by_node: dict):
        self._node_docs = node_docs
        self._edges_by_node = edges_by_node

    async def run(self, query, entity_id=None, **kwargs):
        if "CALL apoc.path.subgraphAll" in query:
            raise ClientError("APOC procedure not found")
        if "RETURN id(n) as node_id, n" in query:
            node = self._node_docs.get(entity_id)
            return _FakeResult([{"n": node, "node_id": 1}] if node else [])
        return _FakeResult(self._edges_by_node.get(entity_id, []))

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeDriver:
    def __init__(self, session: _FakeSession):
        self._session = session

    def session(self, **kwargs):
        return self._session


def _make_storage(node_docs: dict, edges_by_node: dict) -> Neo4JStorage:
    storage = Neo4JStorage(
        namespace="test", global_config={}, embedding_func=None, workspace="ws"
    )
    storage._driver = _FakeDriver(_FakeSession(node_docs, edges_by_node))
    storage._DATABASE = None
    return storage


def _edge(edge_id: str, target_entity_id: str) -> dict:
    return {"r": _FakeRel(), "b": _FakeNode(target_entity_id), "edge_id": edge_id}


@pytest.mark.asyncio
async def test_robust_fallback_reports_truncated_over_cap():
    """Star graph (A + 5 leaves) with max_nodes=3 must be flagged truncated
    and must never return more than max_nodes nodes."""
    leaves = ["B", "C", "D", "E", "F"]
    node_docs = {n: _FakeNode(n) for n in ["A"] + leaves}
    edges_by_node = {"A": [_edge(f"e{i}", leaf) for i, leaf in enumerate(leaves)]}
    for i, leaf in enumerate(leaves):
        edges_by_node[leaf] = [_edge(f"e{i}", "A")]
    storage = _make_storage(node_docs, edges_by_node)

    result = await storage._robust_fallback("A", max_depth=5, max_nodes=3)

    assert result.is_truncated is True
    assert len(result.nodes) <= 3


@pytest.mark.asyncio
async def test_robust_fallback_duplicate_queue_entries_not_falsely_truncated():
    """Diamond graph (A->B, A->C, B->D, C->D) with max_nodes=4: all four
    nodes exactly fill the cap, so the graph must not be reported truncated.
    D is reachable via two parents and gets queued twice before ever being
    visited; the duplicate entry must be dropped by the visited check, not
    misread as a real candidate that didn't fit under the cap."""
    node_docs = {n: _FakeNode(n) for n in ["A", "B", "C", "D"]}
    edges_by_node = {
        "A": [_edge("e1", "B"), _edge("e2", "C")],
        "B": [_edge("e1", "A"), _edge("e3", "D")],
        "C": [_edge("e2", "A"), _edge("e4", "D")],
        "D": [_edge("e3", "B"), _edge("e4", "C")],
    }
    storage = _make_storage(node_docs, edges_by_node)

    result = await storage._robust_fallback("A", max_depth=5, max_nodes=4)

    assert {n.id for n in result.nodes} == {"A", "B", "C", "D"}
    assert result.is_truncated is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("leaf_count", "is_truncated"),
    [(1000, False), (1001, True)],
    ids=["at-scan-limit", "over-scan-limit"],
)
async def test_get_knowledge_graph_reports_neighbor_scan_truncation(
    leaf_count, is_truncated
):
    storage = _make_storage(
        {"Hub": _FakeNode("Hub")},
        {
            "Hub": [_edge(f"e{i}", f"L{i}") for i in range(leaf_count)],
            **{f"L{i}": [_edge(f"e{i}", "Hub")] for i in range(leaf_count)},
        },
    )
    storage.global_config["max_graph_nodes"] = 5000

    graph = await storage.get_knowledge_graph("Hub", max_depth=1, max_nodes=5000)
    payload = graph.model_dump()

    assert (len(payload["nodes"]), len(payload["edges"]), payload["is_truncated"]) == (
        1001,
        1000,
        is_truncated,
    )
