"""
Unit tests for PGGraphStorage.get_knowledge_graph("*") — the full-graph view.

These cover the undirected fix for the degree-based node selection used when the
graph is truncated to ``max_nodes``: the previous implementation counted only
outgoing edges (``OPTIONAL MATCH (n)-[r]->()``), so a node that is mostly an edge
*target* was under-ranked and could be dropped on truncation. The fix selects
top-degree nodes via native SQL that counts both ``start_id`` and ``end_id``
(undirected degree), LEFT JOIN-ing the base vertex table so isolated nodes are
preserved when the graph is not truncated.

The subgraph edge read intentionally stays directed: ``a`` iterates over every
selected node, so ``(a)-[r]->(b)`` already captures every edge whose endpoints
are both selected. These tests guard against an accidental regression there.

All tests mock ``PGGraphStorage._query`` and inspect the SQL it receives.
"""

import pytest
from unittest.mock import MagicMock, patch

from lightrag.kg.postgres_impl import PGGraphStorage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_graph_storage() -> PGGraphStorage:
    """Construct a PGGraphStorage instance with a mocked _query method."""
    storage = PGGraphStorage.__new__(PGGraphStorage)
    storage.workspace = "test_ws"
    storage.namespace = "test_graph"
    storage.graph_name = "test_graph"
    storage.global_config = {"max_graph_nodes": 1000}
    storage.db = MagicMock()
    return storage


class _QueryCapture:
    """Dispatch the three _query calls of the '*' branch by SQL content."""

    def __init__(self, *, total_nodes, degree_rows, subgraph_rows):
        self._total_nodes = total_nodes
        self._degree_rows = degree_rows
        self._subgraph_rows = subgraph_rows
        self.count_sql = None
        self.degree_sql = None
        self.degree_params = None
        self.subgraph_sql = None

    def as_side_effect(self):
        """Return an ``async def`` so AsyncMock awaits it (a callable instance is not)."""

        async def fake_query(query, **kwargs):
            if "count(distinct n)" in query:
                self.count_sql = query
                return [{"total_nodes": self._total_nodes}]
            if "node_degrees" in query:
                self.degree_sql = query
                self.degree_params = kwargs.get("params")
                return self._degree_rows
            # subgraph read
            self.subgraph_sql = query
            return self._subgraph_rows

        return fake_query


def _node(node_id, entity_id):
    return {"id": node_id, "properties": {"entity_id": entity_id}}


def _edge(edge_id, start_id, end_id, weight="1"):
    return {
        "id": edge_id,
        "label": "DIRECTED",
        "start_id": start_id,
        "end_id": end_id,
        "properties": {"weight": weight},
    }


# ---------------------------------------------------------------------------
# degree node selection SQL
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_degree_selection_sql_is_undirected_and_preserves_isolated():
    """The degree query must count start_id + end_id and keep degree-0 nodes."""
    capture = _QueryCapture(
        total_nodes=2,
        degree_rows=[{"node_id": 1, "degree": 3}, {"node_id": 2, "degree": 0}],
        subgraph_rows=[{"a": _node(1, "Alice"), "r": None, "b": None}],
    )
    storage = make_graph_storage()

    with patch.object(storage, "_query", side_effect=capture.as_side_effect()):
        await storage.get_knowledge_graph("*", max_nodes=50)

    sql = capture.degree_sql
    assert sql is not None, "degree selection query was never issued"
    # Undirected: both edge endpoints are counted.
    assert "start_id" in sql
    assert "end_id" in sql
    assert "UNION ALL" in sql
    # Isolated nodes preserved.
    assert "LEFT JOIN" in sql
    assert "COALESCE" in sql
    # Stable ordering with id tie-break.
    assert "ORDER BY degree DESC" in sql
    assert "v.id ASC" in sql
    # The old outgoing-only Cypher must be gone.
    assert "-[r]->()" not in sql
    assert "OPTIONAL MATCH (n)-[r]->()" not in sql


@pytest.mark.asyncio
async def test_degree_selection_limit_is_parameterized():
    """max_nodes must be passed via params, not interpolated into the SQL."""
    capture = _QueryCapture(
        total_nodes=2,
        degree_rows=[{"node_id": 1, "degree": 3}],
        subgraph_rows=[{"a": _node(1, "Alice"), "r": None, "b": None}],
    )
    storage = make_graph_storage()

    with patch.object(storage, "_query", side_effect=capture.as_side_effect()):
        await storage.get_knowledge_graph("*", max_nodes=37)

    assert "LIMIT $1" in capture.degree_sql
    assert "37" not in capture.degree_sql
    assert capture.degree_params == {"limit": 37}


@pytest.mark.asyncio
async def test_isolated_node_survives_end_to_end():
    """A degree-0 node selected by the degree query reaches the final KnowledgeGraph."""
    capture = _QueryCapture(
        total_nodes=2,
        degree_rows=[{"node_id": 1, "degree": 2}, {"node_id": 2, "degree": 0}],
        subgraph_rows=[
            {"a": _node(1, "Alice"), "r": None, "b": None},
            # MATCH (a) still returns the isolated node with null r/b.
            {"a": _node(2, "Bob"), "r": None, "b": None},
        ],
    )
    storage = make_graph_storage()

    with patch.object(storage, "_query", side_effect=capture.as_side_effect()):
        kg = await storage.get_knowledge_graph("*", max_nodes=50)

    # The isolated node id is formatted into the subgraph id list...
    assert "2" in capture.subgraph_sql
    # ...and present in the returned graph.
    assert {node.id for node in kg.nodes} == {"1", "2"}
    assert kg.is_truncated is False


# ---------------------------------------------------------------------------
# subgraph edge read — regression guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subgraph_read_stays_directed_and_dedupes_edges():
    """Subgraph read keeps the directed (a)-[r]->(b) match and dedupes by edge id."""
    duplicate_edge = _edge(10, 1, 2)
    capture = _QueryCapture(
        total_nodes=2,
        degree_rows=[{"node_id": 1, "degree": 1}, {"node_id": 2, "degree": 1}],
        subgraph_rows=[
            {"a": _node(1, "Alice"), "r": duplicate_edge, "b": _node(2, "Bob")},
            # Same AGE edge id appearing twice must collapse to one edge.
            {"a": _node(1, "Alice"), "r": duplicate_edge, "b": _node(2, "Bob")},
        ],
    )
    storage = make_graph_storage()

    with patch.object(storage, "_query", side_effect=capture.as_side_effect()):
        kg = await storage.get_knowledge_graph("*", max_nodes=50)

    assert "(a)-[r]->(b)" in capture.subgraph_sql
    assert len(kg.edges) == 1
    edge = kg.edges[0]
    assert edge.source == "1"
    assert edge.target == "2"
