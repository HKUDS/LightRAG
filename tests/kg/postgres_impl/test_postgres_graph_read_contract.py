"""Unit tests for PGGraphStorage read-path contracts shared with the other
graph backends.

Covers two divergences from ``BaseGraphStorage`` / ``NetworkXStorage``:

1. ``get_popular_labels`` used to rank nodes by inner-joining the vertex table
   against degrees derived from the edge table, so an entity with no relations
   (degree 0) had no row to join against and vanished from the ranking
   entirely.
2. ``get_node_edges`` returned ``[]`` for a node that does not exist, which is
   the same value an existing-but-isolated node yields — collapsing "no such
   node" into "no edges". The contract (and NetworkX/PGOps) is ``None``.
"""

import re
import pytest
from unittest.mock import AsyncMock, MagicMock

from lightrag.kg.postgres_impl import PGGraphStorage


def make_graph_storage() -> PGGraphStorage:
    """Construct a PGGraphStorage instance with a mocked db."""
    storage = PGGraphStorage.__new__(PGGraphStorage)
    storage.workspace = "test_ws"
    storage.namespace = "test_graph"
    storage.graph_name = "test_graph"
    storage.__post_init__()
    storage.db = MagicMock()
    return storage


def _normalize(sql: str) -> str:
    """Collapse whitespace so multi-line SQL can be matched as one string."""
    return re.sub(r"\s+", " ", sql).strip()


# ---------------------------------------------------------------------------
# get_popular_labels: isolated (degree-0) nodes must be ranked, not dropped
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_popular_labels_left_joins_degrees_onto_all_vertices():
    """The vertex table must drive the join so degree-0 nodes survive.

    Regression: the previous query was ``FROM node_degrees d JOIN
    _ag_label_vertex v``. ``node_degrees`` is built purely from the edge table,
    so a node with no relations has no row there and the inner join dropped it.
    A graph of entities with no relations reported ZERO popular labels, while
    NetworkXStorage (dict(graph.degree())) and PGOpsGraphStorage (LEFT JOIN +
    COALESCE) rank every node.
    """
    storage = make_graph_storage()
    storage._query = AsyncMock(return_value=[])

    await storage.get_popular_labels(limit=7)

    sql = _normalize(storage._query.await_args.args[0])

    # Vertex table is the driving (outer) side, degrees are the optional side.
    assert "FROM test_graph._ag_label_vertex v LEFT JOIN node_degrees d" in sql, sql
    # Nodes absent from node_degrees rank as degree 0 rather than disappearing.
    assert "COALESCE(d.degree, 0) AS degree" in sql
    # ... and no inner join back onto the degree set anywhere.
    assert not re.search(r"(?<!LEFT )JOIN node_degrees", sql), sql


@pytest.mark.asyncio
async def test_get_popular_labels_orders_by_degree_then_bytewise_label():
    """Ties break on a byte-order label sort, matching Python's code-point sort.

    With degree-0 nodes included, ties are the common case and decide which
    labels survive the LIMIT — so the tie-break has to agree with the other
    backends instead of following the database's locale collation.
    """
    storage = make_graph_storage()
    storage._query = AsyncMock(return_value=[])

    await storage.get_popular_labels(limit=3)

    sql = _normalize(storage._query.await_args.args[0])
    assert 'ORDER BY degree DESC, label COLLATE "C" ASC' in sql
    assert "LIMIT $1" in sql
    assert storage._query.await_args.kwargs["params"] == {"limit": 3}


@pytest.mark.asyncio
async def test_get_popular_labels_returns_labels_in_query_order():
    """Rows are passed through in the order the database ranked them."""
    storage = make_graph_storage()
    storage._query = AsyncMock(
        return_value=[{"label": "Alpha"}, {"label": "Beta"}, {"label": "Isolated"}]
    )

    assert await storage.get_popular_labels() == ["Alpha", "Beta", "Isolated"]


# ---------------------------------------------------------------------------
# get_node_edges: absent node vs isolated node
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_node_edges_returns_none_for_missing_node():
    """No rows at all means the anchor MATCH failed: the node does not exist.

    Regression: PGGraphStorage returned ``[]`` here while NetworkXStorage and
    PGOpsGraphStorage return ``None``, so callers could not tell a missing
    entity from an isolated one.
    """
    storage = make_graph_storage()
    storage._query = AsyncMock(return_value=[])

    assert await storage.get_node_edges("NoSuchEntity") is None


@pytest.mark.asyncio
async def test_get_node_edges_returns_empty_list_for_isolated_node():
    """An existing node with no relations still yields one row (OPTIONAL MATCH).

    That row carries a NULL connected_id and must degrade to ``[]`` — not to
    ``None``, which would now mean "the node is gone".
    """
    storage = make_graph_storage()
    storage._query = AsyncMock(
        return_value=[{"source_id": "Lonely", "connected_id": None}]
    )

    assert await storage.get_node_edges("Lonely") == []


@pytest.mark.asyncio
async def test_get_node_edges_returns_connected_pairs():
    """Connected nodes come back as (source, connected) tuples."""
    storage = make_graph_storage()
    storage._query = AsyncMock(
        return_value=[
            {"source_id": "Alpha", "connected_id": "Beta"},
            {"source_id": "Alpha", "connected_id": "Gamma"},
        ]
    )

    assert await storage.get_node_edges("Alpha") == [
        ("Alpha", "Beta"),
        ("Alpha", "Gamma"),
    ]
