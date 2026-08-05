"""Unit tests for PGGraphStorage read-path contracts shared with the other
graph backends.

Covers two divergences from ``BaseGraphStorage`` / ``NetworkXStorage``:

1. ``get_popular_labels`` ranked nodes purely from edge-derived degrees, so an
   entity with no relations (degree 0) had no row to join against and vanished
   from the ranking entirely — a graph of unconnected entities reported an
   empty picker. It now tops the result up from the isolated entities, but only
   when the connected ones come up short of ``limit``.
2. ``get_node_edges`` returned ``[]`` for a node that does not exist, which is
   the same value an existing-but-isolated node yields — collapsing "no such
   node" into "no edges". The contract (and NetworkX/PGOps) is ``None``.
"""

import re
import pytest
from unittest.mock import AsyncMock, MagicMock

from lightrag.kg.postgres_impl import PGGraphQueryException, PGGraphStorage


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
# get_popular_labels: connected first, isolated (degree-0) nodes to top up
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_popular_labels_ranks_from_edges_without_scanning_the_vertices():
    """Phase 1 is the cheap edge-derived ranking, and on a graph with enough
    connected entities it is the ONLY query that runs.

    Any real graph holds far more entities than `limit`, so driving the whole
    ranking off the vertex table (one LEFT JOIN) would mean a full vertex scan
    on every call to produce a result phase 1 already had.
    """
    storage = make_graph_storage()
    storage._query = AsyncMock(return_value=[{"label": f"E{i}"} for i in range(7)])

    labels = await storage.get_popular_labels(limit=7)

    assert labels == [f"E{i}" for i in range(7)]
    assert storage._query.await_count == 1, "the limit was filled; no second query"

    sql = _normalize(storage._query.await_args.args[0])
    # Degrees drive the join, and the vertex table is only probed for the ids
    # that actually have edges.
    assert "FROM node_degrees d JOIN test_graph._ag_label_vertex v" in sql, sql
    assert "LIMIT $1" in sql
    assert storage._query.await_args.kwargs["params"] == {"limit": 7}


@pytest.mark.asyncio
async def test_popular_labels_tops_up_from_isolated_when_short():
    """Regression: a graph whose entities carry no relations reported ZERO
    popular labels, because an entity with no edge has no row in node_degrees
    and the inner join dropped it.

    Phase 1 coming up short is exactly that signal — its aggregate is exact, so
    every remaining vertex is isolated — and the top-up is bounded by the
    shortfall, so even a sequential vertex scan stays cheap.
    """
    storage = make_graph_storage()
    storage._query = AsyncMock(
        side_effect=[
            [{"label": "Connected"}],
            [{"label": "Aardvark"}, {"label": "Orphan"}],
        ]
    )

    labels = await storage.get_popular_labels(limit=3)

    # Connected first, then isolated in label order.
    assert labels == ["Connected", "Aardvark", "Orphan"]
    assert storage._query.await_count == 2

    isolated_sql = _normalize(storage._query.await_args_list[1].args[0])
    # Isolated == no edge names it, at either end.
    assert (
        "NOT EXISTS ( SELECT 1 FROM test_graph._ag_label_edge e WHERE e.start_id = v.id )"
        in isolated_sql
    ), isolated_sql
    assert (
        "NOT EXISTS ( SELECT 1 FROM test_graph._ag_label_edge e WHERE e.end_id = v.id )"
        in isolated_sql
    ), isolated_sql
    # Only the shortfall is requested.
    assert storage._query.await_args_list[1].kwargs["params"] == {"limit": 2}


@pytest.mark.asyncio
async def test_popular_labels_orders_by_degree_then_bytewise_label():
    """Ties break on a byte-order label sort, matching Python's code-point sort.

    With degree-0 entities able to fill the tail, ties at the cutoff decide
    which labels survive the LIMIT — so the tie-break has to agree with the
    other backends instead of following the database's locale collation.
    """
    storage = make_graph_storage()
    storage._query = AsyncMock(return_value=[{"label": f"E{i}"} for i in range(3)])

    await storage.get_popular_labels(limit=3)

    sql = _normalize(storage._query.await_args.args[0])
    assert 'ORDER BY degree DESC, label COLLATE "C" ASC' in sql
    # The isolated top-up orders on the label alone (every row is degree 0).
    storage._query = AsyncMock(side_effect=[[], []])
    await storage.get_popular_labels(limit=3)
    isolated_sql = _normalize(storage._query.await_args_list[1].args[0])
    assert 'ORDER BY label COLLATE "C" ASC' in isolated_sql


@pytest.mark.asyncio
async def test_get_popular_labels_returns_labels_in_query_order():
    """Rows are passed through in the order the database ranked them."""
    storage = make_graph_storage()
    storage._query = AsyncMock(
        return_value=[{"label": "Alpha"}, {"label": "Beta"}, {"label": "Isolated"}]
    )

    assert await storage.get_popular_labels(limit=3) == ["Alpha", "Beta", "Isolated"]


# ---------------------------------------------------------------------------
# Label helpers: a database error is not "no labels"
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_popular_labels_raises_on_query_error():
    """A failed query must not be reported as an empty graph.

    Regression: the handler logged and returned []. The
    ``/graph/label/popular`` route already converts an exception into a 500, so
    swallowing it here handed the WebUI a 200 with an empty entity picker while
    the database was unreachable — indistinguishable from a graph that really
    holds no entities.
    """
    storage = make_graph_storage()
    boom = PGGraphQueryException({"message": "connection reset"})
    storage._query = AsyncMock(side_effect=boom)

    with pytest.raises(PGGraphQueryException):
        await storage.get_popular_labels(limit=10)


@pytest.mark.asyncio
async def test_search_labels_raises_on_query_error():
    """Same reasoning: "no match" and "the query failed" are different answers."""
    storage = make_graph_storage()
    storage._query = AsyncMock(side_effect=PGGraphQueryException({"message": "boom"}))

    with pytest.raises(PGGraphQueryException):
        await storage.search_labels("alpha")


@pytest.mark.asyncio
async def test_search_labels_still_short_circuits_on_blank_query():
    """An empty query is a real "nothing to match", not an error."""
    storage = make_graph_storage()
    storage._query = AsyncMock(side_effect=AssertionError("must not be reached"))

    assert await storage.search_labels("   ") == []


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
