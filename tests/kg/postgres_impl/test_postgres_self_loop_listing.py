"""``PGGraphStorage.get_nodes_edges_batch`` must list a self-loop once.

``BaseGraphStorage.get_node_edges``: a self-loop appears ONCE and is never
hidden. This backend runs two DIRECTED Cypher queries -- ``(n)-[]->(connected)``
and ``(n)<-[]-(connected)`` -- and a self-loop satisfies both, so without a
guard one edge is reported as two. ``pgtable_impl``, ``mongo_impl`` and
``opensearch_impl`` all carry the same guard; this was the last holdout.

No in-tree caller is harmed today (``_purge_kg_contributions`` collects into a
set and ``_find_most_related_edges_from_entities`` dedupes on ``seen``), so
this pins the contract rather than a live bug -- which is exactly why it is
worth pinning: the next caller has no reason to expect duplicates.

Only ``_query`` is faked, and the two directed result sets are answered
separately, so the assertion runs against the real assembly loop.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from lightrag.kg.postgres_impl import PGGraphStorage

pytestmark = pytest.mark.offline


def _make_storage(outgoing, incoming):
    s = PGGraphStorage.__new__(PGGraphStorage)
    s.workspace = "test_ws"
    s.namespace = "test_graph"
    s.graph_name = "test_graph"
    s.__post_init__()
    s.db = MagicMock()

    async def _query(query, **kwargs):
        # The outbound pass is issued first; answer each with its own rows so
        # the two directions are not collapsed by the fake.
        return outgoing if "-[]->" in query else incoming

    s._query = AsyncMock(side_effect=_query)
    return s


@pytest.mark.asyncio
async def test_self_loop_is_listed_once():
    """One edge, one tuple: the loop matches both directed queries."""
    s = _make_storage(
        outgoing=[{"node_id": "loop", "connected_id": "loop"}],
        incoming=[{"node_id": "loop", "connected_id": "loop"}],
    )

    result = await s.get_nodes_edges_batch(["loop"])

    assert result["loop"] == [("loop", "loop")]


@pytest.mark.asyncio
async def test_ordinary_edge_still_reaches_both_endpoints():
    """The skip must be scoped to ``src == tgt``: an ordinary edge is still
    listed for its source AND its target, which is what makes the batch
    represent an undirected graph."""
    s = _make_storage(
        outgoing=[{"node_id": "A", "connected_id": "B"}],
        incoming=[{"node_id": "B", "connected_id": "A"}],
    )

    result = await s.get_nodes_edges_batch(["A", "B"])

    assert result["A"] == [("A", "B")]
    assert result["B"] == [("A", "B")]


@pytest.mark.asyncio
async def test_self_loop_alongside_ordinary_edges():
    """A node carrying both kinds keeps every ordinary edge and exactly one
    copy of the loop."""
    s = _make_storage(
        outgoing=[
            {"node_id": "A", "connected_id": "A"},
            {"node_id": "A", "connected_id": "B"},
        ],
        incoming=[
            {"node_id": "A", "connected_id": "A"},
            {"node_id": "A", "connected_id": "C"},
        ],
    )

    result = await s.get_nodes_edges_batch(["A"])

    assert result["A"].count(("A", "A")) == 1
    assert len(result["A"]) == 3
