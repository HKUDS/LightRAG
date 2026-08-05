"""NetworkXStorage.get_popular_labels ranking contract.

NetworkX is the default backend and the reference implementation the
BaseGraphStorage contract points at, so the two things the other backends were
fixed to do must hold here first:

* every node is ranked, including isolated (degree-0) entities;
* ties break on the label ascending, NOT on node insertion order.
"""

import networkx as nx
import pytest

from lightrag.kg.networkx_impl import NetworkXStorage


pytestmark = pytest.mark.offline


def _make_storage(graph: nx.Graph) -> NetworkXStorage:
    storage = NetworkXStorage.__new__(NetworkXStorage)
    storage.workspace = "test"
    storage.namespace = "chunk_entity_relation"

    async def _get_graph():
        return graph

    storage._get_graph = _get_graph
    return storage


@pytest.mark.asyncio
async def test_ties_break_on_label_not_insertion_order():
    """Regression: ties used to come back in node insertion order.

    ``sorted(..., key=degree, reverse=True)`` is stable, so equal degrees kept
    the order the nodes were added in. With more tied labels than ``limit``,
    that decided which ones the caller never sees: inserting "Zeta" before
    "Alpha" made /graph/label/popular?limit=1 answer Zeta and drop Alpha.
    """
    graph = nx.Graph()
    # Insertion order is deliberately the reverse of alphabetical order, and
    # every node has the same degree.
    for name in ["Zeta", "Mu", "Alpha"]:
        graph.add_node(name)
    graph.add_edge("Hub", "Zeta")
    graph.add_edge("Hub", "Mu")
    graph.add_edge("Hub", "Alpha")

    storage = _make_storage(graph)

    # Hub has degree 3; the three leaves all tie at 1 and must come back
    # alphabetically, not Zeta-first.
    assert await storage.get_popular_labels() == ["Hub", "Alpha", "Mu", "Zeta"]

    # And the tie-break decides truncation, which is the part that actually
    # loses data for the caller.
    assert await storage.get_popular_labels(limit=2) == ["Hub", "Alpha"]


@pytest.mark.asyncio
async def test_isolated_nodes_are_ranked_last_but_still_ranked():
    graph = nx.Graph()
    graph.add_edge("Connected", "Peer")
    graph.add_node("Orphan")
    graph.add_node("Aardvark")  # inserted last, must still sort before Orphan

    storage = _make_storage(graph)

    labels = await storage.get_popular_labels()
    assert set(labels) == {"Connected", "Peer", "Orphan", "Aardvark"}
    # Both degree-0 entities are present, in label order.
    assert labels[-2:] == ["Aardvark", "Orphan"]


@pytest.mark.asyncio
async def test_limit_truncates_after_ranking():
    graph = nx.Graph()
    graph.add_edge("A", "B")
    graph.add_edge("A", "C")
    graph.add_node("Z")

    storage = _make_storage(graph)

    assert await storage.get_popular_labels(limit=1) == ["A"]
    assert len(await storage.get_popular_labels(limit=99)) == 4
