"""Self-loop read contract for ``NetworkXStorage``.

NetworkX is the default backend, and the one place where LightRAG's degree
contract deliberately does NOT inherit the library's own semantics:

* ``graph.degree()`` counts a self-loop TWICE -- it measures how many edge
  endpoints a node occupies;
* ``BaseGraphStorage.node_degree`` EXCLUDES it -- degree measures connectivity,
  and a self-loop connects nothing, which is the same reason
  ``_reject_self_loop_relation`` refuses to create one.

So every degree path here subtracts what ``graph.degree()`` added. The opposite
rule governs edge listings: ``get_node_edges`` still reports the self-loop,
because entity deletion, rename, merge and document purge all enumerate a
node's incident edges through it to delete or rewrite the matching relation
rows -- an edge hidden there becomes an orphan relation row.

Only ``_get_graph`` is faked, so the assertions run against the real
``networkx`` object -- same convention as ``test_networkx_popular_labels.py``.
"""

import networkx as nx
import pytest

from lightrag.kg.networkx_impl import NetworkXStorage


pytestmark = pytest.mark.offline


def _make_storage(graph: nx.Graph) -> NetworkXStorage:
    storage = NetworkXStorage.__new__(NetworkXStorage)
    storage.workspace = "test"
    storage.namespace = "chunk_entity_relation"
    storage.global_config = {"max_graph_nodes": 1000}

    async def _get_graph():
        return graph

    storage._get_graph = _get_graph
    return storage


def _self_loop_graph() -> nx.Graph:
    graph = nx.Graph()
    graph.add_edge("Loop", "Loop")
    return graph


def _mixed_graph() -> nx.Graph:
    """``A`` carries two ordinary edges AND a self-loop; ``B``/``C`` are plain."""
    graph = nx.Graph()
    graph.add_edge("A", "B")
    graph.add_edge("C", "A")
    graph.add_edge("A", "A")
    return graph


class TestNodeDegreeExcludesSelfLoops:
    @pytest.mark.asyncio
    async def test_self_loop_only_node_has_degree_zero(self):
        """The library would answer 2 here; the contract answers 0."""
        graph = _self_loop_graph()
        assert graph.degree("Loop") == 2, "networkx semantics assumed by this test"

        storage = _make_storage(graph)

        assert await storage.node_degree("Loop") == 0

    @pytest.mark.asyncio
    async def test_self_loop_does_not_inflate_a_connected_node(self):
        """``A`` has two real neighbours, so its degree is 2 -- not the 4 that
        ``graph.degree()`` reports once the loop's two endpoints are added."""
        graph = _mixed_graph()
        assert graph.degree("A") == 4, "networkx semantics assumed by this test"

        storage = _make_storage(graph)

        assert await storage.node_degree("A") == 2

    @pytest.mark.asyncio
    async def test_ordinary_degrees_are_untouched(self):
        storage = _make_storage(_mixed_graph())

        assert await storage.node_degree("B") == 1
        assert await storage.node_degree("C") == 1

    @pytest.mark.asyncio
    async def test_absent_node_has_degree_zero(self):
        storage = _make_storage(_mixed_graph())

        assert await storage.node_degree("Ghost") == 0

    @pytest.mark.asyncio
    async def test_node_degrees_batch_agrees_with_the_scalar(self):
        """A backend whose scalar and batch paths disagree ranks the same node
        differently depending on which one a caller reaches for."""
        storage = _make_storage(_mixed_graph())

        batch = await storage.node_degrees_batch(["A", "B", "C"])

        assert batch == {"A": 2, "B": 1, "C": 1}
        assert batch["A"] == await storage.node_degree("A")

    @pytest.mark.asyncio
    async def test_edge_degree_sums_both_endpoints(self):
        """``edge_degree`` inherits the rule: (Loop, Loop) is 0 + 0."""
        storage = _make_storage(_self_loop_graph())

        assert await storage.edge_degree("Loop", "Loop") == 0

    @pytest.mark.asyncio
    async def test_edge_degree_on_an_ordinary_edge(self):
        storage = _make_storage(_mixed_graph())

        assert await storage.edge_degree("A", "B") == 3


class TestRankingPathsExcludeSelfLoops:
    @pytest.mark.asyncio
    async def test_get_popular_labels_does_not_rank_a_loop_above_real_edges(self):
        """``Loop`` would outrank ``B`` at degree 2 vs 1 under library
        semantics. It is degree 0, so it sorts last."""
        graph = nx.Graph()
        graph.add_edge("A", "B")
        graph.add_edge("Loop", "Loop")
        storage = _make_storage(graph)

        labels = await storage.get_popular_labels(limit=3)

        assert labels == ["A", "B", "Loop"]

    @pytest.mark.asyncio
    async def test_knowledge_graph_star_ranking_excludes_self_loops(self):
        """The ``*`` path is degree-ranked, so the same rule decides which
        nodes survive ``max_nodes`` truncation."""
        graph = nx.Graph()
        graph.add_edge("Hub", "X")
        graph.add_edge("Hub", "Y")
        graph.add_edge("Loop", "Loop")
        storage = _make_storage(graph)

        kg = await storage.get_knowledge_graph("*", max_depth=1, max_nodes=2)

        assert [n.id for n in kg.nodes] == ["Hub", "X"]


class TestEdgeListingStillShowsSelfLoops:
    @pytest.mark.asyncio
    async def test_get_node_edges_reports_the_self_loop(self):
        """Excluded from DEGREE, never hidden from the LISTING: deletion,
        rename, merge and purge all resolve relation rows through this."""
        storage = _make_storage(_self_loop_graph())

        assert await storage.get_node_edges("Loop") == [("Loop", "Loop")]

    @pytest.mark.asyncio
    async def test_listing_keeps_the_loop_alongside_ordinary_edges(self):
        storage = _make_storage(_mixed_graph())

        edges = await storage.get_node_edges("A")

        assert edges.count(("A", "A")) == 1
        assert len(edges) == 3
