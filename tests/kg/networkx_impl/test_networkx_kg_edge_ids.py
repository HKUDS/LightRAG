import networkx as nx
import pytest

from lightrag.kg.networkx_impl import NetworkXStorage
from lightrag.kg.shared_storage import initialize_share_data


pytestmark = pytest.mark.offline


@pytest.mark.asyncio
async def test_get_knowledge_graph_keeps_edges_whose_names_join_alike(tmp_path):
    """("A", "B-C") and ("A-B", "C") are two edges, not one "A-B-C" edge."""
    initialize_share_data()
    storage = NetworkXStorage(
        namespace="kg_edge_ids",
        workspace="ws",
        global_config={"working_dir": str(tmp_path), "max_graph_nodes": 1000},
        embedding_func=None,
    )
    await storage.initialize()

    g = nx.Graph()
    for name in ["A", "B-C", "A-B", "C", "D"]:
        g.add_node(name, entity_type="CONCEPT", description=name)
    g.add_edge("A", "B-C", keywords="first")
    g.add_edge("A-B", "C", keywords="second")
    g.add_edge("A", "D", keywords="plain")
    storage._graph = g

    result = await storage.get_knowledge_graph("*")

    edges = {(e.source, e.target): e for e in result.edges}
    assert sorted(edges) == [("A", "B-C"), ("A", "D"), ("A-B", "C")]
    assert edges[("A", "B-C")].properties["keywords"] == "first"
    assert edges[("A-B", "C")].properties["keywords"] == "second"
    assert len({e.id for e in result.edges}) == 3
    # A name pair without "-" keeps the plain "source-target" id.
    assert edges[("A", "D")].id == "A-D"
