import networkx as nx
import pytest

from lightrag.kg.networkx_impl import NetworkXStorage
from lightrag.kg.shared_storage import initialize_share_data


@pytest.mark.asyncio
async def test_get_knowledge_graph_star_breaks_degree_ties_by_label(tmp_path):
    """Isolates at the max_nodes cutoff must prefer label order, not insert order."""
    initialize_share_data()
    storage = NetworkXStorage(
        namespace="kg_star_ties",
        workspace="ws",
        global_config={"working_dir": str(tmp_path), "max_graph_nodes": 1000},
        embedding_func=None,
    )
    await storage.initialize()

    g = nx.Graph()
    for name in ["Zeta", "Alpha", "Beta"]:
        g.add_node(name, entity_type="CONCEPT", description=name)
    storage._graph = g

    result = await storage.get_knowledge_graph("*", max_depth=1, max_nodes=2)
    ids = sorted(node.id for node in result.nodes)
    assert ids == ["Alpha", "Beta"]
    assert result.is_truncated is True


@pytest.mark.asyncio
async def test_get_knowledge_graph_bfs_breaks_degree_ties_by_label(tmp_path):
    """Same-depth neighbors with equal degree must prefer label order at cutoff."""
    initialize_share_data()
    storage = NetworkXStorage(
        namespace="kg_bfs_ties",
        workspace="ws",
        global_config={"working_dir": str(tmp_path), "max_graph_nodes": 1000},
        embedding_func=None,
    )
    await storage.initialize()

    g = nx.Graph()
    g.add_node("A", entity_type="CONCEPT", description="A")
    for name in ["Zeta", "Alpha", "Beta"]:
        g.add_node(name, entity_type="CONCEPT", description=name)
        g.add_edge("A", name)
    storage._graph = g

    result = await storage.get_knowledge_graph("A", max_depth=1, max_nodes=3)
    ids = sorted(node.id for node in result.nodes)
    assert ids == ["A", "Alpha", "Beta"]
    assert result.is_truncated is True
