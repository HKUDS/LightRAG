import pytest
import networkx as nx
from lightrag.kg.shared_storage import initialize_share_data
from lightrag.kg.networkx_impl import NetworkXStorage


@pytest.mark.asyncio
async def test_get_knowledge_graph_is_truncated_when_max_nodes_reached(tmp_path):
    initialize_share_data()
    storage = NetworkXStorage(
        namespace="test_graph",
        workspace="ws",
        global_config={"working_dir": str(tmp_path)},
        embedding_func=None,
    )
    await storage.initialize()

    g = nx.Graph()
    for target in ["B", "C", "D", "E", "F"]:
        g.add_edge("A", target)
    storage._graph = g

    result = await storage.get_knowledge_graph("A", max_depth=2, max_nodes=3)
    assert len(result.nodes) == 3
    assert result.is_truncated is True
