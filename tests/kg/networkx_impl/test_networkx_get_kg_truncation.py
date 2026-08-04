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


@pytest.mark.asyncio
async def test_get_knowledge_graph_is_not_truncated_when_all_nodes_returned_with_duplicates(
    tmp_path,
):
    """
    Locks out the bug where duplicate queue entries from multi-parent traversal
    (e.g., diamond graph) caused is_truncated to falsely report True even though
    all reachable nodes were included in the returned graph.
    """
    initialize_share_data()
    storage = NetworkXStorage(
        namespace="test_graph_diamond",
        workspace="ws",
        global_config={"working_dir": str(tmp_path)},
        embedding_func=None,
    )
    await storage.initialize()

    g = nx.Graph()
    # Diamond graph: A -> B, A -> C, B -> D, C -> D
    g.add_edge("A", "B")
    g.add_edge("A", "C")
    g.add_edge("B", "D")
    g.add_edge("C", "D")
    storage._graph = g

    # All 4 nodes (A, B, C, D) are returned, so graph must not be reported as truncated.
    result = await storage.get_knowledge_graph("A", max_depth=2, max_nodes=4)
    assert len(result.nodes) == 4
    assert result.is_truncated is False
