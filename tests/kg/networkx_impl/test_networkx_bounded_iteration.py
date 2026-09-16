import networkx as nx
import pytest

from lightrag.kg.networkx_impl import NetworkXStorage


pytestmark = pytest.mark.offline


def _make_storage(graph: nx.Graph) -> NetworkXStorage:
    storage = NetworkXStorage.__new__(NetworkXStorage)

    async def _get_graph():
        return graph

    storage._get_graph = _get_graph
    return storage


@pytest.mark.asyncio
async def test_graph_iteration_never_exceeds_the_requested_batch_size():
    graph = nx.Graph()
    graph.add_edges_from((f"N{index}", f"N{index + 1}") for index in range(9))
    storage = _make_storage(graph)

    label_batches = [batch async for batch in storage.iter_labels(3)]
    edge_batches = [batch async for batch in storage.iter_edges(4)]

    assert max(map(len, label_batches)) == 3
    assert max(map(len, edge_batches)) == 4
    assert {label for batch in label_batches for label in batch} == set(graph.nodes)
    assert {
        frozenset((edge["source"], edge["target"]))
        for batch in edge_batches
        for edge in batch
    } == {frozenset(edge) for edge in graph.edges}


@pytest.mark.asyncio
async def test_graph_iteration_rejects_non_positive_batch_size():
    storage = _make_storage(nx.Graph())

    with pytest.raises(ValueError, match="positive"):
        await anext(storage.iter_labels(0))
    with pytest.raises(ValueError, match="positive"):
        await anext(storage.iter_edges(0))
