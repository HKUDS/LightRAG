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


@pytest.mark.asyncio
async def test_iter_labels_tolerates_a_node_added_between_batches():
    """A node added between two batches must not raise ``RuntimeError:
    dictionary changed size during iteration`` when the generator resumes.

    No in-tree caller awaits between batches today, so nothing can mutate the
    graph while the generator is suspended; this pins the hazard for a future
    caller that does. Only what ``iter_labels`` promises is asserted: every
    node present at the start is yielded exactly once. Whether the late node
    also appears is left open, as keyset-paginating backends may include it.
    """
    graph = nx.Graph()
    graph.add_edges_from((f"N{index}", f"N{index + 1}") for index in range(5))
    storage = _make_storage(graph)
    original_labels = set(graph.nodes)

    gen = storage.iter_labels(2)
    batches = [await anext(gen)]
    graph.add_node("LATE_ARRIVAL")
    batches.extend([batch async for batch in gen])

    labels = [label for batch in batches for label in batch]
    assert len(labels) == len(set(labels))
    assert original_labels <= set(labels)


@pytest.mark.asyncio
async def test_iter_edges_tolerates_an_edge_added_between_batches():
    """Same hazard and same assertions as the ``iter_labels`` case above."""
    graph = nx.Graph()
    graph.add_edges_from((f"N{index}", f"N{index + 1}") for index in range(5))
    storage = _make_storage(graph)
    original_edges = {frozenset(edge) for edge in graph.edges}

    gen = storage.iter_edges(2)
    batches = [await anext(gen)]
    graph.add_edge("N0", "LATE_ARRIVAL")
    batches.extend([batch async for batch in gen])

    edges = [
        frozenset((edge["source"], edge["target"]))
        for batch in batches
        for edge in batch
    ]
    assert len(edges) == len(set(edges))
    assert original_edges <= set(edges)
