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
    """A mutation landing between two yielded batches must not blow up the
    generator with ``RuntimeError: dictionary changed size during iteration``.

    ``iter_labels`` suspends mid-loop at every ``yield``, so a caller that
    awaits something between batches (the normal shape for a maintenance
    tool walking the whole graph) can let another coroutine in the same
    process add or remove a node in between -- the single-writer invariant
    bars a second concurrent WRITER, not a reader interleaved with one.
    Iterating a live ``NodeView`` across that suspension point raises as
    soon as the loop resumes, instead of finishing the batch it already
    started from a stable snapshot.
    """
    graph = nx.Graph()
    graph.add_edges_from((f"N{index}", f"N{index + 1}") for index in range(5))
    storage = _make_storage(graph)
    original_labels = set(graph.nodes)

    gen = storage.iter_labels(2)
    first_batch = await anext(gen)
    assert len(first_batch) == 2

    # Simulate a concurrent mutation landing while the consumer is between
    # batches (e.g. it awaited something of its own before asking for more).
    graph.add_node("LATE_ARRIVAL")

    remaining_batches = [first_batch]
    async for batch in gen:
        remaining_batches.append(batch)

    all_labels = {label for batch in remaining_batches for label in batch}
    # The snapshot taken when iteration started must win: the node added
    # mid-iteration is neither silently dropped from a torn snapshot nor
    # inconsistently spliced in.
    assert all_labels == original_labels


@pytest.mark.asyncio
async def test_iter_edges_tolerates_an_edge_added_between_batches():
    """Same hazard as ``iter_labels``, for the edge iterator."""
    graph = nx.Graph()
    graph.add_edges_from((f"N{index}", f"N{index + 1}") for index in range(5))
    storage = _make_storage(graph)
    original_edge_count = graph.number_of_edges()

    gen = storage.iter_edges(2)
    first_batch = await anext(gen)
    assert len(first_batch) == 2

    graph.add_edge("N0", "LATE_ARRIVAL")

    remaining_batches = [first_batch]
    async for batch in gen:
        remaining_batches.append(batch)

    assert sum(len(batch) for batch in remaining_batches) == original_edge_count
