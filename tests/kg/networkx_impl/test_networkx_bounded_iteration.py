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


class _CountingGraph:
    """Wraps a graph so a test can see how much of each view was read."""

    def __init__(self, graph: nx.Graph):
        self._graph = graph
        self.reads = 0

    def _counted(self, view):
        for item in view:
            self.reads += 1
            yield item

    def nodes(self):
        return self._counted(self._graph.nodes())

    def edges(self, data=False):
        return self._counted(self._graph.edges(data=data))


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["iter_labels", "iter_edges"])
async def test_first_batch_reads_only_one_batch_of_the_graph(method):
    """``iter_*`` are the BOUNDED readers: the first-batch probes in
    ``vector_space_gate`` (``batch_size=1``) must not pay for a walk over, or
    a copy of, the whole graph.
    """
    graph = nx.Graph()
    graph.add_edges_from((f"N{index}", f"N{index + 1}") for index in range(50))
    counting = _CountingGraph(graph)
    storage = _make_storage(counting)

    gen = getattr(storage, method)(1)
    assert len(await anext(gen)) == 1
    await gen.aclose()

    assert counting.reads == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "mutate"),
    [
        pytest.param(
            "iter_labels", lambda graph: graph.add_node("LATE_ARRIVAL"), id="node"
        ),
        pytest.param(
            "iter_edges",
            lambda graph: graph.add_edge("N0", "LATE_ARRIVAL"),
            id="edge",
        ),
    ],
)
async def test_mutation_between_batches_raises_an_error_naming_the_cause(
    method, mutate
):
    """The graph is read live (see the bounded test above), so a writer that
    runs while the generator is suspended cannot be tolerated. It must fail
    loud with a message that names the iterator and the cause, chained to
    networkx's own "dictionary changed size during iteration".
    """
    graph = nx.Graph()
    graph.add_edges_from((f"N{index}", f"N{index + 1}") for index in range(5))
    storage = _make_storage(graph)

    gen = getattr(storage, method)(2)
    assert len(await anext(gen)) == 2
    mutate(graph)

    with pytest.raises(RuntimeError, match=f"{method}: the graph was modified") as info:
        await anext(gen)
    assert "changed size during iteration" in str(info.value.__cause__)
