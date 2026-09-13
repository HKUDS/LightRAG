"""``OpenSearchGraphStorage.edge_degrees_batch`` must resolve ids once.

The inherited ``BaseGraphStorage`` default calls ``edge_degree`` for every
pair, and each of those is two awaited ``node_degree`` calls -- so the
retrieval path (``_find_most_related_edges_from_entities`` hands it the whole
incident-edge set of the top entities, not ``top_k`` of them) issued four
SERIAL count requests per edge, each preceded by a refresh check.

The override collapses that to a single ``node_degrees_batch`` aggregation over
the DISTINCT endpoint ids. That an aggregation search is more expensive than a
count -- the trade ``test_node_degree_uses_count_api`` pins for the scalar --
does not decide this: it runs once instead of thousands of times.
"""

import pytest
from unittest.mock import AsyncMock

pytest.importorskip(
    "opensearchpy", reason="opensearchpy is required for OpenSearch storage tests"
)

from lightrag.kg.opensearch_impl import OpenSearchGraphStorage

pytestmark = pytest.mark.offline


def _make_storage():
    s = OpenSearchGraphStorage.__new__(OpenSearchGraphStorage)
    s.workspace = "test"
    s.global_config = {"max_graph_nodes": 1000}
    s._nodes_index = "test-nodes"
    s._edges_index = "test-edges"
    s._indices_ready = True
    s._refresh_graph_indices_if_dirty = AsyncMock(return_value=None)
    s.client = AsyncMock()
    return s


PAIRS = [("A", "B"), ("B", "C"), ("A", "C")]


@pytest.mark.asyncio
async def test_resolves_distinct_ids_in_one_batch_call():
    s = _make_storage()
    s.node_degrees_batch = AsyncMock(return_value={"A": 2, "B": 2, "C": 2})
    s.node_degree = AsyncMock(side_effect=AssertionError("scalar must not be used"))

    result = await s.edge_degrees_batch(PAIRS)

    assert result == {("A", "B"): 4, ("B", "C"): 4, ("A", "C"): 4}
    s.node_degrees_batch.assert_awaited_once()
    assert sorted(s.node_degrees_batch.await_args.args[0]) == ["A", "B", "C"]
    s.node_degree.assert_not_awaited()


@pytest.mark.asyncio
async def test_empty_input_issues_no_query():
    s = _make_storage()
    s.node_degrees_batch = AsyncMock(side_effect=AssertionError("must not query"))

    assert await s.edge_degrees_batch([]) == {}


@pytest.mark.asyncio
async def test_missing_endpoint_counts_as_zero():
    s = _make_storage()
    s.node_degrees_batch = AsyncMock(return_value={"A": 3})

    assert await s.edge_degrees_batch([("A", "Ghost")]) == {("A", "Ghost"): 3}


@pytest.mark.asyncio
async def test_node_degrees_batch_answers_every_requested_id():
    """Zero, not a missing key -- see BaseGraphStorage.node_degree."""
    s = _make_storage()
    s.client.search = AsyncMock(
        return_value={
            "aggregations": {
                "source_degrees": {"ids": {"buckets": []}},
                "target_degrees": {"ids": {"buckets": []}},
            }
        }
    )

    assert await s.node_degrees_batch(["A", "B"]) == {"A": 0, "B": 0}
