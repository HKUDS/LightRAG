"""``MongoGraphStorage.edge_degrees_batch`` must resolve ids once, not per pair.

The inherited ``BaseGraphStorage`` default calls ``edge_degree`` for every
pair, and each of those is two awaited ``node_degree`` calls -- so the
retrieval path (``_find_most_related_edges_from_entities`` hands it the whole
incident-edge set of the top entities, not ``top_k`` of them) issued four
SERIAL round trips per edge. This collection carries no index on the endpoint
fields, so each one is a collection scan.

The override collapses that to a single ``node_degrees_batch`` aggregation over
the DISTINCT endpoint ids, which is both fewer round trips and fewer rows
scanned. Same shape as ``pgtable_impl.edge_degrees_batch``.
"""

import pytest
from unittest.mock import AsyncMock, Mock

pytest.importorskip("pymongo", reason="pymongo is required for Mongo storage tests")

from lightrag.kg.mongo_impl import MongoGraphStorage

pytestmark = pytest.mark.offline


class _AsyncCursor:
    def __init__(self, docs):
        self._docs = list(docs)

    def __aiter__(self):
        self._iter = iter(self._docs)
        return self

    async def __anext__(self):
        try:
            return next(self._iter)
        except StopIteration:
            raise StopAsyncIteration


def _make_storage():
    s = MongoGraphStorage.__new__(MongoGraphStorage)
    s.workspace = "test"
    s.namespace = "chunk_entity_relation"
    s._edge_collection_name = "test_edges"
    s.edge_collection = Mock()
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
    # Three pairs, three distinct ids -- the per-pair loop would have issued six
    # node_degree calls, i.e. twelve counts on this backend.
    assert sorted(s.node_degrees_batch.await_args.args[0]) == ["A", "B", "C"]
    s.node_degree.assert_not_awaited()


@pytest.mark.asyncio
async def test_empty_input_issues_no_query():
    s = _make_storage()
    s.node_degrees_batch = AsyncMock(side_effect=AssertionError("must not query"))

    assert await s.edge_degrees_batch([]) == {}


@pytest.mark.asyncio
async def test_missing_endpoint_counts_as_zero():
    """A pair naming an id the batch did not answer for still gets a degree,
    so a caller reading `rank` never sees a KeyError."""
    s = _make_storage()
    s.node_degrees_batch = AsyncMock(return_value={"A": 3})

    assert await s.edge_degrees_batch([("A", "Ghost")]) == {("A", "Ghost"): 3}


@pytest.mark.asyncio
async def test_node_degrees_batch_answers_every_requested_id():
    """Zero, not a missing key: `BaseGraphStorage.node_degree` says the batch
    reports 0 for a node with no edges, which is what makes the sum above
    well-defined without a per-caller default."""
    s = _make_storage()
    s.edge_collection.aggregate = AsyncMock(return_value=_AsyncCursor([]))

    assert await s.node_degrees_batch(["A", "B"]) == {"A": 0, "B": 0}
