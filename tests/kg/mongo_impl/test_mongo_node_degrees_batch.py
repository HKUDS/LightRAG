"""``MongoGraphStorage.node_degrees_batch`` must answer every requested id.

``BaseGraphStorage.node_degrees_batch``: a node with no edges gets ``0``, not
a missing key. This backend builds its result from two grouped aggregations,
and a node with no edges produces no row in either -- so without seeding the
dict first, the id simply disappears from the result and a caller reading
`rank` sees a KeyError rather than the zero ``node_degree`` reports.

``edge_degrees_batch`` is deliberately NOT covered here: Mongo's override
belongs to #3908, which adds it together with ``get_edges_batch`` and the
chunked aggregation. This file pins only the contract that override will rely
on -- that summing two ids out of the batch result is well-defined even when
one of them is isolated.
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


def _make_storage(*pages):
    """Each call to ``aggregate`` returns the next scripted page of buckets."""
    s = MongoGraphStorage.__new__(MongoGraphStorage)
    s.workspace = "test"
    s.namespace = "chunk_entity_relation"
    s._edge_collection_name = "test_edges"
    s.edge_collection = Mock()
    s.edge_collection.aggregate = AsyncMock(
        side_effect=[_AsyncCursor(page) for page in pages]
    )
    return s


@pytest.mark.asyncio
async def test_isolated_node_gets_zero_not_a_missing_key():
    s = _make_storage([], [])

    assert await s.node_degrees_batch(["A", "B"]) == {"A": 0, "B": 0}


@pytest.mark.asyncio
async def test_zero_seed_does_not_mask_real_degrees():
    """The seed must be overwritten, not added to: outbound 2 + inbound 1 is 3."""
    s = _make_storage([{"_id": "A", "degree": 2}], [{"_id": "A", "degree": 1}])

    assert await s.node_degrees_batch(["A", "B"]) == {"A": 3, "B": 0}
