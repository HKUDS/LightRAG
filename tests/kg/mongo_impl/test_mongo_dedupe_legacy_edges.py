"""MongoGraphStorage._dedupe_legacy_edges folds duplicate reciprocal-edge
docs into one survivor at startup, for pre-existing deployments being
migrated onto the canonical (edge_lo, edge_hi) unique index. A duplicate can
contribute new source_ids while carrying a missing or non-numeric legacy
weight; the plain sum used for the survivor's weight then falls below the
merged evidence count, violating the relation weight contract documented in
AGENTS.md (weight >= len(distinct real source IDs)).
"""

import sys

import pytest

pytest.importorskip("pymongo", reason="pymongo is required for Mongo storage tests")

from types import SimpleNamespace
from unittest.mock import AsyncMock

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
    storage = MongoGraphStorage.__new__(MongoGraphStorage)
    storage.workspace = "test"
    storage.edge_collection = SimpleNamespace()
    return storage


@pytest.mark.asyncio
async def test_duplicate_with_missing_weight_does_not_undercut_evidence_floor():
    """Survivor carries weight=1.0 for source_ids=["chunk1"]. The duplicate
    contributes a brand-new source_id but has no weight field at all (a
    pre-contract legacy row). A plain sum would keep the result at 1.0 even
    though the merged evidence count is now 2."""
    storage = _make_storage()
    group = {
        "_id": {"lo": "A", "hi": "B"},
        "docs": [
            {
                "_id": "survivor",
                "source_ids": ["chunk1"],
                "weight": 1.0,
                "created_at": 2,
            },
            {
                "_id": "dup",
                "source_ids": ["chunk2"],
                "created_at": 1,
            },
        ],
        "count": 2,
    }
    storage.edge_collection.aggregate = AsyncMock(return_value=_AsyncCursor([group]))
    storage.edge_collection.update_one = AsyncMock()
    storage.edge_collection.delete_many = AsyncMock()

    removed = await storage._dedupe_legacy_edges()

    assert removed == 1
    set_fields = storage.edge_collection.update_one.await_args[0][1]["$set"]
    assert set_fields["source_ids"] == ["chunk1", "chunk2"]
    assert set_fields["weight"] >= 2


@pytest.mark.asyncio
async def test_all_docs_missing_weight_still_gets_the_evidence_floor():
    """If no doc in the group has a coercible weight at all, the migration
    must still set weight to the evidence count instead of leaving the
    field untouched."""
    storage = _make_storage()
    group = {
        "_id": {"lo": "A", "hi": "B"},
        "docs": [
            {"_id": "survivor", "source_ids": ["chunk1"], "created_at": 2},
            {"_id": "dup", "source_ids": ["chunk2"], "created_at": 1},
        ],
        "count": 2,
    }
    storage.edge_collection.aggregate = AsyncMock(return_value=_AsyncCursor([group]))
    storage.edge_collection.update_one = AsyncMock()
    storage.edge_collection.delete_many = AsyncMock()

    await storage._dedupe_legacy_edges()

    set_fields = storage.edge_collection.update_one.await_args[0][1]["$set"]
    assert set_fields["weight"] == 2


@pytest.mark.asyncio
async def test_weight_sum_overflow_is_clamped_to_the_largest_finite_weight():
    """Each weight is individually finite (1e308 passes _coerce_weight's
    isfinite check), but summing two of them overflows to +inf, which no graph
    backend can store. The migration must clamp to the largest representable
    weight -- keeping the absurd-but-real magnitude -- instead of writing +inf
    or collapsing all the way down to the evidence count."""
    storage = _make_storage()
    group = {
        "_id": {"lo": "A", "hi": "B"},
        "docs": [
            {
                "_id": "survivor",
                "source_ids": ["chunk1"],
                "weight": 1e308,
                "created_at": 2,
            },
            {
                "_id": "dup",
                "source_ids": ["chunk2"],
                "weight": 1e308,
                "created_at": 1,
            },
        ],
        "count": 2,
    }
    storage.edge_collection.aggregate = AsyncMock(return_value=_AsyncCursor([group]))
    storage.edge_collection.update_one = AsyncMock()
    storage.edge_collection.delete_many = AsyncMock()

    await storage._dedupe_legacy_edges()

    set_fields = storage.edge_collection.update_one.await_args[0][1]["$set"]
    assert set_fields["weight"] == sys.float_info.max


@pytest.mark.parametrize("bad_weight", [float("nan"), float("inf"), "NaN", "-Infinity"])
@pytest.mark.asyncio
async def test_non_finite_legacy_weight_is_skipped_not_raised(bad_weight):
    """A NaN/inf legacy weight parses fine through float() but no backend can
    store one, and a NaN would poison the sum/max it reaches. Migration must
    skip it like any other unusable legacy weight, and still floor the survivor
    to the merged evidence count."""
    storage = _make_storage()
    group = {
        "_id": {"lo": "A", "hi": "B"},
        "docs": [
            {
                "_id": "survivor",
                "source_ids": ["chunk1"],
                "weight": bad_weight,
                "created_at": 2,
            },
            {
                "_id": "dup",
                "source_ids": ["chunk2"],
                "created_at": 1,
            },
        ],
        "count": 2,
    }
    storage.edge_collection.aggregate = AsyncMock(return_value=_AsyncCursor([group]))
    storage.edge_collection.update_one = AsyncMock()
    storage.edge_collection.delete_many = AsyncMock()

    await storage._dedupe_legacy_edges()

    set_fields = storage.edge_collection.update_one.await_args[0][1]["$set"]
    assert set_fields["weight"] == 2


@pytest.mark.asyncio
async def test_unstorable_legacy_source_id_does_not_abort_the_migration():
    """A legacy source_id can hold a character no graph backend can store (an
    XML-incompatible control character, say). Routing the floor through
    `apply_relation_weight_floor` -- which validates the whole relation the way
    a caller ingress does -- would raise on that row and abort the one-time
    migration it is supposed to carry through. The evidence count is computed
    directly, so the merge completes and still honors the floor."""
    storage = _make_storage()
    group = {
        "_id": {"lo": "A", "hi": "B"},
        "docs": [
            {
                "_id": "survivor",
                "source_ids": ["chunk-\x0bbad"],
                "weight": 1.0,
                "created_at": 2,
            },
            {"_id": "dup", "source_ids": ["chunk2"], "created_at": 1},
        ],
        "count": 2,
    }
    storage.edge_collection.aggregate = AsyncMock(return_value=_AsyncCursor([group]))
    storage.edge_collection.update_one = AsyncMock()
    storage.edge_collection.delete_many = AsyncMock()

    removed = await storage._dedupe_legacy_edges()

    assert removed == 1
    set_fields = storage.edge_collection.update_one.await_args[0][1]["$set"]
    assert set_fields["weight"] >= 2


@pytest.mark.asyncio
async def test_placeholder_only_evidence_leaves_a_missing_weight_alone():
    """`manual_creation`/`UNKNOWN` are not evidence, so the floor is zero. A
    group whose docs carry no weight at all must keep "weight" out of the
    update: writing 0.0 would replace the 1.0 that `_merge_edges_then_upsert`
    reads for a missing weight, demoting the relation the migration touched."""
    storage = _make_storage()
    group = {
        "_id": {"lo": "A", "hi": "B"},
        "docs": [
            {"_id": "survivor", "source_ids": ["UNKNOWN"], "created_at": 2},
            {"_id": "dup", "source_ids": ["manual_creation"], "created_at": 1},
        ],
        "count": 2,
    }
    storage.edge_collection.aggregate = AsyncMock(return_value=_AsyncCursor([group]))
    storage.edge_collection.update_one = AsyncMock()
    storage.edge_collection.delete_many = AsyncMock()

    await storage._dedupe_legacy_edges()

    set_fields = storage.edge_collection.update_one.await_args[0][1]["$set"]
    assert "weight" not in set_fields
    assert set_fields["source_ids"] == ["UNKNOWN", "manual_creation"]


@pytest.mark.asyncio
async def test_placeholder_only_evidence_keeps_a_real_weight():
    """The guard above must skip only the write it has nothing to say about: a
    source-less relation may carry any non-negative fractional weight, and the
    survivor's own 1.0 has to survive the merge."""
    storage = _make_storage()
    group = {
        "_id": {"lo": "A", "hi": "B"},
        "docs": [
            {
                "_id": "survivor",
                "source_ids": ["UNKNOWN"],
                "weight": 1.0,
                "created_at": 2,
            },
            {"_id": "dup", "source_ids": ["manual_creation"], "created_at": 1},
        ],
        "count": 2,
    }
    storage.edge_collection.aggregate = AsyncMock(return_value=_AsyncCursor([group]))
    storage.edge_collection.update_one = AsyncMock()
    storage.edge_collection.delete_many = AsyncMock()

    await storage._dedupe_legacy_edges()

    set_fields = storage.edge_collection.update_one.await_args[0][1]["$set"]
    assert set_fields["weight"] == 1.0


@pytest.mark.asyncio
async def test_negative_weight_sum_overflow_takes_the_evidence_floor():
    """The overflow clamp must not pull a hugely NEGATIVE sum up to the largest
    positive weight. Summing two finite negatives underflows to -inf, which the
    evidence floor -- not the clamp -- is what absorbs."""
    storage = _make_storage()
    group = {
        "_id": {"lo": "A", "hi": "B"},
        "docs": [
            {
                "_id": "survivor",
                "source_ids": ["chunk1"],
                "weight": -1e308,
                "created_at": 2,
            },
            {
                "_id": "dup",
                "source_ids": ["chunk2"],
                "weight": -1e308,
                "created_at": 1,
            },
        ],
        "count": 2,
    }
    storage.edge_collection.aggregate = AsyncMock(return_value=_AsyncCursor([group]))
    storage.edge_collection.update_one = AsyncMock()
    storage.edge_collection.delete_many = AsyncMock()

    await storage._dedupe_legacy_edges()

    set_fields = storage.edge_collection.update_one.await_args[0][1]["$set"]
    assert set_fields["weight"] == 2
