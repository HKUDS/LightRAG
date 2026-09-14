"""MongoGraphStorage._dedupe_legacy_edges folds duplicate reciprocal-edge
docs into one survivor at startup, for pre-existing deployments being
migrated onto the canonical (edge_lo, edge_hi) unique index. A duplicate can
contribute new source_ids while carrying a missing or non-numeric legacy
weight; the plain sum used for the survivor's weight then falls below the
merged evidence count, violating the relation weight contract documented in
AGENTS.md (weight >= len(distinct real source IDs)).
"""

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
async def test_weight_sum_overflow_falls_back_to_the_evidence_floor():
    """Each weight is individually finite (1e308 passes _coerce_weight's
    isfinite check), but summing two of them overflows to +inf.
    apply_relation_weight_floor rejects a non-finite aggregate outright --
    the migration must fall back to the evidence-count floor instead of
    raising and aborting startup."""
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
    assert set_fields["weight"] == 2


@pytest.mark.parametrize("bad_weight", [float("nan"), float("inf"), "NaN", "-Infinity"])
@pytest.mark.asyncio
async def test_non_finite_legacy_weight_is_skipped_not_raised(bad_weight):
    """A NaN/inf legacy weight parses fine through float() but is rejected by
    apply_relation_weight_floor's storability check. Migration must skip it
    like any other unusable legacy weight, not raise and abort startup."""
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
