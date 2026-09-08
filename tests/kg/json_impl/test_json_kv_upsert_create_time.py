"""Regression tests: JsonKVStorage replacement upserts preserve create_time.

Fixes https://github.com/HKUDS/LightRAG/issues/3870 — callers that replace a
KV value with business fields only must not discard the storage-managed
create_time. update_time advances; legacy rows missing create_time keep the
0/unknown convention rather than inventing an original timestamp.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from lightrag.kg.json_kv_impl import JsonKVStorage
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.namespace import NameSpace

pytestmark = pytest.mark.offline


class _DummyEmbeddingFunc:
    embedding_dim = 1
    max_token_size = 1

    async def __call__(self, texts, **kwargs):
        return [[0.0] for _ in texts]


@pytest.fixture(autouse=True)
def setup_shared_data():
    initialize_share_data()
    yield
    finalize_share_data()


def _make_storage(tmp_path, workspace="ct-ws"):
    return JsonKVStorage(
        namespace=NameSpace.KV_STORE_ENTITY_CHUNKS,
        global_config={"working_dir": str(tmp_path)},
        embedding_func=_DummyEmbeddingFunc(),
        workspace=workspace,
    )


@pytest.mark.asyncio
async def test_replacement_upsert_preserves_create_time(tmp_path):
    storage = _make_storage(tmp_path)
    await storage.initialize()

    with patch("time.time", return_value=1_700_000_000):
        await storage.upsert({"E": {"chunk_ids": ["c1"], "count": 1}})

    created = await storage.get_by_id("E")
    assert created is not None
    assert created["create_time"] == 1_700_000_000
    assert created["update_time"] == 1_700_000_000
    assert created["chunk_ids"] == ["c1"]
    assert created["count"] == 1

    with patch("time.time", return_value=1_700_000_100):
        await storage.upsert({"E": {"chunk_ids": ["c1", "c2"], "count": 2}})

    updated = await storage.get_by_id("E")
    assert updated is not None
    assert updated["create_time"] == 1_700_000_000
    assert updated["update_time"] == 1_700_000_100
    assert updated["chunk_ids"] == ["c1", "c2"]
    assert updated["count"] == 2
    # Business fields are replaced, not merged with the previous value.
    assert "llm_cache_list" not in updated

    await storage.index_done_callback()
    persisted = storage._data["E"]
    assert persisted["create_time"] == 1_700_000_000
    assert persisted["update_time"] == 1_700_000_100


@pytest.mark.asyncio
async def test_legacy_row_missing_create_time_keeps_zero(tmp_path):
    storage = _make_storage(tmp_path)
    await storage.initialize()

    # Simulate a legacy row written before create_time existed.
    storage._data["legacy"] = {
        "chunk_ids": ["c1"],
        "count": 1,
        "update_time": 1_600_000_000,
        "_id": "legacy",
    }

    with patch("time.time", return_value=1_700_000_200):
        await storage.upsert({"legacy": {"chunk_ids": ["c1", "c2"], "count": 2}})

    row = await storage.get_by_id("legacy")
    assert row is not None
    assert row["create_time"] == 0
    assert row["update_time"] == 1_700_000_200
    assert storage._data["legacy"]["create_time"] == 0


@pytest.mark.asyncio
async def test_insert_stamps_both_timestamps(tmp_path):
    storage = _make_storage(tmp_path)
    await storage.initialize()

    with patch("time.time", return_value=1_700_000_300):
        await storage.upsert({"new": {"chunk_ids": ["c9"], "count": 1}})

    row = await storage.get_by_id("new")
    assert row is not None
    assert row["create_time"] == 1_700_000_300
    assert row["update_time"] == 1_700_000_300
