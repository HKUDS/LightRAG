"""Regression tests: RedisKVStorage replacement upserts preserve create_time.

Mirrors the JsonKVStorage invariant from issue #3870 against the in-memory
FakeRedis stand-in (no live Redis required).
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.namespace import NameSpace

from .fake_redis import FakeRedis

pytestmark = pytest.mark.offline


class _DummyEmbeddingFunc:
    embedding_dim = 1
    max_token_size = 1

    async def __call__(self, texts, **kwargs):
        return [[0.0] for _ in texts]


@pytest.fixture(autouse=True)
def _shared():
    initialize_share_data()
    yield
    finalize_share_data()


@pytest.fixture(autouse=True)
def _reset_eviction_cache():
    from lightrag.kg import redis_impl

    cache = getattr(redis_impl, "_eviction_checked", None)
    if cache is not None:
        cache.clear()
    yield
    if cache is not None:
        cache.clear()


@pytest.fixture
def fake(monkeypatch):
    instance = FakeRedis()
    monkeypatch.setattr(
        "lightrag.kg.redis_impl.RedisConnectionManager.get_pool",
        lambda redis_url: MagicMock(name="fake_pool"),
    )
    monkeypatch.setattr(
        "lightrag.kg.redis_impl.Redis", lambda connection_pool=None, **_: instance
    )
    return instance


def _kv_storage(workspace: str = "ct-ws"):
    from lightrag.kg.redis_impl import RedisKVStorage

    return RedisKVStorage(
        namespace=NameSpace.KV_STORE_ENTITY_CHUNKS,
        global_config={},
        embedding_func=_DummyEmbeddingFunc(),
        workspace=workspace,
    )


@pytest.mark.asyncio
async def test_replacement_upsert_preserves_create_time(fake):
    storage = _kv_storage()
    await storage.initialize()

    with patch("time.time", return_value=1_700_000_000):
        await storage.upsert({"E": {"chunk_ids": ["c1"], "count": 1}})

    created = await storage.get_by_id("E")
    assert created is not None
    assert created["create_time"] == 1_700_000_000
    assert created["update_time"] == 1_700_000_000

    with patch("time.time", return_value=1_700_000_100):
        await storage.upsert({"E": {"chunk_ids": ["c1", "c2"], "count": 2}})

    updated = await storage.get_by_id("E")
    assert updated is not None
    assert updated["create_time"] == 1_700_000_000
    assert updated["update_time"] == 1_700_000_100
    assert updated["chunk_ids"] == ["c1", "c2"]
    assert updated["count"] == 2

    raw = fake.store[f"{storage.final_namespace}:E"]
    persisted = json.loads(raw)
    assert persisted["create_time"] == 1_700_000_000
    assert persisted["update_time"] == 1_700_000_100


@pytest.mark.asyncio
async def test_legacy_row_missing_create_time_keeps_zero(fake):
    storage = _kv_storage()
    await storage.initialize()

    key = f"{storage.final_namespace}:legacy"
    fake.store[key] = json.dumps(
        {
            "chunk_ids": ["c1"],
            "count": 1,
            "update_time": 1_600_000_000,
            "_id": "legacy",
        }
    )

    with patch("time.time", return_value=1_700_000_200):
        await storage.upsert({"legacy": {"chunk_ids": ["c1", "c2"], "count": 2}})

    row = await storage.get_by_id("legacy")
    assert row is not None
    assert row["create_time"] == 0
    assert row["update_time"] == 1_700_000_200
    assert json.loads(fake.store[key])["create_time"] == 0


@pytest.mark.asyncio
async def test_insert_stamps_both_timestamps(fake):
    storage = _kv_storage()
    await storage.initialize()

    with patch("time.time", return_value=1_700_000_300):
        await storage.upsert({"new": {"chunk_ids": ["c9"], "count": 1}})

    row = await storage.get_by_id("new")
    assert row is not None
    assert row["create_time"] == 1_700_000_300
    assert row["update_time"] == 1_700_000_300
