"""``RedisKVStorage.delete_by_cache_types`` over the FakeRedis stand-in.

Durable on return (no commit needed), selected by ``parse_cache_key`` on the
namespace-stripped key, scoped to this workspace's namespace, and counted from
what ``DEL`` actually removed.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data

from .fake_redis import FakeRedis

pytestmark = pytest.mark.offline


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


async def _storage(workspace):
    from lightrag.kg.redis_impl import RedisKVStorage

    storage = RedisKVStorage(
        namespace="llm_response_cache",
        workspace=workspace,
        global_config={},
        embedding_func=None,
    )
    await storage.initialize()
    return storage


_ROWS = {
    "mix:query:q1": "query",
    "naive:query:q2": "query",
    "local:keywords:k1": "keywords",
    "default:extract:e1": "extract",
    "default:summary:s1": "summary",
    "not-a-cache-key": "query",
}


async def test_only_matching_rows_in_this_workspace_are_deleted(fake):
    storage = await _storage("ws-a")
    other = await _storage("ws-b")
    rows = {k: {"return": "x", "cache_type": t} for k, t in _ROWS.items()}
    await storage.upsert(rows)
    await other.upsert(rows)

    expected_gone = {"mix:query:q1", "naive:query:q2"}
    deleted = await storage.delete_by_cache_types({"query"})

    assert deleted == len(expected_gone)
    remaining = {k for k in _ROWS if await storage.get_by_id(k) is not None}
    assert remaining == set(_ROWS) - expected_gone
    other_remaining = {k for k in _ROWS if await other.get_by_id(k) is not None}
    assert other_remaining == set(_ROWS), "a sibling workspace was touched"
