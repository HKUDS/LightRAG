"""``RedisKVStorage.iter_rows``: the enumeration surface (scenario 16 in
docs/design/ConfigurationStorage.md) on the Redis backend, over the FakeRedis
stand-in.

SCAN walks the namespace prefix ``count`` keys at a time and each page is read
through ``get_by_ids``, so the rows carry exactly the point-read shape and no
page ever loads the whole namespace.
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


async def _storage(workspace="iterws"):
    from lightrag.kg.redis_impl import RedisKVStorage

    storage = RedisKVStorage(
        namespace="config",
        workspace=workspace,
        global_config={},
        embedding_func=None,
    )
    await storage.initialize()
    return storage


async def test_every_row_is_yielded_across_scan_pages(fake):
    storage = await _storage()
    await storage.upsert({f"k{i:02d}": {"value": {"n": i}} for i in range(7)})
    # A sibling namespace and a sibling workspace must not leak in.
    other = await _storage(workspace="otherws")
    await other.upsert({"k99": {"value": {"n": 99}}})

    scans_before = (
        len([c for c in fake.calls if c[0] == "scan"])
        if hasattr(fake, "calls")
        else None
    )
    rows = [row async for row in storage.iter_rows(page_size=3)]

    assert sorted(r["_id"] for r in rows) == [f"k{i:02d}" for i in range(7)]
    by_id = {r["_id"]: r for r in rows}
    point = await storage.get_by_id("k04")
    point["_id"] = "k04"
    assert by_id["k04"] == point
    if scans_before is not None:
        assert len([c for c in fake.calls if c[0] == "scan"]) - scans_before >= 3


async def test_an_empty_namespace_yields_nothing(fake):
    storage = await _storage()
    assert [row async for row in storage.iter_rows()] == []
