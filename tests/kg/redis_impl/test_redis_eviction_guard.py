"""Redis eviction-policy startup guard.

``doc_status`` rows, ``full_docs`` entries and the derived scheduling sidecar are
all TTL-less keys holding a system of record. Under an ``allkeys-*`` maxmemory
policy Redis may evict any of them, and the loss is both silent and undetectable
— the sidecar's structural probe only asks whether ANY status key exists, so a
partially evicted index looks healthy and never gets rebuilt. The only place this
can be caught is startup, which is what these tests pin.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from redis.exceptions import ResponseError

from lightrag.exceptions import StorageControlPlaneError
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.namespace import NameSpace

from .fake_redis import FakeRedis
from tests._log_capture import capture_lightrag_logs

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
    """The guard checks once per Redis URL per process; isolate that state.

    Tolerant of the cache not existing so a missing guard fails these tests on
    BEHAVIOUR (initialize succeeding when it must refuse) rather than on a
    fixture error.
    """
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


def _doc_status_storage(workspace: str = "evict-ws"):
    from lightrag.kg.redis_impl import RedisDocStatusStorage

    return RedisDocStatusStorage(
        namespace=NameSpace.DOC_STATUS,
        global_config={},
        embedding_func=_DummyEmbeddingFunc(),
        workspace=workspace,
    )


def _kv_storage(workspace: str = "evict-ws"):
    from lightrag.kg.redis_impl import RedisKVStorage

    return RedisKVStorage(
        namespace=NameSpace.KV_STORE_FULL_DOCS,
        global_config={},
        embedding_func=_DummyEmbeddingFunc(),
        workspace=workspace,
    )


@pytest.mark.asyncio
async def test_noeviction_server_initializes(fake):
    storage = _doc_status_storage()
    await storage.initialize()
    assert storage._initialized is True


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["allkeys-lru", "allkeys-lfu", "allkeys-random"])
async def test_allkeys_policy_with_a_memory_cap_refuses_to_start(fake, policy):
    """Every allkeys-* policy can evict a TTL-less key, so all of them refuse."""
    fake.config_values = {"maxmemory": "1073741824", "maxmemory-policy": policy}
    storage = _doc_status_storage()
    with pytest.raises(StorageControlPlaneError, match="evicting cache"):
        await storage.initialize()
    assert storage._initialized is False


@pytest.mark.asyncio
async def test_full_docs_kv_is_guarded_too(fake):
    """full_docs is a system of record as much as doc_status is."""
    fake.config_values = {"maxmemory": "1073741824", "maxmemory-policy": "allkeys-lru"}
    storage = _kv_storage()
    with pytest.raises(StorageControlPlaneError, match="evicting cache"):
        await storage.initialize()


@pytest.mark.asyncio
@pytest.mark.parametrize("first_factory", [_doc_status_storage, _kv_storage])
@pytest.mark.parametrize("retry_factory", [_doc_status_storage, _kv_storage])
async def test_rejected_policy_still_refuses_later_storages(
    fake, monkeypatch, first_factory, retry_factory
):
    """A failed startup must not let later instances bypass the same policy."""
    monkeypatch.delenv("REDIS_ALLOW_EVICTION_POLICY", raising=False)
    fake.config_values = {"maxmemory": "1073741824", "maxmemory-policy": "allkeys-lru"}

    with pytest.raises(StorageControlPlaneError, match="evicting cache"):
        await first_factory().initialize()

    retry = retry_factory(workspace="retry-ws")
    try:
        with pytest.raises(StorageControlPlaneError, match="evicting cache"):
            await retry.initialize()
    finally:
        await retry.finalize()


@pytest.mark.asyncio
@pytest.mark.parametrize("allow_eviction", [False, True])
async def test_rejected_policy_can_recover_on_later_startup(
    fake, monkeypatch, caplog, allow_eviction
):
    monkeypatch.delenv("REDIS_ALLOW_EVICTION_POLICY", raising=False)
    fake.config_values = {"maxmemory": "1073741824", "maxmemory-policy": "allkeys-lru"}
    with pytest.raises(StorageControlPlaneError, match="evicting cache"):
        await _doc_status_storage().initialize()

    if allow_eviction:
        monkeypatch.setenv("REDIS_ALLOW_EVICTION_POLICY", "true")
    else:
        fake.config_values["maxmemory-policy"] = "noeviction"

    retry = _kv_storage()
    try:
        with capture_lightrag_logs(caplog):
            await retry.initialize()
        if allow_eviction:
            assert "configured to EVICT" in caplog.text
    finally:
        await retry.finalize()


@pytest.mark.asyncio
async def test_allkeys_policy_without_a_memory_cap_is_safe(fake):
    """maxmemory=0 means no limit, so the policy never fires."""
    fake.config_values = {"maxmemory": "0", "maxmemory-policy": "allkeys-lru"}
    storage = _doc_status_storage()
    await storage.initialize()
    assert storage._initialized is True


@pytest.mark.asyncio
async def test_volatile_policy_is_safe(fake):
    """volatile-* only evicts keys with a TTL; none of ours carry one, so a full
    instance fails writes loudly instead of dropping data."""
    fake.config_values = {"maxmemory": "1073741824", "maxmemory-policy": "volatile-lru"}
    storage = _doc_status_storage()
    await storage.initialize()
    assert storage._initialized is True


@pytest.mark.asyncio
async def test_unreadable_config_warns_but_starts(fake, caplog):
    """Managed Redis often blocks CONFIG GET. Refusing to start over a policy we
    cannot observe would break working deployments, so this warns instead."""
    import logging

    from lightrag.utils import logger

    fake.fail_next["config_get"] = ResponseError("unknown command 'CONFIG'")
    storage = _doc_status_storage()
    with capture_lightrag_logs(caplog):
        with caplog.at_level(logging.WARNING, logger=logger.name):
            await storage.initialize()
    assert storage._initialized is True
    assert "Could not read the Redis maxmemory policy" in caplog.text


@pytest.mark.asyncio
async def test_explicit_override_downgrades_to_a_warning(fake, monkeypatch, caplog):
    import logging

    from lightrag.utils import logger

    monkeypatch.setenv("REDIS_ALLOW_EVICTION_POLICY", "true")
    fake.config_values = {"maxmemory": "1073741824", "maxmemory-policy": "allkeys-lru"}
    storage = _doc_status_storage()
    with capture_lightrag_logs(caplog):
        with caplog.at_level(logging.WARNING, logger=logger.name):
            await storage.initialize()
    assert storage._initialized is True
    assert "configured to EVICT" in caplog.text


@pytest.mark.asyncio
async def test_policy_is_checked_once_per_url(fake):
    """The policy is a server property shared by every storage on the pool, so a
    second storage on the same URL must not re-issue CONFIG GET."""
    calls = {"n": 0}
    real_config_get = fake.config_get

    async def counting_config_get(pattern="*"):
        calls["n"] += 1
        return await real_config_get(pattern)

    fake.config_get = counting_config_get

    await _doc_status_storage().initialize()
    await _kv_storage().initialize()
    assert calls["n"] == 1
