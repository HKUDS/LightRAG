"""Regression tests for teardown of the Redis KV / DocStatus storages.

``RedisKVStorage`` and ``RedisDocStatusStorage`` own a ``redis.asyncio``
client backed by a *shared*, reference-counted connection pool
(``RedisConnectionManager``). ``finalize()`` drives the same teardown as
``close()``; ``close()`` must be idempotent and re-entrant-safe so that a
double finalize (or finalize-after-``__aexit__``) never releases the shared
pool a second time and never steals a sibling storage's reference.

The reference is taken in ``initialize()`` and given back in ``close()``;
construction takes nothing. That pairing is the whole contract: a reference
taken in ``__post_init__`` has no teardown path, so any construction that
fails afterwards — most of all ``LightRAG.__post_init__``, which builds twelve
storages and then validates — leaks it for the life of the process.

Two flavours of test live here:

* **Mock-based** — the client is an ``AsyncMock`` and
  ``RedisConnectionManager.release_pool`` is patched; these pin the
  ``finalize → close`` wiring and the cancellation contract in isolation.
* **Real-refcount** — no ``release_pool`` mock; a real (never-connected)
  ``ConnectionPool`` or a fake pool is registered in the manager so the
  reference-count arithmetic itself is exercised. These are the ones that
  actually catch a double-release / ref-stealing regression, and the ones
  that catch a reference taken at construction.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lightrag.kg.redis_impl import (
    RedisConnectionManager,
    RedisDocStatusStorage,
    RedisKVStorage,
)
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.namespace import NameSpace

pytestmark = pytest.mark.offline

_REDIS_URL = "redis://localhost:6379/0"

STORAGE_CLASSES = [RedisKVStorage, RedisDocStatusStorage]


class _DummyEmbeddingFunc:
    embedding_dim = 1
    max_token_size = 1

    async def __call__(self, texts, **kwargs):
        return [[0.0] for _ in texts]


def _namespace_for(cls) -> str:
    return (
        NameSpace.KV_STORE_FULL_DOCS if cls is RedisKVStorage else NameSpace.DOC_STATUS
    )


@pytest.fixture
def _shared():
    """``initialize()`` takes the shared-storage data-init lock."""
    initialize_share_data()
    yield
    finalize_share_data()


def _new_storage(cls, url: str = _REDIS_URL, pool=None):
    """Build an *initialized* storage without any live Redis: a client and a
    pool handle attached, as ``initialize()`` would leave them."""
    s = cls.__new__(cls)
    s.workspace = "ws"
    s.namespace = "kv" if cls is RedisKVStorage else "doc_status"
    s._redis = AsyncMock()
    s._redis_url = url
    s._pool = pool if pool is not None else MagicMock()
    s._initialized = True
    return s


def _unconnected_storage(cls, url: str = _REDIS_URL):
    """Build a storage in the state ``__post_init__`` leaves it in: configured,
    holding no pool reference and no client."""
    s = cls.__new__(cls)
    s.workspace = "ws"
    s.namespace = "kv" if cls is RedisKVStorage else "doc_status"
    s.final_namespace = f"ws_{s.namespace}"
    s._redis_url = url
    s._pool = None
    s._redis = None
    s._initialized = False
    return s


def _drop_from_registry(url: str) -> None:
    RedisConnectionManager._pools.pop(url, None)
    RedisConnectionManager._pool_refs.pop(url, None)


# ---------------------------------------------------------------------------
# finalize → close wiring (mock-based)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("cls", STORAGE_CLASSES)
async def test_finalize_closes_client_and_releases_pool(cls):
    s = _new_storage(cls)
    client = s._redis

    with patch.object(
        RedisConnectionManager, "release_pool", new_callable=AsyncMock
    ) as release_pool:
        await s.finalize()

    client.aclose.assert_awaited_once()
    release_pool.assert_awaited_once_with(_REDIS_URL)
    # close() detaches every per-instance resource handle before the first
    # await. _redis_url is configuration, not a handle, and survives.
    assert s._redis is None
    assert s._pool is None
    assert s._initialized is False


@pytest.mark.asyncio
async def test_finalize_is_idempotent():
    """A second finalize() (re-entry on the shutdown path) must be a complete
    no-op: the instance detached its state on the first call."""
    s = _new_storage(RedisKVStorage)

    with patch.object(
        RedisConnectionManager, "release_pool", new_callable=AsyncMock
    ) as release_pool:
        await s.finalize()
        await s.finalize()

    # release_pool ran exactly once despite two finalize() calls
    release_pool.assert_awaited_once_with(_REDIS_URL)
    assert s._redis is None
    assert s._pool is None


@pytest.mark.asyncio
async def test_finalize_after_close_is_safe():
    """finalize() after an explicit close() (e.g. ``__aexit__`` ran first)
    must be a harmless no-op, not a second release."""
    s = _new_storage(RedisDocStatusStorage)

    with patch.object(
        RedisConnectionManager, "release_pool", new_callable=AsyncMock
    ) as release_pool:
        await s.close()
        await s.finalize()

    release_pool.assert_awaited_once_with(_REDIS_URL)
    assert s._redis is None
    assert s._pool is None


# ---------------------------------------------------------------------------
# Real refcount: double close must not steal a sibling's pool reference
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("cls", STORAGE_CLASSES)
async def test_double_close_does_not_steal_sibling_pool_ref(cls):
    """Two storages share one pool (refcount 2). Closing instance A twice must
    release exactly one reference; the sibling's reference and the live pool
    must survive. Fix-proof: without close() detaching ``_pool``, the second
    close() would drop the refcount to 0 and disconnect the shared pool out
    from under the sibling."""
    url = "redis://localhost:6379/15"
    try:
        pool = RedisConnectionManager.get_pool(url)  # refcount 1
        RedisConnectionManager.get_pool(url)  # refcount 2 (sibling B)
        assert RedisConnectionManager._pool_refs[url] == 2

        a = _new_storage(cls, url, pool)
        await a.close()  # A releases once: 2 -> 1
        assert RedisConnectionManager._pool_refs[url] == 1
        assert url in RedisConnectionManager._pools

        await a.close()  # double close: instance already detached -> no-op
        assert RedisConnectionManager._pool_refs[url] == 1
        assert url in RedisConnectionManager._pools

        b = _new_storage(cls, url, pool)
        await b.close()  # B releases the last reference: 1 -> 0
        assert url not in RedisConnectionManager._pools
        assert url not in RedisConnectionManager._pool_refs
    finally:
        _drop_from_registry(url)


# ---------------------------------------------------------------------------
# Manager: pool disconnected only on the last reference
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_release_pool_closes_only_on_last_reference():
    url = "redis://localhost:6379/16"
    fake_pool = MagicMock()
    fake_pool.aclose = AsyncMock()
    try:
        with patch(
            "lightrag.kg.redis_impl.ConnectionPool.from_url", return_value=fake_pool
        ):
            RedisConnectionManager.get_pool(url)  # refcount 1
            RedisConnectionManager.get_pool(url)  # refcount 2

        await RedisConnectionManager.release_pool(url)  # 2 -> 1: no disconnect
        fake_pool.aclose.assert_not_awaited()
        assert url in RedisConnectionManager._pools

        await RedisConnectionManager.release_pool(url)  # 1 -> 0: disconnect
        fake_pool.aclose.assert_awaited_once()
        assert url not in RedisConnectionManager._pools
        assert url not in RedisConnectionManager._pool_refs
    finally:
        _drop_from_registry(url)


@pytest.mark.asyncio
async def test_close_all_pools_disconnects_and_clears_registry():
    url = "redis://localhost:6379/17"
    fake_pool = MagicMock()
    fake_pool.aclose = AsyncMock()
    try:
        with patch(
            "lightrag.kg.redis_impl.ConnectionPool.from_url", return_value=fake_pool
        ):
            RedisConnectionManager.get_pool(url)

        await RedisConnectionManager.close_all_pools()

        fake_pool.aclose.assert_awaited_once()
        assert not RedisConnectionManager._pools
        assert not RedisConnectionManager._pool_refs
    finally:
        _drop_from_registry(url)


# ---------------------------------------------------------------------------
# Construction takes no pool reference (issue #4016)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", STORAGE_CLASSES)
def test_construction_takes_no_pool_reference(cls):
    """A reference taken in ``__post_init__`` could never be given back: a
    constructor has no teardown path, and ``LightRAG.__post_init__`` builds
    twelve storages and then validates, with nothing to unwind the ones already
    built. So construction must leave the manager's registry untouched — the
    pool is acquired in ``initialize()``, which pairs with ``close()``."""
    pools_before = dict(RedisConnectionManager._pools)
    refs_before = dict(RedisConnectionManager._pool_refs)

    storage = cls(
        namespace=_namespace_for(cls),
        global_config={},
        embedding_func=_DummyEmbeddingFunc(),
        workspace="leak-ws",
    )

    assert RedisConnectionManager._pools == pools_before
    assert RedisConnectionManager._pool_refs == refs_before
    assert storage._pool is None
    assert storage._redis is None
    assert storage._initialized is False


@pytest.mark.parametrize("cls", STORAGE_CLASSES)
def test_discarded_storages_leave_no_pool_reference(cls, monkeypatch):
    """The reported symptom. ``LightRAG.__post_init__`` builds its storages and
    then runs a series of validations; when one of those raises, every storage
    already built is dropped on the floor — nothing calls ``close()`` on them,
    and there is no half-built ``LightRAG`` to call ``finalize_storages()`` on.
    Repeating that must not grow the shared pool's refcount, or a later valid
    instance can never bring it to zero and the pool is never disconnected."""
    url = "redis://localhost:6379/20"
    monkeypatch.setenv("REDIS_URI", url)
    try:
        RedisConnectionManager.get_pool(url)  # a live instance's reference
        assert RedisConnectionManager._pool_refs[url] == 1

        for _ in range(3):
            cls(
                namespace=_namespace_for(cls),
                global_config={},
                embedding_func=_DummyEmbeddingFunc(),
                workspace="leak-ws",
            )

        assert RedisConnectionManager._pool_refs[url] == 1
    finally:
        _drop_from_registry(url)


@pytest.mark.asyncio
@pytest.mark.parametrize("cls", STORAGE_CLASSES)
async def test_close_before_initialize_releases_nothing(cls):
    """close() on a never-initialized storage must not release a reference it
    never took — that would disconnect the pool out from under a live sibling."""
    url = "redis://localhost:6379/18"
    try:
        RedisConnectionManager.get_pool(url)  # a sibling's reference
        assert RedisConnectionManager._pool_refs[url] == 1

        s = _unconnected_storage(cls, url)
        await s.close()

        assert RedisConnectionManager._pool_refs[url] == 1
        assert url in RedisConnectionManager._pools
    finally:
        _drop_from_registry(url)


@pytest.mark.asyncio
@pytest.mark.parametrize("cls", STORAGE_CLASSES)
async def test_initialize_client_failure_releases_the_pool_ref(cls, _shared):
    """The acquisition moved to initialize() keeps its error path: if the client
    cannot be built after the reference was taken, that reference is given back
    and the original error propagates unmasked."""
    fake_pool = MagicMock()
    fake_pool.aclose = AsyncMock()
    boom = RuntimeError("unique-init-boom")
    url = "redis://localhost:6379/21"
    s = _unconnected_storage(cls, url)
    try:
        with (
            patch(
                "lightrag.kg.redis_impl.ConnectionPool.from_url",
                return_value=fake_pool,
            ),
            patch("lightrag.kg.redis_impl.Redis", side_effect=boom),
        ):
            with pytest.raises(RuntimeError, match="unique-init-boom"):
                await s.initialize()

        # Refcount went 0 -> 1 -> 0, so the pool was popped and disconnected.
        assert url not in RedisConnectionManager._pools
        assert url not in RedisConnectionManager._pool_refs
        fake_pool.aclose.assert_awaited_once()
        assert s._pool is None
        assert s._redis is None
        assert s._initialized is False
    finally:
        _drop_from_registry(url)


@pytest.mark.asyncio
@pytest.mark.parametrize("cls", STORAGE_CLASSES)
async def test_initialize_ping_failure_releases_the_pool_ref(cls, _shared):
    """Same for a failure after the client exists: initialize() closes itself
    down, so a retry starts from a clean slate instead of a stuck reference."""
    fake_pool = MagicMock()
    fake_pool.aclose = AsyncMock()
    client = AsyncMock()
    client.ping.side_effect = RuntimeError("unique-ping-boom")
    url = "redis://localhost:6379/22"
    s = _unconnected_storage(cls, url)
    try:
        with (
            patch(
                "lightrag.kg.redis_impl.ConnectionPool.from_url",
                return_value=fake_pool,
            ),
            patch("lightrag.kg.redis_impl.Redis", return_value=client),
        ):
            with pytest.raises(RuntimeError, match="unique-ping-boom"):
                await s.initialize()

        client.aclose.assert_awaited_once()
        assert url not in RedisConnectionManager._pools
        assert url not in RedisConnectionManager._pool_refs
        fake_pool.aclose.assert_awaited_once()
        assert s._initialized is False
    finally:
        _drop_from_registry(url)


# ---------------------------------------------------------------------------
# Cancellation safety: a cancelled client close must still release the pool ref
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("cls", STORAGE_CLASSES)
async def test_close_releases_pool_even_if_client_close_cancelled(cls):
    s = _new_storage(cls)
    s._redis = AsyncMock()
    s._redis.aclose.side_effect = asyncio.CancelledError

    with patch.object(
        RedisConnectionManager, "release_pool", new_callable=AsyncMock
    ) as release_pool:
        with pytest.raises(asyncio.CancelledError):
            await s.close()

    # CancelledError propagates, but the pool ref was still released in finally.
    release_pool.assert_awaited_once_with(_REDIS_URL)
    assert s._redis is None
    assert s._pool is None


@pytest.mark.asyncio
async def test_close_cancelled_midflight_still_releases_real_pool_ref():
    """Cancel close() while the client aclose() is in flight; the shared pool's
    refcount must still be decremented (no permanent leak)."""
    url = "redis://localhost:6379/19"
    started = asyncio.Event()
    release = asyncio.Event()  # never set -> aclose blocks until cancelled

    async def blocking_aclose(*args, **kwargs):
        started.set()
        await release.wait()

    try:
        pool = RedisConnectionManager.get_pool(url)  # refcount 1
        RedisConnectionManager.get_pool(url)  # refcount 2 (sibling)

        a = _new_storage(RedisKVStorage, url, pool)
        a._redis = AsyncMock()
        a._redis.aclose = AsyncMock(side_effect=blocking_aclose)

        task = asyncio.create_task(a.close())
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # finally released A's reference: 2 -> 1, pool still alive for sibling.
        assert RedisConnectionManager._pool_refs[url] == 1
        assert url in RedisConnectionManager._pools
    finally:
        _drop_from_registry(url)
