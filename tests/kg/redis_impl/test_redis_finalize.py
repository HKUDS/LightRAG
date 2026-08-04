"""Regression tests for teardown of the Redis KV / DocStatus storages.

``RedisKVStorage`` and ``RedisDocStatusStorage`` own a ``redis.asyncio``
client backed by a *shared*, reference-counted connection pool
(``RedisConnectionManager``). ``finalize()`` drives the same teardown as
``close()``; ``close()`` must be idempotent and re-entrant-safe so that a
double finalize (or finalize-after-``__aexit__``) never releases the shared
pool a second time and never steals a sibling storage's reference.

Two flavours of test live here:

* **Mock-based** — the client is an ``AsyncMock`` and
  ``RedisConnectionManager.release_pool`` is patched; these pin the
  ``finalize → close`` wiring and the cancellation contract in isolation.
* **Real-refcount** — no ``release_pool`` mock; a real (never-connected)
  ``ConnectionPool`` or a fake pool is registered in the manager so the
  reference-count arithmetic itself is exercised. These are the ones that
  actually catch a double-release / ref-stealing regression.
"""

from __future__ import annotations

import asyncio
import gc
import warnings
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lightrag.kg.redis_impl import (
    RedisConnectionManager,
    RedisDocStatusStorage,
    RedisKVStorage,
)

pytestmark = pytest.mark.offline

_REDIS_URL = "redis://localhost:6379/0"

STORAGE_CLASSES = [RedisKVStorage, RedisDocStatusStorage]


def _new_storage(cls, url: str = _REDIS_URL, pool=None):
    """Build a storage instance without running ``__post_init__`` (no live Redis)."""
    s = cls.__new__(cls)
    s.workspace = "ws"
    s.namespace = "kv" if cls is RedisKVStorage else "doc_status"
    s._redis = AsyncMock()
    s._redis_url = url
    s._pool = pool if pool is not None else MagicMock()
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
    # close() detaches all per-instance handles before the first await
    assert s._redis is None
    assert s._pool is None
    assert s._redis_url is None


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
    assert s._redis_url is None


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
    assert s._redis_url is None


# ---------------------------------------------------------------------------
# Real refcount: double close must not steal a sibling's pool reference
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("cls", STORAGE_CLASSES)
async def test_double_close_does_not_steal_sibling_pool_ref(cls):
    """Two storages share one pool (refcount 2). Closing instance A twice must
    release exactly one reference; the sibling's reference and the live pool
    must survive. Fix-proof: without close() detaching ``_redis_url``, the
    second close() would drop the refcount to 0 and disconnect the shared pool
    out from under the sibling."""
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
# Construction-failure path must not mask the original init exception (Fix B)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", STORAGE_CLASSES)
def test_init_error_no_running_loop_does_not_mask_exception(cls):
    """In a sync context with no running loop, the cleanup must not create a
    coroutine (no unawaited-coroutine warning) nor raise a new error that would
    replace the real init failure."""
    fake_pool = MagicMock()
    fake_pool.aclose = AsyncMock()
    boom = RuntimeError("unique-init-boom")
    s = cls.__new__(cls)
    s.workspace = "ws"
    s.namespace = "kv" if cls is RedisKVStorage else "doc_status"
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with (
                patch(
                    "lightrag.kg.redis_impl.ConnectionPool.from_url",
                    return_value=fake_pool,
                ),
                patch("lightrag.kg.redis_impl.Redis", side_effect=boom),
            ):
                with pytest.raises(RuntimeError, match="unique-init-boom"):
                    s.__post_init__()
            # gc.collect() must be OUTSIDE pytest.raises (the exception exits
            # that block immediately) but INSIDE catch_warnings.
            gc.collect()

        assert not any("was never awaited" in str(w.message) for w in caught)
        # No running loop -> nothing scheduled.
        assert not RedisConnectionManager._cleanup_tasks
    finally:
        _drop_from_registry(getattr(s, "_redis_url", ""))


@pytest.mark.asyncio
@pytest.mark.parametrize("cls", STORAGE_CLASSES)
async def test_init_error_with_loop_schedules_pool_close(cls):
    fake_pool = MagicMock()
    fake_pool.aclose = AsyncMock()
    boom = RuntimeError("unique-init-boom2")
    s = cls.__new__(cls)
    s.workspace = "ws"
    s.namespace = "kv" if cls is RedisKVStorage else "doc_status"
    try:
        with (
            patch(
                "lightrag.kg.redis_impl.ConnectionPool.from_url",
                return_value=fake_pool,
            ),
            patch("lightrag.kg.redis_impl.Redis", side_effect=boom),
        ):
            with pytest.raises(RuntimeError, match="unique-init-boom2"):
                s.__post_init__()

        # Let the scheduled task run, then let its done-callback drain the set.
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        fake_pool.aclose.assert_awaited_once()
        assert not RedisConnectionManager._cleanup_tasks
    finally:
        _drop_from_registry(getattr(s, "_redis_url", ""))


@pytest.mark.asyncio
async def test_init_error_background_pool_close_failure_is_swallowed():
    """A failing background ``aclose()`` must not surface as an unretrieved task
    exception, and must not affect the original init error."""
    fake_pool = MagicMock()
    fake_pool.aclose = AsyncMock(side_effect=RuntimeError("pool-close-fail"))
    boom = RuntimeError("unique-init-boom3")
    s = RedisKVStorage.__new__(RedisKVStorage)
    s.workspace = "ws"
    s.namespace = "kv"
    try:
        with (
            patch(
                "lightrag.kg.redis_impl.ConnectionPool.from_url",
                return_value=fake_pool,
            ),
            patch("lightrag.kg.redis_impl.Redis", side_effect=boom),
        ):
            with pytest.raises(RuntimeError, match="unique-init-boom3"):
                s.__post_init__()

        await asyncio.sleep(0)
        await asyncio.sleep(0)

        fake_pool.aclose.assert_awaited_once()
        # Task completed (exception swallowed by _close_pool_safely) and was
        # removed from the tracking set — no "Task exception was never retrieved".
        assert not RedisConnectionManager._cleanup_tasks
    finally:
        _drop_from_registry(getattr(s, "_redis_url", ""))


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
    assert s._redis_url is None
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
