"""Regression tests for ``finalize()`` on the Redis KV / DocStatus storages.

``RedisKVStorage`` and ``RedisDocStatusStorage`` own a ``redis.asyncio``
client backed by a *shared*, reference-counted connection pool
(``RedisConnectionManager``). They define a correct ``close()`` that
releases the per-instance client and decrements the pool reference, but
they used to inherit the base class's no-op ``finalize()`` — so the
standard shutdown path (``LightRAG.finalize_storages``) never released
anything and the shared pool's reference count only ever grew.

These tests pin the contract that ``finalize()`` drives the same teardown
as ``close()``. No live Redis is required: the client is an ``AsyncMock``
and ``RedisConnectionManager.release_pool`` is patched.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lightrag.kg.redis_impl import (
    RedisConnectionManager,
    RedisDocStatusStorage,
    RedisKVStorage,
)

pytestmark = pytest.mark.offline

_REDIS_URL = "redis://localhost:6379/0"


def _make_kv_storage() -> RedisKVStorage:
    s = RedisKVStorage.__new__(RedisKVStorage)
    s.workspace = "ws"
    s.namespace = "kv"
    s._redis = AsyncMock()
    s._redis_url = _REDIS_URL
    s._pool = MagicMock()
    return s


def _make_doc_status_storage() -> RedisDocStatusStorage:
    s = RedisDocStatusStorage.__new__(RedisDocStatusStorage)
    s.workspace = "ws"
    s.namespace = "doc_status"
    s._redis = AsyncMock()
    s._redis_url = _REDIS_URL
    s._pool = MagicMock()
    return s


@pytest.mark.asyncio
async def test_finalize_kv_closes_client_and_releases_pool():
    s = _make_kv_storage()
    client = s._redis

    with patch.object(RedisConnectionManager, "release_pool") as release_pool:
        await s.finalize()

    client.close.assert_awaited_once()
    release_pool.assert_called_once_with(_REDIS_URL)
    # close() nulls the per-instance handles after release
    assert s._redis is None
    assert s._pool is None


@pytest.mark.asyncio
async def test_finalize_doc_status_closes_client_and_releases_pool():
    s = _make_doc_status_storage()
    client = s._redis

    with patch.object(RedisConnectionManager, "release_pool") as release_pool:
        await s.finalize()

    client.close.assert_awaited_once()
    release_pool.assert_called_once_with(_REDIS_URL)
    assert s._redis is None
    assert s._pool is None


@pytest.mark.asyncio
async def test_finalize_is_idempotent():
    """A second finalize() (e.g. a re-entry on the shutdown path) must not
    raise: close() guards on ``self._redis`` and release_pool is a no-op
    once the refcount has already dropped to zero."""
    s = _make_kv_storage()

    with patch.object(RedisConnectionManager, "release_pool"):
        await s.finalize()
        # Second call: client already None, pool ref already released.
        await s.finalize()

    assert s._redis is None


@pytest.mark.asyncio
async def test_finalize_after_close_is_safe():
    """Calling finalize() after an explicit close() (e.g. ``__aexit__`` ran
    first) must be a harmless no-op, not an error."""
    s = _make_doc_status_storage()

    with patch.object(RedisConnectionManager, "release_pool"):
        await s.close()
        await s.finalize()

    assert s._redis is None
