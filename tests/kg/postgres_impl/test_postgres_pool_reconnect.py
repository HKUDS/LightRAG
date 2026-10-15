import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lightrag.kg.postgres_impl import PostgreSQLDB


def make_db() -> PostgreSQLDB:
    return PostgreSQLDB(
        {
            "host": "localhost",
            "port": 5432,
            "user": "postgres",
            "password": "postgres",
            "database": "postgres",
            "workspace": "test_ws",
            "max_connections": 10,
            "connection_retry_attempts": 3,
            "connection_retry_backoff": 0,
            "connection_retry_backoff_max": 0,
            "pool_close_timeout": 5.0,
            "enable_vector": False,
        }
    )


@pytest.mark.asyncio
async def test_ensure_pool_recovers_when_first_pool_creation_fails():
    db = make_db()
    pool = MagicMock()
    create_pool = AsyncMock(side_effect=[ConnectionRefusedError(111, "refused"), pool])

    with patch("lightrag.kg.postgres_impl.asyncpg.create_pool", create_pool):
        await asyncio.wait_for(db._ensure_pool(), timeout=5)

    assert db.pool is pool
    assert create_pool.await_count == 2
    assert not db._pool_reconnect_lock.locked()


@pytest.mark.asyncio
async def test_ensure_pool_raises_after_retries_and_releases_lock():
    db = make_db()
    create_pool = AsyncMock(side_effect=ConnectionRefusedError(111, "refused"))

    with patch("lightrag.kg.postgres_impl.asyncpg.create_pool", create_pool):
        with pytest.raises(ConnectionRefusedError):
            await asyncio.wait_for(db._ensure_pool(), timeout=5)

    assert db.pool is None
    assert create_pool.await_count == 3
    assert not db._pool_reconnect_lock.locked()
