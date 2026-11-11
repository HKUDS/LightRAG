import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lightrag.kg.postgres_impl import PostgreSQLDB

pytestmark = pytest.mark.offline


def make_db(attempts: int = 3) -> PostgreSQLDB:
    return PostgreSQLDB(
        {
            "host": "localhost",
            "port": 5432,
            "user": "postgres",
            "password": "postgres",
            "database": "postgres",
            "workspace": "test_ws",
            "max_connections": 10,
            "connection_retry_attempts": attempts,
            "connection_retry_backoff": 0,
            "connection_retry_backoff_max": 0,
            "pool_close_timeout": 5.0,
            "enable_vector": False,
        }
    )


def make_pool(connection: MagicMock) -> MagicMock:
    @asynccontextmanager
    async def acquire():
        yield connection

    pool = MagicMock()
    pool.acquire = acquire
    pool.close = AsyncMock()
    return pool


def refused() -> ConnectionRefusedError:
    return ConnectionRefusedError(111, "refused")


@pytest.mark.asyncio
async def test_initdb_retry_does_not_wait_on_the_reconnect_lock():
    db = make_db()
    pool = MagicMock()
    create_pool = AsyncMock(side_effect=[refused(), pool])

    with patch("lightrag.kg.postgres_impl.asyncpg.create_pool", create_pool):
        async with db._pool_reconnect_lock:
            await asyncio.wait_for(db.initdb(), timeout=5)

    assert db.pool is pool
    assert create_pool.await_count == 2


@pytest.mark.asyncio
async def test_ensure_pool_makes_one_attempt_and_releases_the_lock():
    db = make_db()
    create_pool = AsyncMock(side_effect=refused())

    with patch("lightrag.kg.postgres_impl.asyncpg.create_pool", create_pool):
        with pytest.raises(ConnectionRefusedError):
            await asyncio.wait_for(db._ensure_pool(), timeout=5)

    assert db.pool is None
    assert create_pool.await_count == 1
    assert not db._pool_reconnect_lock.locked()


@pytest.mark.asyncio
async def test_query_recovers_after_the_database_comes_back():
    db = make_db()
    connection = MagicMock()
    connection.fetch = AsyncMock(return_value=[])
    create_pool = AsyncMock(side_effect=[refused(), refused(), make_pool(connection)])

    with patch("lightrag.kg.postgres_impl.asyncpg.create_pool", create_pool):
        await asyncio.wait_for(db.query("SELECT 1"), timeout=5)

    connection.fetch.assert_awaited_once_with("SELECT 1")
    assert create_pool.await_count == 3
    assert not db._pool_reconnect_lock.locked()


@pytest.mark.asyncio
async def test_query_fails_after_the_configured_attempts():
    db = make_db(attempts=2)
    create_pool = AsyncMock(side_effect=refused())

    with patch("lightrag.kg.postgres_impl.asyncpg.create_pool", create_pool):
        with pytest.raises(ConnectionRefusedError):
            await asyncio.wait_for(db.query("SELECT 1"), timeout=5)

    assert create_pool.await_count == 2
    assert not db._pool_reconnect_lock.locked()


@pytest.mark.asyncio
async def test_failed_reconnect_does_not_close_a_newer_pool():
    db = make_db()
    connection = MagicMock()
    connection.fetch = AsyncMock(return_value=[])
    healthy = make_pool(connection)
    first_attempt_started = asyncio.Event()
    fail_first_attempt = asyncio.Event()
    calls = 0

    async def create_pool(**_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            first_attempt_started.set()
            await fail_first_attempt.wait()
            raise refused()
        return healthy

    with patch("lightrag.kg.postgres_impl.asyncpg.create_pool", create_pool):
        first = asyncio.create_task(db.query("SELECT 1"))
        await first_attempt_started.wait()
        second = asyncio.create_task(db.query("SELECT 2"))
        await asyncio.sleep(0)
        fail_first_attempt.set()
        await asyncio.wait_for(asyncio.gather(first, second), timeout=5)

    healthy.close.assert_not_awaited()
    assert db.pool is healthy
    assert calls == 2


@pytest.mark.asyncio
async def test_reset_pool_closes_only_the_pool_that_failed():
    db = make_db()
    failed = make_pool(MagicMock())
    newer = make_pool(MagicMock())
    db.pool = newer

    await db._reset_pool(failed)

    newer.close.assert_not_awaited()
    assert db.pool is newer

    await db._reset_pool(newer)

    newer.close.assert_awaited_once()
    assert db.pool is None
