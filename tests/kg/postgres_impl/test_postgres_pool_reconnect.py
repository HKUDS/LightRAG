import asyncio
import gc
import logging
import weakref
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import asyncpg
import pytest
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt

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


def gated_create_pool(results):
    """A create_pool stand-in whose first call waits until ``release`` is set.

    ``results`` gives, per call, a pool to return or an exception to raise.
    """
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def create_pool(**_kwargs):
        nonlocal calls
        calls += 1
        result = results[calls - 1]
        if calls == 1:
            started.set()
            await release.wait()
        if isinstance(result, BaseException):
            raise result
        return result

    return create_pool, started, release, lambda: calls


@pytest.mark.asyncio
async def test_waiters_share_the_failure_of_the_attempt_they_waited_on():
    db = make_db()
    create_pool, started, release, calls = gated_create_pool(
        [refused(), make_pool(MagicMock())]
    )

    with patch("lightrag.kg.postgres_impl.asyncpg.create_pool", create_pool):
        leader = asyncio.create_task(db._ensure_pool())
        await started.wait()
        waiters = [asyncio.create_task(db._ensure_pool()) for _ in range(5)]
        await asyncio.sleep(0)
        release.set()
        results = await asyncio.wait_for(
            asyncio.gather(leader, *waiters, return_exceptions=True), timeout=5
        )

        assert all(isinstance(r, ConnectionRefusedError) for r in results)
        assert calls() == 1
        assert db.pool is None

        # A caller that arrives after the failed attempt makes its own.
        await asyncio.wait_for(db._ensure_pool(), timeout=5)

    assert calls() == 2
    assert db.pool is not None
    assert not db._pool_reconnect_lock.locked()


@pytest.mark.asyncio
async def test_waiters_share_the_pool_of_the_attempt_they_waited_on():
    db = make_db()
    healthy = make_pool(MagicMock())
    create_pool, started, release, calls = gated_create_pool([healthy])

    with patch("lightrag.kg.postgres_impl.asyncpg.create_pool", create_pool):
        leader = asyncio.create_task(db._ensure_pool())
        await started.wait()
        waiters = [asyncio.create_task(db._ensure_pool()) for _ in range(5)]
        await asyncio.sleep(0)
        release.set()
        await asyncio.wait_for(asyncio.gather(leader, *waiters), timeout=5)

    assert calls() == 1
    assert db.pool is healthy


@pytest.mark.asyncio
async def test_a_cancelled_attempt_is_not_shared_with_its_waiters():
    db = make_db()
    healthy = make_pool(MagicMock())
    create_pool, started, _release, calls = gated_create_pool([refused(), healthy])

    with patch("lightrag.kg.postgres_impl.asyncpg.create_pool", create_pool):
        leader = asyncio.create_task(db._ensure_pool())
        await started.wait()
        waiter = asyncio.create_task(db._ensure_pool())
        await asyncio.sleep(0)
        leader.cancel()
        await asyncio.wait_for(waiter, timeout=5)

    with pytest.raises(asyncio.CancelledError):
        await leader
    assert calls() == 2
    assert db.pool is healthy


@pytest.mark.asyncio
async def test_a_transient_reconnect_failure_is_a_warning(lightrag_log_records):
    db = make_db()
    create_pool = AsyncMock(side_effect=refused())

    with patch("lightrag.kg.postgres_impl.asyncpg.create_pool", create_pool):
        with pytest.raises(ConnectionRefusedError):
            await db._ensure_pool()

    messages = [(r.levelno, r.getMessage()) for r in lightrag_log_records]
    assert any(
        level == logging.WARNING and "Failed to reconnect database" in msg
        for level, msg in messages
    )
    assert not any(level >= logging.ERROR for level, _ in messages)


@pytest.mark.asyncio
async def test_a_non_transient_reconnect_failure_is_an_error(lightrag_log_records):
    db = make_db()
    create_pool = AsyncMock(
        side_effect=asyncpg.exceptions.InvalidPasswordError("bad password")
    )

    with patch("lightrag.kg.postgres_impl.asyncpg.create_pool", create_pool):
        with pytest.raises(asyncpg.exceptions.InvalidPasswordError):
            await db._ensure_pool()

    assert any(
        r.levelno == logging.ERROR and "Failed to connect database" in r.getMessage()
        for r in lightrag_log_records
    )


@pytest.mark.asyncio
async def test_startup_failure_is_an_error_after_all_attempts(lightrag_log_records):
    db = make_db(attempts=3)
    create_pool = AsyncMock(side_effect=refused())

    with patch("lightrag.kg.postgres_impl.asyncpg.create_pool", create_pool):
        with pytest.raises(ConnectionRefusedError):
            await db.initdb()

    assert create_pool.await_count == 3
    assert any(
        r.levelno == logging.ERROR and "Failed to connect database" in r.getMessage()
        for r in lightrag_log_records
    )


@pytest.mark.asyncio
async def test_retry_log_reports_the_limit_of_the_running_loop(lightrag_log_records):
    db = make_db(attempts=7)

    with pytest.raises(ConnectionRefusedError):
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(2),
            retry=retry_if_exception_type(ConnectionError),
            before_sleep=db._log_transient_failure,
            reraise=True,
        ):
            with attempt:
                raise refused()

    messages = [
        r.getMessage()
        for r in lightrag_log_records
        if "transient connection issue" in r.getMessage()
    ]
    assert len(messages) == 1
    assert "attempt 1/2" in messages[0]


@pytest.mark.asyncio
async def test_failed_reconnects_do_not_retain_earlier_callers():
    """Only the latest failure may stay reachable, not one per failed attempt."""
    db = make_db(attempts=1)
    create_pool = AsyncMock(
        side_effect=lambda **_kwargs: (_ for _ in ()).throw(refused())
    )

    class Payload:
        pass

    refs = []

    async def caller():
        payload = Payload()
        refs.append(weakref.ref(payload))
        with pytest.raises(ConnectionRefusedError):
            await db.query("SELECT 1")

    with patch("lightrag.kg.postgres_impl.asyncpg.create_pool", create_pool):
        for _ in range(20):
            await caller()

    gc.collect()
    assert create_pool.await_count == 20
    assert sum(ref() is not None for ref in refs) <= 1
