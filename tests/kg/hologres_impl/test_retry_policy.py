import asyncio
from contextlib import asynccontextmanager

import pytest

from lightrag.kg.hologres.client import (
    HologresClient,
    HologresOperationError,
    HologresSqlError,
    OperationKind,
    is_connection_loss,
    should_retry_operation,
)
from lightrag.kg.hologres.config import HologresConfig


class DatabaseFailure(Exception):
    def __init__(self, sqlstate: str, message: str = "database failure"):
        super().__init__(message)
        self.sqlstate = sqlstate


class FakeConnection:
    def __init__(self, *, fetch_value=None, execute=None):
        self.fetch_value_result = fetch_value
        self.execute_result = execute
        self.terminated = False

    async def fetchval(self, _sql, *_args, timeout=None):
        if isinstance(self.fetch_value_result, BaseException):
            raise self.fetch_value_result
        return self.fetch_value_result

    async def execute(self, _sql, *_args, timeout=None):
        if isinstance(self.execute_result, BaseException):
            raise self.execute_result
        return self.execute_result

    def terminate(self):
        self.terminated = True


class FailingTerminateConnection(FakeConnection):
    def terminate(self):
        raise RuntimeError("password=terminate-secret")


class FakePool:
    def __init__(self, connections):
        self.connections = list(connections)
        self.acquire_count = 0
        self.released_terminated = []

    @asynccontextmanager
    async def acquire(self, *, timeout=None):
        connection = self.connections[self.acquire_count]
        self.acquire_count += 1
        try:
            yield connection
        finally:
            self.released_terminated.append(connection.terminated)

    async def close(self):
        return None


@pytest.fixture
def config():
    return HologresConfig.from_env(
        {
            "HOLOGRES_HOST": "example.hologres.aliyuncs.com",
            "HOLOGRES_PORT": "80",
            "HOLOGRES_USER": "test_user",
            "HOLOGRES_PASSWORD": "secret",
            "HOLOGRES_DATABASE": "analytics",
            "HOLOGRES_CONNECTION_RETRIES": "1",
            "HOLOGRES_RETRY_BACKOFF": "0",
        }
    )


def test_connection_loss_classification_uses_transport_errors_and_sqlstate_class_08():
    assert is_connection_loss(ConnectionError("socket closed")) is True
    assert is_connection_loss(DatabaseFailure("08006")) is True
    assert is_connection_loss(DatabaseFailure("40001")) is False
    assert is_connection_loss(ValueError("bad query")) is False


def test_asyncpg_interface_error_is_not_automatically_a_connection_loss():
    asyncpg = pytest.importorskip("asyncpg")

    assert is_connection_loss(asyncpg.InterfaceError("operation already in progress")) is False
    assert (
        is_connection_loss(asyncpg.ConnectionDoesNotExistError("connection closed"))
        is True
    )
    assert is_connection_loss(asyncpg.ConnectionFailureError("connection failed")) is True


def test_retry_policy_replays_reads_but_requires_explicit_write_idempotence():
    failure = DatabaseFailure("08003")

    assert should_retry_operation(OperationKind.READ, failure, replay_safe=False) is True
    assert should_retry_operation(OperationKind.WRITE, failure, replay_safe=False) is False
    assert should_retry_operation(OperationKind.WRITE, failure, replay_safe=True) is True
    assert (
        should_retry_operation(
            OperationKind.READ, DatabaseFailure("23505"), replay_safe=True
        )
        is False
    )


@pytest.mark.asyncio
async def test_client_retries_a_read_after_connection_loss(config):
    first = FakeConnection(fetch_value=DatabaseFailure("08006", "secret host text"))
    second = FakeConnection(fetch_value=42)
    pool = FakePool([first, second])
    client = HologresClient(config, pool=pool)

    result = await client.fetch_value(
        "SELECT $1::integer", 42, descriptor="test.read", workspace="tenant-a"
    )

    assert result == 42
    assert pool.acquire_count == 2
    assert first.terminated is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sql",
    [
        "CREATE TABLE snapshot AS SELECT 1",
        "TRUNCATE events",
        "SELECT * INTO snapshot FROM events",
        "WITH source AS (SELECT 1 AS id) SELECT id INTO snapshot FROM source",
        (
            "WITH source AS (SELECT 1 AS id) "
            "INSERT INTO snapshot SELECT id FROM source RETURNING id"
        ),
        "EXPLAIN ANALYZE CREATE TABLE snapshot AS SELECT 1",
        "EXPLAIN (ANALYSE TRUE) CREATE TABLE snapshot AS SELECT 1",
        "EXPLAIN ANALYZE SELECT 1",
        "EXPLAIN (ANALYSE ON) SELECT 1",
    ],
)
async def test_implicit_read_requires_lexically_proven_read_only_sql(config, sql):
    pool = FakePool([FakeConnection(fetch_value="unexpected")])
    client = HologresClient(config, pool=pool)

    with pytest.raises(HologresSqlError):
        await client.fetch_value(sql, descriptor="row.read")

    assert pool.acquire_count == 0


@pytest.mark.asyncio
async def test_row_returning_non_read_command_replays_as_an_explicit_safe_write(config):
    first = FakeConnection(fetch_value=DatabaseFailure("08006"))
    second = FakeConnection(fetch_value="created")
    pool = FakePool([first, second])
    client = HologresClient(config, pool=pool)

    result = await client.fetch_value(
        "CREATE TABLE snapshot AS SELECT 1",
        descriptor="snapshot.create",
        operation_kind=OperationKind.WRITE,
        replay_safe=True,
    )

    assert result == "created"
    assert pool.acquire_count == 2
    assert first.terminated is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1",
        "VALUES (1)",
        "TABLE events",
        "SHOW search_path",
        "WITH source AS (SELECT 1 AS id) SELECT id FROM source",
        "WITH source(id) AS (VALUES (1)) TABLE source",
    ],
)
async def test_lexically_proven_read_commands_remain_replayable(config, sql):
    first = FakeConnection(fetch_value=DatabaseFailure("08006"))
    second = FakeConnection(fetch_value="row")
    pool = FakePool([first, second])
    client = HologresClient(config, pool=pool)

    result = await client.fetch_value(sql, descriptor="row.read")

    assert result == "row"
    assert pool.acquire_count == 2
    assert first.terminated is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sql",
    [
        "EXPLAIN ANALYZE INSERT INTO events(id) VALUES($1)",
        "EXPLAIN ANALYSE INSERT INTO events(id) VALUES($1)",
        "EXPLAIN (ANALYZE, BUFFERS TRUE) UPDATE events SET id = $1",
        "EXPLAIN (ANALYSE TRUE) UPDATE events SET id = $1",
        "EXPLAIN (COSTS OFF, ANALYZE ON, TIMING FALSE) DELETE FROM events",
        (
            "EXPLAIN (ANALYZE TRUE, FORMAT JSON) WITH changed AS ("
            "INSERT INTO events(id) VALUES($1) RETURNING id"
            ") SELECT id FROM changed"
        ),
        (
            "EXPLAIN (ANALYSE TRUE, FORMAT JSON) WITH changed AS ("
            "DELETE FROM events WHERE id = $1 RETURNING id"
            ") SELECT id FROM changed"
        ),
    ],
)
async def test_explain_analyze_dml_is_never_replayed_as_an_implicit_read(config, sql):
    first = FakeConnection(fetch_value=DatabaseFailure("08006"))
    second = FakeConnection(fetch_value="plan")
    pool = FakePool([first, second])
    client = HologresClient(config, pool=pool)

    with pytest.raises(HologresSqlError):
        await client.fetch_value(sql, "event-1", descriptor="explain.read")

    assert pool.acquire_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sql",
    [
        'EXPLAIN ("analyze" TRUE) INSERT INTO events(id) VALUES($1)',
        'EXPLAIN ("ANALYSE" ON) UPDATE events SET id = $1',
        'EXPLAIN ("Analyze" TRUE) CREATE TABLE snapshot AS SELECT 1',
    ],
)
async def test_quoted_explain_analyze_option_rejects_implicit_read_before_acquire(
    config, sql
):
    pool = FakePool([FakeConnection(fetch_value="unexpected")])
    client = HologresClient(config, pool=pool)

    with pytest.raises(HologresSqlError):
        await client.fetch_value(sql, "event-1", descriptor="explain.read")

    assert pool.acquire_count == 0


@pytest.mark.asyncio
async def test_explain_analyse_dml_is_rejected_before_a_lost_read_can_be_replayed(
    config,
):
    first = FakeConnection(fetch_value=DatabaseFailure("08006"))
    second = FakeConnection(fetch_value="plan")
    pool = FakePool([first, second])
    client = HologresClient(config, pool=pool)

    with pytest.raises(HologresSqlError):
        await client.fetch_value(
            "EXPLAIN ANALYSE DELETE FROM events WHERE id = $1",
            "event-1",
            descriptor="explain.read",
        )

    assert pool.acquire_count == 0


@pytest.mark.asyncio
async def test_explain_analyze_dml_replays_only_as_an_explicit_safe_write(config):
    first = FakeConnection(fetch_value=DatabaseFailure("08006"))
    second = FakeConnection(fetch_value="plan")
    pool = FakePool([first, second])
    client = HologresClient(config, pool=pool)

    result = await client.fetch_value(
        "EXPLAIN (ANALYZE TRUE) DELETE FROM events WHERE id = $1",
        "event-1",
        descriptor="explain.write",
        operation_kind=OperationKind.WRITE,
        replay_safe=True,
    )

    assert result == "plan"
    assert pool.acquire_count == 2
    assert first.terminated is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sql",
    [
        "EXPLAIN INSERT INTO events(id) VALUES($1)",
        "EXPLAIN (ANALYZE FALSE, FORMAT JSON) UPDATE events SET id = $1",
        "EXPLAIN (ANALYSE FALSE) DELETE FROM events",
        "EXPLAIN (ANALYZE OFF) CREATE TABLE snapshot AS SELECT 1",
        "EXPLAIN (ANALYSE NO) CREATE TABLE snapshot AS SELECT 1",
        "EXPLAIN (ANALYZE 0, COSTS OFF) DELETE FROM events",
        "EXPLAIN /* ANALYZE DELETE */ INSERT INTO events(id) VALUES($1)",
        "EXPLAIN SELECT 'ANALYZE INSERT'",
        "EXPLAIN SELECT (SELECT 'DELETE FROM events')",
        (
            "EXPLAIN (VERBOSE TRUE, COSTS OFF) WITH source AS ("
            "SELECT 'INSERT' AS value"
            ") SELECT value FROM source"
        ),
    ],
)
async def test_nonexecuting_explain_and_read_only_selects_remain_replayable_reads(
    config, sql
):
    first = FakeConnection(fetch_value=DatabaseFailure("08006"))
    second = FakeConnection(fetch_value="plan")
    pool = FakePool([first, second])
    client = HologresClient(config, pool=pool)

    result = await client.fetch_value(sql, "event-1", descriptor="explain.read")

    assert result == "plan"
    assert pool.acquire_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sql",
    [
        'EXPLAIN ("analyze" FALSE) INSERT INTO events(id) VALUES($1)',
        'EXPLAIN ("analyse" OFF) CREATE TABLE snapshot AS SELECT 1',
    ],
)
async def test_quoted_nonexecuting_explain_options_remain_replayable_reads(config, sql):
    first = FakeConnection(fetch_value=DatabaseFailure("08006"))
    second = FakeConnection(fetch_value="plan")
    pool = FakePool([first, second])
    client = HologresClient(config, pool=pool)

    result = await client.fetch_value(sql, "event-1", descriptor="explain.read")

    assert result == "plan"
    assert pool.acquire_count == 2
    assert first.terminated is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sql",
    [
        'SELECT "into" FROM events',
        'SELECT "analyze" FROM events',
    ],
)
async def test_quoted_keyword_identifiers_remain_replayable_selects(config, sql):
    first = FakeConnection(fetch_value=DatabaseFailure("08006"))
    second = FakeConnection(fetch_value="row")
    pool = FakePool([first, second])
    client = HologresClient(config, pool=pool)

    result = await client.fetch_value(sql, descriptor="row.read")

    assert result == "row"
    assert pool.acquire_count == 2
    assert first.terminated is True


@pytest.mark.asyncio
async def test_row_returning_non_idempotent_write_is_not_replayed(config):
    first = FakeConnection(fetch_value=DatabaseFailure("08006", "secret host text"))
    second = FakeConnection(fetch_value="event-1")
    pool = FakePool([first, second])
    client = HologresClient(config, pool=pool)

    with pytest.raises(HologresOperationError):
        await client.fetch_value(
            "INSERT INTO events(id) VALUES($1) RETURNING id",
            "event-1",
            descriptor="event.insert_returning",
            operation_kind=OperationKind.WRITE,
            replay_safe=False,
        )

    assert pool.acquire_count == 1
    assert first.terminated is True


@pytest.mark.asyncio
async def test_row_returning_write_replays_only_when_explicitly_safe(config):
    first = FakeConnection(fetch_value=DatabaseFailure("08006", "secret host text"))
    second = FakeConnection(fetch_value="event-1")
    pool = FakePool([first, second])
    client = HologresClient(config, pool=pool)

    result = await client.fetch_value(
        "INSERT INTO events(id) VALUES($1) RETURNING id",
        "event-1",
        descriptor="event.upsert_returning",
        operation_kind=OperationKind.WRITE,
        replay_safe=True,
    )

    assert result == "event-1"
    assert pool.acquire_count == 2
    assert first.terminated is True


@pytest.mark.asyncio
async def test_client_does_not_replay_unknown_non_idempotent_write(config):
    secret_message = "connection to password=visible-secret failed"
    first = FakeConnection(execute=DatabaseFailure("08006", secret_message))
    second = FakeConnection(execute="INSERT 0 1")
    pool = FakePool([first, second])
    client = HologresClient(config, pool=pool)

    with pytest.raises(HologresOperationError) as exc_info:
        await client.execute_one(
            "INSERT INTO events(id) VALUES($1)",
            "event-1",
            descriptor="event.insert",
            workspace="tenant-a",
        )

    assert pool.acquire_count == 1
    assert first.terminated is True
    assert pool.released_terminated == [True]
    assert secret_message not in str(exc_info.value)
    assert "visible-secret" not in repr(exc_info.value)


@pytest.mark.asyncio
async def test_client_invalidates_timed_out_non_idempotent_write(config):
    first = FakeConnection(execute=TimeoutError("password=timeout-secret"))
    second = FakeConnection(execute="INSERT 0 1")
    pool = FakePool([first, second])
    client = HologresClient(config, pool=pool)

    with pytest.raises(HologresOperationError) as exc_info:
        await client.execute_one(
            "INSERT INTO events(id) VALUES($1)",
            "event-1",
            descriptor="event.insert",
        )

    assert pool.acquire_count == 1
    assert first.terminated is True
    assert pool.released_terminated == [True]
    assert "timeout-secret" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_client_invalidates_asyncio_timed_out_write(config):
    # asyncio.TimeoutError only aliases the builtin TimeoutError from
    # Python 3.11; on 3.10 it is a distinct class, so this test raises the
    # asyncio class directly to pin the match on every supported version.
    first = FakeConnection(execute=asyncio.TimeoutError())
    second = FakeConnection(execute="INSERT 0 1")
    pool = FakePool([first, second])
    client = HologresClient(config, pool=pool)

    with pytest.raises(HologresOperationError):
        await client.execute_one(
            "INSERT INTO events(id) VALUES($1)",
            "event-1",
            descriptor="event.insert",
        )

    assert pool.acquire_count == 1
    assert first.terminated is True
    assert pool.released_terminated == [True]


@pytest.mark.asyncio
async def test_client_propagates_task_cancellation(config):
    pool = FakePool([FakeConnection(fetch_value=asyncio.CancelledError())])
    client = HologresClient(config, pool=pool)

    with pytest.raises(asyncio.CancelledError):
        await client.fetch_value("SELECT 1", descriptor="test.cancel")


@pytest.mark.asyncio
async def test_client_invalidates_cancelled_non_idempotent_write(config):
    connection = FakeConnection(execute=asyncio.CancelledError())
    pool = FakePool([connection])
    client = HologresClient(config, pool=pool)

    with pytest.raises(asyncio.CancelledError):
        await client.execute_one(
            "INSERT INTO events(id) VALUES($1)",
            "event-1",
            descriptor="event.insert",
        )

    assert connection.terminated is True
    assert pool.released_terminated == [True]


@pytest.mark.asyncio
async def test_connection_invalidation_failure_cannot_leak_driver_text(config):
    pool = FakePool(
        [
            FailingTerminateConnection(
                execute=DatabaseFailure("08006", "password=driver-secret")
            )
        ]
    )
    client = HologresClient(config, pool=pool)

    with pytest.raises(HologresOperationError) as exc_info:
        await client.execute_one(
            "INSERT INTO events(id) VALUES($1)",
            "event-1",
            descriptor="event.insert",
        )

    assert "secret" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_client_rejects_unsafe_sql_before_acquiring_a_connection(config):
    pool = FakePool([FakeConnection(fetch_value=1)])
    client = HologresClient(config, pool=pool)

    with pytest.raises(HologresSqlError):
        await client.fetch_value(
            "SELECT 1; SELECT 2", descriptor="test.invalid", workspace="tenant-a"
        )

    assert pool.acquire_count == 0


def test_client_does_not_expose_transaction_executemany_or_raw_acquire(config):
    client = HologresClient(config, pool=FakePool([]))

    assert hasattr(client, "transaction") is False
    assert hasattr(client, "executemany") is False
    assert hasattr(client, "acquire") is False
