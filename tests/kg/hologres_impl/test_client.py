import asyncio
from contextlib import asynccontextmanager
import sys
from types import SimpleNamespace

import pytest

from lightrag.kg.hologres.capabilities import (
    CapabilityReport,
    HologresVersion,
    ProbeKind,
    ProbeResult,
    ProbeStatus,
)
from lightrag.kg.hologres.client import (
    HologresCapabilityError,
    HologresClient,
    HologresOperationError,
    HologresSqlError,
    OperationKind,
    _default_pool_factory,
)
from lightrag.kg.hologres.config import HologresConfig


class DataConnection:
    def __init__(self):
        self.executed = []

    async def execute(self, sql, *args, timeout=None):
        self.executed.append((sql, args, timeout))
        return f"EXECUTED {len(args)}"

    async def fetchrow(self, _sql, *args, timeout=None):
        return {"bound": args[0], "timeout": timeout}

    async def fetch(self, _sql, *args, timeout=None):
        return [{"bound": value} for value in args]

    async def fetchval(self, _sql, *args, timeout=None):
        return args[0] * 2

    async def copy_records_to_table(
        self, table_name, *, records, columns, schema_name, timeout
    ):
        return {
            "table": table_name,
            "records": list(records),
            "columns": tuple(columns),
            "schema": schema_name,
            "timeout": timeout,
        }


class DataPool:
    def __init__(self, connection):
        self.connection = connection
        self.acquire_count = 0
        self.closed = False
        self.terminate_count = 0

    @asynccontextmanager
    async def acquire(self, *, timeout=None):
        self.acquire_count += 1
        yield self.connection

    async def close(self):
        self.closed = True

    def terminate(self):
        self.terminate_count += 1


class FailingClosePool(DataPool):
    def __init__(self, connection, failure):
        super().__init__(connection)
        self.failure = failure

    async def close(self):
        if self.failure == "hang":
            await asyncio.Event().wait()
        raise self.failure


class FailingInitializationPool:
    def __init__(self, failure):
        self.failure = failure
        self.terminate_count = 0

    def __await__(self):
        async def initialize():
            raise self.failure

        return initialize().__await__()

    def terminate(self):
        self.terminate_count += 1


@pytest.fixture
def base_environment():
    return {
        "HOLOGRES_HOST": "example.hologres.aliyuncs.com",
        "HOLOGRES_PORT": "80",
        "HOLOGRES_USER": "test_user",
        "HOLOGRES_PASSWORD": "secret",
        "HOLOGRES_DATABASE": "analytics",
        "HOLOGRES_SCHEMA": "LightRAG_1",
        "HOLOGRES_COMMAND_TIMEOUT": "19",
    }


@pytest.mark.asyncio
async def test_parameterized_data_apis_return_database_results(base_environment):
    connection = DataConnection()
    client = HologresClient(
        HologresConfig.from_env(base_environment), pool=DataPool(connection)
    )

    assert await client.execute_one(
        "INSERT INTO events(id) VALUES($1)",
        "event-1",
        descriptor="event.insert",
        replay_safe=True,
    ) == "EXECUTED 1"
    assert await client.fetch_one(
        "SELECT $1::text AS bound", "value", descriptor="event.one"
    ) == {"bound": "value", "timeout": 19.0}
    assert await client.fetch_all(
        "SELECT value FROM unnest($1::text[]) value",
        "first",
        "second",
        descriptor="event.all",
    ) == [{"bound": "first"}, {"bound": "second"}]
    assert await client.fetch_value(
        "SELECT $1::integer * 2", 21, descriptor="event.value"
    ) == 42


@pytest.mark.asyncio
@pytest.mark.parametrize("method_name", ["fetch_one", "fetch_all", "fetch_value"])
@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO events(id) VALUES($1) RETURNING id",
        (
            "WITH created AS ("
            "INSERT INTO events(id) VALUES($1) RETURNING id"
            ") SELECT id FROM created"
        ),
        "CALL arbitrary_proc($1)",
    ],
)
async def test_row_returning_apis_reject_writes_declared_as_reads(
    base_environment, method_name, sql
):
    pool = DataPool(DataConnection())
    client = HologresClient(HologresConfig.from_env(base_environment), pool=pool)

    with pytest.raises(HologresSqlError):
        await getattr(client, method_name)(sql, "event-1", descriptor="event.read")

    assert pool.acquire_count == 0


@pytest.mark.asyncio
async def test_row_returning_write_requires_an_explicit_replay_decision(base_environment):
    pool = DataPool(DataConnection())
    client = HologresClient(HologresConfig.from_env(base_environment), pool=pool)

    with pytest.raises(HologresSqlError):
        await client.fetch_value(
            "INSERT INTO events(id) VALUES($1) RETURNING id",
            "event-1",
            descriptor="event.insert_returning",
            operation_kind=OperationKind.WRITE,
        )

    assert pool.acquire_count == 0


def _capabilities_with_stream_copy(status):
    return CapabilityReport(
        version=HologresVersion(5, 0, 0),
        results=(
            ProbeResult(
                kind=ProbeKind.STREAM_COPY,
                status=status,
                blocking=False,
                detail_code="stream_copy_probe",
            ),
        ),
    )


@pytest.mark.asyncio
async def test_copy_rows_requires_both_configuration_and_proven_capability(
    base_environment,
):
    connection = DataConnection()
    pool = DataPool(connection)
    disabled = HologresClient(
        HologresConfig.from_env(base_environment),
        pool=pool,
        capabilities=_capabilities_with_stream_copy(ProbeStatus.PASSED),
    )

    with pytest.raises(HologresCapabilityError):
        await disabled.copy_rows(
            "events", ("id",), [("event-1",)], descriptor="event.copy"
        )

    unproven = HologresClient(
        HologresConfig.from_env(
            {**base_environment, "HOLOGRES_STREAM_COPY_ENABLED": "true"}
        ),
        pool=pool,
        capabilities=_capabilities_with_stream_copy(ProbeStatus.NOT_RUN),
    )
    with pytest.raises(HologresCapabilityError):
        await unproven.copy_rows(
            "events", ("id",), [("event-1",)], descriptor="event.copy"
        )

    assert pool.acquire_count == 0


@pytest.mark.asyncio
async def test_copy_rows_uses_validated_identifiers_after_both_gates_pass(
    base_environment,
):
    connection = DataConnection()
    pool = DataPool(connection)
    client = HologresClient(
        HologresConfig.from_env(
            {**base_environment, "HOLOGRES_STREAM_COPY_ENABLED": "true"}
        ),
        pool=pool,
        capabilities=_capabilities_with_stream_copy(ProbeStatus.PASSED),
    )

    result = await client.copy_rows(
        "events",
        ("id", "payload"),
        [("event-1", "payload-1")],
        descriptor="event.copy",
    )

    assert result == {
        "table": "events",
        "records": [("event-1", "payload-1")],
        "columns": ("id", "payload"),
        "schema": "LightRAG_1",
        "timeout": 19.0,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [RuntimeError("pool initialization failed"), asyncio.CancelledError()],
    ids=["error", "cancellation"],
)
async def test_default_pool_factory_terminates_partially_initialized_pool(
    monkeypatch, failure
):
    pool = FailingInitializationPool(failure)
    fake_asyncpg = SimpleNamespace(create_pool=lambda **_kwargs: pool)
    monkeypatch.setitem(sys.modules, "asyncpg", fake_asyncpg)

    with pytest.raises(type(failure)) as exc_info:
        await _default_pool_factory(host="example.hologres.aliyuncs.com")

    assert exc_info.value is failure
    assert pool.terminate_count == 1


@pytest.mark.asyncio
async def test_pool_hooks_execute_reset_as_separate_guarded_statements(base_environment):
    connection = DataConnection()
    captured = {}

    async def pool_factory(**kwargs):
        captured.update(kwargs)
        await kwargs["setup"](connection)
        await kwargs["reset"](connection)
        return DataPool(connection)

    client = HologresClient(
        HologresConfig.from_env(base_environment), pool_factory=pool_factory
    )
    await client.open()

    statements = [sql for sql, _args, _timeout in connection.executed]
    assert statements == [
        'SET search_path TO "LightRAG_1"',
        "SET hg_experimental_enable_fixed_plan_expression = on",
        "SELECT pg_advisory_unlock_all()",
        "CLOSE ALL",
        "RESET ALL",
        "RESET application_name",
        "SET hg_experimental_enable_fixed_plan_expression = on",
    ]
    assert all(";" not in statement for statement in statements)
    assert "setup" in captured
    assert "reset" in captured


@pytest.mark.asyncio
async def test_open_and_close_are_serialized_without_leaking_a_new_pool(
    base_environment,
):
    factory_started = asyncio.Event()
    allow_factory_return = asyncio.Event()
    created_pool = DataPool(DataConnection())

    async def pool_factory(**_kwargs):
        factory_started.set()
        await allow_factory_return.wait()
        return created_pool

    client = HologresClient(
        HologresConfig.from_env(base_environment), pool_factory=pool_factory
    )
    open_task = asyncio.create_task(client.open())
    await factory_started.wait()
    close_task = asyncio.create_task(client.close())
    await asyncio.sleep(0)
    allow_factory_return.set()
    await asyncio.gather(open_task, close_task)

    assert created_pool.closed is True


@pytest.mark.asyncio
async def test_close_terminates_pool_after_ordinary_failure_and_sanitizes_error(
    base_environment,
):
    failed_pool = FailingClosePool(
        DataConnection(), RuntimeError("password=close-secret")
    )
    replacement_pool = DataPool(DataConnection())

    async def pool_factory(**_kwargs):
        return replacement_pool

    client = HologresClient(
        HologresConfig.from_env(base_environment),
        pool=failed_pool,
        pool_factory=pool_factory,
    )

    with pytest.raises(HologresOperationError) as exc_info:
        await client.close()

    assert failed_pool.terminate_count == 1
    assert "close-secret" not in str(exc_info.value)
    await client.open()
    await client.close()
    assert replacement_pool.closed is True


@pytest.mark.asyncio
async def test_close_terminates_pool_on_timeout(base_environment):
    pool = FailingClosePool(DataConnection(), "hang")
    client = HologresClient(
        HologresConfig.from_env(
            {**base_environment, "HOLOGRES_POOL_CLOSE_TIMEOUT": "0.001"}
        ),
        pool=pool,
    )

    with pytest.raises(HologresOperationError, match="timed out"):
        await client.close()

    assert pool.terminate_count == 1


@pytest.mark.asyncio
async def test_close_terminates_pool_and_propagates_cancellation(base_environment):
    pool = FailingClosePool(DataConnection(), asyncio.CancelledError())
    client = HologresClient(HologresConfig.from_env(base_environment), pool=pool)

    with pytest.raises(asyncio.CancelledError):
        await client.close()

    assert pool.terminate_count == 1
