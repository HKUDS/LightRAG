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


class PreparedStatementStub:
    def __init__(self, sql):
        self.sql = sql
        self._state = f"STATE {sql}"


class CopyProtocolStub:
    def __init__(self):
        self.copy_in_calls = []

    async def copy_in(self, statement, reader, data, records, intro_state, timeout):
        self.copy_in_calls.append(
            (statement, reader, data, records, intro_state, timeout)
        )
        return f"COPY {len(records)}"


class DataConnection:
    def __init__(self):
        self.executed = []
        self.prepared = []
        self._protocol = CopyProtocolStub()

    async def execute(self, sql, *args, timeout=None):
        self.executed.append((sql, args, timeout))
        return f"EXECUTED {len(args)}"

    async def fetchrow(self, _sql, *args, timeout=None):
        return {"bound": args[0], "timeout": timeout}

    async def fetch(self, _sql, *args, timeout=None):
        return [{"bound": value} for value in args]

    async def fetchval(self, _sql, *args, timeout=None):
        return args[0] * 2

    async def prepare(self, sql, *, timeout=None):
        self.prepared.append((sql, timeout))
        return PreparedStatementStub(sql)


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

    intro_sql = 'SELECT "id", "payload" FROM "LightRAG_1"."events" LIMIT 1'
    assert result == "COPY 1"
    assert connection.prepared == [(intro_sql, 19.0)]
    assert connection._protocol.copy_in_calls == [
        (
            'COPY "LightRAG_1"."events" ("id", "payload") FROM STDIN '
            "WITH (format binary, stream_mode true, on_conflict update)",
            None,
            None,
            (("event-1", "payload-1"),),
            f"STATE {intro_sql}",
            19.0,
        )
    ]


@pytest.mark.asyncio
async def test_stream_copy_for_probe_bypasses_gates_and_targets_probe_schema(
    base_environment,
):
    connection = DataConnection()
    pool = DataPool(connection)
    client = HologresClient(HologresConfig.from_env(base_environment), pool=pool)

    result = await client._stream_copy_for_probe(
        "lightrag_test_abc",
        "probe_table",
        ("id", "val"),
        [("conflict-key", "after"), ("fresh-key", "new")],
        descriptor="probe.streamcopy.copy",
    )

    intro_sql = 'SELECT "id", "val" FROM "lightrag_test_abc"."probe_table" LIMIT 1'
    assert result == "COPY 2"
    assert connection.prepared == [(intro_sql, 19.0)]
    assert connection._protocol.copy_in_calls == [
        (
            'COPY "lightrag_test_abc"."probe_table" ("id", "val") FROM STDIN '
            "WITH (format binary, stream_mode true, on_conflict update)",
            None,
            None,
            (("conflict-key", "after"), ("fresh-key", "new")),
            f"STATE {intro_sql}",
            19.0,
        )
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "columns",
    [(), ("id", 'payload"; DROP TABLE x; --')],
    ids=["empty", "injection"],
)
async def test_copy_rows_rejects_unsafe_column_lists_before_touching_the_pool(
    base_environment, columns
):
    pool = DataPool(DataConnection())
    client = HologresClient(
        HologresConfig.from_env(
            {**base_environment, "HOLOGRES_STREAM_COPY_ENABLED": "true"}
        ),
        pool=pool,
        capabilities=_capabilities_with_stream_copy(ProbeStatus.PASSED),
    )

    with pytest.raises(HologresSqlError):
        await client.copy_rows(
            "events", columns, [("event-1",)], descriptor="event.copy"
        )

    assert pool.acquire_count == 0


def test_stream_copy_available_requires_config_and_a_passed_probe(base_environment):
    enabled_config = HologresConfig.from_env(
        {**base_environment, "HOLOGRES_STREAM_COPY_ENABLED": "true"}
    )
    disabled_config = HologresConfig.from_env(base_environment)
    passed = _capabilities_with_stream_copy(ProbeStatus.PASSED)

    cases = [
        (disabled_config, passed, False),
        (enabled_config, None, False),
        (enabled_config, _capabilities_with_stream_copy(ProbeStatus.NOT_RUN), False),
        (enabled_config, _capabilities_with_stream_copy(ProbeStatus.FAILED), False),
        (enabled_config, passed, True),
    ]
    for config, capabilities, expected in cases:
        client = HologresClient(
            config, pool=DataPool(DataConnection()), capabilities=capabilities
        )
        assert client.stream_copy_available is expected


@pytest.mark.asyncio
async def test_call_age_procedure_runs_only_the_fixed_whitelisted_statements(
    base_environment,
):
    connection = DataConnection()
    client = HologresClient(
        HologresConfig.from_env(base_environment), pool=DataPool(connection)
    )

    await client.call_age_procedure(
        "create_graph", "lightrag_age_ws", descriptor="age.graph.create"
    )
    await client.call_age_procedure(
        "create_vlabel", "lightrag_age_ws", "Entity", descriptor="age.vlabel"
    )
    await client.call_age_procedure(
        "create_elabel", "lightrag_age_ws", "DIRECTED", descriptor="age.elabel"
    )
    await client.call_age_procedure(
        "drop_graph", "lightrag_age_ws", True, descriptor="age.graph.drop"
    )

    assert connection.executed == [
        ("CALL ag_catalog.hg_age_create_graph($1)", ("lightrag_age_ws",), 19.0),
        (
            "CALL ag_catalog.hg_age_create_vlabel($1, $2)",
            ("lightrag_age_ws", "Entity"),
            19.0,
        ),
        (
            "CALL ag_catalog.hg_age_create_elabel($1, $2)",
            ("lightrag_age_ws", "DIRECTED"),
            19.0,
        ),
        (
            "CALL ag_catalog.hg_age_drop_graph($1, $2)",
            ("lightrag_age_ws", True),
            19.0,
        ),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("procedure", "values"),
    [
        ("create_schema", ("lightrag_age_ws",)),
        ("create_graph", ()),
        ("create_graph", ("lightrag_age_ws", "extra")),
        ("drop_graph", ("lightrag_age_ws",)),
    ],
    ids=["unknown", "missing", "extra", "drop_missing_cascade"],
)
async def test_call_age_procedure_rejects_unknown_names_and_arity_before_the_pool(
    base_environment, procedure, values
):
    pool = DataPool(DataConnection())
    client = HologresClient(HologresConfig.from_env(base_environment), pool=pool)

    with pytest.raises(HologresSqlError):
        await client.call_age_procedure(
            procedure, *values, descriptor="age.lifecycle"
        )

    assert pool.acquire_count == 0


@pytest.mark.asyncio
async def test_general_sql_surface_still_forbids_call_statements(base_environment):
    pool = DataPool(DataConnection())
    client = HologresClient(HologresConfig.from_env(base_environment), pool=pool)

    with pytest.raises(HologresSqlError):
        await client.execute_one(
            "CALL ag_catalog.hg_age_create_graph($1)",
            "lightrag_age_ws",
            descriptor="age.escape",
        )

    assert pool.acquire_count == 0


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
    # RESET ALL clears the session search_path, so reset must replay it last
    # to restore the pool-wide connection invariant.
    assert statements == [
        'SET search_path TO "LightRAG_1"',
        "SET hg_experimental_enable_fixed_plan_expression = on",
        "SELECT pg_advisory_unlock_all()",
        "CLOSE ALL",
        "RESET ALL",
        "RESET application_name",
        "SET hg_experimental_enable_fixed_plan_expression = on",
        'SET search_path TO "LightRAG_1"',
    ]
    assert all(";" not in statement for statement in statements)
    assert "setup" in captured
    assert "reset" in captured


@pytest.mark.asyncio
async def test_pool_hooks_append_ag_catalog_to_search_path_when_age_enabled(
    base_environment,
):
    connection = DataConnection()

    async def pool_factory(**kwargs):
        await kwargs["setup"](connection)
        await kwargs["reset"](connection)
        return DataPool(connection)

    client = HologresClient(
        HologresConfig.from_env(
            {**base_environment, "HOLOGRES_AGE_SEARCH_PATH": "true"}
        ),
        pool_factory=pool_factory,
    )
    await client.open()

    statements = [sql for sql, _args, _timeout in connection.executed]
    # Hologres rejects a multi-name search_path, so the AGE client's
    # sessions run on ag_catalog alone.
    expected = "SET search_path TO ag_catalog"
    assert statements[0] == expected
    assert statements[-1] == expected


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
