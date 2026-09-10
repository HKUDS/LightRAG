import asyncio
from dataclasses import replace
import inspect
import json
import logging
from pathlib import Path

import pytest

from lightrag.kg.hologres.capabilities import CapabilityReport, HologresVersion
from lightrag.kg.hologres.client import (
    HologresClientManager,
    HologresOperationError,
)
from lightrag.kg.hologres.config import HologresConfig
from lightrag.kg.hologres.kv import HologresKVError, HologresKVStorage
from lightrag.kg.hologres.schema import KV_TABLE_NAME, kv_schema_descriptors
from lightrag.namespace import NameSpace


CONFIG = HologresConfig(
    host="secret-host.example",
    port=80,
    user="secret-user",
    password="secret-password",
    database="secret-database",
    schema="lightrag_test_kv",
    connection_retries=0,
)
ALLOWED_NAMESPACES = {
    NameSpace.KV_STORE_FULL_DOCS,
    NameSpace.KV_STORE_TEXT_CHUNKS,
    NameSpace.KV_STORE_LLM_RESPONSE_CACHE,
    NameSpace.KV_STORE_FULL_ENTITIES,
    NameSpace.KV_STORE_FULL_RELATIONS,
    NameSpace.KV_STORE_ENTITY_CHUNKS,
    NameSpace.KV_STORE_RELATION_CHUNKS,
}


def make_storage(*, namespace=NameSpace.KV_STORE_TEXT_CHUNKS, workspace="", client=None, config=CONFIG):
    return HologresKVStorage(
        namespace=namespace,
        workspace=workspace,
        global_config={},
        embedding_func=None,
        config=config,
        client=client,
    )


class ManagedClient:
    def __init__(self, config, *, open_gate=None, open_error=None, close_error=None):
        self.config = config
        self.open_gate = open_gate
        self.open_error = open_error
        self.close_error = close_error
        self.open_count = 0
        self.close_count = 0

    async def open(self):
        self.open_count += 1
        if self.open_gate is not None:
            await self.open_gate.wait()
        if self.open_error is not None:
            raise self.open_error

    async def close(self):
        self.close_count += 1
        if self.close_error is not None:
            raise self.close_error


class CallClient:
    def __init__(self, config=CONFIG):
        self.config = config
        self.calls = []
        self.handlers = {}
        self.close_count = 0

    async def _call(self, method, sql, values, kwargs, default):
        self.calls.append(
            {
                "method": method,
                "sql": sql,
                "values": values,
                "kwargs": kwargs,
            }
        )
        handler = self.handlers.get(kwargs["descriptor"], default)
        if isinstance(handler, BaseException):
            raise handler
        if callable(handler):
            result = handler(sql, values, kwargs)
            if inspect.isawaitable(result):
                result = await result
            return result
        return handler

    async def execute_one(self, sql, *values, **kwargs):
        return await self._call("execute_one", sql, values, kwargs, "OK")

    async def fetch_one(self, sql, *values, **kwargs):
        return await self._call("fetch_one", sql, values, kwargs, None)

    async def fetch_all(self, sql, *values, **kwargs):
        return await self._call("fetch_all", sql, values, kwargs, [])

    async def fetch_value(self, sql, *values, **kwargs):
        return await self._call("fetch_value", sql, values, kwargs, False)

    async def close(self):
        self.close_count += 1
        raise AssertionError("Injected clients are caller-owned")


@pytest.fixture
def ready_storage(monkeypatch):
    import lightrag.kg.hologres.kv as kv_module

    async def probe(client):
        return CapabilityReport(HologresVersion(5, 0, 0))

    class AppliedSchemaManager:
        def __init__(self, client, *, schema):
            self.client = client
            self.schema = schema

        async def initialize(self, descriptors):
            assert tuple(descriptors) == kv_schema_descriptors(self.schema)
            return ()

    monkeypatch.setattr(kv_module, "probe_production_capabilities", probe)
    monkeypatch.setattr(kv_module, "HologresSchemaManager", AppliedSchemaManager)

    async def factory(
        client,
        *,
        namespace=NameSpace.KV_STORE_TEXT_CHUNKS,
        workspace="workspace-a",
    ):
        storage = make_storage(
            namespace=namespace,
            workspace=workspace,
            client=client,
            config=client.config,
        )
        await storage.initialize()
        return storage

    return factory


def calls_for(client, descriptor):
    return [
        call for call in client.calls if call["kwargs"]["descriptor"] == descriptor
    ]


# ---------------------------------------------------------------------------
# Construction and namespace/workspace safety
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("namespace", sorted(ALLOWED_NAMESPACES))
def test_exact_kv_namespace_allowlist_accepts_supported_names(namespace):
    storage = make_storage(namespace=namespace)

    assert storage.namespace == namespace


@pytest.mark.parametrize(
    "namespace",
    ["doc_status", "entities", "relationships", "chunks", "unknown", ""],
)
def test_namespace_allowlist_rejects_every_other_namespace_before_database_access(
    namespace,
):
    with pytest.raises(ValueError, match="Unsupported Hologres KV namespace") as exc_info:
        make_storage(namespace=namespace, client=object())

    if namespace:
        assert namespace not in str(exc_info.value)


@pytest.mark.parametrize("workspace", [".", "..", "../escape", "bad/name", "bad\\name"])
def test_workspace_validation_precedes_database_access_and_sanitizes_errors(workspace):
    with pytest.raises(ValueError, match="Invalid Hologres KV workspace") as exc_info:
        make_storage(workspace=workspace, client=object())

    assert workspace not in str(exc_info.value)


def test_empty_workspace_remains_a_valid_empty_bound_value():
    storage = make_storage(workspace="")

    assert storage.workspace == ""


def test_storage_advertises_complete_or_raise_point_reads():
    assert HologresKVStorage.supports_strict_point_reads is True


# ---------------------------------------------------------------------------
# Shared restricted-client lifecycle
# ---------------------------------------------------------------------------


async def test_shared_manager_reuses_equivalent_configs_and_reference_counts():
    created = []

    def factory(config):
        client = ManagedClient(config)
        created.append(client)
        return client

    manager = HologresClientManager(client_factory=factory)
    first, second = await asyncio.gather(
        manager.acquire(CONFIG), manager.acquire(replace(CONFIG))
    )

    assert first is second
    assert len(created) == 1
    assert created[0].open_count == 1
    assert manager.snapshot_for_tests() == (1, (2,))

    unrelated = ManagedClient(CONFIG)
    assert await manager.release(CONFIG, unrelated) is False
    assert manager.snapshot_for_tests() == (1, (2,))
    assert await manager.release(CONFIG, first) is False
    assert created[0].close_count == 0
    assert manager.snapshot_for_tests() == (1, (1,))
    assert await manager.release(CONFIG, second) is True
    assert created[0].close_count == 1
    assert manager.snapshot_for_tests() == (0, ())


async def test_shared_manager_publishes_only_one_client_during_concurrent_open():
    open_gate = asyncio.Event()
    factory_called = asyncio.Event()
    created = []

    def factory(config):
        client = ManagedClient(config, open_gate=open_gate)
        created.append(client)
        factory_called.set()
        return client

    manager = HologresClientManager(client_factory=factory)
    acquisitions = [asyncio.create_task(manager.acquire(CONFIG)) for _ in range(8)]
    await factory_called.wait()
    await asyncio.sleep(0)

    assert len(created) == 1
    assert manager.snapshot_for_tests() == (0, ())

    open_gate.set()
    clients = await asyncio.gather(*acquisitions)
    assert all(client is created[0] for client in clients)
    assert manager.snapshot_for_tests() == (1, (8,))
    for client in clients:
        await manager.release(CONFIG, client)
    assert created[0].close_count == 1


@pytest.mark.parametrize(
    "failure",
    [RuntimeError("password=failed-open-secret"), asyncio.CancelledError()],
    ids=["error", "cancellation"],
)
async def test_shared_manager_cleans_failed_and_cancelled_opens(failure):
    attempts = []

    def factory(config):
        client = ManagedClient(config, open_error=failure if not attempts else None)
        attempts.append(client)
        return client

    manager = HologresClientManager(client_factory=factory)
    with pytest.raises(type(failure)) as exc_info:
        await manager.acquire(CONFIG)

    assert attempts[0].close_count == 1
    assert manager.snapshot_for_tests() == (0, ())
    if not isinstance(failure, asyncio.CancelledError):
        assert "failed-open-secret" not in str(exc_info.value)

    recovered = await manager.acquire(CONFIG)
    assert recovered is attempts[1]
    await manager.release(CONFIG, recovered)


async def test_shared_manager_cleans_an_open_cancelled_while_pending():
    open_gate = asyncio.Event()
    factory_called = asyncio.Event()
    attempts = []

    def factory(config):
        client = ManagedClient(
            config,
            open_gate=open_gate if not attempts else None,
        )
        attempts.append(client)
        factory_called.set()
        return client

    manager = HologresClientManager(client_factory=factory)
    acquisition = asyncio.create_task(manager.acquire(CONFIG))
    await factory_called.wait()

    acquisition.cancel()
    with pytest.raises(asyncio.CancelledError):
        await acquisition

    assert attempts[0].close_count == 1
    assert manager.snapshot_for_tests() == (0, ())
    recovered = await manager.acquire(CONFIG)
    assert recovered is attempts[1]
    await manager.release(CONFIG, recovered)


async def test_repeatedly_cancelled_finalize_completes_pending_final_release(
    monkeypatch,
):
    import lightrag.kg.hologres.kv as kv_module

    open_gate = asyncio.Event()
    blocked_factory_called = asyncio.Event()
    created = []
    blocked_config = replace(CONFIG, database="blocked-database")

    def factory(config):
        client = ManagedClient(
            config,
            open_gate=open_gate if config == blocked_config else None,
        )
        created.append(client)
        if config == blocked_config:
            blocked_factory_called.set()
        return client

    async def probe(_client):
        return CapabilityReport(HologresVersion(5, 0, 0))

    class AppliedSchemaManager:
        def __init__(self, _client, *, schema):
            self.schema = schema

        async def initialize(self, descriptors):
            assert tuple(descriptors) == kv_schema_descriptors(self.schema)

    manager = HologresClientManager(client_factory=factory)
    monkeypatch.setattr(kv_module, "_SHARED_CLIENTS", manager)
    monkeypatch.setattr(kv_module, "probe_production_capabilities", probe)
    monkeypatch.setattr(kv_module, "HologresSchemaManager", AppliedSchemaManager)
    storage = make_storage(workspace="workspace-a", client=None)
    await storage.initialize()

    blocking_acquisition = asyncio.create_task(manager.acquire(blocked_config))
    await blocked_factory_called.wait()
    finalization = asyncio.create_task(storage.finalize())
    while storage._initialized:
        await asyncio.sleep(0)

    finalization.cancel()
    await asyncio.sleep(0)
    assert not finalization.done()
    finalization.cancel()
    open_gate.set()
    with pytest.raises(asyncio.CancelledError):
        await finalization

    blocker = await blocking_acquisition
    try:
        assert created[0].close_count == 1
        assert manager.snapshot_for_tests() == (1, (1,))
    finally:
        await manager.release(blocked_config, blocker)


async def test_repeatedly_cancelled_initialization_completes_pending_release(
    monkeypatch,
):
    import lightrag.kg.hologres.kv as kv_module

    probe_started = asyncio.Event()
    probe_gate = asyncio.Event()
    open_gate = asyncio.Event()
    blocked_factory_called = asyncio.Event()
    created = []
    blocked_config = replace(CONFIG, database="blocked-database")

    def factory(config):
        client = ManagedClient(
            config,
            open_gate=open_gate if config == blocked_config else None,
        )
        created.append(client)
        if config == blocked_config:
            blocked_factory_called.set()
        return client

    async def probe(_client):
        probe_started.set()
        await probe_gate.wait()
        return CapabilityReport(HologresVersion(5, 0, 0))

    manager = HologresClientManager(client_factory=factory)
    monkeypatch.setattr(kv_module, "_SHARED_CLIENTS", manager)
    monkeypatch.setattr(kv_module, "probe_production_capabilities", probe)
    storage = make_storage(workspace="workspace-a", client=None)
    initialization = asyncio.create_task(storage.initialize())
    await probe_started.wait()

    blocking_acquisition = asyncio.create_task(manager.acquire(blocked_config))
    await blocked_factory_called.wait()
    initialization.cancel()
    await asyncio.sleep(0)
    assert not initialization.done()
    initialization.cancel()
    open_gate.set()
    with pytest.raises(asyncio.CancelledError):
        await initialization

    blocker = await blocking_acquisition
    try:
        assert created[0].close_count == 1
        assert manager.snapshot_for_tests() == (1, (1,))
    finally:
        await manager.release(blocked_config, blocker)


async def test_cancelled_finalize_chains_sanitized_release_failure(monkeypatch):
    import lightrag.kg.hologres.kv as kv_module

    release_started = asyncio.Event()
    release_gate = asyncio.Event()
    client = CallClient()

    class Manager:
        async def acquire(self, _config):
            return client

        async def release(self, _config, actual_client):
            assert actual_client is client
            release_started.set()
            await release_gate.wait()
            raise HologresOperationError("Hologres shared client close failed")

    async def probe(_client):
        return CapabilityReport(HologresVersion(5, 0, 0))

    class AppliedSchemaManager:
        def __init__(self, _client, *, schema):
            self.schema = schema

        async def initialize(self, descriptors):
            assert tuple(descriptors) == kv_schema_descriptors(self.schema)

    monkeypatch.setattr(kv_module, "_SHARED_CLIENTS", Manager())
    monkeypatch.setattr(kv_module, "probe_production_capabilities", probe)
    monkeypatch.setattr(kv_module, "HologresSchemaManager", AppliedSchemaManager)
    storage = make_storage(workspace="workspace-a", client=None)
    await storage.initialize()

    finalization = asyncio.create_task(storage.finalize())
    await release_started.wait()
    finalization.cancel()
    release_gate.set()
    with pytest.raises(asyncio.CancelledError) as exc_info:
        await finalization

    assert isinstance(exc_info.value.__cause__, HologresOperationError)
    assert str(exc_info.value.__cause__) == "Hologres shared client close failed"


async def test_shared_manager_sanitizes_factory_failure_without_publishing_entry():
    def factory(_config):
        raise RuntimeError("password=factory-secret")

    manager = HologresClientManager(client_factory=factory)

    with pytest.raises(HologresOperationError) as exc_info:
        await manager.acquire(CONFIG)

    assert "factory-secret" not in str(exc_info.value)
    assert manager.snapshot_for_tests() == (0, ())


async def test_shared_manager_removes_entry_before_a_failing_close():
    created = []

    def factory(config):
        client = ManagedClient(
            config,
            close_error=(
                RuntimeError("password=failed-close-secret") if not created else None
            ),
        )
        created.append(client)
        return client

    manager = HologresClientManager(client_factory=factory)
    client = await manager.acquire(CONFIG)

    with pytest.raises(HologresOperationError) as exc_info:
        await manager.release(CONFIG, client)

    assert "failed-close-secret" not in str(exc_info.value)
    assert manager.snapshot_for_tests() == (0, ())
    replacement = await manager.acquire(CONFIG)
    assert replacement is created[1]
    await manager.release(CONFIG, replacement)


def test_shared_manager_introspection_and_repr_never_disclose_config_values():
    manager = HologresClientManager(client_factory=ManagedClient)
    rendered = repr(manager) + repr(manager.snapshot_for_tests())

    for secret in (
        CONFIG.host,
        CONFIG.user,
        CONFIG.password,
        CONFIG.database,
    ):
        assert secret not in rendered


# ---------------------------------------------------------------------------
# Storage lifecycle and schema gate
# ---------------------------------------------------------------------------


async def test_injected_client_is_probed_and_migrated_but_never_acquired_or_closed(
    monkeypatch,
):
    import lightrag.kg.hologres.kv as kv_module

    events = []
    client = CallClient()

    class ForbiddenManager:
        async def acquire(self, _config):
            raise AssertionError("Injected client must not be acquired")

        async def release(self, _config, _client):
            raise AssertionError("Injected client must not be released")

    async def probe(actual_client):
        assert actual_client is client
        events.append("probe")
        return CapabilityReport(HologresVersion(5, 0, 0))

    class RecordingSchemaManager:
        def __init__(self, actual_client, *, schema):
            assert actual_client is client
            assert schema == CONFIG.schema
            events.append("schema-manager")

        async def initialize(self, descriptors):
            assert tuple(descriptors) == kv_schema_descriptors(CONFIG.schema)
            events.append("schema-apply")

    monkeypatch.setattr(kv_module, "_SHARED_CLIENTS", ForbiddenManager())
    monkeypatch.setattr(kv_module, "probe_production_capabilities", probe)
    monkeypatch.setattr(kv_module, "HologresSchemaManager", RecordingSchemaManager)
    storage = make_storage(client=client)

    await asyncio.gather(storage.initialize(), storage.initialize())
    await asyncio.gather(storage.finalize(), storage.finalize())

    assert events == ["probe", "schema-manager", "schema-apply"]
    assert client.close_count == 0


async def test_managed_storage_builds_env_config_acquires_once_and_releases_once(
    monkeypatch,
):
    import lightrag.kg.hologres.kv as kv_module

    events = []
    client = CallClient()

    class Manager:
        async def acquire(self, config):
            assert config is CONFIG
            events.append("acquire")
            return client

        async def release(self, config, actual_client):
            assert config is CONFIG
            assert actual_client is client
            events.append("release")
            return True

    async def probe(actual_client):
        assert actual_client is client
        events.append("probe")
        return CapabilityReport(HologresVersion(5, 0, 0))

    class SchemaManager:
        def __init__(self, actual_client, *, schema):
            assert actual_client is client
            assert schema == CONFIG.schema

        async def initialize(self, descriptors):
            assert tuple(descriptors) == kv_schema_descriptors(CONFIG.schema)
            events.append("schema")

    def from_env(_cls, environment=None):
        assert environment is None
        events.append("from-env")
        return CONFIG

    monkeypatch.setattr(HologresConfig, "from_env", classmethod(from_env))
    monkeypatch.setattr(kv_module, "_SHARED_CLIENTS", Manager())
    monkeypatch.setattr(kv_module, "probe_production_capabilities", probe)
    monkeypatch.setattr(kv_module, "HologresSchemaManager", SchemaManager)
    storage = make_storage(config=None, client=None)

    await asyncio.gather(storage.initialize(), storage.initialize())
    await asyncio.gather(storage.finalize(), storage.finalize())

    assert events == ["from-env", "acquire", "probe", "schema", "release"]


async def test_initialization_failure_releases_once_and_leaves_storage_retryable(
    monkeypatch,
):
    import lightrag.kg.hologres.kv as kv_module

    client = CallClient()
    acquisitions = 0
    releases = 0
    probe_attempts = 0

    class Manager:
        async def acquire(self, config):
            nonlocal acquisitions
            acquisitions += 1
            return client

        async def release(self, config, actual_client):
            nonlocal releases
            releases += 1
            assert actual_client is client
            return True

    async def probe(actual_client):
        nonlocal probe_attempts
        probe_attempts += 1
        if probe_attempts == 1:
            raise RuntimeError("password=probe-secret")
        return CapabilityReport(HologresVersion(5, 0, 0))

    class SchemaManager:
        def __init__(self, actual_client, *, schema):
            pass

        async def initialize(self, descriptors):
            return ()

    monkeypatch.setattr(kv_module, "_SHARED_CLIENTS", Manager())
    monkeypatch.setattr(kv_module, "probe_production_capabilities", probe)
    monkeypatch.setattr(kv_module, "HologresSchemaManager", SchemaManager)
    storage = make_storage(config=CONFIG, client=None)

    with pytest.raises(RuntimeError) as exc_info:
        await storage.initialize()
    assert "probe-secret" in str(exc_info.value)
    assert (acquisitions, releases) == (1, 1)

    await storage.initialize()
    await storage.finalize()
    assert (acquisitions, releases) == (2, 2)


def test_storage_repr_hides_injected_config_and_client_secrets():
    storage = make_storage(client=CallClient())
    rendered = repr(storage)

    for secret in (
        CONFIG.host,
        CONFIG.user,
        CONFIG.password,
        CONFIG.database,
    ):
        assert secret not in rendered
    assert "config=" not in rendered
    assert "client=" not in rendered


# ---------------------------------------------------------------------------
# Fixed physical descriptor
# ---------------------------------------------------------------------------


def test_kv_descriptor_pins_logical_hybrid_table_and_exact_catalog_postcondition():
    (descriptor, columnar_descriptor) = kv_schema_descriptors("lightrag_test_kv")

    assert KV_TABLE_NAME == "lightrag_hologres_kv"
    assert descriptor.identity == ("kv", 1, 1, "shared_table")
    assert descriptor.replay_safe is True
    assert (
        'CREATE TABLE IF NOT EXISTS "lightrag_test_kv"."lightrag_hologres_kv"'
        in descriptor.sql
    )
    for definition in (
        "workspace text NOT NULL",
        "namespace text NOT NULL",
        "id text NOT NULL",
        "payload jsonb NOT NULL",
        "updated_at timestamptz NOT NULL DEFAULT CURRENT_TIMESTAMP",
        "PRIMARY KEY (workspace, namespace, id)",
        "LOGICAL PARTITION BY LIST (workspace)",
        "orientation = 'row,column'",
        "distribution_key = 'namespace,id'",
    ):
        assert definition in descriptor.sql
    assert ";" not in descriptor.sql

    postcondition = descriptor.postcondition_sql
    for catalog in (
        "pg_catalog.pg_class",
        "pg_catalog.pg_namespace",
        "pg_catalog.pg_attribute",
        "pg_catalog.pg_type",
        "pg_catalog.pg_constraint",
    ):
        assert catalog in postcondition
    assert "attisdropped" in postcondition
    assert "jsonb_agg" in postcondition
    assert "WITH ORDINALITY" in postcondition
    assert "contype = 'p'" in postcondition
    assert "count(*)" in postcondition
    assert "hologres.hg_table_properties" in postcondition
    assert "property_value = 'row,column'" in postcondition
    assert descriptor.postcondition_args[:2] == (
        "lightrag_test_kv",
        KV_TABLE_NAME,
    )
    assert json.loads(descriptor.postcondition_args[2]) == [
        ["workspace", "text", True],
        ["namespace", "text", True],
        ["id", "text", True],
        ["payload", "jsonb", True],
        ["updated_at", "timestamptz", True],
    ]
    assert json.loads(descriptor.postcondition_args[3]) == [
        "workspace",
        "namespace",
        "id",
    ]

    assert columnar_descriptor.identity == ("kv", 1, 2, "columnar_payload")
    assert columnar_descriptor.replay_safe is True
    assert columnar_descriptor.sql == (
        'ALTER TABLE "lightrag_test_kv"."lightrag_hologres_kv" '
        "ALTER COLUMN payload SET (enable_columnar_type = on)"
    )
    assert (
        "attoptions @> ARRAY['enable_columnar_type=on']"
        in columnar_descriptor.postcondition_sql
    )
    assert "NOT a.attisdropped" in columnar_descriptor.postcondition_sql
    assert columnar_descriptor.postcondition_args == (
        "lightrag_test_kv",
        KV_TABLE_NAME,
        "payload",
    )


@pytest.mark.parametrize("schema", ['bad"schema', "bad.schema", ""])
def test_kv_descriptor_rejects_invalid_schema_before_building_sql(schema):
    with pytest.raises(Exception):
        kv_schema_descriptors(schema)


# ---------------------------------------------------------------------------
# Point and batch reads
# ---------------------------------------------------------------------------


async def test_point_reads_decode_strings_and_mappings_into_fresh_plain_dicts(
    ready_storage,
):
    client = CallClient()
    storage = await ready_storage(client)
    mapping = {"items": [], "nested": {}}
    responses = iter(
        [
            {"payload": '{"a":1,"empty":[]}'},
            {"payload": mapping},
            None,
        ]
    )
    client.handlers["kv.read.one"] = lambda *_args: next(responses)

    assert await storage.get_by_id("first") == {"a": 1, "empty": []}
    decoded = await storage.get_by_id_strict("second")
    assert decoded == mapping
    assert decoded is not mapping
    assert await storage.get_by_id("missing") is None

    for call in calls_for(client, "kv.read.one"):
        assert call["method"] == "fetch_one"
        assert call["values"][:2] == (storage.workspace, storage.namespace)
        assert call["values"][2] not in call["sql"]
        assert ";" not in call["sql"]


@pytest.mark.parametrize(
    "row",
    [
        {},
        {"payload": None},
        {"payload": "not-json"},
        {"payload": "[]"},
        {"payload": []},
    ],
)
async def test_point_read_rejects_corrupt_rows_without_disclosing_contents(
    ready_storage, row
):
    client = CallClient()
    client.handlers["kv.read.one"] = row
    storage = await ready_storage(client)

    with pytest.raises(HologresKVError, match="corrupt") as exc_info:
        await storage.get_by_id("secret-id")

    message = str(exc_info.value)
    assert "secret-id" not in message
    assert "not-json" not in message


async def test_point_read_sanitizes_transport_errors_instead_of_returning_a_miss(
    ready_storage,
):
    client = CallClient()
    client.handlers["kv.read.one"] = RuntimeError(
        "password=transport-secret id=secret-id"
    )
    storage = await ready_storage(client)

    with pytest.raises(HologresKVError) as exc_info:
        await storage.get_by_id("secret-id")

    assert "transport-secret" not in str(exc_info.value)
    assert "secret-id" not in str(exc_info.value)


async def test_ordered_batch_read_preserves_duplicates_missing_positions_and_chunks(
    ready_storage,
):
    client = CallClient()

    def rows(_sql, values, _kwargs):
        return [
            {
                "ordinality": ordinal,
                "payload": None
                if item == "missing"
                else json.dumps({"id": item}, separators=(",", ":")),
            }
            for ordinal, item in enumerate(values[3], start=1)
        ]

    client.handlers["kv.read.batch"] = rows
    storage = await ready_storage(client)

    assert await storage.get_by_ids(["a", "missing", "a"]) == [
        {"id": "a"},
        None,
        {"id": "a"},
    ]
    identifiers = [f"id-{index}" for index in range(1001)]
    result = await storage.get_by_ids(identifiers)
    assert result[0] == {"id": "id-0"}
    assert result[-1] == {"id": "id-1000"}
    chunk_calls = calls_for(client, "kv.read.batch")[-2:]
    assert [call["values"][2] for call in chunk_calls] == [1000, 1]
    assert [item for call in chunk_calls for item in call["values"][3]] == identifiers

    before = len(client.calls)
    assert await storage.get_by_ids([]) == []
    assert len(client.calls) == before


@pytest.mark.parametrize(
    "rows",
    [
        [{"ordinality": 1, "payload": "{}"}],
        [
            {"ordinality": 2, "payload": "{}"},
            {"ordinality": 1, "payload": "{}"},
        ],
        [
            {"ordinality": 1, "payload": "{}"},
            {"ordinality": 1, "payload": "{}"},
        ],
        [
            {"ordinality": "1", "payload": "{}"},
            {"ordinality": 2, "payload": "{}"},
        ],
        [
            {"ordinality": 1},
            {"ordinality": 2, "payload": "{}"},
        ],
        None,
    ],
)
async def test_ordered_batch_read_fails_closed_on_incomplete_or_malformed_rows(
    ready_storage, rows
):
    client = CallClient()
    client.handlers["kv.read.batch"] = rows
    storage = await ready_storage(client)

    with pytest.raises(HologresKVError, match="corrupt"):
        await storage.get_by_ids(["a", "b"])


async def test_filter_keys_sorts_chunks_and_fails_closed_on_malformed_results(
    ready_storage,
):
    client = CallClient()

    def existing(_sql, values, _kwargs):
        return [{"id": item} for item in values[2] if int(item.split("-")[1]) % 2 == 0]

    client.handlers["kv.filter"] = existing
    storage = await ready_storage(client)
    keys = {f"id-{index:04d}" for index in range(1001)}

    missing = await storage.filter_keys(keys)
    assert missing == {key for key in keys if int(key.split("-")[1]) % 2 == 1}
    calls = calls_for(client, "kv.filter")
    assert [len(call["values"][2]) for call in calls] == [1000, 1]
    flattened = [item for call in calls for item in call["values"][2]]
    assert flattened == sorted(keys)

    before = len(client.calls)
    assert await storage.filter_keys(set()) == set()
    assert len(client.calls) == before

    client.handlers["kv.filter"] = [{"id": "not-requested"}]
    with pytest.raises(HologresKVError, match="corrupt"):
        await storage.filter_keys({"requested"})


@pytest.mark.parametrize(
    "rows",
    [None, [{}], [{"id": None}], [{"id": "a"}, {"id": "a"}]],
)
async def test_filter_keys_rejects_uncertain_responses(ready_storage, rows):
    client = CallClient()
    client.handlers["kv.filter"] = rows
    storage = await ready_storage(client)

    with pytest.raises(HologresKVError, match="corrupt"):
        await storage.filter_keys({"a", "b"})


# ---------------------------------------------------------------------------
# Upsert semantics and bounded chunks
# ---------------------------------------------------------------------------


async def test_generic_upsert_sends_complete_objects_in_replay_safe_replacement_sql(
    ready_storage,
):
    client = CallClient()
    storage = await ready_storage(client, workspace="workspace-secret")
    data = {
        "id-secret": {"value": "payload-secret", "items": []},
        "empty": {"items": [], "mapping": {}},
    }

    await storage.upsert(data)

    (call,) = calls_for(client, "kv.upsert.replace")
    assert call["method"] == "execute_one"
    assert call["kwargs"]["replay_safe"] is True
    assert call["values"][:2] == (storage.workspace, storage.namespace)
    sent = dict(zip(call["values"][2], call["values"][3]))
    assert {k: json.loads(v) for k, v in sent.items()} == data
    assert "unnest($3::text[])" in call["sql"]
    assert "unnest($4::jsonb[])" in call["sql"]
    assert "payload = EXCLUDED.payload" in call["sql"]
    assert "updated_at = CURRENT_TIMESTAMP" in call["sql"]
    for secret in ("workspace-secret", "id-secret", "payload-secret"):
        assert secret not in call["sql"]

    before = len(client.calls)
    await storage.upsert({})
    assert len(client.calls) == before


async def test_full_docs_upsert_sql_pins_every_protected_merge_rule(ready_storage):
    client = CallClient()

    def batch_rows(_sql, values, _kwargs):
        return [
            {"ordinality": ordinal, "payload": None}
            for ordinal, _item in enumerate(values[3], start=1)
        ]

    client.handlers["kv.read.batch"] = batch_rows
    storage = await ready_storage(client, namespace=NameSpace.KV_STORE_FULL_DOCS)

    await storage.upsert(
        {
            "doc": {
                "content": "",
                "doc_name": "",
                "file_path": "",
                "sidecar_location": None,
                "parse_format": "",
                "content_hash": "hash",
                "process_options": None,
                "parse_engine": "engine",
                "chunk_options": {},
                "new_key": "new-value",
            }
        }
    )

    (call,) = calls_for(client, "kv.upsert.full_docs")
    assert call["kwargs"]["replay_safe"] is True
    assert "payload = EXCLUDED.payload" in call["sql"]
    assert "unnest($3::text[])" in call["sql"]
    assert "unnest($4::jsonb[])" in call["sql"]
    sent_payload = json.loads(call["values"][3][0])
    assert sent_payload["content"] == ""
    assert sent_payload["doc_name"] == ""
    assert sent_payload["file_path"] == ""
    assert sent_payload["new_key"] == "new-value"
    assert sent_payload["content_hash"] == "hash"
    assert sent_payload["parse_engine"] == "engine"
    assert "sidecar_location" not in sent_payload
    assert "parse_format" not in sent_payload
    assert "process_options" not in sent_payload
    assert "chunk_options" not in sent_payload


async def test_upsert_keeps_empty_tracking_and_anchor_payloads(ready_storage):
    client = CallClient()
    storage = await ready_storage(client, namespace=NameSpace.KV_STORE_ENTITY_CHUNKS)
    data = {
        "tracking": {"chunk_ids": []},
        "anchor": {"entity_names": [], "metadata": {}},
    }

    await storage.upsert(data)

    (call,) = calls_for(client, "kv.upsert.replace")
    sent = dict(zip(call["values"][2], call["values"][3]))
    assert {k: json.loads(v) for k, v in sent.items()} == data


async def test_upsert_chunks_at_200_records_and_four_mib_with_oversized_progress(
    ready_storage,
):
    client = CallClient()
    storage = await ready_storage(client)

    await storage.upsert({f"id-{index}": {"value": index} for index in range(201)})
    count_calls = calls_for(client, "kv.upsert.replace")
    assert [len(call["values"][2]) for call in count_calls] == [200, 1]

    client.calls.clear()
    large = "x" * (2 * 1024 * 1024 + 100)
    await storage.upsert({"first": {"value": large}, "second": {"value": large}})
    size_calls = calls_for(client, "kv.upsert.replace")
    assert len(size_calls) == 2
    assert all(
        sum(len(v.encode("utf-8")) for v in call["values"][3])
        <= 4 * 1024 * 1024
        for call in size_calls
    )

    client.calls.clear()
    oversized = "x" * (4 * 1024 * 1024 + 1)
    await storage.upsert({"oversized": {"value": oversized}})
    oversized_calls = calls_for(client, "kv.upsert.replace")
    assert len(oversized_calls) == 1
    assert (
        sum(len(v.encode("utf-8")) for v in oversized_calls[0]["values"][3])
        > 4 * 1024 * 1024
    )


@pytest.mark.parametrize(
    "data",
    [
        {"id": None},
        {"id": []},
        {"id": "scalar"},
        {1: {"value": "not-a-text-id"}},
    ],
)
async def test_upsert_rejects_non_object_payloads_and_non_text_ids_without_db_calls(
    ready_storage, data
):
    client = CallClient()
    storage = await ready_storage(client)
    client.calls.clear()

    with pytest.raises(HologresKVError, match="invalid"):
        await storage.upsert(data)

    assert client.calls == []


# ---------------------------------------------------------------------------
# Delete, emptiness, callback, drop, and restricted API
# ---------------------------------------------------------------------------


async def test_delete_is_bounded_scoped_replay_safe_and_empty_is_a_noop(ready_storage):
    client = CallClient()
    storage = await ready_storage(client)
    identifiers = [f"id-{index}" for index in range(1001)]

    await storage.delete(identifiers)

    calls = calls_for(client, "kv.delete")
    assert [len(call["values"][2]) for call in calls] == [1000, 1]
    assert [item for call in calls for item in call["values"][2]] == identifiers
    assert all(call["kwargs"]["replay_safe"] is True for call in calls)
    assert all(
        call["values"][:2] == (storage.workspace, storage.namespace)
        for call in calls
    )

    before = len(client.calls)
    await storage.delete([])
    assert len(client.calls) == before


async def test_is_empty_requires_a_real_boolean_response(ready_storage):
    client = CallClient()
    storage = await ready_storage(client)
    responses = iter([True, False, 1, None])
    client.handlers["kv.empty"] = lambda *_args: next(responses)

    assert await storage.is_empty() is True
    assert await storage.is_empty() is False
    with pytest.raises(HologresKVError, match="corrupt"):
        await storage.is_empty()
    with pytest.raises(HologresKVError, match="corrupt"):
        await storage.is_empty()


async def test_callback_is_noop_and_drop_deletes_only_bound_scope(ready_storage):
    client = CallClient()
    storage = await ready_storage(client)

    assert await storage.index_done_callback() is None
    assert client.calls == []
    assert await storage.drop() == {"status": "success", "message": "data dropped"}

    (call,) = calls_for(client, "kv.drop")
    assert call["method"] == "execute_one"
    assert call["values"] == (storage.workspace, storage.namespace)
    assert call["kwargs"]["replay_safe"] is True
    assert "DROP TABLE" not in call["sql"].upper()
    assert "DROP SCHEMA" not in call["sql"].upper()
    assert "WHERE workspace = $1 AND namespace = $2" in call["sql"]


async def test_drop_database_failure_raises_instead_of_returning_error_dict(ready_storage):
    client = CallClient()
    client.handlers["kv.drop"] = RuntimeError("password=drop-secret")
    storage = await ready_storage(client)

    with pytest.raises(HologresKVError) as exc_info:
        await storage.drop()

    assert "drop-secret" not in str(exc_info.value)


def test_kv_module_has_no_raw_connection_transaction_copy_or_script_escape_hatches():
    source = (
        Path(__file__).resolve().parents[3] / "lightrag/kg/hologres/kv.py"
    ).read_text(encoding="utf-8")

    for forbidden in (
        "asyncpg",
        "transaction(",
        "executemany",
        ".acquire(",
        "copy_rows",
        "COPY ",
        "BEGIN",
        "COMMIT",
        "ROLLBACK",
        "SAVEPOINT",
        "pg_advisory",
    ):
        assert forbidden not in source, forbidden


async def test_every_crud_call_uses_restricted_methods_fixed_descriptors_and_bound_values(
    ready_storage,
):
    client = CallClient()

    def batch_rows(_sql, values, _kwargs):
        return [
            {"ordinality": index, "payload": None}
            for index, _item in enumerate(values[3], start=1)
        ]

    client.handlers.update(
        {
            "kv.read.one": None,
            "kv.read.batch": batch_rows,
            "kv.filter": [],
            "kv.empty": True,
        }
    )
    storage = await ready_storage(client, workspace="workspace-secret")

    await storage.get_by_id("id-secret")
    await storage.get_by_ids(["id-secret"])
    await storage.filter_keys({"id-secret"})
    await storage.upsert({"id-secret": {"payload": "payload-secret"}})
    await storage.delete(["id-secret"])
    await storage.is_empty()
    await storage.drop()

    assert {call["method"] for call in client.calls} <= {
        "execute_one",
        "fetch_one",
        "fetch_all",
        "fetch_value",
    }
    for call in client.calls:
        descriptor = call["kwargs"]["descriptor"]
        assert descriptor.startswith("kv.")
        assert "secret" not in descriptor
        assert ";" not in call["sql"]
        assert call["values"][:2] == (storage.workspace, storage.namespace)
        for secret in (
            "workspace-secret",
            storage.namespace,
            "id-secret",
            "payload-secret",
        ):
            assert secret not in call["sql"]


async def test_errors_repr_and_logs_do_not_expose_payload_ids_or_config_secrets(
    ready_storage, caplog
):
    client = CallClient()
    client.handlers["kv.read.one"] = {"payload": "payload-secret-not-json"}
    storage = await ready_storage(client, workspace="workspace-secret")

    with caplog.at_level(logging.DEBUG, logger="lightrag"):
        with pytest.raises(HologresKVError) as exc_info:
            await storage.get_by_id("id-secret")

    rendered = str(exc_info.value) + repr(storage) + caplog.text
    for secret in (
        "payload-secret",
        "id-secret",
        "workspace-secret",
        storage.namespace,
        CONFIG.host,
        CONFIG.user,
        CONFIG.password,
        CONFIG.database,
    ):
        assert secret not in rendered
