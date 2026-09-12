from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path

import pytest

from lightrag.base import (
    CURSOR_END,
    CURSOR_START,
    CursorAfter,
    DocProcessingStatus,
    DocStatus,
    SourceAbsent,
    SourceConflict,
    SourceUnique,
)
from lightrag.exceptions import (
    SourceConflictRepairCASError,
    StorageControlPlaneError,
    StorageRecordNotFoundError,
)
from lightrag.kg.hologres.capabilities import CapabilityReport, HologresVersion
from lightrag.kg.hologres.client import HologresClientManager
from lightrag.kg.hologres.config import HologresConfig
from lightrag.kg.hologres.doc_status import (
    HologresDocStatusError,
    HologresDocStatusStorage,
    _decode_json,
    _deterministic_json,
    _ID_CHUNK_SIZE,
    _materialize_rows,
    _parse_datetime,
)
from lightrag.kg.hologres.schema import (
    DOC_STATUS_TABLE_NAME,
    doc_status_schema_descriptors,
)
from lightrag.namespace import NameSpace


CONFIG = HologresConfig(
    host="secret-host.example",
    port=80,
    user="secret-user",
    password="secret-password",
    database="secret-database",
    schema="lightrag_test_doc_status",
    connection_retries=0,
)


def make_storage(*, workspace="workspace-a", client=None, config=CONFIG):
    return HologresDocStatusStorage(
        namespace=NameSpace.DOC_STATUS,
        workspace=workspace,
        global_config={},
        embedding_func=None,
        config=config,
        client=client,
    )


def row(
    doc_id="doc-a",
    *,
    status=DocStatus.PENDING.value,
    created_at=None,
    updated_at=None,
    file_path="doc.md",
    metadata=None,
    chunks_list=None,
    **extra,
):
    created_at = created_at or datetime(2026, 1, 1, tzinfo=timezone.utc)
    updated_at = updated_at or created_at
    result = {
        "id": doc_id,
        "content_summary": "body",
        "content_length": 4,
        "file_path": file_path,
        "status": status,
        "created_at": created_at,
        "updated_at": updated_at,
        "track_id": "track-a",
        "chunks_count": 1,
        "chunks_list": ["chunk-a"] if chunks_list is None else chunks_list,
        "error_msg": None,
        "metadata": {} if metadata is None else metadata,
        "multimodal_processed": None,
        "content_hash": "hash-a",
        "extra": {},
    }
    result.update(extra)
    return result


class CallClient:
    def __init__(self, config=CONFIG):
        self.config = config
        self.calls = []
        self.handlers = {}

    async def _call(self, method, sql, values, kwargs, default):
        self.calls.append(
            {"method": method, "sql": sql, "values": values, "kwargs": kwargs}
        )
        handler = self.handlers.get(kwargs["descriptor"], default)
        if (
            isinstance(handler, list)
            and handler
            and (callable(handler[0]) or isinstance(handler[0], BaseException))
        ):
            handler = handler.pop(0)
        if isinstance(handler, BaseException):
            raise handler
        if callable(handler):
            value = handler(sql, values, kwargs)
            if inspect.isawaitable(value):
                value = await value
            return value
        return handler

    async def execute_one(self, sql, *values, **kwargs):
        return await self._call("execute_one", sql, values, kwargs, "OK")

    async def fetch_one(self, sql, *values, **kwargs):
        return await self._call("fetch_one", sql, values, kwargs, None)

    async def fetch_all(self, sql, *values, **kwargs):
        return await self._call("fetch_all", sql, values, kwargs, [])

    async def fetch_value(self, sql, *values, **kwargs):
        return await self._call("fetch_value", sql, values, kwargs, False)


class RecordLike:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, key):
        return self.values[key]


class ManagedClient:
    def __init__(self, config, *, open_gate=None):
        self.config = config
        self.open_gate = open_gate
        self.close_count = 0

    async def open(self):
        if self.open_gate is not None:
            await self.open_gate.wait()

    async def close(self):
        self.close_count += 1


def calls_for(client, descriptor):
    return [call for call in client.calls if call["kwargs"]["descriptor"] == descriptor]


@pytest.fixture
async def ready_storage(monkeypatch):
    import lightrag.kg.hologres.doc_status as module

    client = CallClient()

    async def probe(actual_client):
        assert actual_client is client
        return CapabilityReport(HologresVersion(5, 0, 0))

    class AppliedSchemaManager:
        def __init__(self, actual_client, *, schema):
            assert actual_client is client
            self.schema = schema

        async def initialize(self, descriptors):
            assert tuple(descriptors) == doc_status_schema_descriptors(self.schema)
            return ()

    monkeypatch.setattr(module, "probe_production_capabilities", probe)
    monkeypatch.setattr(module, "HologresSchemaManager", AppliedSchemaManager)
    storage = make_storage(client=client)
    await storage.initialize()
    return storage, client


# Construction, descriptor, and lifecycle.


def test_constructor_validates_namespace_workspace_and_redacts_repr():
    storage = make_storage()
    rendered = repr(storage)
    assert rendered == "HologresDocStatusStorage(<redacted>)"
    for secret in (CONFIG.host, CONFIG.user, CONFIG.password, CONFIG.database):
        assert secret not in rendered

    with pytest.raises(ValueError, match="namespace"):
        make_storage(workspace="workspace", client=object()).__class__(
            namespace="other",
            workspace="workspace",
            global_config={},
            embedding_func=None,
            config=CONFIG,
            client=object(),
        )
    with pytest.raises(ValueError, match="workspace") as exc_info:
        make_storage(workspace="../secret", client=object())
    assert "secret" not in str(exc_info.value)


def test_descriptor_is_exact_partitioned_and_single_statement():
    (descriptor, *columnar_descriptors) = doc_status_schema_descriptors(CONFIG.schema)

    assert descriptor.identity == ("doc_status", 1, 1, "shared_table")
    assert descriptor.replay_safe is True
    assert DOC_STATUS_TABLE_NAME in descriptor.sql
    assert "PRIMARY KEY (workspace, id)" in descriptor.sql
    assert "LOGICAL PARTITION BY LIST (workspace)" in descriptor.sql
    assert "orientation = 'row,column'" in descriptor.sql
    assert "status text NOT NULL" in descriptor.sql
    assert "created_at timestamptz NOT NULL" in descriptor.sql
    assert "metadata jsonb NOT NULL" in descriptor.sql
    assert "chunks_list jsonb NOT NULL" in descriptor.sql
    assert descriptor.sql.count(";") == 0
    assert "BEGIN" not in descriptor.sql.upper()
    assert "COMMIT" not in descriptor.sql.upper()
    assert "hologres.hg_table_properties" in descriptor.postcondition_sql
    assert "property_value = 'row,column'" in descriptor.postcondition_sql
    assert descriptor.postcondition_args[0:2] == (
        CONFIG.schema,
        DOC_STATUS_TABLE_NAME,
    )
    expected_columns = json.loads(descriptor.postcondition_args[2])
    assert expected_columns == [
        ["workspace", "text", True],
        ["id", "text", True],
        ["status", "text", True],
        ["created_at", "timestamptz", True],
        ["updated_at", "timestamptz", True],
        ["file_path", "text", True],
        ["track_id", "text", False],
        ["content_hash", "text", False],
        ["content_summary", "text", True],
        ["content_length", "int8", True],
        ["chunks_count", "int4", False],
        ["chunks_list", "jsonb", True],
        ["error_msg", "text", False],
        ["metadata", "jsonb", True],
        ["multimodal_processed", "bool", False],
        ["extra", "jsonb", True],
    ]
    assert json.loads(descriptor.postcondition_args[3]) == ["workspace", "id"]
    assert "distribution_key = 'id'" in descriptor.sql
    assert (
        "clustering_key = 'status,created_at,id,content_hash,file_path'"
        in descriptor.sql
    )
    assert "a.attnotnull" in descriptor.postcondition_sql
    assert "p.contype = 'p'" in descriptor.postcondition_sql

    assert [d.identity for d in columnar_descriptors] == [
        ("doc_status", 1, 2, "columnar_chunks_list"),
        ("doc_status", 1, 3, "columnar_metadata"),
        ("doc_status", 1, 4, "columnar_extra"),
    ]
    for columnar_descriptor, column in zip(
        columnar_descriptors, ("chunks_list", "metadata", "extra")
    ):
        assert columnar_descriptor.replay_safe is True
        assert columnar_descriptor.sql == (
            f'ALTER TABLE "{CONFIG.schema}"."{DOC_STATUS_TABLE_NAME}" '
            f"ALTER COLUMN {column} SET (enable_columnar_type = on)"
        )
        assert (
            "attoptions @> ARRAY['enable_columnar_type=on']"
            in columnar_descriptor.postcondition_sql
        )
        assert columnar_descriptor.postcondition_args == (
            CONFIG.schema,
            DOC_STATUS_TABLE_NAME,
            column,
        )


async def test_injected_lifecycle_is_idempotent_and_caller_owned(
    ready_storage, monkeypatch
):
    storage, client = ready_storage
    import lightrag.kg.hologres.doc_status as module

    async def forbidden_probe(_client):
        raise AssertionError("second initialize must be a no-op")

    monkeypatch.setattr(module, "probe_production_capabilities", forbidden_probe)
    await storage.initialize()
    await storage.finalize()
    await storage.finalize()
    assert not any(call["method"] == "close" for call in client.calls)


async def test_initialization_failure_releases_manager_owned_client_once(monkeypatch):
    import lightrag.kg.hologres.doc_status as module

    client = CallClient()
    releases = []

    class Manager:
        async def acquire(self, config):
            assert config == CONFIG
            return client

        async def release(self, config, actual_client):
            releases.append((config, actual_client))

    async def failing_probe(_client):
        raise RuntimeError("password=private")

    monkeypatch.setattr(module, "_SHARED_CLIENTS", Manager())
    monkeypatch.setattr(module, "probe_production_capabilities", failing_probe)
    storage = make_storage(client=None)

    with pytest.raises(RuntimeError):
        await storage.initialize()
    assert len(releases) == 1
    assert releases[0] == (CONFIG, client)
    assert "private" not in repr(storage)
    assert storage._initialized is False


async def test_manager_owned_storages_share_client_and_release_once(monkeypatch):
    import lightrag.kg.hologres.doc_status as module

    created = []

    def factory(config):
        client = ManagedClient(config)
        created.append(client)
        return client

    async def probe(_client):
        return CapabilityReport(HologresVersion(5, 0, 0))

    class AppliedSchemaManager:
        def __init__(self, _client, *, schema):
            self.schema = schema

        async def initialize(self, descriptors):
            assert tuple(descriptors) == doc_status_schema_descriptors(self.schema)

    manager = HologresClientManager(client_factory=factory)
    monkeypatch.setattr(module, "_SHARED_CLIENTS", manager)
    monkeypatch.setattr(module, "probe_production_capabilities", probe)
    monkeypatch.setattr(module, "HologresSchemaManager", AppliedSchemaManager)
    first = make_storage(workspace="workspace-a", client=None)
    second = make_storage(workspace="workspace-b", client=None)

    await asyncio.gather(first.initialize(), second.initialize())
    assert len(created) == 1
    assert manager.snapshot_for_tests() == (1, (2,))

    await first.finalize()
    assert created[0].close_count == 0
    await second.finalize()
    assert created[0].close_count == 1
    assert manager.snapshot_for_tests() == (0, ())


async def test_repeatedly_cancelled_finalize_completes_pending_release(monkeypatch):
    import lightrag.kg.hologres.doc_status as module

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
            assert tuple(descriptors) == doc_status_schema_descriptors(self.schema)

    manager = HologresClientManager(client_factory=factory)
    monkeypatch.setattr(module, "_SHARED_CLIENTS", manager)
    monkeypatch.setattr(module, "probe_production_capabilities", probe)
    monkeypatch.setattr(module, "HologresSchemaManager", AppliedSchemaManager)
    storage = make_storage(client=None)
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
    import lightrag.kg.hologres.doc_status as module

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
    monkeypatch.setattr(module, "_SHARED_CLIENTS", manager)
    monkeypatch.setattr(module, "probe_production_capabilities", probe)
    storage = make_storage(client=None)
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


# CRUD, normalization, and strict/best-effort reads.


async def test_upsert_is_bounded_replay_safe_and_preserves_created_at_sql(ready_storage):
    storage, client = ready_storage
    payload = {
        f"doc-{index}": {
            **{k: v for k, v in row(f"doc-{index}").items() if k not in {"id", "extra"}},
            "producer_extension": {"index": index},
        }
        for index in range(205)
    }

    await storage.upsert(payload)

    writes = calls_for(client, "doc_status.upsert")
    assert len(writes) == 205
    assert all(call["kwargs"]["replay_safe"] is True for call in writes)
    assert all("created_at = EXCLUDED.created_at" not in call["sql"] for call in writes)
    assert all(call["values"][0] == storage.workspace for call in writes)
    assert {call["values"][1] for call in writes} == {f"doc-{i}" for i in range(205)}


async def test_upsert_inserts_large_record_without_byte_limit_check(
    ready_storage, monkeypatch
):
    import lightrag.kg.hologres.doc_status as module

    storage, client = ready_storage
    monkeypatch.setattr(module, "_UPSERT_BYTE_LIMIT", 200)
    oversized = {
        key: value
        for key, value in row(chunks_list=["x" * 300]).items()
        if key not in {"id", "extra"}
    }

    await storage.upsert({"doc-large": oversized})
    writes = calls_for(client, "doc_status.upsert")
    assert len(writes) == 1


async def test_nullable_chunks_list_round_trips_as_jsonb_null(ready_storage):
    storage, client = ready_storage
    nullable = {
        key: value
        for key, value in row(chunks_list=[]).items()
        if key not in {"id", "extra"}
    }
    nullable["chunks_list"] = None

    await storage.upsert({"doc-null-chunks": nullable})

    write = calls_for(client, "doc_status.upsert")[-1]
    assert json.loads(write["values"][11]) is None
    assert "chunks_list = EXCLUDED.chunks_list" in write["sql"]

    client.handlers["doc_status.read.one"] = {
        **row("doc-null-chunks"),
        "chunks_list": "null",
    }
    restored = await storage.get_by_id_strict("doc-null-chunks")
    assert restored is not None
    assert restored["chunks_list"] is None


async def test_point_read_accepts_asyncpg_record_shape(ready_storage):
    storage, client = ready_storage
    client.handlers["doc_status.read.one"] = RecordLike(row())

    restored = await storage.get_by_id_strict("doc-a")

    assert restored is not None
    assert restored["status"] == DocStatus.PENDING.value


async def test_point_and_ordered_batch_reads_round_trip_json_and_timestamps(ready_storage):
    storage, client = ready_storage
    stored = row(metadata={"nested": [1]}, chunks_list=["a", "b"])
    client.handlers["doc_status.read.one"] = stored
    client.handlers["doc_status.read.ordered_batch"] = [
        {"ordinality": 1, "requested_id": "doc-a", **stored},
        {"ordinality": 2, "requested_id": "missing", "id": None},
        {"ordinality": 3, "requested_id": "doc-a", **stored},
    ]

    point = await storage.get_by_id_strict("doc-a")
    ordered = await storage.get_by_ids(["doc-a", "missing", "doc-a"])

    assert point["status"] == DocStatus.PENDING.value
    assert point["created_at"] == "2026-01-01T00:00:00+00:00"
    assert point["metadata"] == {"nested": [1]}
    assert point["chunks_list"] == ["a", "b"]
    assert ordered == [point, None, point]


@pytest.mark.parametrize("status", list(DocStatus))
async def test_full_status_decode_uses_every_enum_value(ready_storage, status):
    storage, client = ready_storage
    client.handlers["doc_status.read.statuses"] = [row(status=status.value)]

    result = await storage.get_docs_by_statuses([status], strict=True)

    assert result["doc-a"].status is status


async def test_strict_and_relaxed_status_reads_handle_corruption_but_not_transport(
    ready_storage,
):
    storage, client = ready_storage
    malformed = row(status="not-a-status")
    client.handlers["doc_status.read.statuses"] = [row(), malformed]

    with pytest.raises(HologresDocStatusError, match="corrupt"):
        await storage.get_docs_by_statuses([DocStatus.PENDING], strict=True)
    assert set(await storage.get_docs_by_statuses([DocStatus.PENDING])) == {"doc-a"}

    client.handlers["doc_status.read.statuses"] = RuntimeError("payload-secret")
    with pytest.raises(HologresDocStatusError, match="status read failed") as exc_info:
        await storage.get_docs_by_statuses([DocStatus.PENDING], strict=False)
    assert "payload-secret" not in str(exc_info.value)


async def test_strict_batch_validates_completeness_duplicates_and_projection(
    ready_storage,
):
    storage, client = ready_storage
    good = row()
    client.handlers["doc_status.read.scheduling_batch"] = [
        {"ordinality": 1, "requested_id": "doc-a", **good},
        {"ordinality": 2, "requested_id": "missing", "id": None},
    ]
    projected = await storage.get_docs_by_ids(
        ["doc-a", "doc-a", "missing"], strict=True
    )
    assert set(projected) == {"doc-a"}
    assert not hasattr(projected["doc-a"], "chunks_list")

    client.handlers["doc_status.read.full_batch"] = [
        {"ordinality": 1, "requested_id": "doc-a", **good},
        {"ordinality": 2, "requested_id": "missing", "id": None},
    ]
    full = await storage.get_full_docs_by_ids(
        ["doc-a", "doc-a", "missing"], strict=True
    )
    assert full["doc-a"].chunks_list == ["chunk-a"]

    client.handlers["doc_status.read.scheduling_batch"] = [
        {"ordinality": 1, "requested_id": "doc-a", **good}
    ]
    with pytest.raises(HologresDocStatusError, match="batch response"):
        await storage.get_docs_by_ids(["doc-a", "missing"], strict=True)

    client.handlers["doc_status.read.scheduling_batch"] = [
        {"ordinality": 1, "requested_id": "unexpected", **good}
    ]
    with pytest.raises(HologresDocStatusError, match="batch response"):
        await storage.get_docs_by_ids(["doc-a"], strict=True)


async def test_batches_are_bounded_and_a_later_failure_never_returns_partial_data(
    ready_storage,
):
    storage, client = ready_storage
    requested = [f"doc-{index}" for index in range(1001)]
    responses = [
        lambda _sql, values, _kwargs: [
            {
                "ordinality": index,
                "requested_id": doc_id,
                **row(doc_id),
            }
            for index, doc_id in enumerate(values[1], start=1)
        ],
        RuntimeError("second chunk secret"),
    ]
    client.handlers["doc_status.read.scheduling_batch"] = responses

    with pytest.raises(HologresDocStatusError, match="batch read failed") as exc_info:
        await storage.get_docs_by_ids(requested, strict=True)
    assert "second chunk secret" not in str(exc_info.value)
    assert [len(call["values"][1]) for call in calls_for(client, "doc_status.read.scheduling_batch")] == [1000, 1]


# Scheduling cursor and paging.


async def test_versioned_cursor_ties_boundaries_and_end_semantics(ready_storage):
    storage, client = ready_storage
    stamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    client.handlers["doc_status.read.page"] = [
        row("doc-a", created_at=stamp),
        row("doc-b", created_at=stamp),
    ]
    first = await storage.get_docs_by_statuses_page(
        [DocStatus.PENDING], limit=2, position=CURSOR_START, strict=True
    )
    assert list(first.docs) == ["doc-a", "doc-b"]
    assert isinstance(first.next_position, CursorAfter)
    decoded = json.loads(first.next_position.opaque)
    assert decoded == {
        "created_at": "2026-01-01T00:00:00+00:00",
        "id": "doc-b",
        "v": 1,
    }

    client.handlers["doc_status.read.page"] = []
    second = await storage.get_docs_by_statuses_page(
        [DocStatus.PENDING],
        limit=2,
        position=first.next_position,
        strict=True,
    )
    assert second.docs == {}
    assert second.next_position is CURSOR_END
    page_call = calls_for(client, "doc_status.read.page")[-1]
    assert page_call["values"][2] == stamp
    assert page_call["values"][3] == "doc-b"


@pytest.mark.parametrize(
    "opaque",
    [
        "{}",
        '{"v":2,"created_at":"2026-01-01T00:00:00+00:00","id":"a"}',
        '{"v":true,"created_at":"2026-01-01T00:00:00+00:00","id":"a"}',
        "not-json",
    ],
)
async def test_invalid_cursor_is_rejected_before_database_access(ready_storage, opaque):
    storage, client = ready_storage
    before = len(client.calls)
    with pytest.raises(StorageControlPlaneError, match="cursor"):
        await storage.get_docs_by_statuses_page(
            [DocStatus.PENDING],
            limit=1,
            position=CursorAfter(opaque),
            strict=True,
        )
    assert len(client.calls) == before


async def test_page_strictly_validates_order_scope_and_duplicates(ready_storage):
    storage, client = ready_storage
    stamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for bad_rows in (
        [row("doc-b", created_at=stamp), row("doc-a", created_at=stamp)],
        [row("doc-a", created_at=stamp), row("doc-a", created_at=stamp)],
        [row("doc-a", status=DocStatus.FAILED.value, created_at=stamp)],
        [{**row("doc-a", created_at=stamp, updated_at=stamp), "created_at": None}],
    ):
        client.handlers["doc_status.read.page"] = bad_rows
        with pytest.raises(HologresDocStatusError, match="page response"):
            await storage.get_docs_by_statuses_page(
                [DocStatus.PENDING], limit=10, strict=True
            )


async def test_relaxed_page_consumes_corrupt_row_without_claiming_end(ready_storage):
    storage, client = ready_storage
    bad = row("doc-bad")
    bad["status"] = "bad"
    client.handlers["doc_status.read.page"] = [bad]

    page = await storage.get_docs_by_statuses_page(
        [DocStatus.PENDING], limit=1, strict=False
    )
    assert page.docs == {}
    assert isinstance(page.next_position, CursorAfter)

    end = await storage.get_docs_by_statuses_page(
        [DocStatus.PENDING], limit=1, position=CURSOR_END, strict=True
    )
    assert end.next_position is CURSOR_END
    assert end.docs == {}


async def test_relaxed_page_rejects_an_impossible_null_sort_key(ready_storage):
    storage, client = ready_storage
    client.handlers["doc_status.read.page"] = [{**row("doc-bad"), "created_at": None}]

    with pytest.raises(HologresDocStatusError, match="page response"):
        await storage.get_docs_by_statuses_page(
            [DocStatus.PENDING], limit=10, strict=False
        )


# Content/source resolution and conflict repair.


async def test_content_hash_and_basename_queries_are_deterministic_and_filter_in_sql(
    ready_storage,
):
    storage, client = ready_storage
    client.handlers["doc_status.read.content_hash"] = row("doc-first")
    found = await storage.get_doc_by_content_hash(
        "hash-a", exclude_doc_id="doc-self"
    )
    assert found[0] == "doc-first"
    call = calls_for(client, "doc_status.read.content_hash")[0]
    assert call["values"] == (storage.workspace, "hash-a", "doc-self")
    assert "ORDER BY created_at ASC, id ASC" in call["sql"]
    assert "original_doc_id" in call["sql"]
    before_exclusion = call["sql"].split("AND ($3::text IS NULL", maxsplit=1)[0]
    assert "is_duplicate" in before_exclusion
    assert "LIMIT 1" in call["sql"]

    client.handlers["doc_status.read.basename"] = row("doc-primary")
    assert (await storage.get_doc_by_file_basename("doc.md"))[0] == "doc-primary"
    basename_sql = calls_for(client, "doc_status.read.basename")[0]["sql"]
    assert "is_duplicate" in basename_sql
    assert "ORDER BY created_at ASC, id ASC" in basename_sql
    assert await storage.get_doc_by_file_basename("unknown_source") is None


@pytest.mark.parametrize(
    ("rows", "expected_type"),
    [
        ([], SourceAbsent),
        ([row("doc-a", candidate_count=1)], SourceUnique),
        (
            [
                row("doc-a", candidate_count=2),
                row("doc-b", candidate_count=2),
            ],
            SourceConflict,
        ),
    ],
)
async def test_source_resolution_has_three_fail_closed_states(
    ready_storage, rows, expected_type
):
    storage, client = ready_storage
    client.handlers["doc_status.read.source"] = rows

    result = await storage.resolve_doc_source_strict("doc.md")

    assert isinstance(result, expected_type)
    if isinstance(result, SourceUnique):
        assert result.doc_id == "doc-a"
    if isinstance(result, SourceConflict):
        assert result.candidate_count == len(rows)
        assert result.sample_doc_ids == tuple(item["id"] for item in rows)

    client.handlers["doc_status.read.source"] = RuntimeError("source-secret")
    with pytest.raises(StorageControlPlaneError, match="source resolution") as exc_info:
        await storage.resolve_doc_source_strict("doc.md")
    assert "source-secret" not in str(exc_info.value)

    client.handlers["doc_status.read.source"] = None
    with pytest.raises(StorageControlPlaneError, match="response is corrupt"):
        await storage.resolve_doc_source_strict("doc.md")


@pytest.mark.parametrize(
    "rows",
    [
        [row("doc-a")],
        [row("doc-a", candidate_count=1, file_path="other.md")],
        [row("doc-a", candidate_count=2), row("doc-a", candidate_count=2)],
        [row("doc-b", candidate_count=2), row("doc-a", candidate_count=2)],
        [row("doc-a", candidate_count=3), row("doc-b", candidate_count=3)],
    ],
)
async def test_source_resolution_rejects_incomplete_or_unordered_proof(
    ready_storage, rows
):
    storage, client = ready_storage
    client.handlers["doc_status.read.source"] = rows

    with pytest.raises(StorageControlPlaneError, match="response is corrupt"):
        await storage.resolve_doc_source_strict("doc.md")


async def test_conflict_listing_uses_versioned_keyset_and_exact_counts(ready_storage):
    storage, client = ready_storage
    client.handlers["doc_status.read.conflicts"] = [
        {
            "canonical_source_key": "a.md",
            "candidate_count": 3,
            "sample_doc_ids": ["a", "b", "c"],
        }
    ]
    page = await storage.list_source_conflicts_page(limit=1)
    assert page.conflicts[0].candidate_count == 3
    assert page.conflicts[0].sample_doc_ids == ("a", "b", "c")
    assert isinstance(page.next_position, CursorAfter)
    assert json.loads(page.next_position.opaque) == {"key": "a.md", "v": 1}
    listing_sql = calls_for(client, "doc_status.read.conflicts")[-1]["sql"]
    assert "ROW_NUMBER() OVER" in listing_sql
    assert "FILTER (WHERE sample_rank <= 32)" in listing_sql
    assert ")[1:32]" not in listing_sql

    client.handlers["doc_status.read.conflicts"] = []
    end = await storage.list_source_conflicts_page(
        limit=1, position=page.next_position
    )
    assert end.next_position is CURSOR_END
    assert calls_for(client, "doc_status.read.conflicts")[-1]["values"][1] == "a.md"


async def test_conflict_repair_dry_run_commit_and_stale_cas(ready_storage):
    storage, client = ready_storage
    candidate_rows = [{"id": "doc-a"}, {"id": "doc-b"}, {"id": "doc-c"}]
    client.handlers["doc_status.repair.candidates"] = candidate_rows
    dry = await storage.repair_source_conflict(
        "doc.md",
        primary_doc_id="doc-b",
        expected_candidate_count=0,
        expected_candidate_fingerprint="ignored",
        dry_run=True,
    )
    digest = hashlib.sha256(b"doc-a\0doc-b\0doc-c\0").hexdigest()
    assert dry.fingerprint == digest
    assert dry.demoted_sample_doc_ids == ("doc-a", "doc-c")
    assert not calls_for(client, "doc_status.repair.demote")

    with pytest.raises(SourceConflictRepairCASError):
        await storage.repair_source_conflict(
            "doc.md",
            primary_doc_id="doc-b",
            expected_candidate_count=2,
            expected_candidate_fingerprint=digest,
            dry_run=False,
        )

    client.handlers["doc_status.repair.candidates"] = [
        lambda *_: candidate_rows,
        lambda *_: [{"id": "doc-b"}],
    ]
    client.handlers["doc_status.repair.demote"] = lambda _sql, values, _kwargs: {
        "id": values[1]
    }
    committed = await storage.repair_source_conflict(
        "doc.md",
        primary_doc_id="doc-b",
        expected_candidate_count=3,
        expected_candidate_fingerprint=digest,
        dry_run=False,
    )
    assert committed.committed is True
    demotions = calls_for(client, "doc_status.repair.demote")
    assert [call["values"][1] for call in demotions] == ["doc-a", "doc-c"]
    assert all(call["kwargs"]["replay_safe"] is True for call in demotions)
    assert all("is_duplicate" in call["sql"] for call in demotions)


async def test_conflict_repair_rejects_invalid_primary_and_unproven_result(
    ready_storage,
):
    storage, client = ready_storage
    client.handlers["doc_status.repair.candidates"] = [{"id": "doc-a"}]
    with pytest.raises(ValueError, match="primary_doc_id"):
        await storage.repair_source_conflict(
            "doc.md",
            primary_doc_id="not-current",
            expected_candidate_count=1,
            expected_candidate_fingerprint="x",
        )

    digest = hashlib.sha256(b"doc-a\0doc-b\0").hexdigest()
    client.handlers["doc_status.repair.candidates"] = [
        lambda *_: [{"id": "doc-a"}, {"id": "doc-b"}],
        lambda *_: [{"id": "doc-a"}, {"id": "phantom"}],
    ]
    client.handlers["doc_status.repair.demote"] = {"id": "doc-b"}
    with pytest.raises(StorageControlPlaneError, match="could not be proven"):
        await storage.repair_source_conflict(
            "doc.md",
            primary_doc_id="doc-a",
            expected_candidate_count=2,
            expected_candidate_fingerprint=digest,
            dry_run=False,
        )


# Targeted updates and primitive methods.


async def test_targeted_update_allowlist_noop_missing_and_bound_values(ready_storage):
    storage, client = ready_storage
    before = len(client.calls)
    with pytest.raises(ValueError, match="created_at"):
        await storage.update_doc_status_fields("doc-a", {"created_at": "later"})
    with pytest.raises(ValueError, match="unknown"):
        await storage.update_doc_status_fields(
            "doc-a", {"status = 'failed' --": "value"}
        )
    assert len(client.calls) == before

    client.handlers["doc_status.update"] = {"id": "doc-a"}
    metadata = {"is_duplicate": True, "original_doc_id": "primary", "keep": 1}
    await storage.update_doc_status_fields(
        "doc-a",
        {"status": DocStatus.FAILED, "metadata": metadata, "error_msg": "boom"},
    )
    update = calls_for(client, "doc_status.update")[-1]
    assert update["method"] == "fetch_one"
    assert update["kwargs"]["replay_safe"] is True
    assert "status = $3" in update["sql"]
    assert "metadata = $4::jsonb" in update["sql"]
    assert "boom" not in update["sql"]
    assert update["values"][0:2] == (storage.workspace, "doc-a")
    assert json.loads(update["values"][3]) == metadata

    await storage.update_doc_status_fields("doc-a", {"chunks_list": None})
    nullable_update = calls_for(client, "doc_status.update")[-1]
    assert nullable_update["values"][2] == "null"
    assert "chunks_list = $3::jsonb" in nullable_update["sql"]

    client.handlers["doc_status.exists"] = False
    with pytest.raises(StorageRecordNotFoundError):
        await storage.update_doc_status_fields("missing", {})
    await storage.update_doc_status_fields("missing", {}, missing_ok=True)

    client.handlers["doc_status.update"] = None
    with pytest.raises(StorageRecordNotFoundError):
        await storage.update_doc_status_fields("missing", {"error_msg": None})
    await storage.update_doc_status_fields(
        "missing", {"error_msg": None}, missing_ok=True
    )


async def test_counts_track_lookup_delete_empty_and_workspace_drop(ready_storage):
    storage, client = ready_storage
    client.handlers["doc_status.read.counts"] = [
        {"status": status.value, "count": index}
        for index, status in enumerate(DocStatus)
    ]
    counts = await storage.get_status_counts()
    assert counts == {status.value: index for index, status in enumerate(DocStatus)}
    assert (await storage.get_all_status_counts())["all"] == sum(counts.values())

    client.handlers["doc_status.read.count"] = 2
    assert (
        await storage.count_docs_by_statuses(
            [DocStatus.PENDING, DocStatus.PROCESSING], strict=True
        )
        == 2
    )

    client.handlers["doc_status.read.track"] = [row("doc-track")]
    assert set(await storage.get_docs_by_track_id("track-a")) == {"doc-track"}

    client.handlers["doc_status.empty"] = True
    assert await storage.is_empty() is True
    await storage.delete(["doc-a", "doc-b"])
    await storage.drop()
    assert calls_for(client, "doc_status.delete")[0]["kwargs"]["replay_safe"] is True
    drop = calls_for(client, "doc_status.drop")[0]
    assert drop["values"] == (storage.workspace,)
    assert drop["kwargs"]["replay_safe"] is True
    await storage.index_done_callback()


async def test_paginated_read_validates_options_and_returns_decoded_total(
    ready_storage,
):
    storage, client = ready_storage
    client.handlers["doc_status.read.paginated"] = [row("doc-a")]
    client.handlers["doc_status.read.paginated_count"] = 1

    docs, total = await storage.get_docs_paginated(
        status_filters=[DocStatus.PENDING],
        page=1,
        page_size=10,
        sort_field="created_at",
        sort_direction="asc",
    )
    assert total == 1
    assert docs[0][0] == "doc-a"
    assert isinstance(docs[0][1], DocProcessingStatus)


# Static source guard: this backend must not regain forbidden PostgreSQL mechanics.


def test_doc_status_source_uses_only_restricted_single_statement_client():
    source = Path(
        "lightrag/kg/hologres/doc_status.py"
    ).read_text(encoding="utf-8")
    lowered = source.lower()
    assert ".transaction(" not in lowered
    assert "executemany" not in lowered
    assert "asyncpg.connection" not in lowered
    assert "postgres_impl" not in lowered
    assert "advisory" not in lowered
    assert "with recursive" not in lowered


def test_doc_status_decode_helpers_cover_valid_and_corrupt_inputs():
    assert _materialize_rows((item for item in [{"id": "a"}]), "rows") == [
        {"id": "a"}
    ]
    for corrupt in (None, "rows", b"rows", 7):
        with pytest.raises(HologresDocStatusError, match="corrupt"):
            _materialize_rows(corrupt, "rows")

    assert json.loads(_deterministic_json({"b": 2, "中文": "a"})) == {
        "b": 2,
        "中文": "a",
    }
    for invalid in (float("nan"), object()):
        with pytest.raises(HologresDocStatusError, match="invalid"):
            _deterministic_json(invalid)

    assert _decode_json('{"a":1}', dict, "x") == {"a": 1}
    assert _decode_json({"a": 1}, dict, "x") == {"a": 1}
    assert _decode_json(None, dict, "x", allow_none=True) is None
    with pytest.raises(HologresDocStatusError, match="corrupt"):
        _decode_json(None, dict, "x")
    with pytest.raises(HologresDocStatusError, match="corrupt"):
        _decode_json("[]", dict, "x")
    with pytest.raises(HologresDocStatusError, match="corrupt"):
        _decode_json("{", dict, "x")

    naive = _parse_datetime("2026-01-01T00:00:00", "time")
    assert naive.tzinfo == timezone.utc
    zulu = _parse_datetime("2026-01-01T00:00:00Z", "time")
    assert zulu.utcoffset().total_seconds() == 0
    for invalid in (None, "", "not-time"):
        with pytest.raises(HologresDocStatusError, match="invalid"):
            _parse_datetime(invalid, "time")


async def test_doc_status_environment_client_is_released_when_initialization_fails(
    monkeypatch,
):
    import lightrag.kg.hologres.doc_status as module

    client = CallClient()
    releases = []

    class Manager:
        async def acquire(self, config):
            assert config == CONFIG
            return client

        async def release(self, config, actual_client):
            releases.append((config, actual_client))

    async def failing_probe(_client):
        raise RuntimeError("environment-secret")

    monkeypatch.setattr(module, "_SHARED_CLIENTS", Manager())
    monkeypatch.setattr(module, "probe_production_capabilities", failing_probe)
    monkeypatch.setattr(
        module.HologresConfig,
        "from_env",
        classmethod(lambda cls: CONFIG),
    )
    storage = make_storage(config=None, client=None)

    with pytest.raises(RuntimeError, match="environment-secret"):
        await storage.initialize()
    assert releases == [(CONFIG, client)]

    unconfigured = make_storage(client=object(), config=None)
    with pytest.raises(HologresDocStatusError, match="configuration is unavailable"):
        await unconfigured.initialize()

    uninitialized = make_storage()
    with pytest.raises(HologresDocStatusError, match="not initialized"):
        await uninitialized.get_by_id("doc-a")


async def test_doc_status_filter_keys_chunks_valid_input_and_fails_closed(
    ready_storage,
):
    storage, client = ready_storage
    assert await storage.filter_keys(set()) == set()

    keys = {f"doc-{index:03d}" for index in range(_ID_CHUNK_SIZE + 1)}

    def found(_sql, values, _kwargs):
        return [
            {"id": identifier}
            for identifier in values[1]
            if identifier != "doc-200"
        ]

    client.handlers["doc_status.filter"] = found
    assert await storage.filter_keys(keys) == {"doc-200"}
    filters = calls_for(client, "doc_status.filter")
    assert [len(call["values"][1]) for call in filters] == [_ID_CHUNK_SIZE, 1]

    client.handlers["doc_status.filter"] = RuntimeError("filter-secret")
    with pytest.raises(HologresDocStatusError, match="key filter failed") as error:
        await storage.filter_keys({"doc-a"})
    assert "filter-secret" not in str(error.value)

    client.handlers["doc_status.filter"] = [{"id": "not-requested"}]
    with pytest.raises(HologresDocStatusError, match="filter response is corrupt"):
        await storage.filter_keys({"doc-a"})


async def test_doc_status_point_batch_and_primitive_failures(ready_storage):
    storage, client = ready_storage
    assert await storage.get_by_ids([]) == []
    assert await storage.upsert({}) is None
    assert await storage.delete([]) is None

    stored = row()
    client.handlers["doc_status.read.one"] = stored
    assert (await storage.get_by_id("doc-a"))["file_path"] == "doc.md"
    client.handlers["doc_status.read.one"] = None
    assert await storage.get_by_id("missing") is None
    client.handlers["doc_status.read.one"] = RuntimeError("point-secret")
    with pytest.raises(HologresDocStatusError, match="point read failed"):
        await storage.get_by_id("doc-a")

    client.handlers["doc_status.read.ordered_batch"] = [
        {"ordinality": 1, "requested_id": "doc-a", **stored}
    ]
    assert [item["file_path"] for item in await storage.get_by_ids(["doc-a"])] == [
        "doc.md"
    ]
    client.handlers["doc_status.read.ordered_batch"] = RuntimeError("batch-secret")
    with pytest.raises(HologresDocStatusError, match="ordered batch read failed"):
        await storage.get_by_ids(["doc-a"])

    valid = {key: value for key, value in row().items() if key not in {"id", "extra"}}
    client.handlers.pop("doc_status.upsert", None)
    client.handlers["doc_status.upsert"] = RuntimeError("upsert-secret")
    with pytest.raises(HologresDocStatusError, match="upsert failed"):
        await storage.upsert({"doc-a": valid})

    client.handlers["doc_status.delete"] = RuntimeError("delete-secret")
    with pytest.raises(HologresDocStatusError, match="delete failed"):
        await storage.delete(["doc-a"])

    client.handlers["doc_status.empty"] = RuntimeError("empty-secret")
    with pytest.raises(HologresDocStatusError, match="emptiness check failed"):
        await storage.is_empty()
    client.handlers["doc_status.empty"] = "yes"
    with pytest.raises(HologresDocStatusError, match="emptiness response is corrupt"):
        await storage.is_empty()

    client.handlers["doc_status.drop"] = RuntimeError("drop-secret")
    with pytest.raises(HologresDocStatusError, match="drop failed"):
        await storage.drop()


async def test_doc_status_counts_tracks_and_lookup_helpers_cover_errors(
    ready_storage,
):
    storage, client = ready_storage
    stored = row()

    client.handlers["doc_status.read.counts"] = [
        {"status": DocStatus.PENDING.value, "count": 2}
    ]
    assert await storage.get_status_counts() == {
        **{status.value: 0 for status in DocStatus},
        DocStatus.PENDING.value: 2,
    }
    client.handlers["doc_status.read.counts"] = RuntimeError("counts-secret")
    with pytest.raises(HologresDocStatusError, match="count read failed"):
        await storage.get_status_counts()
    client.handlers["doc_status.read.counts"] = [
        {"status": "invalid", "count": 1}
    ]
    with pytest.raises(HologresDocStatusError, match="counts response is corrupt"):
        await storage.get_status_counts()

    client.handlers["doc_status.read.track"] = [stored]
    assert set(await storage.get_docs_by_track_id("track-a")) == {"doc-a"}
    client.handlers["doc_status.read.track"] = [dict(stored, status="invalid")]
    assert await storage.get_docs_by_track_id("track-a") == {}
    client.handlers["doc_status.read.track"] = RuntimeError("track-secret")
    with pytest.raises(HologresDocStatusError, match="track read failed"):
        await storage.get_docs_by_track_id("track-a")

    client.handlers["doc_status.read.file_path"] = stored
    assert (await storage.get_doc_by_file_path("doc.md"))["file_path"] == "doc.md"
    client.handlers["doc_status.read.file_path"] = None
    assert await storage.get_doc_by_file_path("missing.md") is None
    client.handlers["doc_status.read.file_path"] = RuntimeError("path-secret")
    with pytest.raises(HologresDocStatusError, match="file-path read failed"):
        await storage.get_doc_by_file_path("doc.md")

    assert await storage.get_doc_by_file_basename("") is None
    assert await storage.get_doc_by_file_basename("unknown_source") is None
    client.handlers["doc_status.read.basename"] = stored
    basename_id, basename_raw = await storage.get_doc_by_file_basename("doc.md")
    assert (basename_id, basename_raw["file_path"]) == ("doc-a", "doc.md")
    client.handlers["doc_status.read.basename"] = RuntimeError("basename-secret")
    with pytest.raises(HologresDocStatusError, match="basename read failed"):
        await storage.get_doc_by_file_basename("doc.md")

    assert await storage.get_doc_by_content_hash("") is None
    client.handlers["doc_status.read.content_hash"] = stored
    hash_id, hash_raw = await storage.get_doc_by_content_hash("hash-a")
    assert (hash_id, hash_raw["content_hash"]) == ("doc-a", "hash-a")
    client.handlers["doc_status.read.content_hash"] = None
    assert await storage.get_doc_by_content_hash("hash-a") is None
    client.handlers["doc_status.read.content_hash"] = RuntimeError("hash-secret")
    with pytest.raises(HologresDocStatusError, match="content-hash read failed"):
        await storage.get_doc_by_content_hash("hash-a")


async def test_doc_status_pagination_normalizes_options_and_fails_closed(
    ready_storage,
):
    storage, client = ready_storage
    client.handlers["doc_status.read.paginated_count"] = 1
    client.handlers["doc_status.read.paginated"] = [row()]

    docs, total = await storage.get_docs_paginated(
        status_filter=DocStatus.PENDING,
        page=0,
        page_size=1,
        sort_field="invalid",
        sort_direction="invalid",
    )
    assert total == 1
    assert docs[0][0] == "doc-a"
    count, read = (
        calls_for(client, "doc_status.read.paginated_count")[-1],
        calls_for(client, "doc_status.read.paginated")[-1],
    )
    assert count["values"][1] == [DocStatus.PENDING.value]
    assert "ORDER BY updated_at DESC, id DESC" in read["sql"]
    assert read["values"][-2:] == (10, 0)

    client.handlers["doc_status.read.paginated_count"] = RuntimeError("count-secret")
    with pytest.raises(HologresDocStatusError, match="paginated count failed"):
        await storage.get_docs_paginated()
    client.handlers["doc_status.read.paginated_count"] = "1"
    with pytest.raises(HologresDocStatusError, match="count response is corrupt"):
        await storage.get_docs_paginated()

    client.handlers["doc_status.read.paginated_count"] = 1
    client.handlers["doc_status.read.paginated"] = RuntimeError("page-secret")
    with pytest.raises(HologresDocStatusError, match="paginated read failed"):
        await storage.get_docs_paginated()
    client.handlers["doc_status.read.paginated"] = [dict(row(), id=None)]
    assert await storage.get_docs_paginated() == ([], 1)


async def test_doc_status_scheduling_and_full_batches_validate_rows(ready_storage):
    storage, client = ready_storage
    stored = row()

    def scheduling_rows(_sql, values, _kwargs):
        requested = values[1]
        return [
            {
                "ordinality": ordinal,
                "requested_id": identifier,
                **(
                    dict(stored, id=identifier)
                    if identifier == "doc-a"
                    else {"id": None}
                ),
            }
            for ordinal, identifier in enumerate(requested, start=1)
        ]

    client.handlers["doc_status.read.scheduling_batch"] = scheduling_rows
    assert set(await storage.get_docs_by_ids(["doc-a", "missing", "doc-a"])) == {
        "doc-a"
    }
    client.handlers["doc_status.read.scheduling_batch"] = []
    with pytest.raises(HologresDocStatusError, match="batch response is corrupt"):
        await storage.get_docs_by_ids(["doc-a"], strict=True)
    client.handlers["doc_status.read.scheduling_batch"] = RuntimeError(
        "scheduling-secret"
    )
    with pytest.raises(HologresDocStatusError, match="scheduling batch read failed"):
        await storage.get_docs_by_ids(["doc-a"])

    client.handlers["doc_status.read.full_batch"] = scheduling_rows
    assert set(await storage.get_full_docs_by_ids(["doc-a", "missing", "doc-a"])) == {
        "doc-a"
    }
    client.handlers["doc_status.read.full_batch"] = []
    with pytest.raises(HologresDocStatusError, match="batch response is corrupt"):
        await storage.get_full_docs_by_ids(["doc-a"], strict=True)
    client.handlers["doc_status.read.full_batch"] = RuntimeError("full-secret")
    with pytest.raises(HologresDocStatusError, match="full batch read failed"):
        await storage.get_full_docs_by_ids(["doc-a"])


@pytest.mark.parametrize(
    ("column", "value", "expected"),
    [
        ("status", DocStatus.FAILED, (DocStatus.FAILED.value, "")),
        ("status", "failed", ("failed", "")),
        (
            "updated_at",
            "2026-01-01T00:00:00",
            (datetime(2026, 1, 1, tzinfo=timezone.utc), ""),
        ),
        ("metadata", {"a": 1}, ('{"a":1}', "::jsonb")),
        ("chunks_list", None, ("null", "::jsonb")),
        ("chunks_list", ["a"], ('["a"]', "::jsonb")),
        ("content_length", 2, (2, "")),
        ("chunks_count", None, (None, "")),
        ("chunks_count", 2, (2, "")),
        ("multimodal_processed", True, (True, "")),
        ("multimodal_processed", None, (None, "")),
        ("track_id", "track", ("track", "")),
        ("track_id", None, (None, "")),
    ],
)
def test_doc_status_update_normalizer_accepts_valid_values(
    column, value, expected
):
    assert HologresDocStatusStorage._normalize_update_value(column, value) == expected


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("status", "invalid"),
        ("updated_at", "invalid"),
        ("metadata", []),
        ("chunks_list", "chunk"),
        ("content_length", None),
        ("content_length", True),
        ("chunks_count", True),
        ("multimodal_processed", 1),
        ("track_id", 3),
        ("file_path", None),
        ("content_summary", None),
        ("unknown", "value"),
    ],
)
def test_doc_status_update_normalizer_rejects_invalid_values(column, value):
    with pytest.raises(ValueError, match="Invalid|Unknown"):
        HologresDocStatusStorage._normalize_update_value(column, value)
