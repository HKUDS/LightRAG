import asyncio
from datetime import datetime, timezone
import inspect
import json
import math
from pathlib import Path

import pytest

from lightrag.kg.hologres.capabilities import CapabilityReport, HologresVersion
from lightrag.kg.hologres.client import STREAM_COPY_MIN_ROWS
from lightrag.kg.hologres.config import HologresConfig
from lightrag.kg.hologres.schema import (
    VECTOR_TABLE_NAME,
    vector_schema_descriptors,
)
from lightrag.kg.hologres.vector import (
    HologresVectorError,
    HologresVectorStorage,
)
from lightrag.namespace import NameSpace
from lightrag.utils import compute_mdhash_id


CONFIG = HologresConfig(
    host="secret-vector-host.example",
    port=80,
    user="secret-vector-user",
    password="secret-vector-password",
    database="secret-vector-database",
    schema="lightrag_test_vector",
    connection_retries=0,
)
VECTOR_NAMESPACES = {
    NameSpace.VECTOR_STORE_ENTITIES,
    NameSpace.VECTOR_STORE_RELATIONSHIPS,
    NameSpace.VECTOR_STORE_CHUNKS,
}


class FakeEmbedding:
    def __init__(self, dimension=3, results=None, error=None):
        self.embedding_dim = dimension
        self.results = list(results or [])
        self.error = error
        self.calls = []

    async def __call__(self, texts, **kwargs):
        self.calls.append((list(texts), dict(kwargs)))
        if self.error is not None:
            raise self.error
        if self.results:
            return self.results.pop(0)
        return [[float(index + 1)] + [0.0] * (self.embedding_dim - 1) for index, _ in enumerate(texts)]


class CallClient:
    def __init__(self, config=CONFIG):
        self.config = config
        self.calls = []
        self.handlers = {}
        self.close_count = 0

    async def _call(self, method, sql, values, kwargs, default):
        self.calls.append(
            {"method": method, "sql": sql, "values": values, "kwargs": kwargs}
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


def make_storage(
    *,
    namespace=NameSpace.VECTOR_STORE_CHUNKS,
    workspace="workspace-a",
    embedding=None,
    client=None,
    config=CONFIG,
    batch_size=2,
    threshold=0.2,
):
    return HologresVectorStorage(
        namespace=namespace,
        workspace=workspace,
        global_config={
            "embedding_batch_num": batch_size,
            "vector_db_storage_cls_kwargs": {
                "cosine_better_than_threshold": threshold
            },
        },
        embedding_func=embedding or FakeEmbedding(),
        meta_fields={"content", "full_doc_id", "file_path", "src_id", "tgt_id"},
        config=config,
        client=client,
    )


@pytest.fixture
def ready_storage(monkeypatch):
    import lightrag.kg.hologres.vector as vector_module

    async def probe(client):
        return CapabilityReport(HologresVersion(5, 0, 0))

    class AppliedSchemaManager:
        def __init__(self, client, *, schema):
            self.client = client
            self.schema = schema

        async def initialize(self, descriptors):
            assert tuple(descriptors) == vector_schema_descriptors(self.schema, 3)
            return ()

    monkeypatch.setattr(vector_module, "probe_production_capabilities", probe)
    monkeypatch.setattr(vector_module, "HologresSchemaManager", AppliedSchemaManager)

    async def factory(client, **kwargs):
        storage = make_storage(client=client, config=client.config, **kwargs)
        await storage.initialize()
        return storage

    return factory


def calls_for(client, descriptor):
    return [call for call in client.calls if call["kwargs"]["descriptor"] == descriptor]


# Construction and schema -----------------------------------------------------


@pytest.mark.parametrize("namespace", sorted(VECTOR_NAMESPACES))
def test_exact_vector_namespace_allowlist(namespace):
    assert make_storage(namespace=namespace).namespace == namespace


@pytest.mark.parametrize("namespace", ["full_docs", "doc_status", "unknown", ""])
def test_unsupported_namespace_is_rejected_before_database_access(namespace):
    with pytest.raises(ValueError, match="Unsupported Hologres vector namespace"):
        make_storage(namespace=namespace, client=object())


@pytest.mark.parametrize("workspace", [".", "..", "../escape", "bad/name", "bad\\name"])
def test_invalid_workspace_is_rejected_without_echoing_value(workspace):
    with pytest.raises(ValueError, match="Invalid Hologres vector workspace") as exc_info:
        make_storage(workspace=workspace, client=object())
    assert workspace not in str(exc_info.value)


def test_embedding_function_is_required_and_validated_first():
    with pytest.raises(ValueError, match="embedding_func is required"):
        HologresVectorStorage(
            namespace="unsupported",
            workspace="workspace-a",
            global_config={},
            embedding_func=None,
        )


@pytest.mark.parametrize("dimension", [True, False, 0, -1, 1.5, "3"])
def test_embedding_dimension_must_be_a_positive_integer(dimension):
    with pytest.raises(ValueError, match="embedding dimension"):
        make_storage(embedding=FakeEmbedding(dimension=dimension), client=object())


@pytest.mark.parametrize("batch_size", [True, False, 0, -1, 1.5, "2"])
def test_embedding_batch_size_must_be_a_positive_integer(batch_size):
    with pytest.raises(ValueError, match="embedding batch"):
        make_storage(batch_size=batch_size, client=object())


@pytest.mark.parametrize("threshold", [True, False, math.nan, math.inf, -1.1, 1.1, "0.2"])
def test_cosine_threshold_must_be_finite_and_in_cosine_range(threshold):
    with pytest.raises(ValueError, match="cosine threshold"):
        make_storage(threshold=threshold, client=object())


def test_repr_redacts_configuration_and_runtime_state():
    rendered = repr(make_storage(client=CallClient()))
    assert rendered == "HologresVectorStorage(<redacted>)"
    assert "secret-vector" not in rendered


@pytest.mark.parametrize("dimension", [1, 3, 1536])
def test_vector_descriptor_is_dimension_specific_and_carries_frozen_hgraph_index(dimension):
    (descriptor, columnar_descriptor) = vector_schema_descriptors(
        CONFIG.schema, dimension
    )
    normalized = " ".join(descriptor.sql.split()).lower()

    assert descriptor.component == "vector"
    assert descriptor.replay_safe is True
    assert f'"{CONFIG.schema}"."{VECTOR_TABLE_NAME}"' in descriptor.sql
    assert "embedding float4[] not null" in normalized
    assert f"array_length(embedding, 1) = {dimension}" in normalized
    assert (
        f"check (array_ndims(embedding) = 1 and array_length(embedding, 1) = {dimension})"
        in normalized
    )
    assert "orientation = 'column'" in normalized
    assert "logical partition" not in normalized
    # The vectors table property must stay byte-identical to the live
    # HGraph probe that froze the score contract.
    assert (
        'vectors = \'{"embedding":{"algorithm":"HGraph",'
        '"distance_method":"Cosine","builder_params":{"max_degree":64,'
        '"ef_construction":400,"base_quantization_type":"fp32",'
        '"precise_quantization_type":"fp32","use_reorder":true}}}\''
        in descriptor.sql
    )
    assert "extra_columns" not in descriptor.sql
    assert len(descriptor.postcondition_args) == 4
    assert "contype = 'p'" in descriptor.postcondition_sql
    assert "contype = 'c'" not in descriptor.postcondition_sql
    assert "hologres.hg_table_properties" in descriptor.postcondition_sql
    assert "property_key = 'vectors'" in descriptor.postcondition_sql
    assert "property_value = 'column'" in descriptor.postcondition_sql
    assert "= 'HGraph'" in descriptor.postcondition_sql
    assert "= 'Cosine'" in descriptor.postcondition_sql

    assert columnar_descriptor.identity == ("vector", 1, 2, "columnar_payload")
    assert columnar_descriptor.replay_safe is True
    assert columnar_descriptor.sql == (
        f'ALTER TABLE "{CONFIG.schema}"."{VECTOR_TABLE_NAME}" '
        "ALTER COLUMN payload SET (enable_columnar_type = on)"
    )
    assert (
        "attoptions @> ARRAY['enable_columnar_type=on']"
        in columnar_descriptor.postcondition_sql
    )
    assert columnar_descriptor.postcondition_args == (
        CONFIG.schema,
        VECTOR_TABLE_NAME,
        "payload",
    )


def test_vector_descriptor_digest_changes_with_embedding_dimension():
    first = vector_schema_descriptors(CONFIG.schema, 3)[0]
    second = vector_schema_descriptors(CONFIG.schema, 4)[0]
    assert first.identity == second.identity
    assert first.digest != second.digest


@pytest.mark.parametrize("dimension", [True, 0, -1, 2.5, "3"])
def test_vector_descriptor_rejects_invalid_dimension(dimension):
    with pytest.raises(Exception, match="dimension"):
        vector_schema_descriptors(CONFIG.schema, dimension)


# Lifecycle ------------------------------------------------------------------


async def test_injected_client_initialization_is_idempotent_and_caller_owned(ready_storage):
    client = CallClient()
    storage = await ready_storage(client)

    await storage.initialize()
    await storage.finalize()
    await storage.finalize()

    assert client.close_count == 0


async def test_initialization_failure_leaves_storage_unready_and_sanitizes_error(monkeypatch):
    import lightrag.kg.hologres.vector as vector_module

    client = CallClient()

    async def failed_probe(_client):
        raise RuntimeError("password=probe-secret")

    monkeypatch.setattr(vector_module, "probe_production_capabilities", failed_probe)
    storage = make_storage(client=client)
    with pytest.raises(HologresVectorError) as exc_info:
        await storage.initialize()

    assert "probe-secret" not in str(exc_info.value)
    with pytest.raises(HologresVectorError, match="not initialized"):
        await storage.get_by_id("id")


async def test_shared_client_ownership_reuses_manager_and_reference_counts(monkeypatch):
    import lightrag.kg.hologres.vector as vector_module

    client = CallClient()

    class Manager:
        def __init__(self):
            self.references = 0
            self.acquire_count = 0
            self.release_count = 0

        async def acquire(self, config):
            assert config == CONFIG
            self.acquire_count += 1
            self.references += 1
            return client

        async def release(self, config, actual):
            assert config == CONFIG and actual is client
            self.release_count += 1
            self.references -= 1
            return self.references == 0

    async def probe(_client):
        return CapabilityReport(HologresVersion(5, 0, 0))

    class SchemaManager:
        def __init__(self, _client, *, schema):
            self.schema = schema

        async def initialize(self, descriptors):
            assert tuple(descriptors) == vector_schema_descriptors(self.schema, 3)

    manager = Manager()
    monkeypatch.setattr(vector_module, "_SHARED_CLIENTS", manager)
    monkeypatch.setattr(vector_module, "probe_production_capabilities", probe)
    monkeypatch.setattr(vector_module, "HologresSchemaManager", SchemaManager)

    first = make_storage(client=None)
    second = make_storage(client=None)
    await asyncio.gather(first.initialize(), second.initialize())
    assert manager.acquire_count == 2
    assert manager.references == 2

    await first.finalize()
    assert manager.references == 1
    await second.finalize()
    assert manager.references == 0
    assert manager.release_count == 2


async def test_shared_client_is_released_when_schema_initialization_fails(monkeypatch):
    import lightrag.kg.hologres.vector as vector_module

    client = CallClient()

    class Manager:
        release_count = 0

        async def acquire(self, _config):
            return client

        async def release(self, _config, actual):
            assert actual is client
            self.release_count += 1

    async def probe(_client):
        return CapabilityReport(HologresVersion(5, 0, 0))

    class FailedSchemaManager:
        def __init__(self, _client, *, schema):
            self.schema = schema

        async def initialize(self, _descriptors):
            raise RuntimeError("password=schema-secret")

    manager = Manager()
    monkeypatch.setattr(vector_module, "_SHARED_CLIENTS", manager)
    monkeypatch.setattr(vector_module, "probe_production_capabilities", probe)
    monkeypatch.setattr(vector_module, "HologresSchemaManager", FailedSchemaManager)

    storage = make_storage(client=None)
    with pytest.raises(HologresVectorError) as exc_info:
        await storage.initialize()
    assert manager.release_count == 1
    assert "schema-secret" not in str(exc_info.value)


async def test_initialization_failure_preserves_sanitized_release_failure(monkeypatch):
    import lightrag.kg.hologres.vector as vector_module

    client = CallClient()

    class Manager:
        async def acquire(self, _config):
            return client

        async def release(self, _config, actual):
            assert actual is client
            raise RuntimeError("password=release-secret")

    async def failed_probe(_client):
        raise RuntimeError("password=probe-secret")

    monkeypatch.setattr(vector_module, "_SHARED_CLIENTS", Manager())
    monkeypatch.setattr(vector_module, "probe_production_capabilities", failed_probe)
    storage = make_storage(client=None)

    with pytest.raises(HologresVectorError) as exc_info:
        await storage.initialize()

    assert str(exc_info.value) == "Hologres vector initialization failed"
    assert isinstance(exc_info.value.__cause__, HologresVectorError)
    assert str(exc_info.value.__cause__) == "Hologres vector shared client release failed"
    assert "secret" not in repr(exc_info.value)
    assert "secret" not in repr(exc_info.value.__cause__)


async def test_repeatedly_cancelled_finalize_completes_shared_release(monkeypatch):
    import lightrag.kg.hologres.vector as vector_module

    client = CallClient()
    release_started = asyncio.Event()
    release_gate = asyncio.Event()
    release_finished = asyncio.Event()

    class Manager:
        async def acquire(self, _config):
            return client

        async def release(self, _config, actual):
            assert actual is client
            release_started.set()
            await release_gate.wait()
            release_finished.set()

    async def probe(_client):
        return CapabilityReport(HologresVersion(5, 0, 0))

    class AppliedSchemaManager:
        def __init__(self, _client, *, schema):
            self.schema = schema

        async def initialize(self, descriptors):
            assert tuple(descriptors) == vector_schema_descriptors(self.schema, 3)

    monkeypatch.setattr(vector_module, "_SHARED_CLIENTS", Manager())
    monkeypatch.setattr(vector_module, "probe_production_capabilities", probe)
    monkeypatch.setattr(vector_module, "HologresSchemaManager", AppliedSchemaManager)
    storage = make_storage(client=None)
    await storage.initialize()

    finalization = asyncio.create_task(storage.finalize())
    await release_started.wait()
    finalization.cancel()
    await asyncio.sleep(0)
    assert not finalization.done()
    finalization.cancel()
    release_gate.set()

    with pytest.raises(asyncio.CancelledError):
        await finalization
    assert release_finished.is_set()
    assert storage._initialized is False


async def test_repeatedly_cancelled_initialization_completes_shared_release(
    monkeypatch,
):
    import lightrag.kg.hologres.vector as vector_module

    client = CallClient()
    probe_started = asyncio.Event()
    probe_gate = asyncio.Event()
    release_started = asyncio.Event()
    release_gate = asyncio.Event()
    release_finished = asyncio.Event()

    class Manager:
        async def acquire(self, _config):
            return client

        async def release(self, _config, actual):
            assert actual is client
            release_started.set()
            await release_gate.wait()
            release_finished.set()

    async def probe(_client):
        probe_started.set()
        await probe_gate.wait()
        return CapabilityReport(HologresVersion(5, 0, 0))

    monkeypatch.setattr(vector_module, "_SHARED_CLIENTS", Manager())
    monkeypatch.setattr(vector_module, "probe_production_capabilities", probe)
    storage = make_storage(client=None)
    initialization = asyncio.create_task(storage.initialize())
    await probe_started.wait()

    initialization.cancel()
    await release_started.wait()
    initialization.cancel()
    await asyncio.sleep(0)
    assert not initialization.done()
    release_gate.set()

    with pytest.raises(asyncio.CancelledError):
        await initialization
    assert release_finished.is_set()
    assert storage._initialized is False


# Embedding and upsert --------------------------------------------------------


async def test_empty_upsert_is_a_noop_even_before_initialize():
    storage = make_storage(client=object())
    await storage.upsert({})


async def test_upsert_computes_missing_embeddings_in_bounded_batches(ready_storage):
    embedding = FakeEmbedding(
        results=[
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0]],
        ]
    )
    client = CallClient()
    storage = await ready_storage(client, embedding=embedding, batch_size=2)

    await storage.upsert(
        {
            "a": {"content": "alpha", "nested": {"keep": [1, None]}},
            "b": {"content": "beta", "extension": True},
            "c": {"content": "gamma", "number": 7},
        }
    )

    assert [call[0] for call in embedding.calls] == [
        ["alpha", "beta"],
        ["gamma"],
    ]
    assert all(call[1]["context"] == "document" for call in embedding.calls)
    (write,) = calls_for(client, "vector.upsert")
    assert write["method"] == "execute_one"
    assert write["values"][:2] == ("workspace-a", NameSpace.VECTOR_STORE_CHUNKS)
    ids = write["values"][2]
    embeddings = write["values"][3]
    contents = write["values"][4]
    payloads = write["values"][5]
    count = write["values"][6]
    assert ids == ["a", "b", "c"]
    assert count == 3
    assert embeddings[0] == "{1.0,0.0,0.0}"
    assert embeddings[1] == "{0.0,1.0,0.0}"
    assert embeddings[2] == "{0.0,0.0,1.0}"
    assert contents == ["alpha", "beta", "gamma"]
    assert json.loads(payloads[0]) == {
        "content": "alpha",
        "nested": {"keep": [1, None]},
    }
    assert json.loads(payloads[1]) == {"content": "beta", "extension": True}
    assert json.loads(payloads[2]) == {"content": "gamma", "number": 7}
    assert write["kwargs"]["replay_safe"] is True
    normalized_sql = " ".join(write["sql"].split()).lower()
    assert "on conflict (workspace, namespace, id) do update" in normalized_sql
    assert "embedding = excluded.embedding" in normalized_sql
    assert "content = excluded.content" in normalized_sql
    assert "payload = excluded.payload" in normalized_sql
    assert "generate_series(1, $7::int)" in normalized_sql


async def test_upsert_accepts_supplied_embeddings_without_calling_provider(ready_storage):
    embedding = FakeEmbedding(error=AssertionError("must not embed"))
    client = CallClient()
    storage = await ready_storage(client, embedding=embedding)

    await storage.upsert(
        {
            "a": {"content": "alpha", "embedding": [1, 0, 0], "keep": "yes"},
            "b": {"content": "beta", "__vector__": (0.0, 1.0, 0.0)},
        }
    )

    assert embedding.calls == []
    write = calls_for(client, "vector.upsert")[0]
    ids = write["values"][2]
    payloads = write["values"][5]
    assert ids == ["a", "b"]
    assert json.loads(payloads[0]) == {"content": "alpha", "keep": "yes"}
    assert json.loads(payloads[1]) == {"content": "beta"}


class StreamCopyCallClient(CallClient):
    def __init__(self, config=CONFIG):
        super().__init__(config)
        self.stream_copy_available = True
        self.copies = []

    async def copy_rows(
        self, table, columns, records, *, descriptor, replay_safe=False, timeout=None
    ):
        self.copies.append(
            {
                "table": table,
                "columns": tuple(columns),
                "records": [tuple(record) for record in records],
                "descriptor": descriptor,
                "replay_safe": replay_safe,
            }
        )
        return f"COPY {len(self.copies[-1]['records'])}"


def _bulk_vector_data(count):
    return {
        f"id-{index:04d}": {
            "content": f"content-{index}",
            "embedding": [float(index), 0.0, 0.0],
        }
        for index in range(count)
    }


async def test_bulk_upsert_routes_through_stream_copy_when_gated_and_large(
    ready_storage,
):
    client = StreamCopyCallClient()
    storage = await ready_storage(client)
    data = _bulk_vector_data(STREAM_COPY_MIN_ROWS)

    await storage.upsert(data)

    assert calls_for(client, "vector.upsert") == []
    (copy,) = client.copies
    assert copy["table"] == VECTOR_TABLE_NAME
    assert copy["columns"] == (
        "workspace",
        "namespace",
        "id",
        "embedding",
        "content",
        "payload",
        "updated_at",
    )
    assert copy["descriptor"] == "vector.upsert"
    assert copy["replay_safe"] is True
    assert len(copy["records"]) == len(data)
    for row in copy["records"]:
        workspace, namespace, identifier, embedding, content, payload, updated_at = row
        assert workspace == storage.workspace
        assert namespace == storage.namespace
        assert embedding == data[identifier]["embedding"]
        assert all(isinstance(value, float) for value in embedding)
        assert content == data[identifier]["content"]
        assert json.loads(payload) == {"content": data[identifier]["content"]}
        assert isinstance(updated_at, datetime)
        assert updated_at.tzinfo is timezone.utc


async def test_bulk_upsert_below_threshold_keeps_parameterized_insert(ready_storage):
    client = StreamCopyCallClient()
    storage = await ready_storage(client)
    data = _bulk_vector_data(STREAM_COPY_MIN_ROWS - 1)

    await storage.upsert(data)

    assert client.copies == []
    (write,) = calls_for(client, "vector.upsert")
    assert write["method"] == "execute_one"


async def test_bulk_upsert_without_proven_capability_keeps_parameterized_insert(
    ready_storage,
):
    client = CallClient()
    storage = await ready_storage(client)
    data = _bulk_vector_data(STREAM_COPY_MIN_ROWS)

    await storage.upsert(data)

    (write,) = calls_for(client, "vector.upsert")
    assert write["method"] == "execute_one"


async def test_upsert_uses_bounded_replay_safe_sql_chunks(ready_storage, monkeypatch):
    import lightrag.kg.hologres.vector as vector_module

    monkeypatch.setattr(vector_module, "_UPSERT_RECORD_LIMIT", 2)
    client = CallClient()
    storage = await ready_storage(client)
    data = {
        str(index): {
            "content": f"content-{index}",
            "embedding": [1.0, 0.0, 0.0],
        }
        for index in range(5)
    }

    await storage.upsert(data)

    writes = calls_for(client, "vector.upsert")
    assert [call["values"][6] for call in writes] == [2, 2, 1]
    assert all(call["kwargs"]["replay_safe"] is True for call in writes)


async def test_upsert_rejects_an_oversized_record_before_database_access(
    ready_storage, monkeypatch
):
    import lightrag.kg.hologres.vector as vector_module

    monkeypatch.setattr(vector_module, "_UPSERT_BYTE_LIMIT", 64)
    client = CallClient()
    storage = await ready_storage(client)

    with pytest.raises(HologresVectorError, match="exceeds the upsert batch limit"):
        await storage.upsert(
            {
                "oversized": {
                    "content": "x" * 128,
                    "embedding": [1.0, 0.0, 0.0],
                }
            }
        )
    assert calls_for(client, "vector.upsert") == []


@pytest.mark.parametrize(
    "vector",
    [
        [1.0, 2.0],
        [1.0, 2.0, 3.0, 4.0],
        [True, 0.0, 0.0],
        ["1", 0.0, 0.0],
        [math.nan, 0.0, 0.0],
        [math.inf, 0.0, 0.0],
        [1e100, 0.0, 0.0],
    ],
)
async def test_upsert_rejects_malformed_supplied_vectors_before_database_access(vector):
    client = CallClient()
    storage = make_storage(client=client)
    with pytest.raises(HologresVectorError, match="vector is invalid"):
        await storage.upsert({"secret-id": {"content": "secret", "embedding": vector}})
    assert client.calls == []


async def test_upsert_rejects_conflicting_supplied_vector_fields_before_database_access():
    client = CallClient()
    storage = make_storage(client=client)
    with pytest.raises(HologresVectorError, match="vector is invalid"):
        await storage.upsert(
            {
                "a": {
                    "content": "alpha",
                    "embedding": [1.0, 0.0, 0.0],
                    "__vector__": [0.0, 1.0, 0.0],
                }
            }
        )
    assert client.calls == []


async def test_embedding_count_or_shape_mismatch_fails_before_database_access(ready_storage):
    embedding = FakeEmbedding(results=[[[1.0, 0.0, 0.0]]])
    client = CallClient()
    storage = await ready_storage(client, embedding=embedding)

    with pytest.raises(HologresVectorError, match="embedding result is invalid"):
        await storage.upsert(
            {"a": {"content": "a"}, "b": {"content": "b"}}
        )
    assert calls_for(client, "vector.upsert") == []


async def test_embedding_and_database_errors_are_sanitized(ready_storage):
    embedding = FakeEmbedding(error=RuntimeError("secret-content-from-provider"))
    client = CallClient()
    storage = await ready_storage(client, embedding=embedding)
    with pytest.raises(HologresVectorError) as embed_error:
        await storage.upsert({"secret-id": {"content": "secret-content"}})
    assert "secret" not in str(embed_error.value)

    healthy = await ready_storage(client, embedding=FakeEmbedding())
    client.handlers["vector.upsert"] = RuntimeError("password=database-secret")
    with pytest.raises(HologresVectorError) as database_error:
        await healthy.upsert({"secret-id": {"content": "secret-content"}})
    assert "secret" not in str(database_error.value)


# Reads ----------------------------------------------------------------------


async def test_get_by_id_round_trips_all_payload_fields_without_embedding(ready_storage):
    client = CallClient()
    client.handlers["vector.read.one"] = {
        "id": "a",
        "content": "alpha",
        "payload": {"content": "alpha", "nested": [1, {"ok": True}]},
    }
    storage = await ready_storage(client)

    row = await storage.get_by_id("a")

    assert row == {
        "id": "a",
        "content": "alpha",
        "nested": [1, {"ok": True}],
    }
    call = calls_for(client, "vector.read.one")[0]
    assert call["values"] == ("workspace-a", NameSpace.VECTOR_STORE_CHUNKS, "a")
    assert "embedding" not in call["sql"].lower()


async def test_get_by_ids_preserves_order_missing_slots_and_duplicates(ready_storage):
    client = CallClient()
    client.handlers["vector.read.batch"] = [
        {"ordinality": 1, "id": "a", "content": "alpha", "payload": {"content": "alpha"}},
        {"ordinality": 2, "id": None, "content": None, "payload": None},
        {"ordinality": 3, "id": "a", "content": "alpha", "payload": {"content": "alpha"}},
    ]
    storage = await ready_storage(client)

    assert await storage.get_by_ids(["a", "missing", "a"]) == [
        {"id": "a", "content": "alpha"},
        None,
        {"id": "a", "content": "alpha"},
    ]


@pytest.mark.parametrize(
    "response",
    [
        [{"ordinality": 1, "id": "a", "content": "a", "payload": {}}],
        [
            {"ordinality": 2, "id": "a", "content": "a", "payload": {}},
            {"ordinality": 1, "id": None, "content": None, "payload": None},
        ],
        [
            {"ordinality": 1, "id": "wrong", "content": "a", "payload": {}},
            {"ordinality": 2, "id": None, "content": None, "payload": None},
        ],
        [
            {"ordinality": True, "id": "a", "content": "a", "payload": {}},
            {"ordinality": 2, "id": None, "content": None, "payload": None},
        ],
    ],
)
async def test_get_by_ids_fails_closed_on_incomplete_or_misaligned_responses(ready_storage, response):
    client = CallClient()
    client.handlers["vector.read.batch"] = response
    storage = await ready_storage(client)

    with pytest.raises(HologresVectorError, match="batch response is corrupt"):
        await storage.get_by_ids(["a", "missing"])


@pytest.mark.parametrize(
    "row",
    [
        {"id": "a", "content": "alpha", "payload": None},
        {"id": "a", "content": "alpha", "payload": []},
        {"id": "other", "content": "alpha", "payload": {}},
        {"content": "alpha", "payload": {}},
    ],
)
async def test_get_by_id_fails_closed_on_corrupt_rows(ready_storage, row):
    client = CallClient()
    client.handlers["vector.read.one"] = row
    storage = await ready_storage(client)
    with pytest.raises(HologresVectorError, match="row is corrupt"):
        await storage.get_by_id("a")


async def test_get_vectors_by_ids_returns_only_valid_found_vectors(ready_storage):
    client = CallClient()
    client.handlers["vector.read.vectors"] = [
        {"id": "a", "embedding": [1.0, 0.0, 0.0]},
        {"id": "b", "embedding": (0, 1, 0)},
    ]
    storage = await ready_storage(client)

    assert await storage.get_vectors_by_ids(["a", "missing", "b", "a"]) == {
        "a": [1.0, 0.0, 0.0],
        "b": [0.0, 1.0, 0.0],
    }


@pytest.mark.parametrize(
    "rows",
    [
        [{"id": "other", "embedding": [1.0, 0.0, 0.0]}],
        [{"id": "a", "embedding": [1.0, 0.0]}],
        [{"id": "a", "embedding": [True, 0.0, 0.0]}],
        [{"id": "a", "embedding": [math.nan, 0.0, 0.0]}],
        [
            {"id": "a", "embedding": [1.0, 0.0, 0.0]},
            {"id": "a", "embedding": [1.0, 0.0, 0.0]},
        ],
    ],
)
async def test_get_vectors_by_ids_fails_closed_on_corrupt_rows(ready_storage, rows):
    client = CallClient()
    client.handlers["vector.read.vectors"] = rows
    storage = await ready_storage(client)
    with pytest.raises(HologresVectorError, match="vector response is corrupt"):
        await storage.get_vectors_by_ids(["a"])


# Delete, drop, and callbacks -------------------------------------------------


async def test_delete_is_bounded_bound_and_replay_safe(ready_storage, monkeypatch):
    import lightrag.kg.hologres.vector as vector_module

    monkeypatch.setattr(vector_module, "_ID_CHUNK_SIZE", 2)
    client = CallClient()
    storage = await ready_storage(client)

    await storage.delete(["a", "b", "c", "d", "e"])

    calls = calls_for(client, "vector.delete")
    assert [call["values"][2] for call in calls] == [["a", "b"], ["c", "d"], ["e"]]
    assert all(call["values"][:2] == ("workspace-a", "chunks") for call in calls)
    assert all(call["kwargs"]["replay_safe"] is True for call in calls)


async def test_delete_entity_uses_canonical_entity_id(ready_storage):
    client = CallClient()
    storage = await ready_storage(
        client, namespace=NameSpace.VECTOR_STORE_ENTITIES
    )

    await storage.delete_entity("Alice")

    call = calls_for(client, "vector.delete")[0]
    assert call["values"][2] == [compute_mdhash_id("Alice", prefix="ent-")]


async def test_delete_entity_relation_uses_bound_payload_predicate(ready_storage):
    client = CallClient()
    storage = await ready_storage(
        client, namespace=NameSpace.VECTOR_STORE_RELATIONSHIPS
    )

    await storage.delete_entity_relation("Alice'; password=secret --")

    (call,) = calls_for(client, "vector.delete.relations")
    assert call["values"] == (
        "workspace-a",
        NameSpace.VECTOR_STORE_RELATIONSHIPS,
        "Alice'; password=secret --",
    )
    assert "src_id" in call["sql"] and "tgt_id" in call["sql"]
    assert "Alice" not in call["sql"]
    assert call["kwargs"]["replay_safe"] is True


async def test_drop_deletes_only_current_workspace_and_namespace(ready_storage):
    client = CallClient()
    storage = await ready_storage(client)

    result = await storage.drop()

    assert result == {"status": "success", "message": "data dropped"}
    (call,) = calls_for(client, "vector.drop")
    assert call["values"] == ("workspace-a", NameSpace.VECTOR_STORE_CHUNKS)
    assert "drop table" not in call["sql"].lower()
    assert call["kwargs"]["replay_safe"] is True


async def test_immediate_write_callbacks_are_noops(ready_storage):
    client = CallClient()
    storage = await ready_storage(client)
    before = list(client.calls)
    assert await storage.index_done_callback() is None
    assert await storage.drop_pending_index_ops() is None
    assert client.calls == before


# Query uses the live-frozen HGraph similarity contract ----------------------


@pytest.mark.parametrize("top_k", [True, False, 0, -1, 1.5, "5"])
async def test_query_rejects_invalid_top_k_before_embedding_or_database_access(top_k):
    embedding = FakeEmbedding(error=AssertionError("must not embed"))
    client = CallClient()
    storage = make_storage(embedding=embedding, client=client)
    with pytest.raises(HologresVectorError, match="top_k"):
        await storage.query("secret-query", top_k=top_k, query_embedding=[1, 0, 0])
    assert embedding.calls == []
    assert client.calls == []


async def test_query_with_supplied_embedding_returns_similarity_ordered_payloads(ready_storage):
    embedding = FakeEmbedding(error=AssertionError("must not embed"))
    client = CallClient()
    client.handlers["vector.query"] = [
        {
            "id": "a",
            "content": "alpha",
            "payload": {"content": "alpha", "file_path": "a.txt"},
            "score": 0.98,
            "created_at": 1700000000,
        },
        {
            "id": "b",
            "content": "beta",
            "payload": {"content": "beta"},
            "score": 0.5,
            "created_at": None,
        },
    ]
    storage = await ready_storage(client, embedding=embedding)

    results = await storage.query("secret-query", top_k=3, query_embedding=[1, 0, 0])

    assert results == [
        {
            "id": "a",
            "content": "alpha",
            "file_path": "a.txt",
            "distance": 0.98,
            "created_at": 1700000000,
        },
        {
            "id": "b",
            "content": "beta",
            "distance": 0.5,
            "created_at": None,
        },
    ]
    assert embedding.calls == []
    (call,) = calls_for(client, "vector.query")
    assert call["values"] == (
        "workspace-a",
        NameSpace.VECTOR_STORE_CHUNKS,
        [1.0, 0.0, 0.0],
        0.2,
        3,
    )
    sql = call["sql"]
    assert "approx_cosine_distance(embedding, $3::float4[])" in sql
    assert "> $4::float8" in sql
    assert "ORDER BY score DESC" in sql
    assert "LIMIT $5::int" in sql
    assert "secret-query" not in sql


async def test_query_computes_and_validates_embedding_before_database_access(ready_storage):
    embedding = FakeEmbedding(results=[[[1.0, 0.0, 0.0]]])
    client = CallClient()
    storage = await ready_storage(client, embedding=embedding)

    assert await storage.query("query", top_k=3) == []

    assert embedding.calls == [
        (["query"], {"context": "query", "_priority": 5})
    ]
    (call,) = calls_for(client, "vector.query")
    assert call["values"][2] == [1.0, 0.0, 0.0]


async def test_query_failure_is_sanitized_and_fails_closed(ready_storage):
    client = CallClient()
    client.handlers["vector.query"] = RuntimeError(
        "secret-vector-password [1.0, 2.0, 3.0]"
    )
    storage = await ready_storage(client)

    with pytest.raises(HologresVectorError, match="query failed") as exc_info:
        await storage.query("query", top_k=3, query_embedding=[1, 0, 0])
    assert "secret" not in str(exc_info.value)


@pytest.mark.parametrize(
    "row",
    [
        {"id": "a", "content": "alpha", "payload": {"content": "alpha"}},
        {
            "id": "a",
            "content": "alpha",
            "payload": {"content": "alpha"},
            "score": True,
            "created_at": 1,
        },
        {
            "id": "a",
            "content": "alpha",
            "payload": {"content": "alpha"},
            "score": math.nan,
            "created_at": 1,
        },
        {
            "id": None,
            "content": "alpha",
            "payload": {"content": "alpha"},
            "score": 0.9,
            "created_at": 1,
        },
        {
            "id": "a",
            "content": "alpha",
            "payload": {"content": "mismatch"},
            "score": 0.9,
            "created_at": 1,
        },
    ],
)
async def test_query_fails_closed_on_corrupt_rows(ready_storage, row):
    client = CallClient()
    client.handlers["vector.query"] = [row]
    storage = await ready_storage(client)

    with pytest.raises(HologresVectorError, match="response is corrupt"):
        await storage.query("query", top_k=3, query_embedding=[1, 0, 0])


@pytest.mark.parametrize("query_vector", [[1, 0], [True, 0, 0], [math.inf, 0, 0]])
async def test_query_rejects_invalid_supplied_embedding_before_database_access(query_vector):
    client = CallClient()
    storage = make_storage(client=client)
    with pytest.raises(HologresVectorError, match="vector is invalid"):
        await storage.query("query", top_k=3, query_embedding=query_vector)
    assert client.calls == []


async def test_operational_failures_never_echo_ids_payloads_vectors_or_credentials(ready_storage):
    client = CallClient()
    client.handlers["vector.read.one"] = RuntimeError(
        "secret-id secret-content secret-vector-password [1.0, 2.0, 3.0]"
    )
    storage = await ready_storage(client)

    with pytest.raises(HologresVectorError) as exc_info:
        await storage.get_by_id("secret-id")

    message = str(exc_info.value)
    assert "secret" not in message
    assert "[1.0" not in message


def test_vector_backend_source_uses_only_restricted_client_and_no_postgres_reuse():
    import lightrag.kg.hologres.vector as vector_module

    source = Path(inspect.getsourcefile(vector_module)).read_text(encoding="utf-8")
    lowered = source.lower()
    assert "transaction(" not in lowered
    assert "executemany(" not in lowered
    assert "asyncpg" not in lowered
    assert "postgres_impl" not in lowered
    assert "pgvectorstorage" not in lowered
    assert "raw_connection" not in lowered
    assert "begin;" not in lowered
    assert "commit;" not in lowered
    assert "rollback;" not in lowered
