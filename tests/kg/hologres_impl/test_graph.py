import asyncio
import inspect

import pytest

from lightrag.kg.hologres.capabilities import CapabilityReport, HologresVersion
from lightrag.kg.hologres.config import HologresConfig
from lightrag.kg.hologres.graph import (
    HologresGraphError,
    HologresGraphStorage,
    _payload_batches,
)
from lightrag.kg.hologres.schema import graph_schema_descriptors
from lightrag.namespace import NameSpace


CONFIG = HologresConfig(
    host="secret-graph-host.example",
    port=80,
    user="secret-graph-user",
    password="secret-graph-password",
    database="secret-graph-database",
    schema="lightrag_test_graph",
    connection_retries=0,
)


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
    namespace=NameSpace.GRAPH_STORE_CHUNK_ENTITY_RELATION,
    workspace="workspace-a",
    client=None,
    config=CONFIG,
    global_config=None,
):
    return HologresGraphStorage(
        namespace=namespace,
        workspace=workspace,
        global_config=dict(global_config or {}),
        embedding_func=None,
        config=config,
        client=client,
    )


@pytest.fixture
def ready_storage(monkeypatch):
    import lightrag.kg.hologres.graph as graph_module

    async def probe(client):
        return CapabilityReport(HologresVersion(5, 0, 0))

    class AppliedSchemaManager:
        def __init__(self, client, *, schema):
            self.client = client
            self.schema = schema

        async def initialize(self, descriptors):
            assert tuple(descriptors) == graph_schema_descriptors(self.schema)
            return ()

    monkeypatch.setattr(graph_module, "probe_production_capabilities", probe)
    monkeypatch.setattr(graph_module, "HologresSchemaManager", AppliedSchemaManager)

    async def factory(client, **kwargs):
        storage = make_storage(client=client, config=client.config, **kwargs)
        await storage.initialize()
        return storage

    return factory


def calls_for(client, descriptor):
    return [call for call in client.calls if call["kwargs"]["descriptor"] == descriptor]


# Construction ----------------------------------------------------------------


def test_exact_graph_namespace_allowlist():
    storage = make_storage(namespace=NameSpace.GRAPH_STORE_CHUNK_ENTITY_RELATION)
    assert storage.namespace == NameSpace.GRAPH_STORE_CHUNK_ENTITY_RELATION


@pytest.mark.parametrize("namespace", ["full_docs", "entities", "unknown", "", None])
def test_unsupported_namespace_is_rejected_before_database_access(namespace):
    with pytest.raises(ValueError, match="Unsupported Hologres graph namespace"):
        make_storage(namespace=namespace, client=object())


@pytest.mark.parametrize("workspace", [".", "..", "../escape", "bad/name", "bad\\name"])
def test_invalid_workspace_is_rejected_without_echoing_value(workspace):
    with pytest.raises(ValueError, match="Invalid Hologres graph workspace") as exc_info:
        make_storage(workspace=workspace, client=object())
    assert workspace not in str(exc_info.value)


def test_repr_redacts_configuration_and_runtime_state():
    rendered = repr(make_storage(client=CallClient()))
    assert rendered == "HologresGraphStorage(<redacted>)"
    assert "secret-graph" not in rendered


# Lifecycle ------------------------------------------------------------------


async def test_injected_client_initialization_is_idempotent_and_caller_owned(
    ready_storage,
):
    client = CallClient()
    storage = await ready_storage(client)

    await storage.initialize()
    await storage.finalize()
    await storage.finalize()

    assert client.close_count == 0


async def test_uninitialized_storage_refuses_every_operation():
    storage = make_storage(client=CallClient())

    with pytest.raises(HologresGraphError, match="not initialized"):
        await storage.has_node("node")


async def test_initialization_failure_leaves_storage_unready_and_sanitizes_error(
    monkeypatch,
):
    import lightrag.kg.hologres.graph as graph_module

    client = CallClient()

    async def failed_probe(_client):
        raise RuntimeError("password=probe-secret")

    monkeypatch.setattr(graph_module, "probe_production_capabilities", failed_probe)
    storage = make_storage(client=client)
    with pytest.raises(HologresGraphError) as exc_info:
        await storage.initialize()

    assert "probe-secret" not in str(exc_info.value)
    with pytest.raises(HologresGraphError, match="not initialized"):
        await storage.get_node("id")


async def test_shared_client_ownership_reuses_manager_and_reference_counts(
    monkeypatch,
):
    import lightrag.kg.hologres.graph as graph_module

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
            assert tuple(descriptors) == graph_schema_descriptors(self.schema)

    manager = Manager()
    monkeypatch.setattr(graph_module, "_SHARED_CLIENTS", manager)
    monkeypatch.setattr(graph_module, "probe_production_capabilities", probe)
    monkeypatch.setattr(graph_module, "HologresSchemaManager", SchemaManager)

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
    import lightrag.kg.hologres.graph as graph_module

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
    monkeypatch.setattr(graph_module, "_SHARED_CLIENTS", manager)
    monkeypatch.setattr(graph_module, "probe_production_capabilities", probe)
    monkeypatch.setattr(graph_module, "HologresSchemaManager", FailedSchemaManager)

    storage = make_storage(client=None)
    with pytest.raises(HologresGraphError) as exc_info:
        await storage.initialize()
    assert manager.release_count == 1
    assert "schema-secret" not in str(exc_info.value)


async def test_initialization_failure_preserves_sanitized_release_failure(monkeypatch):
    import lightrag.kg.hologres.graph as graph_module

    client = CallClient()

    class Manager:
        async def acquire(self, _config):
            return client

        async def release(self, _config, actual):
            assert actual is client
            raise RuntimeError("password=release-secret")

    async def failed_probe(_client):
        raise RuntimeError("password=probe-secret")

    monkeypatch.setattr(graph_module, "_SHARED_CLIENTS", Manager())
    monkeypatch.setattr(graph_module, "probe_production_capabilities", failed_probe)
    storage = make_storage(client=None)

    with pytest.raises(HologresGraphError) as exc_info:
        await storage.initialize()

    assert str(exc_info.value) == "Hologres graph initialization failed"
    assert isinstance(exc_info.value.__cause__, HologresGraphError)
    assert (
        str(exc_info.value.__cause__)
        == "Hologres graph shared client release failed"
    )
    assert "secret" not in repr(exc_info.value)
    assert "secret" not in repr(exc_info.value.__cause__)


async def test_index_done_callback_is_a_noop(ready_storage):
    storage = await ready_storage(CallClient())
    assert await storage.index_done_callback() is None


# Node reads -------------------------------------------------------------------


async def test_has_node_reports_presence_from_the_workspace_scoped_row(ready_storage):
    client = CallClient()
    client.handlers["graph.node.exists"] = {"present": 1}
    storage = await ready_storage(client)

    assert await storage.has_node("node-1") is True

    (call,) = calls_for(client, "graph.node.exists")
    assert call["values"] == ("workspace-a", storage.namespace, "node-1")
    assert "workspace = $1 AND namespace = $2 AND id = $3" in call["sql"]

    client.handlers.pop("graph.node.exists")
    assert await storage.has_node("node-1") is False


async def test_get_node_injects_entity_id_and_returns_none_when_absent(ready_storage):
    client = CallClient()
    client.handlers["graph.node.read"] = {"properties": '{"kind":"x","entity_id":"stale"}'}
    storage = await ready_storage(client)

    assert await storage.get_node("node-1") == {"kind": "x", "entity_id": "node-1"}

    client.handlers.pop("graph.node.read")
    assert await storage.get_node("node-1") is None


async def test_node_read_failures_are_sanitized_without_chained_cause(ready_storage):
    client = CallClient()
    client.handlers["graph.node.read"] = RuntimeError("password=read-secret")
    storage = await ready_storage(client)

    with pytest.raises(HologresGraphError) as exc_info:
        await storage.get_node("node-1")

    assert str(exc_info.value) == "Hologres graph node read failed"
    assert exc_info.value.__cause__ is None
    assert "read-secret" not in repr(exc_info.value)


async def test_corrupt_node_row_payload_is_rejected(ready_storage):
    client = CallClient()
    client.handlers["graph.node.read"] = {"properties": "not-json"}
    storage = await ready_storage(client)

    with pytest.raises(HologresGraphError, match="corrupt"):
        await storage.get_node("node-1")


async def test_has_nodes_batch_deduplicates_and_returns_the_existing_subset(
    ready_storage,
):
    client = CallClient()
    client.handlers["graph.node.exists.batch"] = [{"id": "a"}]
    storage = await ready_storage(client)

    assert await storage.has_nodes_batch(["a", "b", "a"]) == {"a"}
    assert await storage.has_nodes_batch([]) == set()

    (call,) = calls_for(client, "graph.node.exists.batch")
    assert call["values"] == ("workspace-a", storage.namespace, ["a", "b"])


async def test_get_nodes_batch_keys_decoded_properties_by_node_id(ready_storage):
    client = CallClient()
    client.handlers["graph.node.read.batch"] = [
        {"id": "a", "properties": '{"k":"v"}'},
        {"id": "b", "properties": {"j": 1}},
    ]
    storage = await ready_storage(client)

    assert await storage.get_nodes_batch(["a", "b"]) == {
        "a": {"k": "v", "entity_id": "a"},
        "b": {"j": 1, "entity_id": "b"},
    }
    assert await storage.get_nodes_batch([]) == {}


async def test_non_iterable_batch_response_is_rejected_as_corrupt(ready_storage):
    client = CallClient()
    client.handlers["graph.node.read.batch"] = {"id": "a"}
    storage = await ready_storage(client)

    with pytest.raises(HologresGraphError, match="corrupt"):
        await storage.get_nodes_batch(["a"])


# Node writes ------------------------------------------------------------------


async def test_upsert_node_merges_properties_and_forces_entity_id(ready_storage):
    client = CallClient()
    storage = await ready_storage(client)

    await storage.upsert_node("node-1", {"entity_id": "stale", "kind": "x"})

    (call,) = calls_for(client, "graph.node.upsert")
    assert call["values"] == (
        "workspace-a",
        storage.namespace,
        ["node-1"],
        ['{"entity_id":"node-1","kind":"x"}'],
        1,
    )
    assert call["kwargs"]["replay_safe"] is True
    assert "current.properties || EXCLUDED.properties" in call["sql"]
    assert "generate_series(1, $5::int)" in call["sql"]
    assert "ON CONFLICT (workspace, namespace, id) DO UPDATE SET" in call["sql"]


async def test_upsert_node_requires_the_entity_id_field(ready_storage):
    storage = await ready_storage(CallClient())

    with pytest.raises(ValueError, match="entity_id"):
        await storage.upsert_node("node-1", {"kind": "x"})


@pytest.mark.parametrize("node_data", [None, "props", ["entity_id"]])
async def test_upsert_node_rejects_non_mapping_properties(ready_storage, node_data):
    storage = await ready_storage(CallClient())

    with pytest.raises(ValueError, match="entity_id"):
        await storage.upsert_node("node-1", node_data)


async def test_upsert_node_rejects_unserializable_payloads(ready_storage):
    storage = await ready_storage(CallClient())

    with pytest.raises(HologresGraphError, match="invalid"):
        await storage.upsert_node("node-1", {"entity_id": "node-1", "v": float("nan")})


async def test_upsert_nodes_batch_dedupes_last_write_wins_in_one_statement(
    ready_storage,
):
    client = CallClient()
    storage = await ready_storage(client)

    await storage.upsert_nodes_batch(
        [
            ("n", {"entity_id": "n", "v": 1}),
            ("m", {"entity_id": "m"}),
            ("n", {"entity_id": "n", "v": 2}),
        ]
    )
    await storage.upsert_nodes_batch([])

    (call,) = calls_for(client, "graph.node.upsert")
    assert call["values"][2] == ["n", "m"]
    assert call["values"][3] == ['{"entity_id":"n","v":2}', '{"entity_id":"m"}']
    assert call["values"][4] == 2


async def test_node_upserts_split_on_the_record_budget(ready_storage, monkeypatch):
    import lightrag.kg.hologres.graph as graph_module

    monkeypatch.setattr(graph_module, "_UPSERT_RECORD_LIMIT", 2)
    client = CallClient()
    storage = await ready_storage(client)

    await storage.upsert_nodes_batch(
        [(f"n{i}", {"entity_id": f"n{i}"}) for i in range(5)]
    )

    batches = calls_for(client, "graph.node.upsert")
    assert [call["values"][4] for call in batches] == [2, 2, 1]


async def test_single_record_over_the_byte_budget_is_rejected(monkeypatch):
    import lightrag.kg.hologres.graph as graph_module

    monkeypatch.setattr(graph_module, "_UPSERT_BYTE_LIMIT", 8)

    with pytest.raises(HologresGraphError, match="batch limit"):
        list(_payload_batches([("n", "x" * 9)], 1))


def test_payload_batches_split_on_the_byte_budget(monkeypatch):
    import lightrag.kg.hologres.graph as graph_module

    monkeypatch.setattr(graph_module, "_UPSERT_BYTE_LIMIT", 8)

    batches = list(
        _payload_batches([("a", "x" * 5), ("b", "y" * 5), ("c", "z" * 2)], 1)
    )

    assert batches == [[("a", "x" * 5)], [("b", "y" * 5), ("c", "z" * 2)]]


async def test_remove_nodes_deletes_incident_edges_before_the_nodes(ready_storage):
    client = CallClient()
    storage = await ready_storage(client)

    await storage.remove_nodes(["b", "a", "b"])
    await storage.remove_nodes([])

    descriptors = [call["kwargs"]["descriptor"] for call in client.calls]
    assert descriptors == ["graph.node.delete.edges", "graph.node.delete"]
    edge_call, node_call = client.calls
    assert edge_call["values"] == ("workspace-a", storage.namespace, ["b", "a"])
    assert "src_id = ANY($3::text[]) OR tgt_id = ANY($3::text[])" in edge_call["sql"]
    assert node_call["values"] == ("workspace-a", storage.namespace, ["b", "a"])
    assert edge_call["kwargs"]["replay_safe"] is True
    assert node_call["kwargs"]["replay_safe"] is True


async def test_remove_nodes_chunks_ids_and_keeps_ordering_per_chunk(
    ready_storage, monkeypatch
):
    import lightrag.kg.hologres.graph as graph_module

    monkeypatch.setattr(graph_module, "_ID_CHUNK_SIZE", 2)
    client = CallClient()
    storage = await ready_storage(client)

    await storage.remove_nodes(["a", "b", "c"])

    descriptors = [call["kwargs"]["descriptor"] for call in client.calls]
    assert descriptors == [
        "graph.node.delete.edges",
        "graph.node.delete",
        "graph.node.delete.edges",
        "graph.node.delete",
    ]
    assert client.calls[0]["values"][2] == ["a", "b"]
    assert client.calls[2]["values"][2] == ["c"]


async def test_remove_nodes_rejects_non_string_ids_before_any_delete(ready_storage):
    client = CallClient()
    storage = await ready_storage(client)

    with pytest.raises(HologresGraphError, match="invalid"):
        await storage.remove_nodes(["a", None])

    assert client.calls == []


async def test_delete_node_delegates_to_remove_nodes(ready_storage):
    client = CallClient()
    storage = await ready_storage(client)

    await storage.delete_node("solo")

    descriptors = [call["kwargs"]["descriptor"] for call in client.calls]
    assert descriptors == ["graph.node.delete.edges", "graph.node.delete"]
    assert client.calls[0]["values"][2] == ["solo"]


# Edge operations ---------------------------------------------------------------


async def test_edge_reads_normalize_to_canonical_min_max_order(ready_storage):
    client = CallClient()
    client.handlers["graph.edge.exists"] = {"present": 1}
    client.handlers["graph.edge.read"] = {"properties": '{"weight":1}'}
    storage = await ready_storage(client)

    assert await storage.has_edge("z", "a") is True
    assert await storage.get_edge("z", "a") == {"weight": 1}

    (exists_call,) = calls_for(client, "graph.edge.exists")
    (read_call,) = calls_for(client, "graph.edge.read")
    assert exists_call["values"][2:] == ("a", "z")
    assert read_call["values"][2:] == ("a", "z")

    client.handlers.pop("graph.edge.exists")
    client.handlers.pop("graph.edge.read")
    assert await storage.has_edge("z", "a") is False
    assert await storage.get_edge("z", "a") is None


async def test_upsert_edge_creates_endpoints_first_then_replaces_properties(
    ready_storage,
):
    client = CallClient()
    storage = await ready_storage(client)

    await storage.upsert_edge("b", "a", {"weight": 1})

    descriptors = [call["kwargs"]["descriptor"] for call in client.calls]
    assert descriptors == ["graph.edge.endpoints", "graph.edge.upsert"]
    endpoints, upsert = client.calls
    assert endpoints["values"] == ("workspace-a", storage.namespace, ["a", "b"])
    assert "ON CONFLICT (workspace, namespace, id) DO NOTHING" in endpoints["sql"]
    assert "jsonb_build_object('entity_id', u.id)" in endpoints["sql"]
    assert endpoints["kwargs"]["replay_safe"] is True

    assert upsert["values"] == (
        "workspace-a",
        storage.namespace,
        ["a"],
        ["b"],
        ['{"weight":1}'],
        1,
    )
    assert upsert["kwargs"]["replay_safe"] is True
    assert "properties = EXCLUDED.properties" in upsert["sql"]
    assert "||" not in upsert["sql"]
    assert "generate_series(1, $6::int)" in upsert["sql"]


@pytest.mark.parametrize(
    ("source", "target", "edge_data"),
    [(None, "a", {}), ("a", 1, {}), ("a", "b", None), ("a", "b", "props")],
)
async def test_upsert_edge_rejects_invalid_inputs_before_any_write(
    ready_storage, source, target, edge_data
):
    client = CallClient()
    storage = await ready_storage(client)

    with pytest.raises(HologresGraphError, match="invalid"):
        await storage.upsert_edge(source, target, edge_data)

    assert client.calls == []


async def test_upsert_edges_batch_dedupes_by_canonical_pair_last_write_wins(
    ready_storage,
):
    client = CallClient()
    storage = await ready_storage(client)

    await storage.upsert_edges_batch(
        [
            ("b", "a", {"w": 1}),
            ("c", "a", {"w": 5}),
            ("a", "b", {"w": 2}),
        ]
    )
    await storage.upsert_edges_batch([])

    (upsert,) = calls_for(client, "graph.edge.upsert")
    assert upsert["values"][2] == ["a", "a"]
    assert upsert["values"][3] == ["b", "c"]
    assert upsert["values"][4] == ['{"w":2}', '{"w":5}']
    assert upsert["values"][5] == 2


async def test_remove_edges_uses_the_generate_series_pair_match(ready_storage):
    client = CallClient()
    storage = await ready_storage(client)

    await storage.remove_edges([("b", "a"), ("a", "b"), ("c", "a")])
    await storage.remove_edges([])

    (call,) = calls_for(client, "graph.edge.delete")
    assert call["values"] == (
        "workspace-a",
        storage.namespace,
        ["a", "a"],
        ["b", "c"],
        2,
    )
    assert call["kwargs"]["replay_safe"] is True
    assert "EXISTS (" in call["sql"]
    assert "generate_series(1, $5::int)" in call["sql"]
    assert "unnest($3::text[], $4::text[])" not in call["sql"]


async def test_remove_edges_rejects_non_string_endpoints(ready_storage):
    client = CallClient()
    storage = await ready_storage(client)

    with pytest.raises(HologresGraphError, match="non-None strings"):
        await storage.remove_edges([("a", None)])

    assert client.calls == []


async def test_get_edges_batch_returns_results_keyed_by_requested_orientation(
    ready_storage,
):
    client = CallClient()
    client.handlers["graph.edge.read.batch"] = [
        {"src_id": "a", "tgt_id": "b", "properties": '{"w":1}'}
    ]
    storage = await ready_storage(client)

    result = await storage.get_edges_batch(
        [
            {"src": "b", "tgt": "a"},
            {"src": "a", "tgt": "b"},
            {"src": "a", "tgt": "missing"},
        ]
    )

    assert result == {("b", "a"): {"w": 1}, ("a", "b"): {"w": 1}}
    (call,) = calls_for(client, "graph.edge.read.batch")
    assert call["values"][2:] == (["a", "a"], ["b", "missing"], 2)
    assert await storage.get_edges_batch([]) == {}


@pytest.mark.parametrize("pair", [None, {"src": 1, "tgt": "a"}, {"src": "a"}])
async def test_get_edges_batch_rejects_malformed_pairs(ready_storage, pair):
    client = CallClient()
    storage = await ready_storage(client)

    with pytest.raises(HologresGraphError, match="invalid"):
        await storage.get_edges_batch([pair])

    assert client.calls == []


async def test_get_node_edges_distinguishes_absent_isolated_and_connected(
    ready_storage,
):
    client = CallClient()
    storage = await ready_storage(client)

    assert await storage.get_node_edges("A") is None

    client.handlers["graph.node.exists"] = {"present": 1}
    assert await storage.get_node_edges("A") == []

    client.handlers["graph.edge.adjacency"] = [
        {"src_id": "Z", "tgt_id": "A"},
        {"src_id": "A", "tgt_id": "M"},
        {"src_id": "A", "tgt_id": "A"},
    ]
    assert await storage.get_node_edges("A") == [
        ("A", "A"),
        ("A", "M"),
        ("A", "Z"),
    ]


async def test_get_node_edges_raises_when_the_backend_cannot_answer(ready_storage):
    client = CallClient()
    client.handlers["graph.node.exists"] = {"present": 1}
    client.handlers["graph.edge.adjacency"] = RuntimeError("password=adj-secret")
    storage = await ready_storage(client)

    with pytest.raises(HologresGraphError) as exc_info:
        await storage.get_node_edges("A")

    assert "adj-secret" not in repr(exc_info.value)


async def test_get_nodes_edges_batch_lists_a_self_loop_once(ready_storage):
    client = CallClient()
    client.handlers["graph.edge.adjacency.batch"] = [
        {"src_id": "A", "tgt_id": "B"},
        {"src_id": "A", "tgt_id": "A"},
        {"src_id": "C", "tgt_id": "B"},
    ]
    storage = await ready_storage(client)

    result = await storage.get_nodes_edges_batch(["A", "B", "unrelated"])

    assert result == {
        "A": [("A", "A"), ("A", "B")],
        "B": [("B", "A"), ("B", "C")],
        "unrelated": [],
    }
    assert await storage.get_nodes_edges_batch([]) == {}


# Degrees ------------------------------------------------------------------------


async def test_node_degree_counts_both_directions(ready_storage):
    client = CallClient()
    client.handlers["graph.degree.node"] = 3
    storage = await ready_storage(client)

    assert await storage.node_degree("A") == 3

    (call,) = calls_for(client, "graph.degree.node")
    assert "UNION ALL" in call["sql"]
    assert call["values"] == ("workspace-a", storage.namespace, "A")


@pytest.mark.parametrize("value", [True, "3", None, 2.5])
async def test_non_integer_degree_responses_are_rejected(ready_storage, value):
    client = CallClient()
    client.handlers["graph.degree.node"] = value
    storage = await ready_storage(client)

    with pytest.raises(HologresGraphError, match="corrupt"):
        await storage.node_degree("A")


async def test_node_degrees_batch_zero_fills_missing_nodes(ready_storage):
    client = CallClient()
    client.handlers["graph.degree.batch"] = [{"id": "A", "degree": 3}]
    storage = await ready_storage(client)

    assert await storage.node_degrees_batch(["A", "B", "A"]) == {"A": 3, "B": 0}
    assert await storage.node_degrees_batch([]) == {}


async def test_edge_degree_sums_both_endpoint_degrees(ready_storage):
    client = CallClient()
    client.handlers["graph.degree.batch"] = [
        {"id": "A", "degree": 3},
        {"id": "B", "degree": 1},
    ]
    storage = await ready_storage(client)

    assert await storage.edge_degree("A", "B") == 4
    assert await storage.edge_degrees_batch([("A", "B"), ("A", "missing")]) == {
        ("A", "B"): 4,
        ("A", "missing"): 3,
    }
    assert await storage.edge_degrees_batch([]) == {}


# Labels ------------------------------------------------------------------------


async def test_get_all_labels_returns_python_sorted_ids(ready_storage):
    client = CallClient()
    client.handlers["graph.labels.all"] = [{"id": "b"}, {"id": "B"}, {"id": "_x"}]
    storage = await ready_storage(client)

    assert await storage.get_all_labels() == ["B", "_x", "b"]


async def test_non_string_label_rows_are_rejected_as_corrupt(ready_storage):
    client = CallClient()
    client.handlers["graph.labels.all"] = [{"id": 7}]
    storage = await ready_storage(client)

    with pytest.raises(HologresGraphError, match="corrupt"):
        await storage.get_all_labels()


async def test_get_popular_labels_ranks_by_degree_including_isolated_nodes(
    ready_storage,
):
    client = CallClient()
    client.handlers["graph.labels.popular"] = [{"id": "hub"}, {"id": "isolated"}]
    storage = await ready_storage(client)

    assert await storage.get_popular_labels(limit=2) == ["hub", "isolated"]

    (call,) = calls_for(client, "graph.labels.popular")
    assert call["values"] == ("workspace-a", storage.namespace, 2)
    assert "LEFT JOIN" in call["sql"]
    assert "ORDER BY degree DESC, n.id ASC" in call["sql"]
    assert "LIMIT $3::int" in call["sql"]
    assert "COLLATE" not in call["sql"]


@pytest.mark.parametrize("limit", [True, False, 0, -1, 1.5, "5"])
async def test_label_limits_must_be_positive_integers(ready_storage, limit):
    storage = await ready_storage(CallClient())

    with pytest.raises(HologresGraphError, match="limit is invalid"):
        await storage.get_popular_labels(limit=limit)
    with pytest.raises(HologresGraphError, match="limit is invalid"):
        await storage.search_labels("query", limit=limit)


async def test_search_labels_escapes_like_wildcards_and_keeps_the_scoring_shape(
    ready_storage,
):
    client = CallClient()
    client.handlers["graph.labels.search"] = [{"id": "match"}]
    storage = await ready_storage(client)

    assert await storage.search_labels("  50%_Off\\Deal  ", limit=5) == ["match"]

    (call,) = calls_for(client, "graph.labels.search")
    escaped = "50\\%\\_off\\\\deal"
    assert call["values"] == (
        "workspace-a",
        storage.namespace,
        "50%_off\\deal",
        f"{escaped}%",
        f"% {escaped}%",
        f"%\\_{escaped}%",
        f"%{escaped}%",
        5,
    )
    sql = call["sql"]
    assert "THEN 1000" in sql
    assert "THEN 500" in sql
    assert "100 - LENGTH(id)" in sql
    assert "THEN 50" in sql
    assert sql.count("ESCAPE E'\\\\'") == 4
    assert "COLLATE" not in sql


async def test_blank_search_queries_short_circuit_without_touching_the_database(
    ready_storage,
):
    client = CallClient()
    storage = await ready_storage(client)

    assert await storage.search_labels("   ") == []
    assert client.calls == []

    with pytest.raises(HologresGraphError, match="invalid"):
        await storage.search_labels(None)


# Exports -----------------------------------------------------------------------


async def test_get_all_nodes_returns_sorted_rows_with_id_and_entity_id(ready_storage):
    client = CallClient()
    client.handlers["graph.export.nodes"] = [
        {"id": "b", "properties": "{}"},
        {"id": "a", "properties": '{"k":1}'},
    ]
    storage = await ready_storage(client)

    assert await storage.get_all_nodes() == [
        {"k": 1, "entity_id": "a", "id": "a"},
        {"entity_id": "b", "id": "b"},
    ]


async def test_get_all_edges_returns_sorted_rows_with_source_and_target(ready_storage):
    client = CallClient()
    client.handlers["graph.export.edges"] = [
        {"src_id": "b", "tgt_id": "c", "properties": '{"w":2}'},
        {"src_id": "a", "tgt_id": "z", "properties": "{}"},
    ]
    storage = await ready_storage(client)

    assert await storage.get_all_edges() == [
        {"source": "a", "target": "z"},
        {"w": 2, "source": "b", "target": "c"},
    ]


# Knowledge graph -----------------------------------------------------------------


async def test_wildcard_knowledge_graph_overfetches_and_reports_truncation(
    ready_storage,
):
    client = CallClient()
    client.handlers["graph.kg.wildcard"] = [
        {"id": "hub", "properties": '{"kind":"hub"}', "degree": 5},
        {"id": "mid", "properties": "{}", "degree": 3},
        {"id": "leaf", "properties": "{}", "degree": 1},
    ]
    client.handlers["graph.kg.edges"] = [
        {"src_id": "hub", "tgt_id": "mid", "properties": '{"w":1}'}
    ]
    storage = await ready_storage(client, global_config={"max_graph_nodes": 2})

    result = await storage.get_knowledge_graph("*", max_nodes=10)

    (wildcard,) = calls_for(client, "graph.kg.wildcard")
    assert wildcard["values"] == ("workspace-a", storage.namespace, 3)

    assert result.is_truncated is True
    assert [node.id for node in result.nodes] == ["hub", "mid"]
    assert result.nodes[0].labels == ["hub"]
    assert result.nodes[0].properties == {"kind": "hub", "entity_id": "hub"}

    (edge,) = result.edges
    assert (edge.id, edge.type, edge.source, edge.target) == (
        "hub-mid",
        "DIRECTED",
        "hub",
        "mid",
    )
    assert edge.properties == {"w": 1}

    (edge_call,) = calls_for(client, "graph.kg.edges")
    assert sorted(edge_call["values"][2]) == ["hub", "mid"]
    assert "src_id = ANY($3::text[]) AND tgt_id = ANY($3::text[])" in edge_call["sql"]


async def test_missing_seed_returns_an_empty_graph_without_hops(ready_storage):
    client = CallClient()
    storage = await ready_storage(client)

    result = await storage.get_knowledge_graph("ghost")

    assert result.nodes == [] and result.edges == []
    assert result.is_truncated is False
    assert calls_for(client, "graph.kg.hop") == []
    assert calls_for(client, "graph.kg.edges") == []


async def test_bfs_pins_the_seed_shrinks_the_level_cap_and_truncates(ready_storage):
    client = CallClient()
    client.handlers["graph.kg.seed"] = {"id": "S", "properties": '{"entity_id":"S"}'}
    hop_rows = [
        [
            {"id": "B", "properties": "{}", "degree": 2},
            {"id": "A", "properties": "{}", "degree": 2},
        ],
        [{"id": "C", "properties": "{}", "degree": 1}],
    ]
    client.handlers["graph.kg.hop"] = lambda sql, values, kwargs: hop_rows.pop(0)
    client.handlers["graph.kg.edges"] = [
        {"src_id": "S", "tgt_id": "B", "properties": "{}"},
        {"src_id": "A", "tgt_id": "S", "properties": "{}"},
    ]
    storage = await ready_storage(client)

    result = await storage.get_knowledge_graph("S", max_depth=2, max_nodes=3)

    hops = calls_for(client, "graph.kg.hop")
    assert len(hops) == 2
    first, second = hops
    # Level cap = remaining budget + 1 overfetch row.
    assert first["values"][2:] == (["S"], ["S"], 3)
    assert second["values"][2] == ["B", "A"]
    assert sorted(second["values"][3]) == ["A", "B", "S"]
    assert second["values"][4] == 1
    assert "NOT EXISTS (SELECT 1 FROM visited" in first["sql"]
    assert "unnest($4::text[])" in first["sql"]
    assert "LIMIT $5::int" in first["sql"]

    # Seed pinned first, then depth, then higher degree, then id; C truncated.
    assert [node.id for node in result.nodes] == ["S", "A", "B"]
    assert result.is_truncated is True
    assert [(edge.source, edge.target) for edge in result.edges] == [
        ("A", "S"),
        ("S", "B"),
    ]


async def test_bfs_stops_at_max_depth_and_reports_a_complete_graph(ready_storage):
    client = CallClient()
    client.handlers["graph.kg.seed"] = {"id": "S", "properties": "{}"}
    client.handlers["graph.kg.hop"] = [
        {"id": "A", "properties": "{}", "degree": 1}
    ]
    storage = await ready_storage(client)

    result = await storage.get_knowledge_graph("S", max_depth=1, max_nodes=10)

    assert len(calls_for(client, "graph.kg.hop")) == 1
    assert [node.id for node in result.nodes] == ["S", "A"]
    assert result.is_truncated is False


async def test_node_budget_is_capped_by_the_global_configuration(ready_storage):
    client = CallClient()
    storage = await ready_storage(client, global_config={"max_graph_nodes": 2})

    await storage.get_knowledge_graph("*", max_nodes=50)
    await storage.get_knowledge_graph("*")

    wildcard_calls = calls_for(client, "graph.kg.wildcard")
    assert [call["values"][2] for call in wildcard_calls] == [3, 3]


# Drop ----------------------------------------------------------------------------


async def test_drop_deletes_edges_before_nodes_with_replay_safe_statements(
    ready_storage,
):
    client = CallClient()
    storage = await ready_storage(client)

    assert await storage.drop() == {"status": "success", "message": "data dropped"}

    descriptors = [call["kwargs"]["descriptor"] for call in client.calls]
    assert descriptors == ["graph.drop.edges", "graph.drop.nodes"]
    for call in client.calls:
        assert call["kwargs"]["replay_safe"] is True
        assert call["values"] == ("workspace-a", storage.namespace)


async def test_drop_failures_raise_a_sanitized_error(ready_storage):
    client = CallClient()
    client.handlers["graph.drop.edges"] = RuntimeError("password=drop-secret")
    storage = await ready_storage(client)

    with pytest.raises(HologresGraphError) as exc_info:
        await storage.drop()

    assert str(exc_info.value) == "Hologres graph drop failed"
    assert "drop-secret" not in repr(exc_info.value)
    assert calls_for(client, "graph.drop.nodes") == []


# SQL hygiene ----------------------------------------------------------------------


async def test_every_emitted_statement_is_single_and_collation_free(ready_storage):
    client = CallClient()
    client.handlers["graph.node.exists"] = {"present": 1}
    client.handlers["graph.kg.seed"] = {"id": "S", "properties": "{}"}
    storage = await ready_storage(client)

    await storage.upsert_node("n", {"entity_id": "n"})
    await storage.upsert_edge("a", "b", {"w": 1})
    await storage.get_node_edges("n")
    await storage.get_nodes_batch(["n"])
    await storage.node_degrees_batch(["n"])
    await storage.get_popular_labels(limit=3)
    await storage.search_labels("n")
    await storage.get_all_nodes()
    await storage.get_all_edges()
    await storage.get_knowledge_graph("S", max_nodes=2)
    await storage.remove_edges([("a", "b")])
    await storage.remove_nodes(["n"])
    await storage.drop()

    assert client.calls
    for call in client.calls:
        assert ";" not in call["sql"]
        assert "COLLATE" not in call["sql"]
        assert "unnest($3::text[], $4::text[])" not in call["sql"]
