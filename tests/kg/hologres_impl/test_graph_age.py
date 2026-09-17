import json
from pathlib import Path

import pytest

from lightrag.kg.hologres.capabilities import (
    CapabilityReport,
    HologresVersion,
    ProbeKind,
    ProbeResult,
    ProbeStatus,
)
from lightrag.kg.hologres.client import OperationKind
from lightrag.kg.hologres.config import HologresConfig
from lightrag.kg.hologres.graph_age import (
    HologresAGEGraphError,
    HologresAGEGraphStorage,
    _agtype_map,
    _agtype_value,
    _decode_agtype,
    _decode_int,
    _decode_map,
    _decode_string,
    _ID_CHUNK_SIZE,
    _materialize_rows,
    _params_json,
)
from lightrag.namespace import NameSpace


CONFIG = HologresConfig(
    host="secret-age-host.example",
    port=80,
    user="secret-age-user",
    password="secret-age-password",
    database="secret-age-database",
    schema="lightrag_test_age",
    connection_retries=0,
)


def make_storage(*, workspace="ws1", client=None, config=CONFIG):
    return HologresAGEGraphStorage(
        namespace=NameSpace.GRAPH_STORE_CHUNK_ENTITY_RELATION,
        workspace=workspace,
        global_config={"max_graph_nodes": 1000},
        embedding_func=None,
        config=config,
        client=client,
    )


class FakeAgeClient:
    def __init__(self, config=CONFIG):
        self.config = config
        self.calls = []
        self.procedures = []
        self.handlers = {
            "age.graph.check": True,
            "age.graph.label.check": True,
        }
        self.close_count = 0

    def _resolve(self, descriptor, sql, values, default):
        handler = self.handlers.get(descriptor, default)
        if isinstance(handler, BaseException):
            raise handler
        if callable(handler):
            return handler(sql, values)
        return handler

    async def fetch_all(
        self, sql, *values, descriptor, operation_kind=None, replay_safe=None
    ):
        self.calls.append(
            {
                "method": "fetch_all",
                "sql": sql,
                "values": values,
                "descriptor": descriptor,
                "operation_kind": operation_kind,
                "replay_safe": replay_safe,
            }
        )
        return self._resolve(descriptor, sql, values, [])

    async def fetch_value(self, sql, *values, descriptor):
        self.calls.append(
            {
                "method": "fetch_value",
                "sql": sql,
                "values": values,
                "descriptor": descriptor,
            }
        )
        return self._resolve(descriptor, sql, values, None)

    async def call_age_procedure(
        self, procedure, *values, descriptor, replay_safe=False
    ):
        self.calls.append(
            {
                "method": "call_age_procedure",
                "procedure": procedure,
                "values": values,
                "descriptor": descriptor,
                "replay_safe": replay_safe,
            }
        )
        self.procedures.append((procedure, values, replay_safe))
        return "CALL"

    async def close(self):
        self.close_count += 1


@pytest.fixture
def ready_storage(monkeypatch):
    import lightrag.kg.hologres.graph_age as module

    async def probe_version(client):
        return CapabilityReport(HologresVersion(5, 0, 0))

    async def probe_age(client):
        return ProbeResult(
            kind=ProbeKind.AGE,
            status=ProbeStatus.PASSED,
            blocking=False,
            detail_code="age_graph_semantics_frozen",
        )

    monkeypatch.setattr(module, "probe_production_capabilities", probe_version)
    monkeypatch.setattr(module, "probe_age_graph_capability", probe_age)

    async def factory(client, **kwargs):
        storage = make_storage(client=client, **kwargs)
        await storage.initialize()
        return storage

    return factory


def calls_for(client, descriptor):
    return [
        call for call in client.calls if call.get("descriptor") == descriptor
    ]


# ---------------------------------------------------------------------------
# Construction and literal rendering
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "namespace", ["full_docs", "doc_status", "entities", "unknown", ""]
)
def test_unsupported_namespace_is_rejected_before_database_access(namespace):
    with pytest.raises(ValueError, match="Unsupported Hologres AGE graph namespace"):
        HologresAGEGraphStorage(
            namespace=namespace,
            workspace="ws1",
            global_config={},
            embedding_func=None,
        )


@pytest.mark.parametrize("workspace", ["bad/ws", "bad\\ws", "..", "bad-ws", "bad ws"])
def test_workspace_must_form_a_valid_graph_identifier(workspace):
    with pytest.raises(ValueError):
        make_storage(workspace=workspace)


def test_graph_name_is_prefixed_and_workspace_scoped():
    assert make_storage(workspace="ws1")._graph == "lightrag_age_ws1"


@pytest.mark.parametrize(
    ("value", "rendered"),
    [
        (None, "null"),
        (True, "true"),
        (False, "false"),
        (7, "7"),
        (2.5, "2.5"),
        ("plain", "'plain'"),
        ("it's", "'it\\'s'"),
        ("back\\slash", "'back\\\\slash'"),
        ("line\nbreak\ttab\rcr", "'line\\nbreak\\ttab\\rcr'"),
        ("中文 🚀", "'中文 🚀'"),
        (
            "'}) DETACH DELETE (n {x:'",
            "'\\'}) DETACH DELETE (n {x:\\''",
        ),
    ],
)
def test_agtype_literals_escape_every_injection_vector(value, rendered):
    assert _agtype_value(value) == rendered


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), [1, 2], {"nested": 1}, object()]
)
def test_non_scalar_or_non_finite_property_values_fail_closed(value):
    with pytest.raises(HologresAGEGraphError):
        _agtype_value(value)


@pytest.mark.parametrize("key", ["bad-key", "1key", "k y", "", "k;y"])
def test_property_keys_outside_the_identifier_alphabet_fail_closed(key):
    with pytest.raises(HologresAGEGraphError):
        _agtype_map({key: "v"})


def test_agtype_map_renders_sorted_keys():
    assert (
        _agtype_map({"weight": 1.0, "entity_id": "a"})
        == "{entity_id: 'a', weight: 1.0}"
    )


async def test_dollar_tag_collisions_are_rejected(ready_storage):
    client = FakeAgeClient()
    storage = await ready_storage(client)

    with pytest.raises(HologresAGEGraphError):
        await storage.upsert_node(
            "$lightrag_age$", {"entity_id": "$lightrag_age$"}
        )


# ---------------------------------------------------------------------------
# Initialization: AGE path, fallback path
# ---------------------------------------------------------------------------


async def test_initialize_checks_graph_and_labels_before_creating(ready_storage):
    client = FakeAgeClient()
    client.handlers["age.graph.check"] = False
    client.handlers["age.graph.label.check"] = False

    await ready_storage(client)

    assert client.procedures == [
        ("create_graph", ("lightrag_age_ws1",), False),
        ("create_vlabel", ("lightrag_age_ws1", "Entity"), False),
        ("create_elabel", ("lightrag_age_ws1", "DIRECTED"), False),
    ]


async def test_initialize_skips_creation_when_graph_and_labels_exist(ready_storage):
    client = FakeAgeClient()

    await ready_storage(client)

    assert client.procedures == []


async def test_failed_probe_falls_back_to_the_two_table_storage(monkeypatch):
    import lightrag.kg.hologres.graph_age as module

    async def probe_version(client):
        return CapabilityReport(HologresVersion(5, 0, 0))

    async def probe_age(client):
        return ProbeResult(
            kind=ProbeKind.AGE,
            status=ProbeStatus.FAILED,
            blocking=False,
            detail_code="age_extension_missing",
        )

    created = {}

    class RecordingDelegate:
        def __init__(self, *, namespace, workspace, global_config, embedding_func, config):
            created.update(
                namespace=namespace,
                workspace=workspace,
                config=config,
            )
            self.initialized = False
            self.calls = []

        async def initialize(self):
            self.initialized = True

        async def has_node(self, node_id):
            self.calls.append(("has_node", node_id))
            return True

        async def drop(self):
            self.calls.append(("drop",))
            return {"status": "success", "message": "data dropped"}

        async def finalize(self):
            self.calls.append(("finalize",))

    monkeypatch.setattr(module, "probe_production_capabilities", probe_version)
    monkeypatch.setattr(module, "probe_age_graph_capability", probe_age)
    monkeypatch.setattr(module, "HologresGraphStorage", RecordingDelegate)

    client = FakeAgeClient()
    storage = make_storage(client=client)
    await storage.initialize()

    assert storage._delegate is not None
    assert storage._delegate.initialized is True
    assert created["namespace"] == NameSpace.GRAPH_STORE_CHUNK_ENTITY_RELATION
    assert created["workspace"] == "ws1"
    assert created["config"] is CONFIG

    assert await storage.has_node("x") is True
    assert await storage.drop() == {"status": "success", "message": "data dropped"}
    await storage.finalize()
    assert ("finalize",) in storage._delegate.calls if storage._delegate else True


async def test_indeterminate_probe_failure_fails_instead_of_falling_back(
    monkeypatch,
):
    # Any non-PASSED probe outcome other than a definitively missing AGE
    # extension is indeterminate; falling back would send writes to a
    # different physical store and hide an existing AGE graph.
    import lightrag.kg.hologres.graph_age as module

    async def probe_version(client):
        return CapabilityReport(HologresVersion(5, 0, 0))

    async def probe_age(client):
        return ProbeResult(
            kind=ProbeKind.AGE,
            status=ProbeStatus.FAILED,
            blocking=False,
            detail_code="age_graph_probe_failed",
        )

    monkeypatch.setattr(module, "probe_production_capabilities", probe_version)
    monkeypatch.setattr(module, "probe_age_graph_capability", probe_age)

    client = FakeAgeClient()
    storage = make_storage(client=client)
    with pytest.raises(HologresAGEGraphError, match="age_graph_probe_failed"):
        await storage.initialize()

    assert storage._delegate is None
    assert storage._initialized is False


# ---------------------------------------------------------------------------
# CRUD statement pinning
# ---------------------------------------------------------------------------


def _agtype_rows(*values):
    return [tuple(values)]


async def test_upsert_node_merges_then_set_plus_equals(ready_storage):
    client = FakeAgeClient()
    storage = await ready_storage(client)

    await storage.upsert_node(
        "alpha", {"entity_id": "ignored", "description": "d'1"}
    )

    merge, update = calls_for(client, "age.node.merge") + calls_for(
        client, "age.node.set"
    )
    assert merge["operation_kind"] is OperationKind.WRITE
    assert merge["replay_safe"] is True
    assert (
        "MERGE (n:Entity {entity_id: 'alpha'})" in merge["sql"]
    )
    assert update["replay_safe"] is True
    assert (
        "MATCH (n:Entity {entity_id: 'alpha'}) "
        "SET n += {description: 'd\\'1', entity_id: 'alpha'}"
    ) in update["sql"]
    assert update["sql"].startswith("SELECT * FROM ag_catalog.cypher(")
    assert "$lightrag_age$" in update["sql"]


async def test_upsert_node_requires_entity_id(ready_storage):
    client = FakeAgeClient()
    storage = await ready_storage(client)

    with pytest.raises(ValueError, match="entity_id"):
        await storage.upsert_node("alpha", {"description": "x"})


async def test_upsert_edge_normalizes_direction_and_replaces_properties(
    ready_storage,
):
    client = FakeAgeClient()
    storage = await ready_storage(client)

    await storage.upsert_edge("zeta", "alpha", {"weight": 2.5, "keywords": "k"})

    endpoints = calls_for(client, "age.edge.endpoint")
    assert [
        "MERGE (n:Entity {entity_id: 'alpha'})" in call["sql"]
        for call in endpoints
    ] == [True, False]
    assert "MERGE (n:Entity {entity_id: 'zeta'})" in endpoints[1]["sql"]
    # Edge creation and property replacement are one statement: an
    # interruption must never expose an edge without its properties.
    (upsert,) = calls_for(client, "age.edge.upsert")
    assert (
        "MATCH (a:Entity {entity_id: 'alpha'}), (b:Entity {entity_id: 'zeta'}) "
        "MERGE (a)-[r:DIRECTED]->(b) "
        "SET r = {keywords: 'k', weight: 2.5}"
    ) in upsert["sql"]


async def test_self_loop_edge_creates_the_endpoint_once(ready_storage):
    client = FakeAgeClient()
    storage = await ready_storage(client)

    await storage.upsert_edge("alpha", "alpha", {"weight": 1.0})

    assert len(calls_for(client, "age.edge.endpoint")) == 1


async def test_reads_bind_parameters_instead_of_literals(ready_storage):
    client = FakeAgeClient()
    client.handlers["age.node.read.batch"] = _agtype_rows(
        json.dumps("alpha"), json.dumps({"entity_id": "alpha", "d": "x"})
    )
    storage = await ready_storage(client)

    node = await storage.get_node("alpha")

    assert node == {"entity_id": "alpha", "d": "x"}
    (read,) = calls_for(client, "age.node.read.batch")
    assert read["sql"].endswith(
        "$lightrag_age$, $1) AS (id ag_catalog.agtype, props ag_catalog.agtype)"
    )
    assert read["values"] == ('{"entity_ids":["alpha"]}',)
    assert "IN $entity_ids" in read["sql"]


async def test_degrees_add_directed_self_loop_compensation(ready_storage):
    client = FakeAgeClient()
    client.handlers["age.degree.base"] = _agtype_rows(
        json.dumps("beta"), json.dumps(2)
    )
    client.handlers["age.degree.loops"] = _agtype_rows(
        json.dumps("beta"), json.dumps(1)
    )
    storage = await ready_storage(client)

    # The undirected cypher match sees a self-loop once; networkx counts it
    # twice, so the directed self-loop count is added on top.
    assert await storage.node_degree("beta") == 3
    (base,) = calls_for(client, "age.degree.base")
    assert "OPTIONAL MATCH (n)-[r:DIRECTED]-()" in base["sql"]
    (loops,) = calls_for(client, "age.degree.loops")
    assert "MATCH (n:Entity)-[r:DIRECTED]->(n)" in loops["sql"]


async def test_remove_nodes_embeds_escaped_literal_id_lists(ready_storage):
    client = FakeAgeClient()
    storage = await ready_storage(client)

    await storage.remove_nodes(["alpha", "o'brien"])

    (delete,) = calls_for(client, "age.node.delete")
    assert (
        "WHERE n.entity_id IN ['alpha', 'o\\'brien'] DETACH DELETE n"
    ) in delete["sql"]
    assert delete["replay_safe"] is True
    assert delete["values"] == ()


async def test_remove_edges_deletes_each_canonical_pair_once(ready_storage):
    client = FakeAgeClient()
    storage = await ready_storage(client)

    await storage.remove_edges([("zeta", "alpha"), ("alpha", "zeta")])

    (delete,) = calls_for(client, "age.edge.delete")
    assert (
        "MATCH (a:Entity {entity_id: 'alpha'})-[r:DIRECTED]->"
        "(b:Entity {entity_id: 'zeta'}) DELETE r"
    ) in delete["sql"]


async def test_get_node_edges_reports_three_outcomes(ready_storage):
    client = FakeAgeClient()
    client.handlers["age.edge.adjacency"] = []
    client.handlers["age.node.exists"] = _agtype_rows(json.dumps(0))
    storage = await ready_storage(client)

    assert await storage.get_node_edges("ghost") is None

    client.handlers["age.node.exists"] = _agtype_rows(json.dumps(1))
    assert await storage.get_node_edges("isolated") == []

    client.handlers["age.edge.adjacency"] = _agtype_rows(
        json.dumps("alpha"), json.dumps("beta")
    )
    assert await storage.get_node_edges("alpha") == [("alpha", "beta")]


async def test_search_labels_replicates_the_reference_scoring(ready_storage):
    client = FakeAgeClient()
    client.handlers["age.labels.search"] = [
        (json.dumps("Apple"),),
        (json.dumps("apple"),),
        (json.dumps("pineapple juice with apple"),),
        (json.dumps("crab_apple"),),
    ]
    storage = await ready_storage(client)

    ranked = await storage.search_labels("apple", limit=3)

    (search,) = calls_for(client, "age.labels.search")
    assert "toLower(n.entity_id) CONTAINS $needle" in search["sql"]
    assert search["values"] == ('{"needle":"apple"}',)
    # Both case-folded exacts score 1000; the tie-break is id ascending, so
    # "Apple" (A < a) ranks first, matching ORDER BY score DESC, id ASC.
    assert ranked == ["Apple", "apple", "crab_apple"]


async def test_drop_detach_deletes_the_workspace_graph(ready_storage):
    client = FakeAgeClient()
    storage = await ready_storage(client)

    assert await storage.drop() == {
        "status": "success",
        "message": "data dropped",
    }
    (drop,) = calls_for(client, "age.drop")
    assert "MATCH (n:Entity) DETACH DELETE n" in drop["sql"]
    assert drop["replay_safe"] is True


async def test_knowledge_graph_hydrates_edges_across_id_chunks(ready_storage):
    from lightrag.kg.hologres.graph_age import _ID_CHUNK_SIZE

    neighbours = [f"n{index:03d}" for index in range(_ID_CHUNK_SIZE)]
    star_edges = {("hub", neighbour) for neighbour in neighbours}
    every_node = ["hub", *neighbours]

    def requested(values):
        return json.loads(values[0])

    def hop(_sql, values):
        seeds = set(requested(values)["entity_ids"])
        peers = {
            peer
            for src, tgt in star_edges
            for peer in ((tgt,) if src in seeds else ())
        }
        peers |= {src for src, tgt in star_edges if tgt in seeds}
        return [(json.dumps(peer),) for peer in sorted(peers)]

    def degrees(_sql, values):
        asked = requested(values)["entity_ids"]
        return [
            (
                json.dumps(node_id),
                json.dumps(len(neighbours) if node_id == "hub" else 1),
            )
            for node_id in asked
        ]

    def node_rows(_sql, values):
        return [
            (json.dumps(node_id), json.dumps({"entity_id": node_id}))
            for node_id in requested(values)["entity_ids"]
        ]

    def edge_rows(sql, values):
        params = requested(values)
        # Falling back to entity_ids emulates the buggy same-chunk contract.
        targets = set(params.get("target_ids", params.get("entity_ids", [])))
        sources = set(params.get("source_ids", targets))
        if "source_ids" in params:
            assert "a.entity_id IN $source_ids" in sql
            assert "b.entity_id IN $target_ids" in sql
        return [
            (json.dumps(src), json.dumps(tgt), json.dumps({"weight": 1}))
            for src, tgt in sorted(star_edges)
            if src in sources and tgt in targets
        ]

    client = FakeAgeClient()
    client.handlers.update(
        {
            "age.node.exists": _agtype_rows(json.dumps(1)),
            "age.kg.hop": hop,
            "age.degree.base": degrees,
            "age.degree.loops": [],
            "age.node.read.batch": node_rows,
            "age.kg.edges": edge_rows,
        }
    )
    storage = await ready_storage(client)

    graph = await storage.get_knowledge_graph("hub", max_depth=1, max_nodes=300)

    assert len(graph.nodes) == len(every_node)
    assert graph.is_truncated is False
    # A same-chunk endpoint filter would omit the final neighbour's edge.
    assert {(edge.source, edge.target) for edge in graph.edges} == star_edges
    hydration = calls_for(client, "age.kg.edges")
    assert len(hydration) == 2
    for call in hydration:
        params = requested(call["values"])
        assert params["target_ids"] == sorted(every_node)
        assert len(params["source_ids"]) <= _ID_CHUNK_SIZE


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (json.dumps({"a": 1}), {"a": 1}),
        (json.dumps("alpha"), "alpha"),
        (json.dumps(3), 3),
    ],
)
def test_agtype_decoder_accepts_json_scalars_and_maps(value, expected):
    assert _decode_agtype(value) == expected
    if isinstance(expected, str):
        assert _decode_string(value) == expected
    elif isinstance(expected, int):
        assert _decode_int(value) == expected
    else:
        assert _decode_map(value) == expected


@pytest.mark.parametrize("value", [None, 7, b"{}", "not-json"])
def test_agtype_decoder_rejects_corrupt_values(value):
    with pytest.raises(HologresAGEGraphError, match="corrupt"):
        _decode_agtype(value)


def test_agtype_specific_decoders_reject_wrong_json_types():
    with pytest.raises(HologresAGEGraphError, match="corrupt"):
        _decode_string(json.dumps(3))
    with pytest.raises(HologresAGEGraphError, match="corrupt"):
        _decode_int(json.dumps("3"))
    with pytest.raises(HologresAGEGraphError, match="corrupt"):
        _decode_int(json.dumps(True))
    with pytest.raises(HologresAGEGraphError, match="corrupt"):
        _decode_map(json.dumps([]))


def test_age_row_and_params_helpers_fail_closed():
    with pytest.raises(HologresAGEGraphError, match="corrupt"):
        _materialize_rows(7)
    with pytest.raises(HologresAGEGraphError, match="invalid"):
        _params_json({"ids": float("nan")})
    assert _materialize_rows((row for row in [(1, 2)])) == [(1, 2)]


async def test_initialize_rejects_an_injected_client_without_config():
    storage = make_storage(client=object(), config=None)

    with pytest.raises(HologresAGEGraphError, match="configuration is unavailable"):
        await storage.initialize()


async def test_owned_age_client_is_released_once_and_close_failure_is_reported(
    monkeypatch,
):
    import lightrag.kg.hologres.graph_age as module

    class OwnedClient(FakeAgeClient):
        def __init__(self, *, close_error=None):
            super().__init__()
            self.close_error = close_error
            self.close_count = 0

        async def close(self):
            self.close_count += 1
            if self.close_error is not None:
                raise self.close_error

    async def probe_version(_client):
        return CapabilityReport(HologresVersion(5, 0, 0))

    async def probe_age(_client):
        return ProbeResult(
            kind=ProbeKind.AGE,
            status=ProbeStatus.PASSED,
            blocking=False,
            detail_code="age_graph_semantics_frozen",
        )

    monkeypatch.setattr(module, "probe_production_capabilities", probe_version)
    monkeypatch.setattr(module, "probe_age_graph_capability", probe_age)

    clients = []

    def factory(config):
        assert config.age_search_path is True
        client = OwnedClient()
        clients.append(client)
        return client

    monkeypatch.setattr(module, "HologresClient", factory)
    storage = make_storage()
    await storage.initialize()
    await storage.initialize()
    await storage.finalize()
    await storage.finalize()
    assert clients[0].close_count == 1

    failing = OwnedClient(close_error=RuntimeError("secret-close-failure"))
    monkeypatch.setattr(module, "HologresClient", lambda _config: failing)
    second = make_storage(workspace="ws2")
    await second.initialize()
    with pytest.raises(HologresAGEGraphError, match="client close failed") as error:
        await second.finalize()
    assert "secret-close-failure" not in str(error.value)


async def test_age_graph_initialization_and_read_write_errors_are_wrapped(
    ready_storage,
):
    client = FakeAgeClient()
    client.handlers["age.graph.check"] = RuntimeError("catalog-secret")
    with pytest.raises(HologresAGEGraphError, match="graph initialization failed"):
        await ready_storage(client)
    assert "catalog-secret" not in client.calls[0]["sql"]

    client = FakeAgeClient()
    storage = await ready_storage(client)
    client.handlers["age.node.exists"] = RuntimeError("transport-secret")
    with pytest.raises(HologresAGEGraphError, match="graph read failed") as error:
        await storage.has_node("alpha")
    assert "transport-secret" not in str(error.value)

    client.handlers.pop("age.node.exists")
    client.handlers["age.node.merge"] = RuntimeError("write-secret")
    with pytest.raises(HologresAGEGraphError, match="graph write failed") as error:
        await storage.upsert_node("alpha", {"entity_id": "alpha"})
    assert "write-secret" not in str(error.value)


async def test_age_batch_node_reads_validate_input_and_chunk_large_batches(
    ready_storage,
):
    storage = await ready_storage(FakeAgeClient())
    client = storage._active_client
    with pytest.raises(HologresAGEGraphError, match="input is invalid"):
        await storage.get_nodes_batch(["alpha", 3])

    ids = [f"n{index:03d}" for index in range(_ID_CHUNK_SIZE + 1)]

    def nodes(_sql, values):
        requested = set(json.loads(values[0])["entity_ids"])
        return [
            (
                json.dumps(node_id),
                json.dumps({"entity_id": node_id, "description": node_id}),
            )
            for node_id in sorted(requested)
        ]

    client.handlers["age.node.read.batch"] = nodes
    result = await storage.get_nodes_batch(ids)
    assert set(result) == set(ids)
    read_sizes = [
        len(json.loads(call["values"][0])["entity_ids"])
        for call in calls_for(client, "age.node.read.batch")
    ]
    assert read_sizes == [_ID_CHUNK_SIZE, 1]


async def test_age_edge_batch_reads_canonicalize_dedupe_and_chunk(ready_storage):
    client = FakeAgeClient()
    edges = {
        ("alpha", "beta"): {"weight": 1},
        ("delta", "gamma"): {"weight": 2},
    }

    def edge_rows(_sql, values):
        requested = set(json.loads(values[0])["entity_ids"])
        return [
            (
                json.dumps(src),
                json.dumps(tgt),
                json.dumps(properties),
            )
            for (src, tgt), properties in sorted(edges.items())
            if src in requested and tgt in requested
        ]

    client.handlers["age.edge.read.batch"] = edge_rows
    storage = await ready_storage(client)

    result = await storage.get_edges_batch(
        [
            {"src": "beta", "tgt": "alpha"},
            {"src": "alpha", "tgt": "beta"},
            {"src": "gamma", "tgt": "delta"},
            {"src": "missing", "tgt": "gamma"},
        ]
    )
    assert result == {
        ("beta", "alpha"): {"weight": 1},
        ("alpha", "beta"): {"weight": 1},
        ("gamma", "delta"): {"weight": 2},
    }
    reads = calls_for(client, "age.edge.read.batch")
    assert len({call["values"][0] for call in reads}) == 1

    with pytest.raises(HologresAGEGraphError, match="input is invalid"):
        await storage.get_edges_batch([{"src": "alpha"}])
    with pytest.raises(HologresAGEGraphError, match="input is invalid"):
        await storage.get_edges_batch(["alpha,beta"])


async def test_age_edge_upsert_and_remove_batches_dedupe_canonical_pairs(
    ready_storage,
):
    client = FakeAgeClient()
    storage = await ready_storage(client)

    await storage.upsert_edges_batch(
        [
            ("zeta", "alpha", {"weight": 1}),
            ("alpha", "zeta", {"weight": 2}),
        ]
    )
    upserts = calls_for(client, "age.edge.upsert")
    assert len(upserts) == 1
    assert "{weight: 2}" in upserts[0]["sql"]

    await storage.remove_edges([("zeta", "alpha"), ("alpha", "zeta")])
    deletes = calls_for(client, "age.edge.delete")
    assert len(deletes) == 1

    with pytest.raises(HologresAGEGraphError, match="non-None strings"):
        await storage.remove_edges([(None, "alpha")])


async def test_age_degrees_and_adjacency_batches_cover_empty_and_chunked_inputs(
    ready_storage,
):
    client = FakeAgeClient()

    def degrees(_sql, values):
        requested = json.loads(values[0])["entity_ids"]
        return [
            (json.dumps(node_id), json.dumps(2 if node_id != "hub" else 4))
            for node_id in requested
        ]

    client.handlers["age.degree.base"] = degrees
    client.handlers["age.degree.loops"] = [
        (json.dumps("hub"), json.dumps(1)),
    ]
    client.handlers["age.edge.adjacency"] = lambda _sql, values: [
        (json.dumps(node_id), json.dumps("peer"))
        for node_id in json.loads(values[0])["entity_ids"]
    ]
    storage = await ready_storage(client)

    assert await storage.node_degrees_batch([]) == {}
    assert await storage.edge_degrees_batch([]) == {}
    assert await storage.node_degrees_batch(["a", "a"]) == {"a": 2}
    assert await storage.edge_degrees_batch([("a", "a")]) == {("a", "a"): 4}
    assert await storage.get_nodes_edges_batch([]) == {}
    assert await storage.get_nodes_edges_batch(["a", "a"]) == {
        "a": [("a", "peer")]
    }


async def test_age_label_export_and_wildcard_knowledge_graph_paths(ready_storage):
    client = FakeAgeClient()
    labels = ["zeta", "alpha", "alpha", "beta"]
    client.handlers["age.labels.all"] = [(json.dumps(label),) for label in labels]
    client.handlers["age.labels.popular"] = [(json.dumps(label),) for label in labels]
    client.handlers["age.kg.wildcard"] = [(json.dumps(label),) for label in labels]

    def degrees(_sql, values):
        requested = json.loads(values[0])["entity_ids"]
        return [
            (
                json.dumps(node_id),
                json.dumps({"alpha": 3, "beta": 2, "zeta": 1}.get(node_id, 0)),
            )
            for node_id in requested
        ]

    client.handlers["age.degree.base"] = degrees
    client.handlers["age.degree.loops"] = []

    def node_rows(_sql, values):
        requested = json.loads(values[0])["entity_ids"]
        return [
            (json.dumps(node_id), json.dumps({"entity_id": node_id}))
            for node_id in requested
        ]

    client.handlers["age.node.read.batch"] = node_rows
    client.handlers["age.export.nodes"] = [
        (json.dumps("beta"), json.dumps({"entity_id": "beta"})),
        (json.dumps("alpha"), json.dumps({"entity_id": "alpha"})),
    ]
    client.handlers["age.export.edges"] = [
        (
            json.dumps("beta"),
            json.dumps("alpha"),
            json.dumps({"weight": 1}),
        )
    ]
    client.handlers["age.kg.edges"] = [
        (
            json.dumps("beta"),
            json.dumps("alpha"),
            json.dumps({"weight": 1}),
        ),
        (
            json.dumps("beta"),
            json.dumps("alpha"),
            json.dumps({"weight": 1}),
        ),
    ]
    storage = await ready_storage(client)

    assert await storage.get_all_labels() == ["alpha", "alpha", "beta", "zeta"]
    assert await storage.get_popular_labels(limit=2) == ["alpha", "alpha"]
    with pytest.raises(HologresAGEGraphError, match="limit is invalid"):
        await storage.get_popular_labels(limit=0)
    with pytest.raises(HologresAGEGraphError, match="input is invalid"):
        await storage.search_labels(7)
    assert await storage.search_labels("   ") == []

    assert [node["id"] for node in await storage.get_all_nodes()] == [
        "alpha",
        "beta",
    ]
    assert await storage.get_all_edges() == [
        {"weight": 1, "source": "beta", "target": "alpha"}
    ]

    graph = await storage.get_knowledge_graph("*", max_nodes=2)
    assert [node.id for node in graph.nodes] == ["alpha", "beta"]
    assert graph.is_truncated is True
    assert [(edge.source, edge.target) for edge in graph.edges] == [
        ("beta", "alpha")
    ]


async def test_every_age_public_operation_delegates_after_capability_fallback():
    class FullDelegate:
        def __init__(self):
            self.calls = []

        def __getattr__(self, name):
            async def method(*args, **kwargs):
                self.calls.append((name, args, kwargs))
                if name == "has_node":
                    return True
                if name == "get_node":
                    return {"entity_id": "alpha"}
                if name == "get_nodes_batch":
                    return {"alpha": {"entity_id": "alpha"}}
                if name == "has_nodes_batch":
                    return {"alpha"}
                if name in {"has_edge", "get_node_edges"}:
                    return True if name == "has_edge" else [("a", "b")]
                if name == "get_edge":
                    return {"weight": 1}
                if name == "get_edges_batch":
                    return {("a", "b"): {"weight": 1}}
                if name == "get_nodes_edges_batch":
                    return {"a": [("a", "b")]}
                if name in {"node_degree", "edge_degree"}:
                    return 2
                if name == "node_degrees_batch":
                    return {"a": 2}
                if name == "edge_degrees_batch":
                    return {("a", "b"): 4}
                if name in {"get_all_labels", "get_popular_labels", "search_labels"}:
                    return ["alpha"]
                if name == "get_all_nodes":
                    return [{"id": "alpha"}]
                if name == "get_all_edges":
                    return [{"source": "a", "target": "b"}]
                if name == "get_knowledge_graph":
                    return "knowledge-graph"
                if name == "drop":
                    return {"status": "success"}
                return None

            return method

    delegate = FullDelegate()
    storage = make_storage()
    storage._delegate = delegate
    storage._initialized = True

    assert repr(storage) == "HologresAGEGraphStorage(<redacted>)"
    assert await storage.index_done_callback() is None
    assert await storage.has_node("a") is True
    assert await storage.get_node("a") == {"entity_id": "alpha"}
    assert await storage.get_nodes_batch(["a"]) == {
        "alpha": {"entity_id": "alpha"}
    }
    assert await storage.has_nodes_batch(["a"]) == {"alpha"}
    await storage.upsert_node("a", {"entity_id": "a"})
    await storage.upsert_nodes_batch([("a", {"entity_id": "a"})])
    await storage.delete_node("a")
    await storage.remove_nodes(["a"])
    assert await storage.has_edge("a", "b") is True
    assert await storage.get_edge("a", "b") == {"weight": 1}
    await storage.upsert_edge("a", "b", {"weight": 1})
    await storage.upsert_edges_batch([("a", "b", {"weight": 1})])
    await storage.remove_edges([("a", "b")])
    assert await storage.get_edges_batch([{"src": "a", "tgt": "b"}]) == {
        ("a", "b"): {"weight": 1}
    }
    assert await storage.get_node_edges("a") == [("a", "b")]
    assert await storage.get_nodes_edges_batch(["a"]) == {"a": [("a", "b")]}
    assert await storage.node_degree("a") == 2
    assert await storage.edge_degree("a", "b") == 2
    assert await storage.node_degrees_batch(["a"]) == {"a": 2}
    assert await storage.edge_degrees_batch([("a", "b")]) == {("a", "b"): 4}
    assert await storage.get_all_labels() == ["alpha"]
    assert await storage.get_popular_labels() == ["alpha"]
    assert await storage.search_labels("a") == ["alpha"]
    assert await storage.get_all_nodes() == [{"id": "alpha"}]
    assert await storage.get_all_edges() == [{"source": "a", "target": "b"}]
    assert (
        await storage.get_knowledge_graph("a", max_depth=1, max_nodes=2)
        == "knowledge-graph"
    )
    assert await storage.drop() == {"status": "success"}

    delegated_names = [call[0] for call in delegate.calls]
    assert delegated_names == [
        "index_done_callback",
        "has_node",
        "get_node",
        "get_nodes_batch",
        "has_nodes_batch",
        "upsert_node",
        "upsert_nodes_batch",
        "delete_node",
        "remove_nodes",
        "has_edge",
        "get_edge",
        "upsert_edge",
        "upsert_edges_batch",
        "remove_edges",
        "get_edges_batch",
        "get_node_edges",
        "get_nodes_edges_batch",
        "node_degree",
        "edge_degree",
        "node_degrees_batch",
        "edge_degrees_batch",
        "get_all_labels",
        "get_popular_labels",
        "search_labels",
        "get_all_nodes",
        "get_all_edges",
        "get_knowledge_graph",
        "drop",
    ]


# ---------------------------------------------------------------------------
# Static source guard
# ---------------------------------------------------------------------------


def test_graph_age_module_uses_only_the_restricted_client_surface():
    source = (
        Path(__file__).resolve().parents[3] / "lightrag/kg/hologres/graph_age.py"
    ).read_text(encoding="utf-8")
    lowered = source.lower()
    assert "asyncpg" not in lowered
    assert ".transaction(" not in lowered
    assert "executemany" not in lowered
    assert ".acquire(" not in lowered
    assert "postgres_impl" not in lowered
    assert "pg_advisory" not in lowered
    assert "begin;" not in lowered
    assert "commit;" not in lowered
    # Graph lifecycle CALLs go only through the whitelisted channel; the
    # module never builds a raw CALL statement itself.
    assert '"CALL ' not in source
    assert "'CALL " not in source
