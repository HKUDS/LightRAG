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
    (merge,) = calls_for(client, "age.edge.merge")
    assert (
        "MATCH (a:Entity {entity_id: 'alpha'}), (b:Entity {entity_id: 'zeta'}) "
        "MERGE (a)-[:DIRECTED]->(b)"
    ) in merge["sql"]
    assert "[r:" not in merge["sql"]
    (set_call,) = calls_for(client, "age.edge.set")
    assert (
        "MATCH (a:Entity {entity_id: 'alpha'})-[r:DIRECTED]->"
        "(b:Entity {entity_id: 'zeta'}) SET r = {keywords: 'k', weight: 2.5}"
    ) in set_call["sql"]


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
