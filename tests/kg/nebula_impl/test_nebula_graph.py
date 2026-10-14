"""
Tests for NebulaStorage (graph storage backend).

These tests verify that the NebulaStorage class correctly implements
the BaseGraphStorage interface using ISO GQL.

Run offline (mock) tests:
    ./scripts/test.sh tests/kg/nebula_impl/

Run integration tests (requires live NebulaGraph):
    LIGHTRAG_RUN_INTEGRATION=true ./scripts/test.sh tests/kg/nebula_impl/ -m integration
"""

import os

import pytest
import numpy as np

from lightrag.kg.nebula_impl import NebulaStorage
from lightrag.kg.shared_storage import initialize_share_data


# ── mock embedding for integration tests ──────────────────────────────
async def _mock_embedding(texts):
    return np.random.rand(len(texts), 10)


# ══════════════════════════════════════════════════════════════════════
# Mock fixtures (offline tests — no real DB needed)
# ══════════════════════════════════════════════════════════════════════


class _FakeResultSet:
    """Mock NebulaGraph ResultSet for testing (v5.3.0 API)."""

    def __init__(self, rows_data=None):
        self._rows_data = rows_data or []

    def raise_on_error(self):
        return self

    def as_primitive_by_row(self):
        yield from self._rows_data


class _FakeAsyncClient:
    """Mock AsyncNebulaClient."""

    def __init__(self):
        self.statements: list[str] = []

    async def _init_client(self):
        pass

    async def execute(self, statement: str):
        self.statements.append(statement)
        return _FakeResultSet()

    async def execute_with_timeout(self, statement: str, timeout_ms: int):
        self.statements.append(statement)
        return _FakeResultSet()

    async def close(self):
        pass


@pytest.fixture
def mock_client():
    return _FakeAsyncClient()


@pytest.fixture
def nebula_storage(mock_client):
    """Create a NebulaStorage instance with a mock client."""
    storage = NebulaStorage.__new__(NebulaStorage)
    storage.workspace = "test"
    storage.global_config = {"max_graph_nodes": 1000}
    storage.embedding_func = None
    storage._client = mock_client
    storage._graph = "test_space"
    storage._entity_label = "entity_test"
    storage._edge_label = "directed_test"

    # Patch _query for tests that need structured results
    storage._query_results = None

    async def fake_query(gql):
        if storage._query_results is not None:
            return storage._query_results
        return []

    storage._query = fake_query
    return storage


# ══════════════════════════════════════════════════════════════════════
# Integration fixture (real NebulaGraph connection)
# ══════════════════════════════════════════════════════════════════════


@pytest.fixture
async def nebula_integration_storage():
    """Initialize a real NebulaStorage backed by a live NebulaGraph instance.

    Uses a dedicated ``lightrag_test`` graph space so production data in
    the default ``lightrag`` space is never touched.
    Skips if NEBULA_GRAPH_HOSTS is not set.
    """
    if not os.getenv("NEBULA_GRAPH_HOSTS"):
        pytest.skip("NebulaGraph not configured (NEBULA_GRAPH_HOSTS not set)")

    # isolate test runs from the default production graph space
    os.environ["NEBULA_GRAPH"] = "lightrag_test"

    initialize_share_data()

    global_config = {
        "embedding_batch_num": 10,
        "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.5},
        "working_dir": os.environ.get("WORKING_DIR", "./rag_storage"),
        "max_graph_nodes": 1000,
    }

    storage = NebulaStorage(
        namespace="test_nebula_integration",
        workspace="test_nebula",
        global_config=global_config,
        embedding_func=_mock_embedding,
    )

    await storage.initialize()

    yield storage

    try:
        await storage.drop()
    except Exception:
        pass
    await storage.finalize()


# ══════════════════════════════════════════════════════════════════════
# Offline tests (mock-based, fast, no DB)
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.offline
@pytest.mark.asyncio
async def test_has_node_returns_false_when_no_match(nebula_storage):
    nebula_storage._query_results = []
    result = await nebula_storage.has_node("nonexistent")
    assert result is False


@pytest.mark.offline
@pytest.mark.asyncio
async def test_has_node_returns_true_when_match(nebula_storage):
    nebula_storage._query_results = [{"n": {"entity_id": "test-node"}}]
    result = await nebula_storage.has_node("test-node")
    assert result is True


@pytest.mark.offline
@pytest.mark.asyncio
async def test_has_edge_returns_true(nebula_storage):
    nebula_storage._query_results = [{"r": {}}]
    result = await nebula_storage.has_edge("src", "tgt")
    assert result is True


@pytest.mark.offline
@pytest.mark.asyncio
async def test_has_edge_returns_false(nebula_storage):
    nebula_storage._query_results = []
    result = await nebula_storage.has_edge("src", "tgt")
    assert result is False


@pytest.mark.offline
@pytest.mark.asyncio
async def test_get_node_returns_properties(nebula_storage):
    nebula_storage._query_results = [
        {"n": {"entity_id": "n1", "entity_type": "Person", "description": "desc"}}
    ]
    result = await nebula_storage.get_node("n1")
    assert result == {"entity_id": "n1", "entity_type": "Person", "description": "desc"}


@pytest.mark.offline
@pytest.mark.asyncio
async def test_get_node_returns_none(nebula_storage):
    nebula_storage._query_results = []
    result = await nebula_storage.get_node("n1")
    assert result is None


@pytest.mark.offline
@pytest.mark.asyncio
async def test_node_degree(nebula_storage):
    nebula_storage._query_results = [{"degree": 5}]
    result = await nebula_storage.node_degree("n1")
    assert result == 5


@pytest.mark.offline
@pytest.mark.asyncio
async def test_get_edge_returns_properties(nebula_storage):
    nebula_storage._query_results = [
        {"r": {"weight": 1.0, "description": "knows", "keywords": "friend"}}
    ]
    result = await nebula_storage.get_edge("src", "tgt")
    assert result["weight"] == 1.0
    assert result["description"] == "knows"
    assert result["keywords"] == "friend"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_get_edge_defaults_missing_keys(nebula_storage):
    nebula_storage._query_results = [{"r": {}}]
    result = await nebula_storage.get_edge("src", "tgt")
    assert result["weight"] == 1.0
    assert result["source_id"] is None


@pytest.mark.offline
@pytest.mark.asyncio
async def test_get_node_edges(nebula_storage):
    nebula_storage._query_results = [
        {"src_entity_id": "n1", "tgt_entity_id": "n2"},
        {"src_entity_id": "n1", "tgt_entity_id": "n3"},
    ]
    result = await nebula_storage.get_node_edges("n1")
    assert ("n1", "n2") in result
    assert ("n1", "n3") in result


@pytest.mark.offline
@pytest.mark.asyncio
async def test_upsert_node_raises_without_entity_id(nebula_storage):
    with pytest.raises(ValueError, match="entity_id"):
        await nebula_storage.upsert_node("n1", {"description": "missing entity_id"})


@pytest.mark.offline
@pytest.mark.asyncio
async def test_upsert_node_sends_iso_gql(nebula_storage, mock_client):
    await nebula_storage.upsert_node(
        "n1",
        {"entity_id": "n1", "entity_type": "Person", "description": "A person"},
    )
    assert len(mock_client.statements) > 0
    last_stmt = mock_client.statements[-1]
    assert "INSERT OR REPLACE" in last_stmt
    assert "entity_test" in last_stmt


@pytest.mark.offline
@pytest.mark.asyncio
async def test_upsert_edge_sends_iso_gql(nebula_storage, mock_client):
    await nebula_storage.upsert_edge(
        "src",
        "tgt",
        {"weight": 1.0, "description": "knows", "keywords": "friend"},
    )
    assert len(mock_client.statements) > 0
    last_stmt = mock_client.statements[-1]
    assert "INSERT OR REPLACE" in last_stmt
    assert "MATCH" in last_stmt
    assert "directed_test" in last_stmt


@pytest.mark.offline
@pytest.mark.asyncio
async def test_get_all_labels(nebula_storage):
    nebula_storage._query_results = [
        {"entity_type": "Person"},
        {"entity_type": "Organization"},
        {"entity_type": "Person"},
    ]
    result = await nebula_storage.get_all_labels()
    assert "Person" in result
    assert "Organization" in result


@pytest.mark.offline
@pytest.mark.asyncio
async def test_get_nodes_batch(nebula_storage):
    nebula_storage._query_results = [
        {"n": {"entity_id": "n1", "entity_type": "Person"}},
        {"n": {"entity_id": "n2", "entity_type": "Org"}},
    ]
    result = await nebula_storage.get_nodes_batch(["n1", "n2"])
    assert "n1" in result
    assert "n2" in result
    assert result["n1"]["entity_type"] == "Person"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_drop_returns_success(nebula_storage, mock_client):
    result = await nebula_storage.drop()
    assert result["status"] == "success"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_get_knowledge_graph_star_mode(nebula_storage):
    nebula_storage._query_results = [{"total": 1}]

    async def multi_query(gql):
        if "count(n) AS total" in gql:
            return [{"total": 1}]
        if "ORDER BY degree" in gql:
            return [{"n": {"entity_id": "n1", "entity_type": "Person"}}]
        if "WHERE a.entity_id IN" in gql:
            return []
        return []

    nebula_storage._query = multi_query

    result = await nebula_storage.get_knowledge_graph("*")
    assert len(result.nodes) == 1
    assert result.nodes[0].id == "n1"


@pytest.mark.offline
def test_escape_iso_gql_string(nebula_storage):
    assert nebula_storage._escape_iso_gql_string("hello") == "hello"
    assert nebula_storage._escape_iso_gql_string("it's") == "it\\'s"
    assert nebula_storage._escape_iso_gql_string("back\\slash") == "back\\\\slash"


@pytest.mark.offline
def test_format_iso_props(nebula_storage):
    props = {"name": "John", "age": 30, "active": True, "score": 3.14, "extra": None}
    result = nebula_storage._format_iso_props(props)
    assert "name: 'John'" in result
    assert "age: 30" in result
    assert "active: true" in result
    assert "score: 3.14" in result
    assert "extra: NULL" in result


@pytest.mark.offline
def test_workspace_label_generation():
    """Test workspace label sanitization."""
    storage = NebulaStorage.__new__(NebulaStorage)
    storage.workspace = "my-workspace"
    label = storage._get_workspace_label()
    assert label == "my_workspace"

    storage.workspace = ""
    label = storage._get_workspace_label()
    assert label == "base"

    storage.workspace = "123invalid"
    label = storage._get_workspace_label()
    assert label.startswith("ws_")


# ══════════════════════════════════════════════════════════════════════
# Integration tests (real NebulaGraph — requires --run-integration)
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration_initialize_creates_graph(nebula_integration_storage):
    """Verify that initialize() connects and creates graph + element types."""
    s = nebula_integration_storage
    # The fixture already called initialize() — just check labels are set
    assert s._entity_label.startswith("entity_")
    assert s._edge_label.startswith("directed_")
    assert s._client is not None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration_upsert_and_get_node(nebula_integration_storage):
    """Upsert a node then read it back."""
    s = nebula_integration_storage

    await s.upsert_node(
        "integration-test-node",
        {
            "entity_id": "integration-test-node",
            "entity_type": "Person",
            "description": "Integration test entity",
            "source_id": "chunk-001",
            "file_path": "test.pdf",
            "created_at": 1234567890,
        },
    )

    result = await s.get_node("integration-test-node")
    assert result is not None
    assert result["entity_id"] == "integration-test-node"
    assert result["entity_type"] == "Person"
    assert result["description"] == "Integration test entity"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration_has_node(nebula_integration_storage):
    s = nebula_integration_storage

    await s.upsert_node(
        "has-node-test",
        {"entity_id": "has-node-test", "entity_type": "Test"},
    )

    assert await s.has_node("has-node-test") is True
    assert await s.has_node("nonexistent-node") is False


@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration_upsert_nodes_batch(nebula_integration_storage):
    s = nebula_integration_storage

    nodes = [
        (
            "batch-n1",
            {
                "entity_id": "batch-n1",
                "entity_type": "Person",
                "description": "batch 1",
            },
        ),
        (
            "batch-n2",
            {"entity_id": "batch-n2", "entity_type": "Org", "description": "batch 2"},
        ),
        (
            "batch-n3",
            {
                "entity_id": "batch-n3",
                "entity_type": "Person",
                "description": "batch 3",
            },
        ),
    ]
    await s.upsert_nodes_batch(nodes)

    result = await s.get_nodes_batch(["batch-n1", "batch-n2", "batch-n3"])
    assert len(result) == 3
    assert result["batch-n1"]["description"] == "batch 1"
    assert result["batch-n2"]["entity_type"] == "Org"
    assert result["batch-n3"]["description"] == "batch 3"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration_upsert_and_get_edge(nebula_integration_storage):
    s = nebula_integration_storage

    # Create both endpoint nodes first
    await s.upsert_node("edge-src", {"entity_id": "edge-src", "entity_type": "Person"})
    await s.upsert_node("edge-tgt", {"entity_id": "edge-tgt", "entity_type": "Person"})

    await s.upsert_edge(
        "edge-src",
        "edge-tgt",
        {
            "weight": 2.0,
            "description": "knows",
            "keywords": "friend, colleague",
            "source_id": "chunk-002",
        },
    )

    edge = await s.get_edge("edge-src", "edge-tgt")
    assert edge is not None
    assert edge["weight"] == 2.0
    assert edge["description"] == "knows"
    assert "friend" in edge["keywords"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration_has_edge(nebula_integration_storage):
    s = nebula_integration_storage

    await s.upsert_node("he-src", {"entity_id": "he-src", "entity_type": "Test"})
    await s.upsert_node("he-tgt", {"entity_id": "he-tgt", "entity_type": "Test"})
    await s.upsert_edge("he-src", "he-tgt", {"weight": 1.0, "description": "test"})

    assert await s.has_edge("he-src", "he-tgt") is True
    assert await s.has_edge("he-tgt", "he-src") is True  # undirected
    assert await s.has_edge("he-src", "nonexistent") is False


@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration_node_degree(nebula_integration_storage):
    s = nebula_integration_storage

    await s.upsert_node("deg-center", {"entity_id": "deg-center", "entity_type": "Hub"})
    await s.upsert_node("deg-1", {"entity_id": "deg-1", "entity_type": "Node"})
    await s.upsert_node("deg-2", {"entity_id": "deg-2", "entity_type": "Node"})
    await s.upsert_node("deg-3", {"entity_id": "deg-3", "entity_type": "Node"})

    await s.upsert_edge("deg-center", "deg-1", {"weight": 1.0, "description": "edge"})
    await s.upsert_edge("deg-center", "deg-2", {"weight": 1.0, "description": "edge"})
    await s.upsert_edge("deg-center", "deg-3", {"weight": 1.0, "description": "edge"})

    assert await s.node_degree("deg-center") == 3
    assert await s.node_degree("deg-1") == 1
    assert await s.node_degree("nonexistent") == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration_get_node_edges(nebula_integration_storage):
    s = nebula_integration_storage

    await s.upsert_node("edges-n1", {"entity_id": "edges-n1", "entity_type": "X"})
    await s.upsert_node("edges-n2", {"entity_id": "edges-n2", "entity_type": "X"})
    await s.upsert_node("edges-n3", {"entity_id": "edges-n3", "entity_type": "X"})

    await s.upsert_edge("edges-n1", "edges-n2", {"weight": 1.0, "description": "a"})
    await s.upsert_edge("edges-n1", "edges-n3", {"weight": 1.0, "description": "b"})

    edges = await s.get_node_edges("edges-n1")
    assert edges is not None
    assert ("edges-n1", "edges-n2") in edges
    assert ("edges-n1", "edges-n3") in edges


@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration_delete_node(nebula_integration_storage):
    s = nebula_integration_storage

    await s.upsert_node("del-node", {"entity_id": "del-node", "entity_type": "Temp"})
    assert await s.has_node("del-node") is True

    await s.delete_node("del-node")
    assert await s.has_node("del-node") is False


@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration_get_all_labels(nebula_integration_storage):
    s = nebula_integration_storage

    await s.upsert_node("lbl-1", {"entity_id": "lbl-1", "entity_type": "Animal"})
    await s.upsert_node("lbl-2", {"entity_id": "lbl-2", "entity_type": "Animal"})
    await s.upsert_node("lbl-3", {"entity_id": "lbl-3", "entity_type": "Plant"})

    labels = await s.get_all_labels()
    assert "Animal" in labels
    assert "Plant" in labels


@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration_drop_clears_data(nebula_integration_storage):
    s = nebula_integration_storage

    await s.upsert_node("drop-1", {"entity_id": "drop-1", "entity_type": "X"})
    await s.upsert_node("drop-2", {"entity_id": "drop-2", "entity_type": "X"})

    result = await s.drop()
    assert result["status"] == "success"

    assert await s.has_node("drop-1") is False
    assert await s.has_node("drop-2") is False


@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration_get_knowledge_graph(nebula_integration_storage):
    s = nebula_integration_storage

    await s.upsert_node("kg-root", {"entity_id": "kg-root", "entity_type": "Root"})
    await s.upsert_node("kg-a", {"entity_id": "kg-a", "entity_type": "A"})
    await s.upsert_node("kg-b", {"entity_id": "kg-b", "entity_type": "B"})

    await s.upsert_edge("kg-root", "kg-a", {"weight": 1.0, "description": "r-a"})
    await s.upsert_edge("kg-root", "kg-b", {"weight": 1.0, "description": "r-b"})

    kg = await s.get_knowledge_graph("kg-root", max_depth=2)
    assert len(kg.nodes) == 3
    assert len(kg.edges) == 2
    node_ids = {n.id for n in kg.nodes}
    assert node_ids == {"kg-root", "kg-a", "kg-b"}
    s.workspace = "my-workspace"
    label = s._get_workspace_label()
    assert label == "my_workspace"

    s.workspace = ""
    label = s._get_workspace_label()
    assert label == "base"

    s.workspace = "123invalid"
    label = s._get_workspace_label()
    assert label.startswith("ws_")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration_node_degrees_batch(nebula_integration_storage):
    s = nebula_integration_storage

    await s.upsert_node("ndb-1", {"entity_id": "ndb-1", "entity_type": "Hub"})
    await s.upsert_node("ndb-2", {"entity_id": "ndb-2", "entity_type": "Node"})
    await s.upsert_node("ndb-3", {"entity_id": "ndb-3", "entity_type": "Node"})

    await s.upsert_edge("ndb-1", "ndb-2", {"weight": 1.0, "description": "e1"})
    await s.upsert_edge("ndb-1", "ndb-3", {"weight": 1.0, "description": "e2"})

    degrees = await s.node_degrees_batch(["ndb-1", "ndb-2", "ndb-3"])
    assert degrees["ndb-1"] == 2
    assert degrees["ndb-2"] == 1
    assert degrees["ndb-3"] == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration_edge_degree(nebula_integration_storage):
    s = nebula_integration_storage

    await s.upsert_node("ed-src", {"entity_id": "ed-src", "entity_type": "S"})
    await s.upsert_node("ed-tgt", {"entity_id": "ed-tgt", "entity_type": "T"})
    await s.upsert_node("ed-other", {"entity_id": "ed-other", "entity_type": "O"})

    await s.upsert_edge("ed-src", "ed-tgt", {"weight": 1.0, "description": "main"})
    await s.upsert_edge("ed-src", "ed-other", {"weight": 1.0, "description": "extra"})

    degree = await s.edge_degree("ed-src", "ed-tgt")
    assert degree == 3  # src(2) + tgt(1)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration_edge_degrees_batch(nebula_integration_storage):
    s = nebula_integration_storage

    await s.upsert_node("eb-s", {"entity_id": "eb-s", "entity_type": "S"})
    await s.upsert_node("eb-a", {"entity_id": "eb-a", "entity_type": "A"})
    await s.upsert_node("eb-b", {"entity_id": "eb-b", "entity_type": "B"})

    await s.upsert_edge("eb-s", "eb-a", {"weight": 1.0, "description": "sa"})
    await s.upsert_edge("eb-s", "eb-b", {"weight": 1.0, "description": "sb"})

    result = await s.edge_degrees_batch([("eb-s", "eb-a"), ("eb-s", "eb-b")])
    assert result[("eb-s", "eb-a")] >= 2
    assert result[("eb-s", "eb-b")] >= 2


@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration_has_nodes_batch(nebula_integration_storage):
    s = nebula_integration_storage

    await s.upsert_node("hnb-1", {"entity_id": "hnb-1", "entity_type": "A"})
    await s.upsert_node("hnb-2", {"entity_id": "hnb-2", "entity_type": "B"})

    existing = await s.has_nodes_batch(["hnb-1", "hnb-2", "hnb-missing"])
    assert "hnb-1" in existing
    assert "hnb-2" in existing
    assert "hnb-missing" not in existing


@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration_upsert_edges_batch(nebula_integration_storage):
    s = nebula_integration_storage

    await s.upsert_node("ueb-a", {"entity_id": "ueb-a", "entity_type": "A"})
    await s.upsert_node("ueb-b", {"entity_id": "ueb-b", "entity_type": "B"})
    await s.upsert_node("ueb-c", {"entity_id": "ueb-c", "entity_type": "C"})

    edges = [
        ("ueb-a", "ueb-b", {"weight": 1.0, "description": "ab"}),
        ("ueb-a", "ueb-c", {"weight": 2.0, "description": "ac"}),
    ]
    await s.upsert_edges_batch(edges)

    assert await s.has_edge("ueb-a", "ueb-b") is True
    assert await s.has_edge("ueb-a", "ueb-c") is True


@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration_get_edges_batch(nebula_integration_storage):
    s = nebula_integration_storage

    await s.upsert_node("geb-a", {"entity_id": "geb-a", "entity_type": "A"})
    await s.upsert_node("geb-b", {"entity_id": "geb-b", "entity_type": "B"})

    await s.upsert_edge(
        "geb-a", "geb-b", {"weight": 3.0, "description": "knows", "keywords": "friend"}
    )

    result = await s.get_edges_batch(
        [
            {"src": "geb-a", "tgt": "geb-b"},
            {"src": "geb-a", "tgt": "geb-nonexistent"},
        ]
    )
    assert result[("geb-a", "geb-b")]["weight"] == 3.0
    assert result[("geb-a", "geb-b")]["description"] == "knows"
    # Non-existent edge returns defaults
    assert result[("geb-a", "geb-nonexistent")]["weight"] == 1.0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration_get_nodes_edges_batch(nebula_integration_storage):
    s = nebula_integration_storage

    await s.upsert_node("gneb-1", {"entity_id": "gneb-1", "entity_type": "Hub"})
    await s.upsert_node("gneb-2", {"entity_id": "gneb-2", "entity_type": "Node"})
    await s.upsert_node("gneb-3", {"entity_id": "gneb-3", "entity_type": "Node"})

    await s.upsert_edge("gneb-1", "gneb-2", {"weight": 1.0, "description": "e1"})
    await s.upsert_edge("gneb-1", "gneb-3", {"weight": 1.0, "description": "e2"})

    result = await s.get_nodes_edges_batch(["gneb-1", "gneb-2"])
    assert len(result["gneb-1"]) == 2
    assert len(result["gneb-2"]) == 1
    # gneb-1 edges should connect to gneb-2 and gneb-3
    targets = {tgt for _, tgt in result["gneb-1"]}
    assert "gneb-2" in targets
    assert "gneb-3" in targets


@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration_remove_nodes(nebula_integration_storage):
    s = nebula_integration_storage

    await s.upsert_node("rn-1", {"entity_id": "rn-1", "entity_type": "A"})
    await s.upsert_node("rn-2", {"entity_id": "rn-2", "entity_type": "B"})
    await s.upsert_node("rn-3", {"entity_id": "rn-3", "entity_type": "C"})

    await s.remove_nodes(["rn-1", "rn-2"])

    assert await s.has_node("rn-1") is False
    assert await s.has_node("rn-2") is False
    assert await s.has_node("rn-3") is True


@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration_remove_edges(nebula_integration_storage):
    s = nebula_integration_storage

    await s.upsert_node("re-a", {"entity_id": "re-a", "entity_type": "A"})
    await s.upsert_node("re-b", {"entity_id": "re-b", "entity_type": "B"})
    await s.upsert_node("re-c", {"entity_id": "re-c", "entity_type": "C"})

    await s.upsert_edge("re-a", "re-b", {"weight": 1.0, "description": "ab"})
    await s.upsert_edge("re-a", "re-c", {"weight": 1.0, "description": "ac"})

    await s.remove_edges([("re-a", "re-b")])

    assert await s.has_edge("re-a", "re-b") is False
    assert await s.has_edge("re-a", "re-c") is True


@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration_get_popular_labels(nebula_integration_storage):
    s = nebula_integration_storage

    await s.upsert_node("gpl-1", {"entity_id": "gpl-1", "entity_type": "Cat"})
    await s.upsert_node("gpl-2", {"entity_id": "gpl-2", "entity_type": "Cat"})
    await s.upsert_node("gpl-3", {"entity_id": "gpl-3", "entity_type": "Cat"})
    await s.upsert_node("gpl-4", {"entity_id": "gpl-4", "entity_type": "Dog"})
    await s.upsert_node("gpl-5", {"entity_id": "gpl-5", "entity_type": "Bird"})

    labels = await s.get_popular_labels(limit=10)
    assert len(labels) >= 1
    # "Cat" (3) should come before "Dog" (1) and "Bird" (1)
    assert labels[0] == "Cat"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration_search_labels(nebula_integration_storage):
    s = nebula_integration_storage

    await s.upsert_node("sl-1", {"entity_id": "sl-1", "entity_type": "Python"})
    await s.upsert_node("sl-2", {"entity_id": "sl-2", "entity_type": "Pythonista"})
    await s.upsert_node("sl-3", {"entity_id": "sl-3", "entity_type": "Rust"})

    # Substring search via contains()
    results = await s.search_labels("Py", limit=10)
    assert "Python" in results
    assert "Pythonista" in results
    assert "Rust" not in results

    # Non-matching query returns empty
    results2 = await s.search_labels("zzz_no_match", limit=10)
    assert results2 == []


@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration_get_all_nodes(nebula_integration_storage):
    s = nebula_integration_storage

    await s.upsert_node("gan-1", {"entity_id": "gan-1", "entity_type": "X"})
    await s.upsert_node("gan-2", {"entity_id": "gan-2", "entity_type": "Y"})

    nodes = await s.get_all_nodes()
    entity_ids = {n.get("entity_id") for n in nodes if isinstance(n, dict)}
    assert "gan-1" in entity_ids
    assert "gan-2" in entity_ids


@pytest.mark.integration
@pytest.mark.asyncio
async def test_integration_get_all_edges(nebula_integration_storage):
    s = nebula_integration_storage

    await s.upsert_node("gae-a", {"entity_id": "gae-a", "entity_type": "A"})
    await s.upsert_node("gae-b", {"entity_id": "gae-b", "entity_type": "B"})

    await s.upsert_edge(
        "gae-a", "gae-b", {"weight": 5.0, "description": "special", "keywords": "test"}
    )

    edges = await s.get_all_edges()
    assert len(edges) >= 1
    assert any(e.get("src") == "gae-a" and e.get("tgt") == "gae-b" for e in edges)
