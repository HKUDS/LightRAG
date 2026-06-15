"""Offline unit tests for PgRcteGraphStorage.

Uses AsyncMock to patch _fetch/_fetchrow/_execute so no real PostgreSQL
connection is required. Marked @offline so these run in CI.
"""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

pytestmark = pytest.mark.offline


def make_storage(global_config=None):
    """Build a PgRcteGraphStorage with a mocked db."""
    from lightrag.kg.pg_rcte_impl import PgRcteGraphStorage

    storage = object.__new__(PgRcteGraphStorage)
    storage.workspace = "test"
    storage.global_config = global_config or {}
    mock_db = MagicMock()
    storage.db = mock_db
    return storage


# ---------------------------------------------------------------------------
# get_node_edges — queried node must be tuple[0]
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_node_edges_queried_node_first_as_src():
    """When the queried node is stored as src_id, it must still be first."""
    storage = make_storage()
    # Edge stored as (A, Z) — canonical order because A < Z
    with patch.object(storage, "has_node", new=AsyncMock(return_value=True)), \
         patch.object(storage, "_fetch", new=AsyncMock(return_value=[
             {"src_id": "A", "tgt_id": "Z"}
         ])):
        result = await storage.get_node_edges("A")
    assert result == [("A", "Z")]


@pytest.mark.asyncio
async def test_get_node_edges_queried_node_first_as_tgt():
    """When the queried node is stored as tgt_id, it must be moved to first."""
    storage = make_storage()
    # Edge stored as (A, Z) — querying Z, so Z must be first
    with patch.object(storage, "has_node", new=AsyncMock(return_value=True)), \
         patch.object(storage, "_fetch", new=AsyncMock(return_value=[
             {"src_id": "A", "tgt_id": "Z"}
         ])):
        result = await storage.get_node_edges("Z")
    assert result == [("Z", "A")]


@pytest.mark.asyncio
async def test_get_node_edges_returns_none_for_missing_node():
    storage = make_storage()
    with patch.object(storage, "has_node", new=AsyncMock(return_value=False)):
        result = await storage.get_node_edges("ghost")
    assert result is None


# ---------------------------------------------------------------------------
# get_nodes_edges_batch — queried node first
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_nodes_edges_batch_queried_node_first():
    """Batch version must also put the queried node first."""
    storage = make_storage()
    # Edge (A, Z): both A and Z are queried
    with patch.object(storage, "_fetch", new=AsyncMock(return_value=[
        {"src_id": "A", "tgt_id": "Z"}
    ])):
        result = await storage.get_nodes_edges_batch(["A", "Z"])
    assert ("A", "Z") in result["A"]
    assert ("Z", "A") in result["Z"]  # Z is queried node, must be first


# ---------------------------------------------------------------------------
# JSONB round-trip
# ---------------------------------------------------------------------------

def test_jsonb_roundtrip():
    """json.dumps then json.loads must preserve dict structure."""
    original = {"entity_id": "node1", "name": "Alice", "weight": 1.5}
    serialized = json.dumps(original)
    assert isinstance(serialized, str)
    restored = json.loads(serialized)
    assert restored == original


# ---------------------------------------------------------------------------
# entity_id validation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_upsert_node_raises_without_entity_id():
    storage = make_storage()
    with pytest.raises(ValueError, match="entity_id"):
        await storage.upsert_node("node1", {"name": "Alice"})  # missing entity_id


@pytest.mark.asyncio
async def test_upsert_node_succeeds_with_entity_id():
    storage = make_storage()
    with patch.object(storage, "_execute", new=AsyncMock()):
        await storage.upsert_node("node1", {"entity_id": "node1", "name": "Alice"})


# ---------------------------------------------------------------------------
# get_knowledge_graph — max_nodes config
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_knowledge_graph_uses_global_config_when_max_nodes_none():
    storage = make_storage(global_config={"max_graph_nodes": 42})
    with patch.object(storage, "_fetch", new=AsyncMock(return_value=[])):
        kg = await storage.get_knowledge_graph("*", max_nodes=None)
    assert kg.nodes == []


@pytest.mark.asyncio
async def test_get_knowledge_graph_wildcard_uses_flat_limited_query():
    """Wildcard should bound DB work instead of running recursive traversal."""
    storage = make_storage(global_config={"max_graph_nodes": 100})
    fetch = AsyncMock(return_value=[])

    with patch.object(storage, "_fetch", new=fetch):
        await storage.get_knowledge_graph("*", max_nodes=7)

    sql, workspace, fetch_limit = fetch.call_args.args
    assert workspace == "test"
    assert fetch_limit == 8
    assert "WITH RECURSIVE" not in sql
    assert "LIMIT $2" in sql


@pytest.mark.asyncio
async def test_get_knowledge_graph_clamps_max_nodes():
    """Caller-supplied max_nodes > config value must be clamped to config."""
    storage = make_storage(global_config={"max_graph_nodes": 10})
    rows = [{"id": str(i), "properties": json.dumps({"entity_id": str(i)})} for i in range(20)]
    # First _fetch call returns nodes; second (edge query) returns no edges.
    with patch.object(storage, "_fetch", new=AsyncMock(side_effect=[rows, []])), \
         patch.object(storage, "node_degrees_batch", new=AsyncMock(return_value={str(i): i for i in range(20)})):
        kg = await storage.get_knowledge_graph("*", max_nodes=999)
    assert len(kg.nodes) <= 10
    assert kg.is_truncated is True


@pytest.mark.asyncio
async def test_get_knowledge_graph_truncates_by_degree_then_id():
    """Truncation must not depend on unordered PostgreSQL row order."""
    storage = make_storage(global_config={"max_graph_nodes": 100})
    rows = [
        {"id": "z_low", "properties": json.dumps({"entity_id": "z_low"})},
        {"id": "tie_b", "properties": json.dumps({"entity_id": "tie_b"})},
        {"id": "hub", "properties": json.dumps({"entity_id": "hub"})},
        {"id": "tie_a", "properties": json.dumps({"entity_id": "tie_a"})},
    ]
    degrees = {"z_low": 1, "tie_b": 3, "hub": 9, "tie_a": 3}

    with patch.object(storage, "_fetch", new=AsyncMock(side_effect=[rows, []])), \
         patch.object(storage, "node_degrees_batch", new=AsyncMock(return_value=degrees)):
        kg = await storage.get_knowledge_graph("*", max_nodes=3)

    assert [node.id for node in kg.nodes] == ["hub", "tie_a", "tie_b"]
    assert kg.is_truncated is True


# ---------------------------------------------------------------------------
# self-loop degree consistency
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_self_loop_degree_consistency():
    """node_degree and node_degrees_batch must agree for self-loop nodes."""
    storage = make_storage()

    async def fake_node_degree(node_id):
        # Self-loop: src_id == tgt_id == node_id, count = 1
        return 1

    async def fake_node_degrees_batch(node_ids):
        return {nid: 1 for nid in node_ids}

    with patch.object(storage, "node_degree", new=fake_node_degree), \
         patch.object(storage, "node_degrees_batch", new=fake_node_degrees_batch):
        single = await storage.node_degree("A")
        batch = await storage.node_degrees_batch(["A"])
    assert single == batch["A"]


# ---------------------------------------------------------------------------
# remove_edges — NULL guard
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_remove_edges_raises_on_none_id():
    storage = make_storage()
    with pytest.raises((ValueError, TypeError)):
        await storage.remove_edges([(None, "B")])  # type: ignore[arg-type]
