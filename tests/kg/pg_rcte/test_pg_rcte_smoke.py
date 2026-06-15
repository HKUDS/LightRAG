"""Real-PostgreSQL smoke tests for PgRcteGraphStorage.

These tests exercise the actual DB glue that all offline unit tests mock:
  - _execute  → PostgreSQLDB.execute(data={str(i): v})   positional binding
  - _fetch    → PostgreSQLDB.query(..., multirows=True)   JSONB via asyncpg
  - json.loads() on JSONB columns returned by asyncpg

Requires a live PostgreSQL instance.  The tests are gated behind the
`pg_smoke` pytest marker and skip automatically when POSTGRES_PASSWORD
is absent so they never block offline CI.

The CI gate is .github/workflows/pg-smoke.yml which uses a GitHub Actions
PostgreSQL service container.
"""
import json
import os
import time
import uuid

import pytest

pytestmark = pytest.mark.pg_smoke

# ---------------------------------------------------------------------------
# Skip the whole module if no real PostgreSQL is configured
# ---------------------------------------------------------------------------

PG_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "")
if not PG_PASSWORD:
    pytest.skip(
        "POSTGRES_PASSWORD not set — skipping pg_smoke tests",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Fixture: isolated PgRcteGraphStorage with a unique workspace per test
# ---------------------------------------------------------------------------


@pytest.fixture
async def store():
    """Yield an initialized PgRcteGraphStorage and clean up after the test."""
    from lightrag.kg.pg_rcte_impl import PgRcteGraphStorage

    workspace = f"smoke_{uuid.uuid4().hex[:8]}"
    storage = PgRcteGraphStorage(
        namespace="graph",
        workspace=workspace,
        global_config={"max_graph_nodes": 1000},
    )
    await storage.initialize()
    try:
        yield storage
    finally:
        await storage.drop()
        await storage.finalize()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _node(entity_id: str) -> dict:
    return {
        "entity_id": entity_id,
        "entity_type": "SMOKE",
        "description": f"smoke node {entity_id}",
        "source_id": "smoke-chunk",
        "file_path": "smoke.txt",
        "created_at": int(time.time()),
    }


def _edge(weight: float = 1.0) -> dict:
    return {
        "weight": weight,
        "description": "smoke edge",
        "keywords": "smoke",
        "source_id": "smoke-chunk",
        "file_path": "smoke.txt",
        "created_at": int(time.time()),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_and_get_node(store):
    """upsert_node round-trips through positional binding and JSONB decode."""
    await store.upsert_node("Alice", _node("Alice"))

    row = await store.get_node("Alice")

    assert row is not None, "get_node must return the upserted node"
    assert row["entity_id"] == "Alice"
    assert row["entity_type"] == "SMOKE"
    assert row["description"] == "smoke node Alice"


@pytest.mark.asyncio
async def test_upsert_node_is_idempotent(store):
    """A second upsert with updated data replaces the first."""
    await store.upsert_node("Bob", _node("Bob"))
    updated = dict(_node("Bob"), description="updated")
    await store.upsert_node("Bob", updated)

    row = await store.get_node("Bob")
    assert row is not None
    assert row["description"] == "updated"


@pytest.mark.asyncio
async def test_has_node(store):
    await store.upsert_node("Carol", _node("Carol"))

    assert await store.has_node("Carol") is True
    assert await store.has_node("ghost_xyz") is False


@pytest.mark.asyncio
async def test_upsert_and_get_edge(store):
    """upsert_edge round-trips through positional binding and JSONB decode."""
    await store.upsert_node("A", _node("A"))
    await store.upsert_node("B", _node("B"))
    await store.upsert_edge("A", "B", _edge(2.5))

    row = await store.get_edge("A", "B")
    assert row is not None
    # Weight is stored as JSON number; asyncpg decodes JSONB to Python dict
    assert float(row["weight"]) == pytest.approx(2.5)

    # Edge is undirected — reverse lookup must also work
    row_rev = await store.get_edge("B", "A")
    assert row_rev is not None
    assert float(row_rev["weight"]) == pytest.approx(2.5)


@pytest.mark.asyncio
async def test_has_edge(store):
    await store.upsert_node("X", _node("X"))
    await store.upsert_node("Y", _node("Y"))
    await store.upsert_edge("X", "Y", _edge())

    assert await store.has_edge("X", "Y") is True
    assert await store.has_edge("Y", "X") is True  # undirected
    assert await store.has_edge("X", "ghost") is False


@pytest.mark.asyncio
async def test_get_node_edges_queried_node_first(store):
    """The ordering contract: get_node_edges(Q) must always have Q at [0].

    This exercises the real `_fetch` path, which is where the canonical
    (min, max) storage order would break the contract if the Python
    normalization were missing or wrong.
    """
    await store.upsert_node("ZNode", _node("ZNode"))
    await store.upsert_node("ANode", _node("ANode"))
    await store.upsert_node("MNode", _node("MNode"))
    await store.upsert_edge("ZNode", "ANode", _edge())
    await store.upsert_edge("ZNode", "MNode", _edge())

    edges = await store.get_node_edges("ZNode")
    assert edges is not None
    assert len(edges) == 2
    for src, tgt in edges:
        assert src == "ZNode", f"queried node must be first; got ({src!r}, {tgt!r})"
    assert {tgt for _, tgt in edges} == {"ANode", "MNode"}

    # Also verify from the neighbor's perspective
    edges_a = await store.get_node_edges("ANode")
    assert edges_a is not None
    assert len(edges_a) == 1
    assert edges_a[0] == ("ANode", "ZNode")


@pytest.mark.asyncio
async def test_get_nodes_edges_batch_queried_node_first(store):
    """Batch version must also normalise queried node to position 0."""
    await store.upsert_node("Hub", _node("Hub"))
    await store.upsert_node("Spoke1", _node("Spoke1"))
    await store.upsert_node("Spoke2", _node("Spoke2"))
    await store.upsert_edge("Hub", "Spoke1", _edge())
    await store.upsert_edge("Hub", "Spoke2", _edge())

    result = await store.get_nodes_edges_batch(["Hub", "Spoke1", "Spoke2"])

    for queried, edge_list in result.items():
        for src, tgt in edge_list:
            assert src == queried, (
                f"queried node {queried!r} must be first; got ({src!r}, {tgt!r})"
            )


@pytest.mark.asyncio
async def test_node_degree(store):
    await store.upsert_node("Center", _node("Center"))
    await store.upsert_node("N1", _node("N1"))
    await store.upsert_node("N2", _node("N2"))
    await store.upsert_node("N3", _node("N3"))
    await store.upsert_edge("Center", "N1", _edge())
    await store.upsert_edge("Center", "N2", _edge())
    await store.upsert_edge("Center", "N3", _edge())

    assert await store.node_degree("Center") == 3
    assert await store.node_degree("N1") == 1
    assert await store.node_degree("ghost") == 0


@pytest.mark.asyncio
async def test_jsonb_unicode_and_special_chars(store):
    """JSONB round-trip must preserve Unicode and apostrophes without corruption.

    Guards against SQL-injection / quoting bugs in the positional param path.
    """
    node_data = dict(
        _node("Special"),
        description="O'Brien's café — 日本語 <script>",
    )
    await store.upsert_node("Special", node_data)

    row = await store.get_node("Special")
    assert row is not None
    assert row["description"] == "O'Brien's café — 日本語 <script>"
