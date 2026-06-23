"""Real-PostgreSQL smoke tests for PGTableGraphStorage.

These tests exercise the actual DB glue that all offline unit tests mock:
  - _execute  → PostgreSQLDB.execute(data={str(i): v})   positional binding
  - _fetch    → PostgreSQLDB.query(..., multirows=True)   JSONB via asyncpg
  - json.loads() on JSONB columns returned by asyncpg

Requires a live PostgreSQL instance.  The tests carry both the ``pg_smoke``
and ``integration`` markers so that:

* Normal ``pytest`` runs skip them automatically via conftest's
  ``pytest_collection_modifyitems`` (no ``--run-integration`` flag → skip),
  even when ``POSTGRES_PASSWORD`` happens to be set in the developer's env.
* The CI workflow (pg-smoke.yml) passes ``--run-integration`` to opt in, then
  the module-level ``POSTGRES_PASSWORD`` guard provides a second safety net.
"""

import os
import time
import uuid

import pytest

pytestmark = [pytest.mark.pg_smoke, pytest.mark.integration]

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
# Fixture: isolated PGTableGraphStorage with a unique workspace per test
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def init_shared_storage():
    from lightrag.kg.shared_storage import initialize_share_data

    initialize_share_data(workers=1)


@pytest.fixture
async def store():
    """Yield an initialized PGTableGraphStorage and clean up after the test."""
    from lightrag.kg.pgtable_impl import PGTableGraphStorage

    workspace = f"smoke_{uuid.uuid4().hex[:8]}"
    storage = PGTableGraphStorage(
        namespace="graph",
        workspace=workspace,
        global_config={"max_graph_nodes": 1000},
        embedding_func=None,
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
async def test_upsert_edge_creates_missing_endpoint_nodes(store):
    """Missing endpoint edges follow NetworkX add_edge semantics."""
    await store.upsert_node("OnlyA", _node("OnlyA"))

    await store.upsert_edge("OnlyA", "MissingB", _edge())
    await store.upsert_edges_batch(
        [
            ("OnlyA", "MissingC", _edge()),
            ("MissingD", "OnlyA", _edge()),
        ]
    )

    assert await store.has_edge("OnlyA", "MissingB") is True
    assert await store.has_edge("OnlyA", "MissingC") is True
    assert await store.has_edge("MissingD", "OnlyA") is True
    assert await store.get_node("MissingB") == {"entity_id": "MissingB"}
    assert await store.get_node("MissingC") == {"entity_id": "MissingC"}
    assert await store.get_node("MissingD") == {"entity_id": "MissingD"}


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
async def test_self_loop_degree_matches_networkx(store):
    await store.upsert_edge("Loop", "Loop", _edge())

    assert await store.node_degree("Loop") == 2
    assert (await store.node_degrees_batch(["Loop"]))["Loop"] == 2
    assert await store.get_popular_labels(limit=1) == ["Loop"]


@pytest.mark.asyncio
async def test_get_knowledge_graph_bfs(store):
    """Exercise the WITH RECURSIVE BFS path against a real PostgreSQL instance.

    Builds a chain A-B-C-D and verifies that get_knowledge_graph("A", max_depth=2)
    returns A, B, C but not D (3 hops away).
    """
    for name in ("A", "B", "C", "D"):
        await store.upsert_node(name, _node(name))
    await store.upsert_edge("A", "B", _edge(1.0))
    await store.upsert_edge("B", "C", _edge(1.0))
    await store.upsert_edge("C", "D", _edge(1.0))

    kg = await store.get_knowledge_graph("A", max_depth=2, max_nodes=100)

    node_ids = {n.id for n in kg.nodes}
    assert "A" in node_ids
    assert "B" in node_ids
    assert "C" in node_ids
    assert "D" not in node_ids, "D is 3 hops from A; must not appear at max_depth=2"
    assert kg.is_truncated is False

    edge = next(e for e in kg.edges if {e.source, e.target} == {"A", "B"})
    assert edge.id == "A-B"
    assert edge.type == "DIRECTED"


@pytest.mark.asyncio
async def test_get_all_edges_key_shape(store):
    """get_all_edges must return 'source'/'target' keys, not 'src_id'/'tgt_id'."""
    await store.upsert_node("X", _node("X"))
    await store.upsert_node("Y", _node("Y"))
    await store.upsert_edge(
        "X",
        "Y",
        dict(_edge(0.5), source="property-source", target="property-target"),
    )

    edges = await store.get_all_edges()

    assert len(edges) >= 1
    edge = next(e for e in edges if {e.get("source"), e.get("target")} == {"X", "Y"})
    assert "source" in edge
    assert "target" in edge
    assert edge["source"] == "X"
    assert edge["target"] == "Y"
    assert "src_id" not in edge
    assert "tgt_id" not in edge


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


# ---------------------------------------------------------------------------
# get_knowledge_graph — frontier-capped BFS traversal (algorithm change)
#
# Exercise the iterative frontier BFS that replaced the UNION ALL recursive CTE:
# per-depth correctness, cyclic-graph termination, bounded blast radius on dense
# graphs (regression for the CTE's path explosion), and truncation boundary.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_knowledge_graph_depth_levels(store):
    """Frontier BFS returns exactly the nodes reachable within max_depth hops."""
    chain = ["n0", "n1", "n2", "n3", "n4"]
    for x in chain:
        await store.upsert_node(x, _node(x))
    for i in range(len(chain) - 1):
        await store.upsert_edge(chain[i], chain[i + 1], _edge())

    for depth, expected in [
        (1, {"n0", "n1"}),
        (2, {"n0", "n1", "n2"}),
        (3, {"n0", "n1", "n2", "n3"}),
    ]:
        kg = await store.get_knowledge_graph("n0", max_depth=depth, max_nodes=100)
        assert {n.id for n in kg.nodes} == expected, f"depth={depth}"


@pytest.mark.asyncio
async def test_get_knowledge_graph_cyclic_terminates(store):
    """A cycle must not loop forever; each node is visited exactly once."""
    ring = ["c0", "c1", "c2", "c3", "c4"]
    for x in ring:
        await store.upsert_node(x, _node(x))
    for i in range(len(ring)):
        await store.upsert_edge(ring[i], ring[(i + 1) % len(ring)], _edge())

    kg = await store.get_knowledge_graph("c0", max_depth=10, max_nodes=100)
    ids = [n.id for n in kg.nodes]
    assert set(ids) == set(ring)
    assert len(ids) == len(ring)  # no duplicates despite cyclic paths
    assert kg.is_truncated is False


@pytest.mark.asyncio
async def test_get_knowledge_graph_dense_blast_radius_bounded(store):
    """On a complete graph, frontier BFS stays bounded by max_nodes and finishes
    fast. The prior UNION ALL recursive CTE re-materialized shared nodes once per
    simple path and would explode here — regression guard for the traversal
    change."""
    n = 60
    ks = [f"k{i}" for i in range(n)]
    await store.upsert_nodes_batch([(x, _node(x)) for x in ks])
    await store.upsert_edges_batch(
        [(ks[i], ks[j], _edge()) for i in range(n) for j in range(i + 1, n)]
    )

    started = time.perf_counter()
    kg = await store.get_knowledge_graph("k0", max_depth=3, max_nodes=10)
    elapsed = time.perf_counter() - started

    assert len(kg.nodes) == 10
    assert kg.is_truncated is True
    # A path-exploding traversal would not finish quickly on K60 (1770 edges).
    assert elapsed < 5.0, f"frontier BFS too slow on dense graph: {elapsed:.2f}s"


@pytest.mark.asyncio
async def test_get_knowledge_graph_truncation_boundary(store):
    """is_truncated reflects whether reachable nodes exceed max_nodes, and the
    seed is always retained at position 0."""
    star = [f"s{i}" for i in range(6)]
    for x in star:
        await store.upsert_node(x, _node(x))
    for x in star[1:]:
        await store.upsert_edge("s0", x, _edge())  # hub s0 + 5 spokes = 6 nodes

    kg_full = await store.get_knowledge_graph("s0", max_depth=1, max_nodes=100)
    assert len(kg_full.nodes) == 6
    assert kg_full.is_truncated is False

    kg_trunc = await store.get_knowledge_graph("s0", max_depth=1, max_nodes=3)
    assert len(kg_trunc.nodes) == 3
    assert kg_trunc.is_truncated is True
    assert kg_trunc.nodes[0].id == "s0"  # seed pinned even under truncation


@pytest.mark.asyncio
async def test_get_knowledge_graph_truncation_prefers_high_degree(store):
    """When a depth level overflows max_nodes, frontier BFS must keep the
    highest-degree neighbours (degree-priority), not whatever order the DB
    returned. Matches NetworkX's degree-ordered BFS and the prior recursive CTE
    that degree-sorted the full reachable set before cutting."""
    for x in ("S", "hi", "mid", "lo"):
        await store.upsert_node(x, _node(x))
    # depth-1 spokes off the seed
    await store.upsert_edge("S", "hi", _edge())
    await store.upsert_edge("S", "mid", _edge())
    await store.upsert_edge("S", "lo", _edge())
    # boost degrees: hi -> 6, mid -> 3, lo -> 1 (extra endpoints are depth 2)
    for i in range(5):
        await store.upsert_edge("hi", f"hx{i}", _edge())
    for i in range(2):
        await store.upsert_edge("mid", f"mx{i}", _edge())

    # budget = seed + 2 neighbours: must retain hi (6) and mid (3), drop lo (1).
    kg = await store.get_knowledge_graph("S", max_depth=1, max_nodes=3)
    ids = {n.id for n in kg.nodes}
    assert ids == {"S", "hi", "mid"}, f"degree-priority truncation failed: {ids}"
    assert kg.is_truncated is True
