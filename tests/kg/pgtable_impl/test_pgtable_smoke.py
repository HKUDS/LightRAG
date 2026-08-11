"""Real-PostgreSQL smoke tests for PGTableGraphStorage.

These tests exercise the actual DB glue that all offline unit tests mock:
  - _execute  → PostgreSQLDB.execute(data={str(i): v})   positional binding
  - _fetch    → PostgreSQLDB.query(..., multirows=True)   JSONB via asyncpg
  - json.loads() on JSONB columns returned by asyncpg

The second half of the module carries the backend-agnostic end-to-end graph
contract ported from tests/kg/test_graph_storage.py (that suite only reaches
PGTableGraphStorage when LIGHTRAG_GRAPH_STORAGE happens to point at it), with the
parts already covered above merged in rather than duplicated — see the section
banner further down for the mapping.

Requires a live PostgreSQL instance.  The tests carry both the ``pg_smoke``
and ``integration`` markers so that:

* Normal ``pytest`` runs skip them automatically via conftest's
  ``pytest_collection_modifyitems`` (no ``--run-integration`` flag → skip),
  even when ``POSTGRES_PASSWORD`` happens to be set in the developer's env.
* The CI workflow (pg-smoke.yml) passes ``--run-integration`` to opt in, then
  the module-level ``POSTGRES_PASSWORD`` guard provides a second safety net.
"""

import asyncio
import json
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
        # Deliberately no "vector_storage" key: this exercises the
        # no-vector-backend-configured fallback in initialize(), which is what
        # keeps the pool from requiring pgvector. CI runs these against a plain
        # postgres image, so a regression here fails loudly rather than silently
        # depending on an extension.
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
    data = dict(_node("Alice"), keywords="AI,Machine Learning,Deep Learning")
    await store.upsert_node("Alice", data)

    row = await store.get_node("Alice")

    assert row is not None, "get_node must return the upserted node"
    assert row["entity_id"] == "Alice"
    assert row["entity_type"] == "SMOKE"
    assert row["description"] == "smoke node Alice"
    # Every key of the payload must survive the JSONB round-trip, not just the
    # three the original smoke test spot-checked.
    assert row == data


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
    """upsert_edge round-trips through positional binding and JSONB decode.

    Also pins full-payload fidelity and strict forward/reverse equality — the
    undirected contract asserted by ``test_graph_basic`` in
    tests/kg/test_graph_storage.py, folded in here so the PG path carries it too.
    """
    await store.upsert_node("A", _node("A"))
    await store.upsert_node("B", _node("B"))
    edge_data = dict(
        _edge(2.5),
        relationship="includes",
        description="A includes the subfield B.",
    )
    await store.upsert_edge("A", "B", edge_data)

    row = await store.get_edge("A", "B")
    assert row is not None
    # Weight is stored as JSON number; asyncpg decodes JSONB to Python dict
    assert float(row["weight"]) == pytest.approx(2.5)
    assert row["relationship"] == edge_data["relationship"]
    assert row["description"] == edge_data["description"]
    assert row["keywords"] == edge_data["keywords"]

    # Edge is undirected — reverse lookup must return the identical property set
    row_rev = await store.get_edge("B", "A")
    assert row_rev is not None
    assert float(row_rev["weight"]) == pytest.approx(2.5)
    assert row_rev == row, "forward and reverse edge properties must be identical"


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
    """Exercise the frontier-capped iterative BFS path against a real PostgreSQL
    instance (the traversal is _bfs_frontier, not a recursive CTE).

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
    """JSONB round-trip must preserve Unicode, quotes and backslashes verbatim.

    Guards against SQL-injection / quoting bugs in the positional param path.

    Merged with ``test_graph_special_characters`` from
    tests/kg/test_graph_storage.py: single/double quotes and backslashes in both
    node ids and property values, on nodes and on edges, including a value that
    looks like a SQL injection attempt.
    """
    quoted_id = "Node with 'single quotes'"
    dquoted_id = 'Node with "double quotes"'
    backslash_id = "Node with \\backslashes\\"

    node_payloads = {
        "Special": dict(
            _node("Special"), description="O'Brien's café — 日本語 <script>"
        ),
        quoted_id: dict(
            _node(quoted_id),
            description="Contains 'single quotes', \"double quotes\" and \\backslashes",
        ),
        dquoted_id: dict(
            _node(dquoted_id),
            description="Both 'single quotes' and \"double quotes\" and \\a\\path",
        ),
        backslash_id: dict(
            _node(backslash_id),
            description="Windows path C:\\Program Files\\ and escapes \\n\\t",
        ),
    }
    for node_id, payload in node_payloads.items():
        await store.upsert_node(node_id, payload)

    for node_id, payload in node_payloads.items():
        row = await store.get_node(node_id)
        assert row is not None, f"node {node_id!r} must round-trip"
        assert row["entity_id"] == node_id
        assert row["description"] == payload["description"]

    # Edge properties take the same path and must survive it identically.
    edge_payload = dict(
        _edge(0.8),
        relationship='complex "relationship"\\type',
        description=("SQL injection attempt: SELECT * FROM users WHERE name='admin'--"),
    )
    await store.upsert_edge(quoted_id, backslash_id, edge_payload)

    edge = await store.get_edge(quoted_id, backslash_id)
    assert edge is not None
    assert edge["relationship"] == edge_payload["relationship"]
    assert edge["description"] == edge_payload["description"]
    assert await store.get_edge(backslash_id, quoted_id) == edge


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


@pytest.mark.asyncio
async def test_get_knowledge_graph_seed_leads_and_survives_truncation(store):
    """The seed leads the result and survives truncation on its own merit.

    _bfs_frontier deliberately does not fetch the seed's degree, so `degrees`
    carries no entry for it and the ordering falls back to `.get(seed, 0)`.
    This is the adversarial shape for that: the seed has the LOWEST real degree
    in the graph, and the budget forces a cut. If the degree term ever preceded
    the pin/depth terms, the seed would sort last on a 0 default and be the
    first row dropped. Only a live DB proves it, because the neighbour degrees
    here come from the hop query's real aggregate rather than a mock.
    """
    for x in ("seedlow", "big", "mid"):
        await store.upsert_node(x, _node(x))
    await store.upsert_edge("seedlow", "big", _edge())
    await store.upsert_edge("seedlow", "mid", _edge())
    # seedlow degree 2; boost the neighbours well past it (extras sit at depth 2)
    for i in range(6):
        await store.upsert_edge("big", f"bx{i}", _edge())
    for i in range(3):
        await store.upsert_edge("mid", f"mx{i}", _edge())

    kg = await store.get_knowledge_graph("seedlow", max_depth=1, max_nodes=2)

    assert kg.nodes[0].id == "seedlow", (
        f"seed must lead regardless of its degree: {[n.id for n in kg.nodes]}"
    )
    # Budget 2 = seed + the single highest-degree neighbour.
    assert [n.id for n in kg.nodes] == ["seedlow", "big"]
    assert kg.is_truncated is True


@pytest.mark.asyncio
async def test_get_knowledge_graph_self_loop_seed(store):
    """A seed whose only edge is a self-loop still returns cleanly.

    The hop query finds the seed as its own neighbour, the visited anti-join
    removes it, and the hop yields zero rows. The seed's degree is 2 here (a
    self-loop counts on both the src and tgt arms) yet nothing in the traversal
    reads it — the degenerate path that any scheme recovering the seed degree
    from hop output would get wrong.
    """
    await store.upsert_node("loop", _node("loop"))
    await store.upsert_edge("loop", "loop", _edge())
    assert await store.node_degree("loop") == 2

    kg = await store.get_knowledge_graph("loop", max_depth=3, max_nodes=10)

    assert [n.id for n in kg.nodes] == ["loop"]
    assert kg.is_truncated is False


@pytest.mark.asyncio
async def test_get_knowledge_graph_isolated_seed(store):
    """An isolated seed returns itself, with no degree round trip to fall back on."""
    await store.upsert_node("lonely", _node("lonely"))

    kg = await store.get_knowledge_graph("lonely", max_depth=3, max_nodes=10)

    assert [n.id for n in kg.nodes] == ["lonely"]
    assert kg.edges == []
    assert kg.is_truncated is False


# ---------------------------------------------------------------------------
# Real-PG semantics the mock-based unit tests cannot exercise
# (FK visibility inside a data-modifying CTE, concurrent JSONB merge, atomic drop)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_edge_creates_both_missing_endpoints(store):
    """Both endpoints absent: the data-modifying CTE must insert the two nodes
    AND the edge in one statement without tripping the FK.

    The offline unit test only asserts the SQL *text* contains
    ``jsonb_build_object(...)``; only a live PG proves the FK is satisfied by
    rows inserted in the same CTE (FK checks use transaction, not snapshot,
    visibility), so this is the real regression guard for that CTE.
    """
    await store.upsert_edge("Ghost1", "Ghost2", _edge())

    assert await store.has_edge("Ghost1", "Ghost2") is True
    assert await store.get_node("Ghost1") == {"entity_id": "Ghost1"}
    assert await store.get_node("Ghost2") == {"entity_id": "Ghost2"}


@pytest.mark.asyncio
async def test_concurrent_node_merge_preserves_all_keys(store):
    """Concurrent upsert_node on one id must not lose updates: the ``||`` JSONB
    merge is serialized by the row lock ON CONFLICT takes, so every distinct key
    survives. Guards the 'properties are MERGED, not replaced' contract under
    real concurrency (impossible to show with a mocked _execute).
    """
    await store.upsert_node("M", _node("M"))

    await asyncio.gather(
        *[
            store.upsert_node("M", dict(_node("M"), **{f"k{i}": str(i)}))
            for i in range(20)
        ]
    )

    row = await store.get_node("M")
    assert row is not None
    survived = {k for k in row if k.startswith("k")}
    assert survived == {f"k{i}" for i in range(20)}, (
        f"lost update under concurrent merge: {sorted(survived)}"
    )


@pytest.mark.asyncio
async def test_drop_clears_nodes_and_edges_atomically(store):
    """drop() deletes edges and nodes in a single data-modifying CTE. Verify it
    leaves neither orphan edges nor stray nodes (the edge delete is explicit, so
    it holds even without FK CASCADE).
    """
    await store.upsert_edge("P", "Q", _edge())
    await store.upsert_node("R", _node("R"))

    result = await store.drop()

    assert result["status"] == "success", result
    assert await store.get_all_nodes() == []
    assert await store.get_all_edges() == []
    assert await store.has_edge("P", "Q") is False


# ---------------------------------------------------------------------------
# Legacy-migration paths — exercised on real PG, never via mocks (these are the
# one-time paths no general contract suite can reach: they only run on
# pre-existing adversarial data, and unit tests mock them away).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_normalize_legacy_edges_collapses_reversed_duplicate(store):
    """Regression (review item #1): a legacy table can hold both the canonical
    edge (A,B) and its reversed duplicate (B,A). _normalize_legacy_edges must
    collapse them into one canonical row.

    The earlier single statement put the canonical tuple in BOTH the CTE delete
    set and the ON CONFLICT DO UPDATE target — modifying the same row twice in one
    data-modifying CTE, which Postgres documents as unspecified ("not supported").
    On PG 15-18 it happens to yield the correct single row, but the result is not
    guaranteed across versions/plans. The fix makes the delete set disjoint from
    the INSERT target so the outcome is well-defined everywhere; this test pins
    that outcome. Also covers the deterministic survivor (#3): the later write
    wins on an updated_at tie.
    """
    await store.upsert_node("A", _node("A"))
    await store.upsert_node("B", _node("B"))
    await store.upsert_edge("A", "B", _edge(weight=1.0))  # canonical (A, B)
    # Inject the reversed (B, A) row directly to simulate pre-canonical legacy
    # data (src_id "B" > tgt_id "A" trips the normalization guard).
    await store._execute(
        "INSERT INTO lightrag_graph_edges "
        "(workspace, namespace, src_id, tgt_id, properties, updated_at) "
        "VALUES ($1, $2, $3, $4, $5::jsonb, now())",
        store.workspace,
        store.namespace,
        "B",
        "A",
        json.dumps(dict(_edge(weight=2.0), relationship="reversed")),
    )
    assert len(await store.get_all_edges()) == 2  # both directions present

    await store._normalize_legacy_edges()  # must not raise

    edges = await store.get_all_edges()
    assert len(edges) == 1, "reversed duplicate must collapse to one canonical row"
    assert {edges[0]["source"], edges[0]["target"]} == {"A", "B"}
    # undirected: queryable from either direction after the collapse
    assert await store.get_edge("A", "B") is not None
    assert await store.get_edge("B", "A") is not None


@pytest.mark.asyncio
async def test_initialize_dedupes_before_rebuilding_pk(store):
    """Regression (N3): a legacy table whose PK lacks ``namespace`` can hold
    duplicate rows for the new (workspace, namespace, id) key. The DDL migration
    must de-dupe before ``ADD PRIMARY KEY``, otherwise the rebuild fails on the
    duplicate and aborts startup. Simulate by dropping the node PK, injecting a
    duplicate, then re-applying the DDL (which rebuilds the namespaced PK).
    """
    from lightrag.kg.pgtable_impl import _DDL

    ws, ns = store.workspace, store.namespace
    # Drop the table-level PK so duplicate (ws, ns, id) rows can be inserted,
    # mimicking a legacy table that predates the namespaced primary key. The edge
    # FKs depend on the node PK index, so drop them first (the DDL migration does
    # the same, and re-creates them after the rebuild).
    await store.db.execute(
        "ALTER TABLE lightrag_graph_edges "
        "DROP CONSTRAINT IF EXISTS fk_lightrag_graph_edges_src"
    )
    await store.db.execute(
        "ALTER TABLE lightrag_graph_edges "
        "DROP CONSTRAINT IF EXISTS fk_lightrag_graph_edges_tgt"
    )
    await store.db.execute(
        "ALTER TABLE lightrag_graph_nodes DROP CONSTRAINT lightrag_graph_nodes_pkey"
    )
    for tag in ("older", "newer"):
        await store._execute(
            "INSERT INTO lightrag_graph_nodes "
            "(workspace, namespace, id, properties, updated_at) "
            "VALUES ($1, $2, $3, $4::jsonb, now())",
            ws,
            ns,
            "DUP",
            json.dumps({"entity_id": "DUP", "tag": tag}),
        )

    # Re-apply the DDL: the migration sees the PK lacks namespace, de-dupes, and
    # rebuilds it. Must NOT raise on the duplicate rows.
    await store.db.execute(_DDL)

    assert await store.get_node("DUP") is not None
    rows = await store._fetch(
        "SELECT id FROM lightrag_graph_nodes "
        "WHERE workspace = $1 AND namespace = $2 AND id = $3",
        ws,
        ns,
        "DUP",
    )
    assert len(rows) == 1, "duplicates must collapse to a single surviving row"


@pytest.mark.asyncio
async def test_upsert_edge_replaces_properties(store):
    """Edge upsert REPLACES properties (node upsert merges). A second upsert with
    a different key set must drop keys absent from the new payload — matching
    PGGraphStorage and the documented contract. A regression to `||` (merge) on
    edges would otherwise go unnoticed."""
    await store.upsert_node("A", _node("A"))
    await store.upsert_node("B", _node("B"))
    await store.upsert_edge("A", "B", dict(_edge(weight=1.0), extra="present"))
    first = await store.get_edge("A", "B")
    assert first["extra"] == "present"

    # Re-upsert without "extra" -> replace semantics drops it (merge would keep it).
    await store.upsert_edge("A", "B", _edge(weight=2.0))
    second = await store.get_edge("A", "B")
    assert second["weight"] == 2.0
    assert "extra" not in second, "edge upsert must replace, not merge, properties"


@pytest.mark.asyncio
async def test_get_node_edges_self_loop_appears_once(store):
    """get_node_edges (single-node path, distinct from get_nodes_edges_batch)
    returns a self-loop (A, A) exactly once."""
    await store.upsert_node("A", _node("A"))
    await store.upsert_edge("A", "A", _edge())  # self-loop: min/max canonical (A, A)

    edges = await store.get_node_edges("A")
    assert edges == [("A", "A")], f"self-loop must appear exactly once: {edges}"


# ---------------------------------------------------------------------------
# SQL-vs-Python equivalence — the SQL rewrites must match their references
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_labels_sql_scoring_matches_search_score_reference(store):
    """The SQL CASE must score/rank exactly like the _search_score reference.

    Scoring moved from Python into SQL so the LIMIT could be applied server-side
    (the Python version transferred every id matching %q% to return `limit` of
    them). _search_score stays the definition of NetworkXStorage's semantics, so
    the two must agree — including the subtle part: the +50 word-boundary bonus
    applies ONLY to the contains branch, never to an exact or prefix match.

    Scope: every label below folds identically under SQL LOWER() and Python
    str.lower(), which is true of all but a handful of code points. Those are a
    separate, documented divergence — see
    test_search_labels_folds_case_with_the_database_not_python. Keep this corpus
    free of them, or this test stops testing the scoring rules and starts
    testing case folding.
    """
    labels = [
        "foo",
        "FOO",
        "foobar",
        "foo foo",  # prefix match that ALSO has a boundary hit -> must stay 500
        "xx_foo",
        "bar foo",
        "barfoo",
        "foo_bar",
        "prefix_foo_suffix",
        "x" * 120 + "foo",  # negative score (100 - len)
        "_foo",
        "café foo",
        "北京 foo",
        "foo%bar",  # literal % must not act as a wildcard
        "a_foo",  # literal _ must not act as a wildcard
        "zzfoo",
    ]
    await store.upsert_nodes_batch([(lbl, _node(lbl)) for lbl in labels])

    for query in ("foo", "f", "oo", "bar", "%", "_", "FOO"):
        normalized = query.strip().lower()
        for limit in (1, 3, 50):
            got = await store.search_labels(query, limit=limit)

            matched = [lbl for lbl in labels if normalized in lbl.lower()]
            expected = sorted(
                matched, key=lambda lbl: (-store._search_score(lbl, normalized), lbl)
            )[:limit]

            assert got == expected, (
                f"SQL scoring diverged from _search_score for "
                f"query={query!r} limit={limit}: {got} != {expected}"
            )


@pytest.mark.asyncio
async def test_search_labels_folds_case_with_the_database_not_python(store):
    """Pin the one place SQL scoring is allowed to disagree with _search_score.

    SQL LOWER() folds one character to one character; Python str.lower() applies
    full Unicode case mapping. So LOWER('İ') is 'i' — an EXACT match for query
    'i', worth 1000 — while 'İ'.lower() is 'i̇', only a prefix match worth
    500, which the code-point tie-break then puts behind 'ia'.

    This is not new authority: the pre-scoring implementation already folded with
    LOWER() in the same WHERE clause, so which labels match has always been the
    database's call. PGGraphStorage folds the same way, so the divergence is
    against the NetworkX/JSON storages, not between the PostgreSQL ones. Asserted
    explicitly here so it stays a decision rather than drifting into a surprise —
    if a future change unifies the folding, this test is the one to update.
    """
    # LOWER() is lc_ctype-dependent: a server initdb'd with the pure "C" locale
    # does not fold non-ASCII at all, so there is no divergence to pin there.
    # Ask this server rather than hard-coding one libc's answer.
    server_lower = await store._fetchval("SELECT LOWER($1)", "İ")
    if server_lower != "i":
        pytest.skip(
            f"server folds 'İ' to {server_lower!r}, not 'i' "
            "(lc_ctype=C?) — nothing to diverge from Python here"
        )

    labels = ["İ", "ia"]
    await store.upsert_nodes_batch([(lbl, _node(lbl)) for lbl in labels])

    # What the database's folding produces, in full and truncated to one.
    assert await store.search_labels("i", limit=50) == ["İ", "ia"]
    assert await store.search_labels("i", limit=1) == ["İ"]

    # ... and that this is a real divergence from the Python reference, not an
    # accident of ordering: the reference disagrees on both the tier and the rank.
    assert store._search_score("İ", "i") == 500
    assert store._search_score("ia", "i") == 500
    reference = sorted(labels, key=lambda lbl: (-store._search_score(lbl, "i"), lbl))
    assert reference == ["ia", "İ"]


@pytest.mark.asyncio
async def test_bfs_matches_degree_ordered_reference_traversal(store):
    """SQL-side rank+cap BFS must equal the Python reference on a hub graph.

    The per-hop unvisited filter, degree ranking and budget cap all moved into
    SQL. This pins the observable result — which nodes are retained at which BFS
    depth, and their degrees — against a plain in-memory reference walk, on a
    graph with a hub whose degree far exceeds the node budget (the case the
    rewrite targets).
    """
    nodes = [f"n{i:03d}" for i in range(120)]
    hub = nodes[0]
    edges = {(min(hub, n), max(hub, n)) for n in nodes[1:80]}
    # A second, lower-degree cluster reachable only at depth 2.
    edges |= {(min(nodes[80], n), max(nodes[80], n)) for n in nodes[81:100]}
    edges.add((min(hub, nodes[80]), max(hub, nodes[80])))
    edges.add((nodes[5], nodes[5]))  # self-loop: counts twice, like node_degree

    await store.upsert_nodes_batch([(n, _node(n)) for n in nodes])
    await store.upsert_edges_batch([(s, t, _edge()) for s, t in sorted(edges)])

    adjacency: dict[str, set[str]] = {}
    degree: dict[str, int] = {}
    for src, tgt in edges:
        adjacency.setdefault(src, set()).add(tgt)
        adjacency.setdefault(tgt, set()).add(src)
        degree[src] = degree.get(src, 0) + 1
        degree[tgt] = degree.get(tgt, 0) + 1

    def reference(seed, max_depth, budget):
        """Degree-ordered BFS, admitting one past the budget like the real one."""
        collected = {seed: 0}
        frontier, depth = [seed], 0
        while frontier and depth < max_depth and len(collected) <= budget:
            depth += 1
            candidates = {
                nb
                for node in frontier
                for nb in adjacency.get(node, ())
                if nb not in collected
            }
            if not candidates:
                break
            ordered = sorted(candidates, key=lambda n: (-degree.get(n, 0), n))
            nxt = []
            for node in ordered:
                collected[node] = depth
                nxt.append(node)
                if len(collected) > budget:
                    break
            frontier = nxt
        return collected

    for seed, max_depth, budget in [
        (hub, 1, 1000),
        (hub, 2, 1000),
        (hub, 3, 25),
        (hub, 2, 10),
        (nodes[80], 2, 1000),
        (nodes[99], 3, 40),
        (nodes[5], 2, 1000),
    ]:
        rows, degrees = await store._bfs_frontier(seed, max_depth, budget)
        expected = reference(seed, max_depth, budget)

        assert {r["id"] for r in rows} == set(expected), (
            f"node set diverged for seed={seed} depth={max_depth} budget={budget}"
        )
        assert {r["id"]: r["depth"] for r in rows} == expected, (
            f"BFS depths diverged for seed={seed} depth={max_depth} budget={budget}"
        )
        # Degrees cover every collected node except the seed, whose degree the
        # traversal deliberately never fetches (the caller pins it ahead of the
        # degree term — see _bfs_frontier's Returns note). Assert the exclusion
        # explicitly so a reintroduced round trip shows up as a failure here.
        assert seed not in degrees, f"seed degree should not be fetched: {seed}"
        for row in rows:
            if row["id"] == seed:
                continue
            assert degrees[row["id"]] == degree.get(row["id"], 0), (
                f"degree diverged for {row['id']}"
            )


@pytest.mark.asyncio
async def test_reinitialize_is_idempotent_and_keeps_data(store):
    """Re-running the DDL must not disturb existing rows.

    The orphan sweep is now gated behind "an FK is missing", so a second
    initialize() must skip it entirely — and must certainly not delete live
    edges. Guards the gating change against a regression that would make
    startup destructive.
    """
    await store.upsert_node("A", _node("A"))
    await store.upsert_node("B", _node("B"))
    await store.upsert_edge("A", "B", _edge(weight=3.5))

    from lightrag.kg.pgtable_impl import _DDL

    # Second pass over the schema DDL + legacy normalization.
    await store._db.execute(_DDL)
    await store._normalize_legacy_edges()

    assert await store.get_node("A") is not None
    assert await store.get_node("B") is not None
    edge = await store.get_edge("A", "B")
    assert edge is not None, "re-running the DDL must not delete live edges"
    assert edge["weight"] == 3.5


@pytest.mark.asyncio
async def test_orphan_sweep_still_runs_when_foreign_keys_are_missing(store):
    """Gating the sweep must not break the legacy migration it exists for.

    Drop both FKs, plant an orphan edge that the constraints would have
    forbidden, then re-run the DDL: the sweep must remove the orphan and both
    constraints must come back. Without the sweep, ADD CONSTRAINT would fail and
    initialize() would abort.
    """
    from lightrag.kg.pgtable_impl import _DDL

    await store.upsert_node("A", _node("A"))
    await store.upsert_node("B", _node("B"))
    await store.upsert_edge("A", "B", _edge())

    async def _plant_orphan(conn):
        await conn.execute(
            "ALTER TABLE lightrag_graph_edges "
            "DROP CONSTRAINT IF EXISTS fk_lightrag_graph_edges_src"
        )
        await conn.execute(
            "ALTER TABLE lightrag_graph_edges "
            "DROP CONSTRAINT IF EXISTS fk_lightrag_graph_edges_tgt"
        )
        await conn.execute(
            "INSERT INTO lightrag_graph_edges "
            "(workspace, namespace, src_id, tgt_id, properties) "
            "VALUES ($1, $2, 'A', 'GHOST', '{}'::jsonb)",
            store.workspace,
            store.namespace,
        )

    await store._db._run_with_retry(_plant_orphan)
    assert await store.has_edge("A", "GHOST") is True

    await store._db.execute(_DDL)

    assert await store.has_edge("A", "GHOST") is False, (
        "orphan edge must be swept before the FKs are recreated"
    )
    assert await store.get_edge("A", "B") is not None, "live edge must survive"

    async def _count_fks(conn):
        return await conn.fetchval(
            "SELECT COUNT(*) FROM pg_constraint "
            "WHERE conrelid = 'lightrag_graph_edges'::regclass "
            "AND conname IN ('fk_lightrag_graph_edges_src', "
            "'fk_lightrag_graph_edges_tgt')"
        )

    assert await store._db._run_with_retry(_count_fks) == 2, (
        "both foreign keys must be recreated after the sweep"
    )


# ---------------------------------------------------------------------------
# flush-time batching against a real server
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chunked_batch_writes_and_deletes_round_trip(store, monkeypatch):
    """Multi-chunk upserts and deletes must be indistinguishable from single ones.

    Caps are set absurdly low so every batch path splits, then the full data set is
    read back. This is the live counterpart to the offline chunking tests: it proves
    the per-chunk statements are valid SQL, that each edge chunk satisfies the
    endpoint foreign keys on its own, and that a chunked delete removes exactly the
    requested set.
    """
    monkeypatch.setenv("POSTGRES_UPSERT_MAX_RECORDS_PER_BATCH", "4")
    monkeypatch.setenv("POSTGRES_UPSERT_MAX_PAYLOAD_BYTES", "512")
    monkeypatch.setenv("POSTGRES_DELETE_MAX_RECORDS_PER_BATCH", "3")
    # The fixture may already have cached the default caps.
    store._cached_batch_limits = None

    nodes = [f"n{i:03d}" for i in range(37)]
    # Chain plus a few cross edges, all endpoints among `nodes`.
    edges = [(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]
    edges += [(nodes[0], nodes[20]), (nodes[5], nodes[30])]

    await store.upsert_nodes_batch([(n, _node(n)) for n in nodes])
    await store.upsert_edges_batch([(s, t, _edge()) for s, t in edges])

    fetched = await store.get_nodes_batch(nodes)
    assert set(fetched) == set(nodes), "every node must survive chunked upsert"
    for n in nodes:
        assert fetched[n]["description"] == f"smoke node {n}"

    all_edges = await store.get_all_edges()
    expected_pairs = {(min(s, t), max(s, t)) for s, t in edges}
    assert {(e["source"], e["target"]) for e in all_edges} == expected_pairs

    # Chunked edge delete removes exactly the requested pairs.
    to_drop = edges[:7]
    await store.remove_edges(list(to_drop))
    remaining = {(e["source"], e["target"]) for e in await store.get_all_edges()}
    assert remaining == expected_pairs - {(min(s, t), max(s, t)) for s, t in to_drop}

    # Chunked node delete removes exactly the requested nodes (and cascades edges).
    await store.remove_nodes(nodes[:11])
    still_there = await store.get_nodes_batch(nodes)
    assert set(still_there) == set(nodes[11:])
    for edge in await store.get_all_edges():
        assert edge["source"] in still_there and edge["target"] in still_there, (
            "cascade must not leave an edge pointing at a deleted node"
        )


@pytest.mark.asyncio
async def test_failed_upsert_chunk_leaves_no_edge_without_endpoints(store, monkeypatch):
    """A chunk failure mid-batch must not strand an edge without its endpoints.

    Upsert chunks are separate statements, so a crash between them leaves earlier
    chunks committed. That is acceptable only because each chunk creates its own
    endpoints in the same statement: the partial result must still satisfy the
    foreign keys, and replaying the whole batch must converge (upsert is
    idempotent). Without the per-chunk endpoint set, a later chunk's edge could
    reference a node an earlier failed chunk never wrote.
    """
    monkeypatch.setenv("POSTGRES_UPSERT_MAX_RECORDS_PER_BATCH", "2")
    store._cached_batch_limits = None

    edges = [(f"s{i}", f"t{i}", _edge()) for i in range(6)]
    real_execute = store._execute
    seen = {"count": 0}

    async def _flaky(sql, *args):
        seen["count"] += 1
        if seen["count"] == 2:
            raise RuntimeError("injected chunk failure")
        return await real_execute(sql, *args)

    store._execute = _flaky
    with pytest.raises(RuntimeError, match="injected chunk failure"):
        await store.upsert_edges_batch(edges)
    store._execute = real_execute

    persisted_nodes = {n["id"] for n in await store.get_all_nodes()}
    partial = await store.get_all_edges()
    assert partial, "the chunk before the failure must have committed"
    for edge in partial:
        assert (
            edge["source"] in persisted_nodes and edge["target"] in persisted_nodes
        ), "a committed chunk left an edge whose endpoint node is missing"

    # Replaying the full batch converges on the complete edge set.
    await store.upsert_edges_batch(edges)
    assert len(await store.get_all_edges()) == len(edges)


# ---------------------------------------------------------------------------
# End-to-end graph-storage contract, ported from tests/kg/test_graph_storage.py
#
# That module is the backend-agnostic end-to-end suite selected by
# LIGHTRAG_GRAPH_STORAGE, so it only exercises PGTableGraphStorage when a
# developer happens to point .env at it. The contract it asserts is ported here
# so the pg-smoke CI job runs it unconditionally against a live server.
#
# Already covered above and therefore NOT re-added:
#   test_graph_basic            -> test_upsert_and_get_node / _edge (extended
#                                  there with full-payload and forward==reverse
#                                  assertions)
#   node_degree, get_node_edges -> test_node_degree,
#                                  test_get_node_edges_queried_node_first
#   test_graph_special_chars    -> merged into
#                                  test_jsonb_unicode_and_special_chars
#   search_labels               -> test_search_labels_sql_scoring_matches_
#                                  search_score_reference (strictly stronger)
#   test_graph_undirected_property -> its unique assertions (undirected
#                                  edge_degree, undirected get_edges_batch,
#                                  undirected deletion) are folded into the
#                                  degree / batch-read / deletion tests below
#                                  rather than duplicated as a seventh walk over
#                                  the same graph.
# ---------------------------------------------------------------------------


def _kg_node(entity_id: str, description: str, keywords: str) -> dict:
    """A node payload shaped like the end-to-end suite's (keywords included)."""
    return dict(_node(entity_id), description=description, keywords=keywords)


async def _build_ai_graph(store) -> tuple[dict[str, dict], dict[tuple[str, str], dict]]:
    """Build the 5-node / 6-edge topology the ported batch tests assert on.

    Degrees: AI=3, ML=2, DL=3, NLP=2, CV=2.
    """
    nodes = {
        "AI": _kg_node("AI", "Artificial intelligence", "AI,ML,DL"),
        "ML": _kg_node("ML", "Machine learning", "supervised,unsupervised"),
        "DL": _kg_node("DL", "Deep learning", "neural networks,CNN,RNN"),
        "NLP": _kg_node("NLP", "Natural language processing", "NLP,text"),
        "CV": _kg_node("CV", "Computer vision", "CV,image recognition"),
    }
    for node_id, payload in nodes.items():
        await store.upsert_node(node_id, payload)

    edges = {
        ("AI", "ML"): dict(_edge(1.0), relationship="includes", description="ai>ml"),
        ("ML", "DL"): dict(_edge(1.0), relationship="includes", description="ml>dl"),
        ("AI", "NLP"): dict(_edge(1.0), relationship="includes", description="ai>nlp"),
        ("AI", "CV"): dict(_edge(1.0), relationship="includes", description="ai>cv"),
        ("DL", "NLP"): dict(
            _edge(0.8), relationship="applied to", description="dl>nlp"
        ),
        ("DL", "CV"): dict(_edge(0.8), relationship="applied to", description="dl>cv"),
    }
    for (src, tgt), payload in edges.items():
        await store.upsert_edge(src, tgt, payload)

    return nodes, edges


@pytest.mark.asyncio
async def test_edge_degree_sums_endpoints_and_is_symmetric(store):
    """edge_degree(a, b) == node_degree(a) + node_degree(b), in either direction.

    Ported from test_graph_advanced / test_graph_undirected_property: the smoke
    suite covered node_degree but never edge_degree, whose undirected symmetry is
    part of the storage contract.
    """
    for name in ("AI", "ML", "DL"):
        await store.upsert_node(name, _node(name))
    await store.upsert_edge("AI", "ML", _edge())
    await store.upsert_edge("ML", "DL", _edge())

    assert await store.node_degree("AI") == 1
    assert await store.node_degree("ML") == 2
    assert await store.node_degree("DL") == 1

    forward = await store.edge_degree("AI", "ML")
    assert forward == 3, f"edge_degree should be 1 + 2 = 3, got {forward}"
    assert await store.edge_degree("ML", "AI") == forward, (
        "edge_degree must be direction-independent on an undirected graph"
    )


@pytest.mark.asyncio
async def test_get_all_labels_lists_every_node_id(store):
    """get_all_labels returns every node id, sorted. Ported from
    test_graph_advanced; no smoke test touched this method."""
    nodes, _ = await _build_ai_graph(store)

    labels = await store.get_all_labels()

    assert labels == sorted(nodes), f"get_all_labels mismatch: {labels}"


@pytest.mark.asyncio
async def test_get_knowledge_graph_wildcard_returns_whole_graph(store):
    """The "*" wildcard path returns every node and edge, with entity_id intact.

    The smoke suite only exercised seeded BFS (`get_knowledge_graph("A", ...)`);
    the wildcard branch is a completely separate SQL query. The entity_id
    assertion is the WebUI regression guard from test_graph_advanced: the
    property panel reads properties["entity_id"] to render a node's name.
    """
    from lightrag.types import KnowledgeGraph

    nodes, edges = await _build_ai_graph(store)

    kg = await store.get_knowledge_graph("*", max_depth=2, max_nodes=10)

    assert isinstance(kg, KnowledgeGraph)
    assert {n.id for n in kg.nodes} == set(nodes)
    assert len(kg.edges) == len(edges)
    assert kg.is_truncated is False
    for kg_node in kg.nodes:
        assert kg_node.properties.get("entity_id") == kg_node.id, (
            f"node {kg_node.id} lost entity_id in properties: {kg_node.properties}"
        )


@pytest.mark.asyncio
async def test_delete_node_and_remove_nodes_and_edges(store):
    """delete_node / remove_edges / remove_nodes on a small live graph.

    Ported from test_graph_advanced plus the deletion half of
    test_graph_undirected_property: removing an edge in one direction must remove
    it in both, and deleting a node must take its incident edges with it (here via
    the FK cascade). The existing chunked-delete smoke test drives the same
    methods but only checks bulk arithmetic, never these semantics.
    """
    await _build_ai_graph(store)

    # delete_node also cascades the node's incident edges.
    await store.delete_node("CV")
    assert await store.get_node("CV") is None
    assert await store.has_edge("AI", "CV") is False
    assert await store.has_edge("DL", "CV") is False

    # remove_edges is undirected: both directions disappear.
    await store.remove_edges([("ML", "DL")])
    assert await store.get_edge("ML", "DL") is None
    assert await store.get_edge("DL", "ML") is None, (
        "reverse edge must be deleted too — deletion is undirected"
    )
    ml_edges = await store.get_node_edges("ML")
    assert ml_edges == [("ML", "AI")], f"ML should keep only its AI edge: {ml_edges}"

    # remove_nodes drops several nodes at once and leaves the rest untouched.
    await store.remove_nodes(["ML", "DL"])
    assert await store.get_node("ML") is None
    assert await store.get_node("DL") is None
    assert await store.get_node("AI") is not None
    assert await store.get_all_labels() == ["AI", "NLP"]
    surviving = {(e["source"], e["target"]) for e in await store.get_all_edges()}
    assert surviving == {("AI", "NLP")}, (
        f"only the AI-NLP edge should survive the cascade: {surviving}"
    )


@pytest.mark.asyncio
async def test_batch_read_helpers_round_trip_and_stay_undirected(store):
    """get_nodes_batch / node_degrees_batch / edge_degrees_batch /
    get_edges_batch / get_nodes_edges_batch against a real server.

    Ported from test_graph_batch_operations, with the reverse-pair assertions of
    test_graph_undirected_property folded in. Only get_nodes_batch and
    get_nodes_edges_batch had any live coverage here before, and neither checked
    property fidelity or degree arithmetic.
    """
    nodes, edges = await _build_ai_graph(store)

    # 1. get_nodes_batch — properties survive the batch read path.
    nodes_dict = await store.get_nodes_batch(["AI", "ML", "DL"])
    assert set(nodes_dict) == {"AI", "ML", "DL"}
    for node_id, payload in nodes_dict.items():
        assert payload["description"] == nodes[node_id]["description"]
        assert payload["keywords"] == nodes[node_id]["keywords"]
        assert payload["entity_id"] == node_id

    # 2. node_degrees_batch — matches the per-node degrees.
    degrees = await store.node_degrees_batch(["AI", "ML", "DL", "NLP", "CV", "ghost"])
    assert degrees == {
        "AI": 3,
        "ML": 2,
        "DL": 3,
        "NLP": 2,
        "CV": 2,
        "ghost": 0,
    }, f"node_degrees_batch mismatch: {degrees}"

    # 3. edge_degrees_batch — sum of endpoint degrees per pair.
    pairs = [("AI", "ML"), ("ML", "DL"), ("DL", "NLP")]
    edge_degrees = await store.edge_degrees_batch(pairs)
    assert edge_degrees == {
        ("AI", "ML"): 5,
        ("ML", "DL"): 5,
        ("DL", "NLP"): 5,
    }, f"edge_degrees_batch mismatch: {edge_degrees}"

    # 4. get_edges_batch — properties keyed by the pair AS ASKED, both directions.
    forward = await store.get_edges_batch([{"src": s, "tgt": t} for s, t in pairs])
    reverse = await store.get_edges_batch([{"src": t, "tgt": s} for s, t in pairs])
    assert set(forward) == set(pairs)
    assert set(reverse) == {(t, s) for s, t in pairs}
    for src, tgt in pairs:
        expected = edges[(src, tgt)]
        assert forward[(src, tgt)]["relationship"] == expected["relationship"]
        assert forward[(src, tgt)]["description"] == expected["description"]
        assert reverse[(tgt, src)] == forward[(src, tgt)], (
            f"edge {src}<->{tgt} not undirected-consistent in get_edges_batch"
        )

    # 5. get_nodes_edges_batch — every incident edge, regardless of direction.
    nodes_edges = await store.get_nodes_edges_batch(["AI", "DL"])
    assert set(nodes_edges) == {"AI", "DL"}
    assert {tgt for _, tgt in nodes_edges["AI"]} == {"ML", "NLP", "CV"}
    assert {tgt for _, tgt in nodes_edges["DL"]} == {"ML", "NLP", "CV"}


@pytest.mark.asyncio
async def test_batch_upsert_dedup_and_existence_checks(store, monkeypatch):
    """upsert_nodes_batch / upsert_edges_batch: same-batch dedup, has_node,
    has_nodes_batch, reciprocal edge collapse.

    Ported from test_graph_batch_upsert. The chunking smoke tests above drive the
    same two methods but never feed them a duplicate, and has_nodes_batch had no
    live coverage at all. The record cap is lowered to 2 so the duplicate and its
    original land in different chunks: dedup has to happen before chunking, or the
    earlier payload wins in its own statement.
    """
    monkeypatch.setenv("POSTGRES_UPSERT_MAX_RECORDS_PER_BATCH", "2")
    # The fixture may already have cached the default caps.
    store._cached_batch_limits = None

    nodes = [
        ("E1", _kg_node("E1", "first", "k")),
        ("E2", _kg_node("E2", "second", "k")),
        ("E3", _kg_node("E3", "third", "k")),
        ("E4", _kg_node("E4", "fourth", "k")),
        ("E5", _kg_node("E5", "fifth", "k")),
        # duplicate of E1 later in the same batch -> this payload must win
        ("E1", _kg_node("E1", "first-updated", "k")),
    ]
    await store.upsert_nodes_batch(nodes)

    for node_id in ("E1", "E2", "E3", "E4", "E5"):
        assert await store.has_node(node_id) is True

    assert await store.has_nodes_batch(["E1", "E3", "E5", "DOES_NOT_EXIST"]) == {
        "E1",
        "E3",
        "E5",
    }

    e1 = await store.get_node("E1")
    assert e1 is not None and e1["description"] == "first-updated", (
        "same-batch node dedup must keep the last write"
    )

    edges = [
        ("E1", "E2", dict(_edge(1.0), relationship="r12")),
        ("E2", "E3", dict(_edge(1.0), relationship="r23")),
        ("E3", "E4", dict(_edge(1.0), relationship="r34")),
        ("E4", "E5", dict(_edge(1.0), relationship="r45")),
        # reciprocal of (E1, E2): undirected -> last write wins
        ("E2", "E1", dict(_edge(2.0), relationship="r12-updated")),
    ]
    await store.upsert_edges_batch(edges)

    for a, b in (("E1", "E2"), ("E2", "E3"), ("E3", "E4"), ("E4", "E5")):
        fwd = await store.get_edge(a, b)
        rev = await store.get_edge(b, a)
        assert fwd is not None, f"edge {a}->{b} missing after upsert_edges_batch"
        assert fwd == rev, f"edge {a}<->{b} not undirected-consistent"

    e12 = await store.get_edge("E1", "E2")
    assert e12["relationship"] == "r12-updated", "reciprocal edge dedup lost"
    assert float(e12["weight"]) == pytest.approx(2.0), (
        "reciprocal edge dedup kept the wrong payload"
    )

    # Exactly four distinct edges were written, so the degrees are unambiguous.
    assert await store.node_degree("E1") == 1
    assert await store.node_degree("E2") == 2
    assert await store.node_degree("E3") == 2
    assert await store.node_degree("E5") == 1


@pytest.mark.asyncio
async def test_get_all_nodes_and_popular_labels_rank_by_degree(store):
    """get_all_nodes carries "id"; get_popular_labels orders by degree.

    Ported from test_graph_query_helpers. get_all_nodes was only ever asserted
    empty here (in the drop test), and get_popular_labels only on a single
    self-loop — neither pinned the "id" key or the degree ordering.
    """
    star = ["Alpha", "Beta", "Gamma", "Alphabet"]
    for node_id in star:
        await store.upsert_node(node_id, _node(node_id))
    star_edges = [("Alpha", "Beta"), ("Alpha", "Gamma"), ("Alpha", "Alphabet")]
    for src, tgt in star_edges:
        await store.upsert_edge(src, tgt, _edge())

    all_nodes = await store.get_all_nodes()
    assert {n["id"] for n in all_nodes} == set(star)
    assert all("entity_id" in n for n in all_nodes)

    all_edges = await store.get_all_edges()
    assert {frozenset((e["source"], e["target"])) for e in all_edges} == {
        frozenset(pair) for pair in star_edges
    }

    # Alpha has degree 3, every spoke has 1 — the hub must come first.
    popular = await store.get_popular_labels(limit=2)
    assert len(popular) == 2
    assert popular[0] == "Alpha", f"highest-degree label must lead: {popular}"
    assert popular[1] in {"Alphabet", "Beta", "Gamma"}


@pytest.mark.asyncio
async def test_escaped_entity_ids_round_trip_across_every_path(store):
    """Quoted / backslash-laden entity IDs through every read and write path.

    Ported from test_graph_string_escaping_regressions. Entity ids (not just
    property values) go into positional parameters and into ANY()/unnest() arrays,
    so each read shape needs its own pass; the ids here are the ones that broke
    the Cypher-escaping backends.
    """
    center_id = 'Danh mục "bài toán lớn"'
    backslash_id = r"C:\Program Files\LightRAG"
    mixed_id = 'Path "C:\\RAG\\docs"'
    single_quote_id = "Node with 'single quotes'"
    ids = [center_id, backslash_id, mixed_id, single_quote_id]

    node_payloads = {
        center_id: _kg_node(
            center_id,
            'Quoted entity with JSON-ish payload {"path": "C:\\\\temp"}',
            'quotes,"double quotes",unicode',
        ),
        backslash_id: _kg_node(
            backslash_id,
            r"Windows path C:\Program Files\LightRAG\bin",
            r"paths,C:\temp,backslashes",
        ),
        mixed_id: _kg_node(
            mixed_id,
            'Mixed quotes "and" slashes \\ in one entity id',
            r'mixed,"quoted",C:\RAG\docs',
        ),
        single_quote_id: _kg_node(
            single_quote_id,
            "Single quotes stay literal in entity identifiers",
            "single quotes,escaping",
        ),
    }
    for node_id, payload in node_payloads.items():
        await store.upsert_node(node_id, payload)

    edge_payloads = {
        (center_id, backslash_id): dict(
            _edge(1.0),
            relationship=r'contains "path"\edge',
            description=r'Links "quoted" title to C:\Program Files\LightRAG',
        ),
        (center_id, mixed_id): dict(
            _edge(0.8),
            relationship='references "docs"',
            description=r'Contains both "quotes" and \\backslashes\\',
        ),
        (center_id, single_quote_id): dict(
            _edge(0.6),
            relationship="mentions 'alias'",
            description='Single quote entity linked to "quoted" center node',
        ),
    }
    for (src_id, tgt_id), payload in edge_payloads.items():
        await store.upsert_edge(src_id, tgt_id, payload)

    # Single-node reads.
    for node_id, payload in node_payloads.items():
        node = await store.get_node(node_id)
        assert node is not None, f"expected node {node_id!r} to round-trip"
        assert node["entity_id"] == node_id
        assert node["description"] == payload["description"]

    # Batch reads (ANY() array binding).
    nodes_batch = await store.get_nodes_batch(ids)
    assert set(nodes_batch) == set(ids)
    for node_id, payload in node_payloads.items():
        assert nodes_batch[node_id]["entity_id"] == node_id
        assert nodes_batch[node_id]["description"] == payload["description"]

    assert await store.has_nodes_batch(ids) == set(ids)
    assert await store.node_degrees_batch(ids) == {
        center_id: 3,
        backslash_id: 1,
        mixed_id: 1,
        single_quote_id: 1,
    }

    def connects(edges, a, b):
        """Undirected: accept either endpoint order."""
        return any(
            (src == a and tgt == b) or (src == b and tgt == a) for src, tgt in edges
        )

    center_edges = await store.get_node_edges(center_id)
    assert center_edges is not None
    for other in (backslash_id, mixed_id, single_quote_id):
        assert connects(center_edges, center_id, other), (
            f"center_edges should contain connection to {other!r}"
        )

    batch_edges = await store.get_nodes_edges_batch(ids)
    assert set(batch_edges) == set(ids)
    for other in (backslash_id, mixed_id, single_quote_id):
        assert connects(batch_edges[center_id], center_id, other)
        assert connects(batch_edges[other], center_id, other)

    # Edge reads, forward and reverse.
    for (src_id, tgt_id), payload in edge_payloads.items():
        fwd = await store.get_edge(src_id, tgt_id)
        rev = await store.get_edge(tgt_id, src_id)
        assert fwd is not None, f"get_edge({src_id!r}, {tgt_id!r}) returned None"
        assert rev == fwd, f"edge {src_id!r}<->{tgt_id!r} not undirected-consistent"
        assert fwd["relationship"] == payload["relationship"]
        assert fwd["description"] == payload["description"]
        assert await store.has_edge(src_id, tgt_id) is True
        assert await store.has_edge(tgt_id, src_id) is True

    forward_batch = await store.get_edges_batch(
        [{"src": src_id, "tgt": tgt_id} for src_id, tgt_id in edge_payloads]
    )
    reverse_batch = await store.get_edges_batch(
        [{"src": tgt_id, "tgt": src_id} for src_id, tgt_id in edge_payloads]
    )
    assert set(forward_batch) == set(edge_payloads)
    for pair, payload in edge_payloads.items():
        assert forward_batch[pair]["relationship"] == payload["relationship"]
        assert forward_batch[pair]["description"] == payload["description"]
        assert reverse_batch[(pair[1], pair[0])] == forward_batch[pair]

    # Write paths: escaped ids must be deletable too.
    await store.remove_edges([(center_id, mixed_id)])
    assert await store.get_edge(center_id, mixed_id) is None
    assert await store.get_edge(mixed_id, center_id) is None
    remaining = await store.get_node_edges(center_id)
    assert remaining is not None
    assert not connects(remaining, center_id, mixed_id)

    await store.delete_node(single_quote_id)
    assert await store.get_node(single_quote_id) is None

    await store.remove_nodes([center_id, backslash_id])
    assert await store.get_node(center_id) is None
    assert await store.get_node(backslash_id) is None
    assert await store.get_node(mixed_id) is not None
