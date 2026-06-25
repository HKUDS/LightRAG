"""Offline unit tests for PGTableGraphStorage.

Uses AsyncMock to patch _fetch/_fetchrow/_execute so no real PostgreSQL
connection is required. Marked @offline so these run in CI.
"""

import json
from contextlib import asynccontextmanager
from pathlib import Path
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

pytestmark = pytest.mark.offline

GRAPH_NAMESPACE = "chunk_entity_relation"


def make_storage(global_config=None, namespace=GRAPH_NAMESPACE):
    """Build a PGTableGraphStorage with a mocked db."""
    from lightrag.kg.pgtable_impl import PGTableGraphStorage

    storage = object.__new__(PGTableGraphStorage)
    storage.workspace = "test"
    storage.namespace = namespace
    storage.global_config = global_config or {}
    mock_db = MagicMock()
    storage.db = mock_db
    return storage


def make_uninitialized_storage(global_config=None, namespace=GRAPH_NAMESPACE):
    storage = make_storage(global_config=global_config, namespace=namespace)
    storage.db = None
    return storage


@asynccontextmanager
async def fake_init_lock():
    yield


def test_constructor_validates_workspace_like_other_backends():
    from lightrag.kg.pgtable_impl import PGTableGraphStorage

    with pytest.raises(ValueError, match="workspace"):
        PGTableGraphStorage(
            namespace=GRAPH_NAMESPACE,
            workspace="../bad",
            global_config={},
            embedding_func=MagicMock(),
        )


@pytest.mark.parametrize(
    ("global_config", "expected_vector_storage"),
    [
        ({}, "PGTableGraphStorage"),
        ({"vector_storage": "NanoVectorDBStorage"}, "NanoVectorDBStorage"),
    ],
)
@pytest.mark.asyncio
async def test_initialize_passes_pgtable_safe_vector_storage_to_client_manager(
    global_config, expected_vector_storage
):
    storage = make_uninitialized_storage(global_config=global_config)
    db = MagicMock()
    db.workspace = None
    db.execute = AsyncMock()

    with (
        patch(
            "lightrag.kg.postgres_impl.ClientManager.get_client",
            new=AsyncMock(return_value=db),
        ) as get_client,
        patch(
            "lightrag.kg.pgtable_impl.get_data_init_lock",
            new=MagicMock(return_value=fake_init_lock()),
        ),
        patch.object(storage, "_normalize_legacy_edges", new=AsyncMock()),
    ):
        await storage.initialize()

    get_client.assert_awaited_once_with(vector_storage=expected_vector_storage)


# ---------------------------------------------------------------------------
# get_node_edges — queried node must be tuple[0]
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_node_edges_queried_node_first_as_src():
    """When the queried node is stored as src_id, it must still be first."""
    storage = make_storage()
    # Edge stored as (A, Z) — canonical order because A < Z
    with (
        patch.object(storage, "has_node", new=AsyncMock(return_value=True)),
        patch.object(
            storage,
            "_fetch",
            new=AsyncMock(return_value=[{"src_id": "A", "tgt_id": "Z"}]),
        ),
    ):
        result = await storage.get_node_edges("A")
    assert result == [("A", "Z")]


@pytest.mark.asyncio
async def test_get_node_edges_queried_node_first_as_tgt():
    """When the queried node is stored as tgt_id, it must be moved to first."""
    storage = make_storage()
    # Edge stored as (A, Z) — querying Z, so Z must be first
    with (
        patch.object(storage, "has_node", new=AsyncMock(return_value=True)),
        patch.object(
            storage,
            "_fetch",
            new=AsyncMock(return_value=[{"src_id": "A", "tgt_id": "Z"}]),
        ),
    ):
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
    with patch.object(
        storage, "_fetch", new=AsyncMock(return_value=[{"src_id": "A", "tgt_id": "Z"}])
    ):
        result = await storage.get_nodes_edges_batch(["A", "Z"])
    assert ("A", "Z") in result["A"]
    assert ("Z", "A") in result["Z"]  # Z is queried node, must be first


@pytest.mark.asyncio
async def test_get_nodes_edges_batch_self_loop_counted_once():
    """A self-loop (A, A) is one edge — it must appear once, not twice
    (matches get_node_edges() and NetworkX)."""
    storage = make_storage()
    with patch.object(
        storage, "_fetch", new=AsyncMock(return_value=[{"src_id": "A", "tgt_id": "A"}])
    ):
        result = await storage.get_nodes_edges_batch(["A"])
    assert result["A"] == [("A", "A")]


@pytest.mark.asyncio
async def test_get_nodes_edges_batch_uses_index_friendly_union():
    storage = make_storage()
    with patch.object(storage, "_fetch", new=AsyncMock(return_value=[])) as fetch:
        await storage.get_nodes_edges_batch(["A", "Z"])

    sql, workspace, namespace, node_ids = fetch.call_args.args
    normalized_sql = " ".join(sql.split()).upper()
    compact_sql = normalized_sql.replace(" ", "")
    assert workspace == "test"
    assert namespace == GRAPH_NAMESPACE
    assert node_ids == ["A", "Z"]
    assert " OR " not in normalized_sql
    assert "UNION" in normalized_sql
    assert "SRC_ID=ANY($3)" in compact_sql
    assert "TGT_ID=ANY($3)" in compact_sql


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
# DDL — referential integrity
# ---------------------------------------------------------------------------


def test_ddl_adds_cascading_edge_foreign_keys():
    """Edges must not survive after endpoint nodes are deleted."""
    from lightrag.kg.pgtable_impl import _DDL

    assert "PRIMARY KEY (workspace, namespace, id)" in _DDL
    assert "PRIMARY KEY (workspace, namespace, src_id, tgt_id)" in _DDL
    assert "fk_lightrag_graph_edges_src" in _DDL
    assert "fk_lightrag_graph_edges_tgt" in _DDL
    assert "FOREIGN KEY (workspace, namespace, src_id)" in _DDL
    assert "REFERENCES lightrag_graph_nodes (workspace, namespace, id)" in _DDL
    assert "ON DELETE CASCADE" in _DDL
    assert "DELETE FROM lightrag_graph_edges e" in _DDL
    assert "n.namespace = e.namespace" in _DDL


def test_postgres_impl_loads_pgvector_lazily_for_vector_connections():
    """PGTable imports postgres_impl but must not trigger pgvector installation."""
    src = (
        Path(__file__).parents[3] / "lightrag" / "kg" / "postgres_impl.py"
    ).read_text()
    module_preamble = src.split("class PostgreSQLDB", 1)[0]

    assert "pgvector.asyncpg" not in module_preamble
    assert 'pm.install("pgvector")' not in module_preamble
    assert "if self.enable_vector:" in src
    assert 'pm.install("pgvector")' in src
    assert "from pgvector.asyncpg import register_vector" in src


# ---------------------------------------------------------------------------
# entity_id normalization
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_node_raises_without_entity_id():
    """Match PGGraphStorage: a node payload missing entity_id is rejected, not
    silently patched, so malformed callers surface immediately."""
    storage = make_storage()
    with pytest.raises(ValueError, match="entity_id"):
        await storage.upsert_node("node1", {"name": "Alice"})


@pytest.mark.asyncio
async def test_upsert_node_overwrites_mismatched_entity_id():
    storage = make_storage()
    execute = AsyncMock()

    with patch.object(storage, "_execute", new=execute):
        await storage.upsert_node("node1", {"entity_id": "wrong", "name": "Alice"})

    *_, props = execute.call_args.args
    assert json.loads(props)["entity_id"] == "node1"


# ---------------------------------------------------------------------------
# node upsert — merge (not replace) semantics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_node_merges_properties():
    """Node upsert must MERGE properties (jsonb ||), not replace, matching
    NetworkXStorage.add_node and PGGraphStorage SET n += so omitted keys
    survive a partial update."""
    storage = make_storage()
    execute = AsyncMock()

    with patch.object(storage, "_execute", new=execute):
        await storage.upsert_node("n1", {"entity_id": "n1", "entity_type": "person"})

    sql, *_, props = execute.call_args.args
    assert "lightrag_graph_nodes.properties || EXCLUDED.properties" in sql
    assert "= EXCLUDED.properties," not in sql  # not wholesale replace
    assert json.loads(props)["entity_id"] == "n1"


@pytest.mark.asyncio
async def test_upsert_nodes_batch_merges_properties():
    storage = make_storage()
    execute = AsyncMock()

    with patch.object(storage, "_execute", new=execute):
        await storage.upsert_nodes_batch(
            [("n1", {"entity_id": "n1", "entity_type": "person"})]
        )

    sql, *_ = execute.call_args.args
    assert "lightrag_graph_nodes.properties || EXCLUDED.properties" in sql
    assert "= EXCLUDED.properties," not in sql


@pytest.mark.asyncio
async def test_upsert_nodes_batch_last_write_wins_on_duplicate():
    """A node_id appearing twice in one batch keeps only the LAST payload (not a
    merge of both), matching PGGraphStorage.upsert_nodes_batch and the shared
    batch-ordering tests. The surviving payload is still jsonb-merged with any
    pre-existing DB row by ON CONFLICT — but the discarded earlier batch payload
    does not contribute."""
    storage = make_storage()
    execute = AsyncMock()

    with patch.object(storage, "_execute", new=execute):
        await storage.upsert_nodes_batch(
            [
                ("n", {"entity_id": "n", "a": "1"}),
                ("n", {"entity_id": "n", "b": "2"}),
            ]
        )

    *_, ids, props = execute.call_args.args
    assert ids == ["n"]
    last = json.loads(props[0])
    assert "a" not in last  # earlier batch payload discarded (last-write-wins)
    assert last["b"] == "2"
    assert last["entity_id"] == "n"


# ---------------------------------------------------------------------------
# _execute — transient write-error retry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_retries_transient_write_error():
    """deadlock/serialization/lock/cancel are query-level transient errors that
    PostgreSQLDB._run_with_retry does not cover; _execute must retry them."""
    # asyncpg is genuinely required by this test, not lazily avoidable: the retry
    # path under test (_is_transient_write_error) matches by isinstance against
    # real asyncpg.exceptions.* classes, so a fabricated exception object can't
    # exercise it (only the real classes satisfy the isinstance check).
    # importorskip only GUARDS (skips) this one test when asyncpg is absent — it
    # does not drop the dependency. The offline CI installs asyncpg via the
    # offline-storage extra, so this test runs there; a bare `test`-extra env
    # without asyncpg skips it instead of erroring.
    asyncpg = pytest.importorskip("asyncpg")

    storage = make_storage()
    storage.db.execute = AsyncMock(
        side_effect=[
            asyncpg.exceptions.DeadlockDetectedError("deadlock"),
            asyncpg.exceptions.SerializationError("serialize"),
            None,
        ]
    )
    await storage._execute("INSERT ...", "a")
    assert storage.db.execute.call_count == 3


@pytest.mark.asyncio
async def test_execute_does_not_retry_non_transient_error():
    storage = make_storage()
    storage.db.execute = AsyncMock(side_effect=ValueError("boom"))
    with pytest.raises(ValueError):
        await storage._execute("INSERT ...", "a")
    assert storage.db.execute.call_count == 1


# ---------------------------------------------------------------------------
# upsert_edge — endpoint existence policy
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_edge_creates_missing_endpoints_in_sql():
    """Missing endpoint edges follow NetworkX add_edge semantics."""
    storage = make_storage()
    execute = AsyncMock()

    with patch.object(storage, "_execute", new=execute):
        await storage.upsert_edge("A", "B", {"weight": 1.0})

    sql, workspace, namespace, src, tgt, props, endpoint_ids = execute.call_args.args
    assert workspace == "test"
    assert namespace == GRAPH_NAMESPACE
    assert src == "A"
    assert tgt == "B"
    assert json.loads(props)["weight"] == 1.0
    assert endpoint_ids == ["A", "B"]
    assert "jsonb_build_object('entity_id', u.id)" in sql
    assert "DO NOTHING" in sql
    assert "WHERE EXISTS" not in sql


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
    storage = make_storage(global_config={"max_graph_nodes": 100})
    all_rows = [
        {
            "id": f"node_{i}",
            "properties": json.dumps({"entity_id": f"node_{i}"}),
            "degree": i,
        }
        for i in range(20)
    ]

    async def fetch_rows(sql, *args):
        normalized_sql = " ".join(sql.split()).upper()
        if "FROM LIGHTRAG_GRAPH_NODES" in normalized_sql:
            if "LIMIT" in normalized_sql:
                return all_rows[: args[2]]
            return all_rows
        return []

    fetch = AsyncMock(side_effect=fetch_rows)
    node_degrees = AsyncMock(
        return_value={f"node_{i}": i for i in range(len(all_rows))}
    )

    with (
        patch.object(storage, "_fetch", new=fetch),
        patch.object(storage, "node_degrees_batch", new=node_degrees),
    ):
        await storage.get_knowledge_graph("*", max_nodes=7)

    sql, *args = fetch.call_args_list[0].args
    normalized_sql = " ".join(sql.split()).upper()
    assert args == ["test", GRAPH_NAMESPACE, 8]
    assert "WITH RECURSIVE" not in sql
    assert "LIMIT" in normalized_sql
    assert "$3" in normalized_sql
    node_degrees.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_knowledge_graph_clamps_max_nodes():
    """Caller-supplied max_nodes > config value must be clamped to config."""
    storage = make_storage(global_config={"max_graph_nodes": 10})
    rows = [
        {"id": str(i), "properties": json.dumps({"entity_id": str(i)})}
        for i in range(20)
    ]
    # First _fetch call returns nodes; second (edge query) returns no edges.
    with (
        patch.object(storage, "_fetch", new=AsyncMock(side_effect=[rows, []])),
        patch.object(
            storage,
            "node_degrees_batch",
            new=AsyncMock(return_value={str(i): i for i in range(20)}),
        ),
    ):
        kg = await storage.get_knowledge_graph("*", max_nodes=999)
    assert len(kg.nodes) <= 10
    assert kg.is_truncated is True


@pytest.mark.asyncio
async def test_get_knowledge_graph_truncates_by_degree_then_id():
    """Truncation must not depend on unordered PostgreSQL row order."""
    storage = make_storage(global_config={"max_graph_nodes": 100})
    rows = [
        {"id": "z_low", "properties": json.dumps({"entity_id": "z_low"}), "degree": 1},
        {"id": "tie_b", "properties": json.dumps({"entity_id": "tie_b"}), "degree": 3},
        {"id": "hub", "properties": json.dumps({"entity_id": "hub"}), "degree": 9},
        {"id": "tie_a", "properties": json.dumps({"entity_id": "tie_a"}), "degree": 3},
    ]
    degrees = {"z_low": 1, "tie_b": 3, "hub": 9, "tie_a": 3}

    with (
        patch.object(storage, "_fetch", new=AsyncMock(side_effect=[rows, []])),
        patch.object(
            storage, "node_degrees_batch", new=AsyncMock(return_value=degrees)
        ),
    ):
        kg = await storage.get_knowledge_graph("*", max_nodes=3)

    assert [node.id for node in kg.nodes] == ["hub", "tie_a", "tie_b"]
    assert kg.is_truncated is True


@pytest.mark.asyncio
async def test_get_knowledge_graph_seed_never_dropped_on_truncation():
    """Exact-label seed must survive truncation even when it has a low degree.

    Regression for the P2 bug where degree-sort ran before truncation without
    pinning the seed, so a low-degree seed connected to high-degree neighbors
    could be evicted when max_nodes was tight.
    """
    storage = make_storage(global_config={"max_graph_nodes": 100})
    # BFS returns seed "low_seed" (degree 1) + two high-degree neighbors.
    bfs_rows = [
        {
            "id": "low_seed",
            "properties": json.dumps({"entity_id": "low_seed"}),
            "depth": 0,
        },
        {
            "id": "big_hub_1",
            "properties": json.dumps({"entity_id": "big_hub_1"}),
            "depth": 1,
        },
        {
            "id": "big_hub_2",
            "properties": json.dumps({"entity_id": "big_hub_2"}),
            "depth": 1,
        },
    ]
    degrees = {"low_seed": 1, "big_hub_1": 99, "big_hub_2": 97}

    with (
        # _bfs_frontier returns (rows, degrees); the caller reuses those degrees
        # for ordering instead of a second node_degrees_batch.
        patch.object(
            storage, "_bfs_frontier", new=AsyncMock(return_value=(bfs_rows, degrees))
        ),
        patch.object(storage, "_fetch", new=AsyncMock(return_value=[])),  # edge fetch
    ):
        # max_nodes=2 forces truncation; without the seed-pin fix, low_seed drops out.
        kg = await storage.get_knowledge_graph("low_seed", max_nodes=2)

    node_ids = [node.id for node in kg.nodes]
    assert "low_seed" in node_ids, (
        f"seed 'low_seed' was dropped by truncation; got {node_ids}"
    )
    assert len(node_ids) == 2
    assert kg.is_truncated is True
    # Seed is at position 0 (pinned), then the highest-degree neighbor.
    assert node_ids[0] == "low_seed"
    assert node_ids[1] == "big_hub_1"


@pytest.mark.asyncio
async def test_get_knowledge_graph_wildcard_does_not_pin_literal_star():
    """Wildcard has no seed: a node whose id is literally '*' must rank by
    degree, not be pinned first."""
    storage = make_storage()
    node_rows = [
        {"id": "*", "properties": json.dumps({"entity_id": "*"}), "degree": 1},
        {"id": "hub", "properties": json.dumps({"entity_id": "hub"}), "degree": 99},
    ]
    degrees = {"*": 1, "hub": 99}
    with (
        patch.object(storage, "_fetch", new=AsyncMock(side_effect=[node_rows, []])),
        patch.object(
            storage, "node_degrees_batch", new=AsyncMock(return_value=degrees)
        ),
    ):
        kg = await storage.get_knowledge_graph("*", max_nodes=10)
    node_ids = [n.id for n in kg.nodes]
    # hub (degree 99) outranks the literal '*' (degree 1); '*' is NOT pinned.
    assert node_ids[0] == "hub"
    assert "*" in node_ids


@pytest.mark.asyncio
async def test_get_knowledge_graph_bfs_depth_beats_degree_on_truncation():
    """Truncation must follow BFS-level order, not global degree.

    Regression for the design bug where the exact-label path sorted the full
    reachable set by global degree before truncating. That can drop a low-degree
    immediate neighbor in favour of a high-degree node two hops away, yielding a
    subgraph that is disconnected from the requested seed. The reference backends
    (networkx / AGE _bfs_subgraph) always retain nearer nodes first, so a depth-1
    neighbour must outrank a depth-2 hub regardless of degree.
    """
    storage = make_storage(global_config={"max_graph_nodes": 100})
    # Topology:  seed --- near (depth 1, low degree) --- far_hub (depth 2, huge degree)
    # _bfs_frontier returns rows carrying their shortest-hop depth.
    rows = [
        {"id": "seed", "properties": json.dumps({"entity_id": "seed"}), "depth": 0},
        {"id": "near", "properties": json.dumps({"entity_id": "near"}), "depth": 1},
        {
            "id": "far_hub",
            "properties": json.dumps({"entity_id": "far_hub"}),
            "depth": 2,
        },
    ]
    # far_hub has the highest degree by far; pure degree-sort would keep it.
    degrees = {"seed": 1, "near": 1, "far_hub": 99}
    # Edge fetch (second _fetch call) returns only the seed--near edge, since
    # after truncation to {seed, near} that is the only intra-set edge.
    edge_rows = [
        {
            "src_id": "seed",
            "tgt_id": "near",
            "properties": json.dumps({"weight": 1.0}),
        }
    ]

    with (
        patch.object(
            storage, "_bfs_frontier", new=AsyncMock(return_value=(rows, degrees))
        ),
        patch.object(storage, "_fetch", new=AsyncMock(return_value=edge_rows)),
    ):
        # max_nodes=2: budget covers depth 0 + depth 1; depth 2 must be excluded.
        kg = await storage.get_knowledge_graph("seed", max_nodes=2)

    node_ids = [node.id for node in kg.nodes]
    assert node_ids == [
        "seed",
        "near",
    ], f"expected BFS-nearest nodes; degree leaked into selection: {node_ids}"
    assert "far_hub" not in node_ids, "depth-2 hub must not displace a depth-1 neighbor"
    assert kg.is_truncated is True
    # The retained subgraph stays connected to the seed.
    assert len(kg.edges) == 1
    assert {kg.edges[0].source, kg.edges[0].target} == {"seed", "near"}


# ---------------------------------------------------------------------------
# search_labels — literal SQL pattern semantics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_labels_strips_and_relevance_orders():
    storage = make_storage()
    fetch = AsyncMock(return_value=[])

    with patch.object(storage, "_fetch", new=fetch):
        assert await storage.search_labels("   ") == []
        await storage.search_labels(" Foo ", limit=7)

    sql, workspace, namespace, contains = fetch.call_args.args
    assert workspace == "test"
    assert namespace == GRAPH_NAMESPACE
    assert contains == "%foo%"
    assert "ORDER BY" not in sql
    assert "ESCAPE '\\'" in sql


@pytest.mark.asyncio
async def test_search_labels_escapes_like_wildcards():
    storage = make_storage()
    fetch = AsyncMock(return_value=[])

    with patch.object(storage, "_fetch", new=fetch):
        await storage.search_labels(r"a_%\b")

    _sql, _workspace, namespace, contains = fetch.call_args.args
    assert namespace == GRAPH_NAMESPACE
    assert contains == r"%a\_\%\\b%"


@pytest.mark.asyncio
async def test_search_labels_relevance_sorted_in_python():
    storage = make_storage()
    with patch.object(
        storage,
        "_fetch",
        new=AsyncMock(
            return_value=[
                {"id": "xx_foo"},
                {"id": "foo"},
                {"id": "bar foo"},
                {"id": "foobar"},
            ]
        ),
    ):
        result = await storage.search_labels("foo", limit=3)

    assert result == ["foo", "foobar", "xx_foo"]


# ---------------------------------------------------------------------------
# self-loop degree consistency
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_self_loop_degree_consistency():
    """node_degree SQL must count self-loops like NetworkX degree: two."""
    storage = make_storage()
    fetchval = AsyncMock(return_value=2)

    with patch.object(storage, "_fetchval", new=fetchval):
        assert await storage.node_degree("A") == 2

    sql, workspace, namespace, node_id = fetchval.call_args.args
    assert workspace == "test"
    assert namespace == GRAPH_NAMESPACE
    assert node_id == "A"
    assert "CASE WHEN src_id = $3 THEN 1 ELSE 0 END" in sql
    assert "CASE WHEN tgt_id = $3 THEN 1 ELSE 0 END" in sql


@pytest.mark.asyncio
async def test_get_popular_labels_counts_self_loop_twice():
    storage = make_storage()
    fetch = AsyncMock(return_value=[{"id": "A", "degree": 2}])

    with patch.object(storage, "_fetch", new=fetch):
        labels = await storage.get_popular_labels(limit=5)

    sql, workspace, namespace = fetch.call_args.args
    assert workspace == "test"
    assert namespace == GRAPH_NAMESPACE
    assert labels == ["A"]
    assert "src_id <> tgt_id" not in sql
    assert "ORDER BY" not in sql


@pytest.mark.asyncio
async def test_get_popular_labels_includes_isolated_nodes():
    """Isolated (degree 0) nodes must rank too, matching NetworkX which counts
    every node via dict(graph.degree()). Ranking from the edge table alone
    would silently drop them."""
    storage = make_storage()
    fetch = AsyncMock(
        return_value=[{"id": "hub", "degree": 3}, {"id": "iso", "degree": 0}]
    )

    with patch.object(storage, "_fetch", new=fetch):
        labels = await storage.get_popular_labels(limit=5)

    sql, *_ = fetch.call_args.args
    assert "lightrag_graph_nodes" in sql  # ranks from node table, not edges alone
    assert "LEFT JOIN" in sql
    assert labels == ["hub", "iso"]  # degree-0 node retained, ranked last


def test_search_score_bonus_only_on_contains_branch():
    """_search_score must mirror NetworkXStorage.search_labels: the
    word-boundary +50 bonus applies ONLY to the contains branch, never to
    exact or prefix matches."""
    score = make_storage()._search_score
    assert score("foo", "foo") == 1000  # exact
    assert score("foobar", "foo") == 500  # prefix, no bonus
    # Regression: prefix match that ALSO contains a boundary occurrence must
    # stay 500 (the bonus previously leaked into the prefix branch -> 550).
    assert score("foo foo", "foo") == 500
    # Contains branch keeps the boundary bonus.
    assert score("x_foo", "foo") == 100 - len("x_foo") + 50
    assert score("xxfoo", "foo") == 100 - len("xxfoo")


# ---------------------------------------------------------------------------
# legacy edge normalization — idempotency guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_normalize_legacy_edges_skips_when_already_canonical():
    """Once all edges are canonical (no src_id > tgt_id row), the migration must
    short-circuit on a cheap EXISTS guard without scanning/regrouping the whole
    edge table on every initialize(). The guard compares with COLLATE "C" to
    match the Python min/max canonical order."""
    storage = make_storage()
    fetchval = AsyncMock(return_value=False)
    fetch = AsyncMock()

    with (
        patch.object(storage, "_fetchval", new=fetchval),
        patch.object(storage, "_fetch", new=fetch),
    ):
        await storage._normalize_legacy_edges()

    guard_sql, *_ = fetchval.call_args.args
    assert 'src_id COLLATE "C" > tgt_id COLLATE "C"' in guard_sql
    fetch.assert_not_called()  # no full scan / Python regroup on the fast path


@pytest.mark.asyncio
async def test_normalize_legacy_edges_scans_when_denormalized_present():
    """When a non-canonical row exists, the migration proceeds to the full
    scan-and-rewrite path."""
    storage = make_storage()
    fetchval = AsyncMock(return_value=True)
    run_retry = AsyncMock()
    storage.db._run_with_retry = run_retry

    with patch.object(storage, "_fetchval", new=fetchval):
        await storage._normalize_legacy_edges()

    # guard True -> migration runs inside a single advisory-locked transaction
    run_retry.assert_called_once()


# ---------------------------------------------------------------------------
# remove_edges — NULL guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_remove_edges_raises_on_none_id():
    storage = make_storage()
    with pytest.raises((ValueError, TypeError)):
        await storage.remove_edges([(None, "B")])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# get_all_edges — key contract (source/target, not src_id/tgt_id)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_all_edges_returns_source_target_keys():
    """get_all_edges must emit 'source'/'target' keys to match BaseGraphStorage
    contract consumed by storage_migrations.py."""
    storage = make_storage()
    with patch.object(
        storage,
        "_fetch",
        new=AsyncMock(
            return_value=[
                {
                    "src_id": "A",
                    "tgt_id": "B",
                    "properties": json.dumps(
                        {"weight": 1.0, "source": "wrong", "target": "wrong"}
                    ),
                }
            ]
        ),
    ):
        edges = await storage.get_all_edges()

    assert len(edges) == 1
    assert "source" in edges[0]
    assert "target" in edges[0]
    assert edges[0]["source"] == "A"
    assert edges[0]["target"] == "B"
    assert edges[0]["weight"] == 1.0
    assert "src_id" not in edges[0]
    assert "tgt_id" not in edges[0]
