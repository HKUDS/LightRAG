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
        # Forwarded verbatim in every case, exactly like the sibling PG storages,
        # so the process-wide pool signature can never diverge. An unspecified
        # backend stays None: ClientManager maps that to enable_vector=False, so a
        # bare global_config does not demand pgvector and this backend needs no
        # sentinel name to stay installable on stock PostgreSQL. Bare configs are
        # real — tests/kg/test_graph_storage.py, the cross-backend contract suite,
        # is one, and CI runs it against a plain postgres image.
        ({}, None),
        ({"vector_storage": None}, None),
        ({"vector_storage": "NanoVectorDBStorage"}, "NanoVectorDBStorage"),
        ({"vector_storage": "PGVectorStorage"}, "PGVectorStorage"),
    ],
)
@pytest.mark.asyncio
async def test_initialize_forwards_configured_vector_storage_verbatim(
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


def test_ddl_gates_orphan_sweep_behind_missing_foreign_keys():
    """The table-wide orphan sweep must not run once both FKs already exist.

    It is an anti-join over the WHOLE edge table (all workspaces — the FK is a
    global composite constraint), it runs inside the transaction holding
    pg_advisory_xact_lock, and with the FKs enforced it can never match a row.
    As a top-level statement every process paid it, serialized, on every start.
    """
    from lightrag.kg.pgtable_impl import _DDL

    # The sweep must live inside the *second* DO block (the FK creator), after
    # its both-present early return and before the first ADD CONSTRAINT.
    fk_block = _DDL.split("DO $$", 2)[2]
    early_return = fk_block.index("RETURN;")
    sweep = fk_block.index("DELETE FROM lightrag_graph_edges e")
    add_src_fk = fk_block.index("ADD CONSTRAINT fk_lightrag_graph_edges_src")
    assert early_return < sweep < add_src_fk, (
        "orphan sweep must sit between the early return and ADD CONSTRAINT"
    )
    assert "IF has_src_fk AND has_tgt_fk THEN" in fk_block
    # ...and must not additionally exist as an ungated top-level statement.
    assert _DDL.count("DELETE FROM lightrag_graph_edges e") == 1


def test_normalize_guard_predicate_matches_the_partial_index():
    """The guard and its partial index must share one predicate, verbatim.

    The planner can only serve the guard from
    idx_lightrag_graph_edges_noncanonical if it can prove the query predicate
    implies the index predicate. Any drift between the two spellings silently
    turns a ~0.04 ms index-only probe back into a full scan of the edge table on
    every process start, with no error to notice.
    """
    import inspect

    from lightrag.kg.pgtable_impl import _DDL, PGTableGraphStorage

    predicate = 'src_id COLLATE "C" > tgt_id COLLATE "C"'

    assert "idx_lightrag_graph_edges_noncanonical" in _DDL
    assert predicate in _DDL
    guard_src = inspect.getsource(PGTableGraphStorage._normalize_legacy_edges)
    assert predicate in guard_src


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


@pytest.mark.asyncio
async def test_upsert_nodes_batch_raises_without_entity_id():
    """upsert_nodes_batch applies the same entity_id requirement as upsert_node;
    one malformed entry in the batch rejects the whole call."""
    storage = make_storage()
    with pytest.raises(ValueError, match="entity_id"):
        await storage.upsert_nodes_batch(
            [("n1", {"entity_id": "n1"}), ("n2", {"name": "missing id"})]
        )


@pytest.mark.asyncio
async def test_upsert_nodes_batch_overwrites_mismatched_entity_id():
    """A mismatched entity_id is forced to node_id in the batch path too."""
    storage = make_storage()
    execute = AsyncMock()
    with patch.object(storage, "_execute", new=execute):
        await storage.upsert_nodes_batch([("n1", {"entity_id": "wrong", "k": "v"})])

    *_, sorted_ids, props = execute.call_args.args
    assert sorted_ids == ["n1"]
    assert json.loads(props[0])["entity_id"] == "n1"


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
    # path under test (_is_transient_error) matches by isinstance against
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
# _fetch* — read paths retry the same query-level transient errors as writes
# (under SERIALIZABLE / REPEATABLE READ a read racing ingestion can raise one).
# See note on asyncpg import in test_execute_retries_transient_write_error.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetchrow_retries_transient_error():
    asyncpg = pytest.importorskip("asyncpg")
    storage = make_storage()
    storage.db.query = AsyncMock(
        side_effect=[
            asyncpg.exceptions.SerializationError("serialize"),
            {"id": "n"},
        ]
    )
    row = await storage._fetchrow("SELECT ...", "a")
    assert row == {"id": "n"}
    assert storage.db.query.call_count == 2


@pytest.mark.asyncio
async def test_fetch_retries_transient_error():
    asyncpg = pytest.importorskip("asyncpg")
    storage = make_storage()
    storage.db.query = AsyncMock(
        side_effect=[
            asyncpg.exceptions.DeadlockDetectedError("deadlock"),
            [{"id": "n"}],
        ]
    )
    rows = await storage._fetch("SELECT ...", "a")
    assert rows == [{"id": "n"}]
    assert storage.db.query.call_count == 2


@pytest.mark.asyncio
async def test_fetch_does_not_retry_non_transient_error():
    storage = make_storage()
    storage.db.query = AsyncMock(side_effect=ValueError("boom"))
    with pytest.raises(ValueError):
        await storage._fetch("SELECT ...", "a")
    assert storage.db.query.call_count == 1


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
    """max_nodes=None must fall back to global_config['max_graph_nodes'] as the
    budget. Return MORE rows than the cap so the fallback is actually exercised:
    mocking _fetch -> [] would hit the empty-result early-exit before the cap is
    ever read, passing even if the None -> global_config fallback were broken."""
    storage = make_storage(global_config={"max_graph_nodes": 42})
    rows = [
        {"id": str(i), "properties": json.dumps({"entity_id": str(i)}), "degree": 0}
        for i in range(50)
    ]
    fetch = AsyncMock(side_effect=[rows, []])  # 1st call: nodes, 2nd: edges
    with (
        patch.object(storage, "_fetch", new=fetch),
        patch.object(storage, "node_degrees_batch", new=AsyncMock(return_value={})),
    ):
        kg = await storage.get_knowledge_graph("*", max_nodes=None)

    # The node query asked for cap + 1 (42 + 1) rows -> the None -> global_config
    # fallback supplied the budget...
    assert fetch.call_args_list[0].args[3] == 43
    # ...and the result is truncated to the cap, with is_truncated set.
    assert len(kg.nodes) == 42
    assert kg.is_truncated is True


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
# _bfs_frontier — per-hop work bounded in SQL
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bfs_hop_ranks_and_caps_in_sql_without_any_degree_roundtrip():
    """Each hop must rank + cap server-side and cost ONE query.

    Previously a hop fetched every neighbour (full JSONB properties included),
    then issued a second node_degrees_batch round trip over that whole
    neighbourhood, then ranked and truncated in Python. On a hub that meant
    shipping the entire neighbour set just to discard all but `node_budget` of
    it.

    node_degrees_batch must now not be awaited at all. It previously ran once
    more, after the loop, to fill in the seed's own degree "for the ordering
    tie-break" — but the seed's degree cannot affect that ordering.
    get_knowledge_graph sorts on the tuple

        (r["id"] != node_label, depth, -degrees[id], id)

    and the seed wins on both leading elements independently: the first is False
    only for the seed, and the seed's depth is 0 while every other collected row
    is at depth >= 1. The comparison therefore short-circuits before the degree
    element. That round trip also cost an O(deg(seed)) edge enumeration inside
    node_degrees_batch — a full hub scan for a value no caller reads.
    """
    storage = make_storage()

    seed_row = {"id": "seed", "properties": json.dumps({"entity_id": "seed"})}
    # Two levels, each already degree-ranked and capped by the DB.
    hop_rows = [
        [
            {"id": "a", "properties": json.dumps({"entity_id": "a"}), "degree": 9},
            {"id": "b", "properties": json.dumps({"entity_id": "b"}), "degree": 4},
        ],
        [{"id": "c", "properties": json.dumps({"entity_id": "c"}), "degree": 1}],
        [],
    ]
    fetch = AsyncMock(side_effect=hop_rows)
    degrees_batch = AsyncMock(return_value={"seed": 7})

    with (
        patch.object(storage, "_fetchrow", new=AsyncMock(return_value=seed_row)),
        patch.object(storage, "_fetch", new=fetch),
        patch.object(storage, "node_degrees_batch", new=degrees_batch),
    ):
        rows, degrees = await storage._bfs_frontier("seed", max_depth=3, node_budget=10)

    # Never — not once per level, and not once more for the seed.
    degrees_batch.assert_not_awaited()
    # Degrees come from the hop query itself, and carry no seed entry.
    assert degrees == {"a": 9, "b": 4, "c": 1}
    assert [r["id"] for r in rows] == ["seed", "a", "b", "c"]
    assert [r["depth"] for r in rows] == [0, 1, 1, 2]

    # Every hop must push the unvisited filter, the degree ranking and the
    # remaining-budget cap into SQL.
    first_sql = fetch.await_args_list[0].args[0]
    assert "ORDER BY COALESCE(d.degree, 0) DESC" in first_sql
    assert "LIMIT $5" in first_sql
    assert "NOT EXISTS" in first_sql

    # The cap is the remaining budget + 1, so the caller still sees the one
    # overfetched row that distinguishes "exactly full" from "truncated".
    # collected grows 1 -> 3 -> 4, so the cap is 10-1+1, 10-3+1, 10-4+1.
    caps = [call.args[5] for call in fetch.await_args_list]
    assert caps == [10, 8, 7]
    # The visited set is passed so the DB can anti-join it.
    visited_per_hop = [call.args[4] for call in fetch.await_args_list]
    assert visited_per_hop[0] == ["seed"]
    assert visited_per_hop[1] == ["seed", "a", "b"]


@pytest.mark.asyncio
async def test_bfs_hop_cap_never_exceeds_remaining_budget():
    """A hop must not admit more than budget+1 nodes even if SQL returned more."""
    storage = make_storage()
    seed_row = {"id": "seed", "properties": json.dumps({"entity_id": "seed"})}
    fetch = AsyncMock(
        return_value=[
            {"id": "a", "properties": json.dumps({"entity_id": "a"}), "degree": 3},
        ]
    )

    with (
        patch.object(storage, "_fetchrow", new=AsyncMock(return_value=seed_row)),
        patch.object(storage, "_fetch", new=fetch),
    ):
        await storage._bfs_frontier("seed", max_depth=5, node_budget=1)

    # budget 1, seed already collected -> cap 1, and the loop stops after it.
    assert fetch.await_args_list[0].args[5] == 1
    assert fetch.await_count == 1


def _bfs_mocks(hop_rows):
    """Build mocks for the three DB entry points _bfs_frontier can reach.

    Returns (fetchrow, fetch, degrees_batch) so a test can count round trips.
    """
    fetchrow = AsyncMock(
        return_value={"id": "seed", "properties": json.dumps({"entity_id": "seed"})}
    )
    fetch = AsyncMock(side_effect=hop_rows)
    degrees_batch = AsyncMock(return_value={"seed": 99})
    return fetchrow, fetch, degrees_batch


@pytest.mark.asyncio
async def test_bfs_costs_one_round_trip_per_hop_plus_the_seed_lookup():
    """Total DB round trips must be max_depth + 1, not max_depth + 2.

    One _fetchrow for the seed row, one _fetch per hop, and nothing else. The
    trailing node_degrees_batch call for the seed's own degree used to make this
    max_depth + 2 for every single traversal.
    """
    storage = make_storage()
    hop_rows = [
        [{"id": "a", "properties": json.dumps({"entity_id": "a"}), "degree": 3}],
        [{"id": "b", "properties": json.dumps({"entity_id": "b"}), "degree": 2}],
        [{"id": "c", "properties": json.dumps({"entity_id": "c"}), "degree": 1}],
    ]
    fetchrow, fetch, degrees_batch = _bfs_mocks(hop_rows)

    with (
        patch.object(storage, "_fetchrow", new=fetchrow),
        patch.object(storage, "_fetch", new=fetch),
        patch.object(storage, "node_degrees_batch", new=degrees_batch),
    ):
        await storage._bfs_frontier("seed", max_depth=3, node_budget=100)

    assert fetchrow.await_count == 1
    assert fetch.await_count == 3
    degrees_batch.assert_not_awaited()
    total = fetchrow.await_count + fetch.await_count + degrees_batch.await_count
    assert total == 3 + 1, f"expected max_depth + 1 round trips, got {total}"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "max_depth,hop_rows,expected_fetches",
    [
        # No hop ever runs: the while condition fails before the first query.
        pytest.param(0, [], 0, id="max_depth_0"),
        # A seed whose neighbours are all already visited (or has none) — the
        # hop query runs but returns zero rows, so no candidate row exists.
        pytest.param(3, [[]], 1, id="isolated_seed"),
    ],
)
async def test_bfs_degenerate_seeds_issue_no_degree_round_trip(
    max_depth, hop_rows, expected_fetches
):
    """Degenerate seeds must not fall back to a degree round trip either.

    These are the two cases where the seed's degree cannot be carried on a hop
    row: with max_depth=0 no hop query is issued at all, and an isolated seed's
    hop returns no rows to attach it to. A self-loop-only seed lands in the
    second case while having a non-zero degree (self-loops count twice), so any
    scheme that recovers the seed degree from hop output would be wrong here.
    Not computing it at all is well-defined in both.
    """
    storage = make_storage()
    fetchrow, fetch, degrees_batch = _bfs_mocks(hop_rows)

    with (
        patch.object(storage, "_fetchrow", new=fetchrow),
        patch.object(storage, "_fetch", new=fetch),
        patch.object(storage, "node_degrees_batch", new=degrees_batch),
    ):
        rows, degrees = await storage._bfs_frontier(
            "seed", max_depth=max_depth, node_budget=10
        )

    assert fetchrow.await_count == 1
    assert fetch.await_count == expected_fetches
    degrees_batch.assert_not_awaited()
    # The seed is still returned as a row; only its degree entry is absent.
    assert [r["id"] for r in rows] == ["seed"]
    assert degrees == {}


# ---------------------------------------------------------------------------
# search_labels — literal SQL pattern semantics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_labels_strips_and_ranks_and_limits_in_sql():
    storage = make_storage()
    fetch = AsyncMock(return_value=[])

    with patch.object(storage, "_fetch", new=fetch):
        assert await storage.search_labels("   ") == []
        await storage.search_labels(" Foo ", limit=7)

    sql, workspace, namespace, exact, prefix, space_bnd, us_bnd, contains, limit = (
        fetch.call_args.args
    )
    assert workspace == "test"
    assert namespace == GRAPH_NAMESPACE
    # $3 is compared with '=', so it must be the raw lowercased query with no
    # LIKE escaping applied.
    assert exact == "foo"
    assert prefix == "foo%"
    assert space_bnd == "% foo%"
    # '_' is a LIKE wildcard: the "_query" word-boundary probe must escape it.
    assert us_bnd == r"%\_foo%"
    assert contains == "%foo%"
    # Ranking and truncation must happen server-side, not after transferring the
    # whole match set (that was the point of the change).
    assert limit == 7
    assert "ORDER BY score DESC" in sql
    assert "LIMIT $8" in sql
    # E'' escape literal: unambiguous one-char escape regardless of
    # standard_conforming_strings (see search_labels).
    assert "ESCAPE E'\\\\'" in sql


@pytest.mark.asyncio
async def test_search_labels_escapes_like_wildcards():
    storage = make_storage()
    fetch = AsyncMock(return_value=[])

    with patch.object(storage, "_fetch", new=fetch):
        await storage.search_labels(r"a_%\b")

    args = fetch.call_args.args
    namespace, exact, prefix, space_bnd, us_bnd, contains = args[2:8]
    assert namespace == GRAPH_NAMESPACE
    assert exact == r"a_%\b"
    assert prefix == r"a\_\%\\b%"
    assert space_bnd == r"% a\_\%\\b%"
    assert us_bnd == r"%\_a\_\%\\b%"
    assert contains == r"%a\_\%\\b%"


@pytest.mark.asyncio
async def test_search_labels_preserves_sql_row_order():
    """Ordering is the DB's job now; the method must not re-sort or re-slice.

    Rows are handed back in the order SQL returned them (the equivalence of that
    SQL ordering to _search_score is proven against a live PostgreSQL by
    tests/kg/pgtable_impl/test_pgtable_smoke.py).
    """
    storage = make_storage()
    # Deliberately NOT in _search_score order: a lingering Python sort would
    # reorder these, and a lingering [:limit] would drop the last one.
    sql_order = [{"id": "zz"}, {"id": "aa"}, {"id": "mm"}]
    with patch.object(storage, "_fetch", new=AsyncMock(return_value=sql_order)):
        result = await storage.search_labels("a", limit=1)

    assert result == ["zz", "aa", "mm"]


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

    sql, workspace, namespace, limit_arg = fetch.call_args.args
    assert workspace == "test"
    assert namespace == GRAPH_NAMESPACE
    assert limit_arg == 5
    assert labels == ["A"]
    assert "src_id <> tgt_id" not in sql  # self-loop counted twice (no guard)
    # Ranking + truncation are pushed into SQL (ORDER BY degree DESC, LIMIT) so
    # large graphs do not transfer every node just to slice in Python.
    assert "ORDER BY degree DESC" in sql
    assert "LIMIT" in sql


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


# ---------------------------------------------------------------------------
# flush-time batching — POSTGRES_UPSERT_* / POSTGRES_DELETE_* caps
# ---------------------------------------------------------------------------


class _FakeConn:
    """Records execute() calls and the transaction nesting they ran under."""

    def __init__(self):
        self.calls: list[tuple[str, tuple, int]] = []
        self.depth = 0
        self.max_depth = 0

    def transaction(self):
        outer = self

        class _Txn:
            async def __aenter__(self):
                outer.depth += 1
                outer.max_depth = max(outer.max_depth, outer.depth)

            async def __aexit__(self, *exc):
                outer.depth -= 1
                return False

        return _Txn()

    async def execute(self, sql, *args):
        self.calls.append((sql, args, self.depth))


def _wire_delete_capture(storage) -> _FakeConn:
    """Record every statement a delete path emits, whichever route it takes.

    Both the transactional `_run_with_retry` route and a plain `_execute` land in
    the same recorder, so the assertions are about observable behaviour — how many
    statements, and whether they ran inside a transaction — rather than about which
    private helper happens to exist. A single-statement implementation shows up as
    one call at depth 0.
    """
    conn = _FakeConn()

    async def _run(fn):
        return await fn(conn)

    async def _execute(sql, *args):
        await conn.execute(sql, *args)

    storage.db._run_with_retry = _run
    storage._execute = _execute
    return conn


@pytest.mark.asyncio
async def test_upsert_nodes_batch_splits_by_record_cap(monkeypatch):
    """A large node batch must not go out as one unbounded statement."""
    monkeypatch.setenv("POSTGRES_UPSERT_MAX_RECORDS_PER_BATCH", "3")
    monkeypatch.setenv("POSTGRES_UPSERT_MAX_PAYLOAD_BYTES", "16777216")
    storage = make_storage()
    nodes = [(f"n{i:02d}", {"entity_id": f"n{i:02d}"}) for i in range(7)]

    with patch.object(storage, "_execute", new=AsyncMock()) as execute:
        await storage.upsert_nodes_batch(nodes)

    assert execute.await_count == 3, "7 nodes at cap 3 must be 3 statements"
    per_chunk_ids = [call.args[3] for call in execute.await_args_list]
    assert [len(ids) for ids in per_chunk_ids] == [3, 3, 1]
    # Every node written exactly once, in id order, with parallel props arrays.
    assert [nid for ids in per_chunk_ids for nid in ids] == [n[0] for n in nodes]
    for call in execute.await_args_list:
        assert len(call.args[3]) == len(call.args[4])


@pytest.mark.asyncio
async def test_upsert_nodes_batch_splits_by_payload_bytes(monkeypatch):
    """The byte budget is the primary limiter, independent of the record cap."""
    monkeypatch.setenv("POSTGRES_UPSERT_MAX_RECORDS_PER_BATCH", "1000")
    monkeypatch.setenv("POSTGRES_UPSERT_MAX_PAYLOAD_BYTES", "400")
    storage = make_storage()
    # ~300 bytes of properties each, so only one fits per chunk.
    nodes = [
        (f"n{i}", {"entity_id": f"n{i}", "description": "x" * 300}) for i in range(4)
    ]

    with patch.object(storage, "_execute", new=AsyncMock()) as execute:
        await storage.upsert_nodes_batch(nodes)

    assert execute.await_count == 4, "byte cap must split even under the record cap"


@pytest.mark.asyncio
async def test_upsert_edges_batch_chunk_carries_its_own_endpoints(monkeypatch):
    """Each edge chunk must create the endpoints for its OWN edges.

    Chunks are separate statements, so an edge whose endpoint node was only going
    to be created by a different chunk would violate the FK.
    """
    monkeypatch.setenv("POSTGRES_UPSERT_MAX_RECORDS_PER_BATCH", "2")
    monkeypatch.setenv("POSTGRES_UPSERT_MAX_PAYLOAD_BYTES", "16777216")
    storage = make_storage()
    edges = [(f"s{i}", f"t{i}", {"weight": 1.0}) for i in range(5)]

    with patch.object(storage, "_execute", new=AsyncMock()) as execute:
        await storage.upsert_edges_batch(edges)

    assert execute.await_count == 3
    for call in execute.await_args_list:
        srcs, tgts, props, endpoint_ids = call.args[3:7]
        assert len(srcs) == len(tgts) == len(props)
        # endpoint_ids is exactly this chunk's endpoint set — no more, no less.
        assert set(endpoint_ids) == set(srcs) | set(tgts)


def _edge_statement_bytes(call) -> int:
    """UTF-8 bytes an edge-upsert statement actually binds for its chunk.

    All four text arrays, including the $6 endpoint list — the point being that
    $6 is real bound data, not free, so the chunk splitter has to have paid for
    it.
    """
    srcs, tgts, props, endpoint_ids = call.args[3:7]
    return sum(
        len(value.encode("utf-8"))
        for array in (srcs, tgts, props, endpoint_ids)
        for value in array
    )


@pytest.mark.asyncio
async def test_upsert_edges_batch_payload_cap_accounts_for_the_endpoint_array(
    monkeypatch,
):
    """No edge chunk may bind more bytes than POSTGRES_UPSERT_MAX_PAYLOAD_BYTES.

    The chunk splitter used to size an edge by src + tgt + properties, but the
    statement also sends the chunk's endpoint ids again in $6. With distinct
    endpoints and small properties — long entity names, thin edges, i.e. the
    normal shape — the id bytes went over the wire twice while being budgeted
    once, so a chunk that "fit" bound close to double the cap.

    Long ids and tiny properties here so the id bytes dominate and the
    under-count is the whole difference: at this cap the old accounting emitted a
    4-edge chunk binding 1660 bytes.
    """
    cap = 1000
    monkeypatch.setenv("POSTGRES_UPSERT_MAX_PAYLOAD_BYTES", str(cap))
    monkeypatch.setenv("POSTGRES_UPSERT_MAX_RECORDS_PER_BATCH", "1000")
    storage = make_storage()
    edges = [
        (f"s{i}" + "x" * 98, f"t{i}" + "y" * 98, {"weight": 1.0}) for i in range(6)
    ]

    with patch.object(storage, "_execute", new=AsyncMock()) as execute:
        await storage.upsert_edges_batch(edges)

    sizes = [_edge_statement_bytes(call) for call in execute.await_args_list]
    counts = [len(call.args[3]) for call in execute.await_args_list]
    for size, count in zip(sizes, counts):
        # A single item over budget is passed through by design (the server is the
        # final arbiter); a multi-item chunk over budget is an accounting bug.
        assert count == 1 or size <= cap, (
            f"chunk of {count} edges bound {size} bytes against a {cap}-byte cap"
        )
    # Guard against the assertion above passing vacuously through the
    # oversized-single-item exemption.
    assert max(counts) > 1, "expected multi-edge chunks at this cap"
    assert sum(counts) == len(edges)


@pytest.mark.asyncio
async def test_remove_nodes_chunks_inside_one_transaction(monkeypatch):
    """Delete chunking must stay all-or-nothing across chunks."""
    monkeypatch.setenv("POSTGRES_DELETE_MAX_RECORDS_PER_BATCH", "2")
    storage = make_storage()
    conn = _wire_delete_capture(storage)

    await storage.remove_nodes([f"n{i}" for i in range(5)])

    assert len(conn.calls) == 3, "5 ids at cap 2 must be 3 statements"
    assert conn.max_depth == 1, "all chunks must run in ONE transaction"
    for sql, args, depth in conn.calls:
        assert depth == 1, "every chunk must execute inside the transaction"
        assert args[0] == "test" and args[1] == GRAPH_NAMESPACE
    assert [len(args[2]) for _sql, args, _d in conn.calls] == [2, 2, 1]
    assert [nid for _s, args, _d in conn.calls for nid in args[2]] == [
        f"n{i}" for i in range(5)
    ]


@pytest.mark.asyncio
async def test_remove_edges_chunks_canonically_inside_one_transaction(monkeypatch):
    monkeypatch.setenv("POSTGRES_DELETE_MAX_RECORDS_PER_BATCH", "2")
    storage = make_storage()
    conn = _wire_delete_capture(storage)

    # Mixed orientation: canonicalization must survive chunking.
    await storage.remove_edges([("z", "a"), ("b", "y"), ("m", "c")])

    assert len(conn.calls) == 2
    assert conn.max_depth == 1
    pairs = [pair for _sql, args, _d in conn.calls for pair in zip(args[2], args[3])]
    assert pairs == [("a", "z"), ("b", "y"), ("c", "m")]


@pytest.mark.asyncio
async def test_non_positive_caps_disable_chunking(monkeypatch):
    """A non-positive cap means "one statement", matching the sibling storages."""
    monkeypatch.setenv("POSTGRES_UPSERT_MAX_RECORDS_PER_BATCH", "0")
    monkeypatch.setenv("POSTGRES_UPSERT_MAX_PAYLOAD_BYTES", "0")
    monkeypatch.setenv("POSTGRES_DELETE_MAX_RECORDS_PER_BATCH", "0")

    storage = make_storage()
    with patch.object(storage, "_execute", new=AsyncMock()) as execute:
        await storage.upsert_nodes_batch(
            [(f"n{i}", {"entity_id": f"n{i}"}) for i in range(50)]
        )
    assert execute.await_count == 1

    delete_storage = make_storage()
    conn = _wire_delete_capture(delete_storage)
    await delete_storage.remove_nodes([f"n{i}" for i in range(50)])
    assert len(conn.calls) == 1


@pytest.mark.asyncio
async def test_batch_limits_come_from_the_shared_pg_resolver(monkeypatch):
    """The caps must be the shared POSTGRES_* ones, not a private re-read.

    Guards against this backend growing its own env names or defaults and
    drifting from the rest of the PostgreSQL write paths.
    """
    monkeypatch.setenv("POSTGRES_UPSERT_MAX_PAYLOAD_BYTES", "12345")
    monkeypatch.setenv("POSTGRES_UPSERT_MAX_RECORDS_PER_BATCH", "67")
    monkeypatch.setenv("POSTGRES_DELETE_MAX_RECORDS_PER_BATCH", "89")

    from lightrag.kg.postgres_impl import _resolve_pg_batch_limits

    assert make_storage()._batch_limits() == _resolve_pg_batch_limits()
    assert make_storage()._batch_limits() == (12345, 67, 89)
