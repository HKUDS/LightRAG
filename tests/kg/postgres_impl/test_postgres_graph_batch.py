"""Unit tests for PGGraphStorage chunk-level transaction batch paths.

Covers the PR-2 chunk-level fallback: ``upsert_nodes_batch`` /
``upsert_edges_batch`` group per-row Cypher into payload/record-bounded chunks
run in a single transaction; ``remove_nodes`` runs all chunks in ONE
transaction (all-or-nothing); ``remove_edges`` runs one transaction per chunk.
"""

import pytest
from unittest.mock import AsyncMock

from lightrag.kg.postgres_impl import PGGraphStorage


# ---------------------------------------------------------------------------
# Capture harness
# ---------------------------------------------------------------------------


class _Capture:
    def __init__(self) -> None:
        self.calls: list[dict] = []  # every connection.execute(sql, *args)
        self.tx_count = 0  # connection.transaction() opens
        self.run_count = 0  # _run_with_retry invocations (= chunks issued)


class _FakeTransaction:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeConnection:
    def __init__(self, capture: _Capture):
        self._capture = capture

    def transaction(self):
        self._capture.tx_count += 1
        return _FakeTransaction()

    async def execute(self, sql, *args):
        self._capture.calls.append({"sql": sql, "args": args})
        return ""


def make_graph_storage(
    *,
    max_upsert_records: int | None = None,
    max_upsert_payload: int | None = None,
    max_delete_records: int | None = None,
) -> tuple[PGGraphStorage, _Capture]:
    storage = PGGraphStorage.__new__(PGGraphStorage)
    storage.workspace = "test_ws"
    storage.namespace = "test_graph"
    storage.graph_name = "test_graph"
    storage.__post_init__()  # resolves the chunk-level batch limits
    if max_upsert_records is not None:
        storage._max_upsert_records_per_batch = max_upsert_records
    if max_upsert_payload is not None:
        storage._max_upsert_payload_bytes = max_upsert_payload
    if max_delete_records is not None:
        storage._max_delete_records_per_batch = max_delete_records

    capture = _Capture()
    conn = _FakeConnection(capture)

    async def fake_run_with_retry(operation, **_kwargs):
        capture.run_count += 1
        return await operation(conn)

    storage.db = AsyncMock()
    storage.db._run_with_retry = AsyncMock(side_effect=fake_run_with_retry)
    return storage, capture


def _sql_count(capture: _Capture, needle: str) -> int:
    return sum(1 for c in capture.calls if needle in c["sql"])


# ---------------------------------------------------------------------------
# upsert_nodes_batch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_nodes_batch_splits_into_chunk_transactions():
    storage, cap = make_graph_storage(max_upsert_records=2)
    nodes = [(f"n{i}", {"entity_id": f"n{i}", "v": str(i)}) for i in range(5)]

    await storage.upsert_nodes_batch(nodes)

    # 5 nodes / cap 2 => 3 chunks, each its own _run_with_retry + transaction.
    assert cap.run_count == 3
    assert cap.tx_count == 3
    # One MERGE per node.
    assert _sql_count(cap, "MERGE (n:base") == 5


@pytest.mark.asyncio
async def test_upsert_nodes_batch_dedupes_last_write_wins():
    storage, cap = make_graph_storage()
    await storage.upsert_nodes_batch(
        [
            ("A", {"entity_id": "A", "weight": "first"}),
            ("A", {"entity_id": "A", "weight": "second"}),
        ]
    )

    merge_calls = [c for c in cap.calls if "MERGE (n:base" in c["sql"]]
    assert len(merge_calls) == 1
    assert '"second"' in merge_calls[0]["sql"]
    assert '"first"' not in merge_calls[0]["sql"]


@pytest.mark.asyncio
async def test_upsert_nodes_batch_splits_by_payload_bytes():
    # Disable the record cap, force a small byte budget so each big node lands
    # in its own chunk.
    storage, cap = make_graph_storage(max_upsert_records=0, max_upsert_payload=200)
    nodes = [(f"n{i}", {"entity_id": f"n{i}", "blob": "X" * 150}) for i in range(3)]

    await storage.upsert_nodes_batch(nodes)

    assert cap.run_count == 3


@pytest.mark.asyncio
async def test_upsert_nodes_batch_missing_entity_id_raises_before_tx():
    storage, cap = make_graph_storage()
    with pytest.raises(ValueError, match="entity_id"):
        await storage.upsert_nodes_batch([("A", {"foo": "bar"})])
    # The builder runs before the transaction opens, so nothing was issued.
    assert cap.run_count == 0


@pytest.mark.asyncio
async def test_upsert_nodes_batch_empty_noop():
    storage, cap = make_graph_storage()
    await storage.upsert_nodes_batch([])
    assert cap.run_count == 0


# ---------------------------------------------------------------------------
# upsert_edges_batch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_edges_batch_splits_into_chunk_transactions():
    storage, cap = make_graph_storage(max_upsert_records=2)
    edges = [(f"s{i}", f"t{i}", {"w": str(i)}) for i in range(5)]

    await storage.upsert_edges_batch(edges)

    # 5 edges / cap 2 => 3 chunks; each chunk = one transaction.
    assert cap.run_count == 3
    assert cap.tx_count == 3
    # Per edge: one advisory lock + one CREATE cypher.
    assert _sql_count(cap, "pg_advisory_xact_lock") == 5
    assert _sql_count(cap, "CREATE (source)-[r:DIRECTED") == 5


@pytest.mark.asyncio
async def test_upsert_edges_batch_lock_precedes_cypher_per_edge():
    storage, cap = make_graph_storage()
    await storage.upsert_edges_batch([("A", "B", {"w": "1"})])

    # Order within the chunk: lock, then cypher.
    assert "pg_advisory_xact_lock" in cap.calls[0]["sql"]
    assert "CREATE (source)-[r:DIRECTED" in cap.calls[1]["sql"]


@pytest.mark.asyncio
async def test_upsert_edges_batch_empty_noop():
    storage, cap = make_graph_storage()
    await storage.upsert_edges_batch([])
    assert cap.run_count == 0


# ---------------------------------------------------------------------------
# remove_nodes — all chunks in ONE transaction (all-or-nothing)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_remove_nodes_chunks_share_one_transaction():
    storage, cap = make_graph_storage(max_delete_records=2)
    await storage.remove_nodes([f"n{i}" for i in range(5)])

    # One transaction wraps all chunks (preserves single-statement atomicity).
    assert cap.run_count == 1
    assert cap.tx_count == 1
    # 5 ids / cap 2 => 3 bounded IN [...] DETACH DELETE statements.
    assert _sql_count(cap, "DETACH DELETE n") == 3


@pytest.mark.asyncio
async def test_remove_nodes_empty_noop():
    storage, cap = make_graph_storage()
    await storage.remove_nodes([])
    assert cap.run_count == 0


# ---------------------------------------------------------------------------
# remove_edges — one transaction per chunk
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_remove_edges_one_transaction_per_chunk():
    storage, cap = make_graph_storage(max_delete_records=2)
    edges = [(f"s{i}", f"t{i}") for i in range(5)]

    await storage.remove_edges(edges)

    # 5 edges / cap 2 => 3 chunks, each its own transaction.
    assert cap.run_count == 3
    assert cap.tx_count == 3
    # One DELETE r statement per edge.
    assert _sql_count(cap, "DELETE r") == 5


@pytest.mark.asyncio
async def test_remove_edges_empty_noop():
    storage, cap = make_graph_storage()
    await storage.remove_edges([])
    assert cap.run_count == 0
