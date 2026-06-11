"""Unit tests for PGVectorStorage deferred-embedding contract.

PGVectorStorage now buffers upserts/deletes in process-local pending buffers
and embeds + persists only during ``index_done_callback()`` / ``finalize()``.
This mirrors OpenSearchVectorDBStorage and NanoVectorDBStorage.

These tests use the same ``MagicMock``-based DB stub as
``test_postgres_upsert.py``, plus a counting embedding function adapted from
``tests/kg/opensearch_impl/test_opensearch_storage.py``.
"""

import asyncio
import datetime

import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock

from lightrag.kg.postgres_impl import (
    PGVectorStorage,
    _PendingPGVectorDoc,
)
from lightrag.namespace import NameSpace
from lightrag.utils import EmbeddingFunc, compute_mdhash_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class CountingEmbed:
    """Embedding test double that records calls and can fail N times first."""

    def __init__(self, dim: int = 3, fail_times: int = 0):
        self.embedding_dim = dim
        self.max_token_size = 512
        self.model_name = "test_model"
        self.fail_times = fail_times
        self.call_count = 0
        self.batches: list[list[str]] = []

    async def __call__(self, texts, **kwargs):
        self.call_count += 1
        batch = list(texts)
        self.batches.append(batch)
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("embedding failed")
        return np.array(
            [[float(self.call_count), 0.0, 0.0] for _ in batch], dtype=np.float32
        )


def _make_storage(
    namespace: str = NameSpace.VECTOR_STORE_CHUNKS,
    embed: CountingEmbed | None = None,
    embedding_batch_num: int = 10,
    fail_run_with_retry: bool = False,
) -> PGVectorStorage:
    """Construct a PGVectorStorage with a stubbed DB and embedding func."""
    db = MagicMock()
    captured_executemany: list[tuple] = []
    captured_execute: list[tuple] = []
    retry_kwargs: list[dict] = []
    retry_call_count = {"n": 0}

    async def fake_run_with_retry(operation, **kwargs):
        retry_kwargs.append(kwargs)
        retry_call_count["n"] += 1
        if fail_run_with_retry:
            raise RuntimeError("simulated PG failure")
        mock_conn = AsyncMock()
        tx_cm = AsyncMock()
        tx_cm.__aenter__.return_value = None
        tx_cm.__aexit__.return_value = None
        mock_conn.transaction = MagicMock(return_value=tx_cm)
        await operation(mock_conn)
        for call in mock_conn.executemany.call_args_list:
            captured_executemany.append((call.args[0], call.args[1]))
        for call in mock_conn.execute.call_args_list:
            captured_execute.append((call.args[0], call.args[1:]))

    db._run_with_retry = AsyncMock(side_effect=fake_run_with_retry)

    # db.execute is used by delete_entity, delete_entity_relation, drop.
    db.execute = AsyncMock(return_value=None)
    db.query = AsyncMock(return_value=[])
    db.workspace = "test_ws"

    embedding = embed or CountingEmbed()
    embedding_func = EmbeddingFunc(
        embedding_dim=embedding.embedding_dim,
        func=embedding,
        model_name=embedding.model_name,
    )
    storage = PGVectorStorage(
        namespace=namespace,
        workspace="test_ws",
        global_config={
            "embedding_batch_num": embedding_batch_num,
            "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.5},
        },
        embedding_func=embedding_func,
    )
    storage.db = db
    storage._flush_lock = asyncio.Lock()
    storage._counting_embed = embedding
    storage._captured_executemany = captured_executemany
    storage._captured_execute = captured_execute
    storage._retry_kwargs = retry_kwargs
    storage._retry_call_count = retry_call_count
    return storage


def _chunk_data(**overrides):
    base = {
        "tokens": 1,
        "chunk_order_index": 0,
        "full_doc_id": "doc-1",
        "content": "alpha",
        "file_path": "/a.txt",
    }
    base.update(overrides)
    return base


def _entity_data(name: str = "Alice", **overrides):
    base = {
        "entity_name": name,
        "content": f"{name} content",
        "source_id": "chunk-1",
        "file_path": "/e.txt",
    }
    base.update(overrides)
    return base


def _relation_data(src: str = "Alice", tgt: str = "Bob", **overrides):
    base = {
        "src_id": src,
        "tgt_id": tgt,
        "content": f"{src}->{tgt}",
        "source_id": "chunk-1",
        "file_path": "/r.txt",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# 1. upsert() buffers only
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_buffers_without_embedding_or_db_call():
    storage = _make_storage()
    await storage.upsert({"c1": _chunk_data(content="alpha")})

    assert storage._counting_embed.call_count == 0
    assert storage._retry_call_count["n"] == 0
    assert "c1" in storage._pending_vector_docs
    pending = storage._pending_vector_docs["c1"]
    assert isinstance(pending, _PendingPGVectorDoc)
    assert pending.vector is None
    assert pending.item["__id__"] == "c1"
    assert pending.item["content"] == "alpha"


# ---------------------------------------------------------------------------
# 2. Deferred batching across many upsert() calls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_many_upserts_flush_in_one_executemany():
    storage = _make_storage(embedding_batch_num=3)
    for i in range(5):
        await storage.upsert({f"c{i}": _chunk_data(content=f"doc {i}")})

    assert storage._counting_embed.call_count == 0
    await storage.index_done_callback()

    # Embedding split only by embedding_batch_num (3 + 2).
    assert [len(b) for b in storage._counting_embed.batches] == [3, 2]
    # One executemany for 5 records (not one per upsert call).
    assert len(storage._captured_executemany) == 1
    sql, rows = storage._captured_executemany[0]
    assert len(rows) == 5
    assert "LIGHTRAG_VDB_CHUNKS" in sql


# ---------------------------------------------------------------------------
# 3. Same-id upsert overwrite
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_same_id_upsert_overwrites():
    storage = _make_storage()
    await storage.upsert({"x": _chunk_data(content="a")})
    await storage.upsert({"x": _chunk_data(content="b")})
    await storage.index_done_callback()

    rows = storage._captured_executemany[0][1]
    assert len(rows) == 1
    # The chunk tuple position 6 ($6) is content.
    assert rows[0][5] == "b"


# ---------------------------------------------------------------------------
# 4. Lazy vector cache: get_vectors_by_ids embeds, flush reuses
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lazy_vector_cache_reused_by_flush():
    storage = _make_storage()
    await storage.upsert({"c1": _chunk_data(content="alpha")})
    vecs = await storage.get_vectors_by_ids(["c1"])
    assert "c1" in vecs
    assert storage._counting_embed.call_count == 1

    await storage.index_done_callback()
    # Flush must not re-embed; total call count stays 1.
    assert storage._counting_embed.call_count == 1
    # The vector that landed in the executemany row equals what get_vectors_by_ids returned.
    rows = storage._captured_executemany[0][1]
    persisted_vec = rows[0][6]  # chunks tuple: $7 is content_vector
    assert list(np.asarray(persisted_vec, dtype=np.float32)) == list(
        np.asarray(vecs["c1"], dtype=np.float32)
    )


# ---------------------------------------------------------------------------
# 5. Upsert after lazy cache discards the cached vector
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_after_lazy_cache_discards_cached_vector():
    storage = _make_storage()
    await storage.upsert({"c1": _chunk_data(content="a")})
    await storage.get_vectors_by_ids(["c1"])  # embed call #1
    assert storage._pending_vector_docs["c1"].vector is not None

    await storage.upsert({"c1": _chunk_data(content="b")})  # discards cache
    assert storage._pending_vector_docs["c1"].vector is None

    await storage.index_done_callback()
    # Two embed calls total: one lazy, one for the new content during flush.
    assert storage._counting_embed.call_count == 2
    # And the persisted content is "b".
    rows = storage._captured_executemany[0][1]
    assert rows[0][5] == "b"


# ---------------------------------------------------------------------------
# 6. Embedding failure leaves buffers intact
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_embedding_failure_leaves_pending_for_retry():
    embed = CountingEmbed(fail_times=1)
    storage = _make_storage(embed=embed)
    await storage.upsert({"c1": _chunk_data(content="retry me")})

    with pytest.raises(RuntimeError, match="embedding failed"):
        await storage.index_done_callback()

    assert storage._retry_call_count["n"] == 0
    assert "c1" in storage._pending_vector_docs
    assert storage._pending_vector_docs["c1"].vector is None

    # Next flush succeeds; embed called twice total (one failure + one success).
    await storage.index_done_callback()
    assert embed.call_count == 2
    assert storage._pending_vector_docs == {}
    assert len(storage._captured_executemany) == 1


# ---------------------------------------------------------------------------
# 7. _run_with_retry failure leaves buffers + cached vectors intact
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_persistence_failure_keeps_buffers_and_cached_vectors():
    storage = _make_storage(fail_run_with_retry=True)
    await storage.upsert({"c1": _chunk_data(content="alpha")})

    with pytest.raises(RuntimeError, match="simulated PG failure"):
        await storage.index_done_callback()

    # Buffer intact, vector cached (so next flush won't re-embed).
    assert "c1" in storage._pending_vector_docs
    assert storage._pending_vector_docs["c1"].vector is not None
    embed_calls_before = storage._counting_embed.call_count

    # Repair the DB and flush again.
    storage.db._run_with_retry.side_effect = None
    storage.db._run_with_retry.return_value = None

    # We need to actually persist this time; re-attach a working side_effect.
    captured_em = storage._captured_executemany
    captured_ex = storage._captured_execute

    async def working_retry(operation, **kwargs):
        mock_conn = AsyncMock()
        tx_cm = AsyncMock()
        tx_cm.__aenter__.return_value = None
        tx_cm.__aexit__.return_value = None
        mock_conn.transaction = MagicMock(return_value=tx_cm)
        await operation(mock_conn)
        for call in mock_conn.executemany.call_args_list:
            captured_em.append((call.args[0], call.args[1]))
        for call in mock_conn.execute.call_args_list:
            captured_ex.append((call.args[0], call.args[1:]))

    storage.db._run_with_retry.side_effect = working_retry

    await storage.index_done_callback()
    # No re-embed thanks to the cached vector.
    assert storage._counting_embed.call_count == embed_calls_before
    assert storage._pending_vector_docs == {}
    assert len(captured_em) == 1


# ---------------------------------------------------------------------------
# 8. Delete cancels pending upsert
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_cancels_pending_upsert():
    storage = _make_storage()
    await storage.upsert({"c1": _chunk_data()})
    await storage.delete(["c1"])

    assert "c1" not in storage._pending_vector_docs
    assert "c1" in storage._pending_vector_deletes

    await storage.index_done_callback()
    # Only a delete went out, no upsert executemany.
    assert storage._captured_executemany == []
    assert len(storage._captured_execute) == 1
    sql, args = storage._captured_execute[0]
    assert "DELETE FROM" in sql
    assert args[0] == "test_ws"
    assert args[1] == ["c1"]


# ---------------------------------------------------------------------------
# 9. Upsert cancels pending delete
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_cancels_pending_delete():
    storage = _make_storage()
    await storage.delete(["c1"])
    await storage.upsert({"c1": _chunk_data(content="new")})

    assert "c1" in storage._pending_vector_docs
    assert "c1" not in storage._pending_vector_deletes

    await storage.index_done_callback()
    assert len(storage._captured_executemany) == 1
    # And no DELETE in the same flush.
    assert storage._captured_execute == []


# ---------------------------------------------------------------------------
# 10. delete_entity prunes pending docs and runs SQL predicate under lock
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_entity_prunes_pending_and_runs_sql():
    storage = _make_storage(namespace=NameSpace.VECTOR_STORE_ENTITIES)
    entity_id = compute_mdhash_id("Alice", prefix="ent-")
    # Pending entity keyed by the hash id.
    await storage.upsert({entity_id: _entity_data(name="Alice")})

    await storage.delete_entity("Alice")

    # Pending pruned.
    assert entity_id not in storage._pending_vector_docs
    # SQL predicate fired against db.execute (the immediate path).
    storage.db.execute.assert_awaited_once()
    sql_arg = storage.db.execute.await_args.args[0]
    params_arg = storage.db.execute.await_args.args[1]
    assert "entity_name=$2" in sql_arg
    assert params_arg == {"workspace": "test_ws", "entity_name": "Alice"}


# ---------------------------------------------------------------------------
# 11. delete_entity_relation prunes pending relation docs + runs SQL predicate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_entity_relation_prunes_pending_and_runs_sql():
    storage = _make_storage(namespace=NameSpace.VECTOR_STORE_RELATIONSHIPS)
    await storage.upsert(
        {
            "r1": _relation_data(src="Alice", tgt="Bob"),
            "r2": _relation_data(src="Carol", tgt="Alice"),
            "r3": _relation_data(src="Eve", tgt="Mallory"),
        }
    )

    await storage.delete_entity_relation("Alice")

    assert "r1" not in storage._pending_vector_docs
    assert "r2" not in storage._pending_vector_docs
    assert "r3" in storage._pending_vector_docs

    storage.db.execute.assert_awaited_once()
    sql_arg = storage.db.execute.await_args.args[0]
    assert "source_id=$2 OR target_id=$2" in sql_arg


# ---------------------------------------------------------------------------
# 12. drop() clears buffers and runs workspace delete
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_drop_clears_buffers_and_runs_delete():
    storage = _make_storage()
    await storage.upsert({"c1": _chunk_data()})
    await storage.delete(["c2"])
    assert storage._pending_vector_docs and storage._pending_vector_deletes

    result = await storage.drop()
    assert result["status"] == "success"
    assert storage._pending_vector_docs == {}
    assert storage._pending_vector_deletes == set()
    storage.db.execute.assert_awaited_once()


# ---------------------------------------------------------------------------
# 13. Read-your-writes: get_by_id, get_by_ids, get_vectors_by_ids
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_by_id_returns_pending_and_hides_deletes():
    storage = _make_storage()
    await storage.upsert({"c1": _chunk_data(content="hello")})

    doc = await storage.get_by_id("c1")
    assert doc is not None
    assert doc["id"] == "c1"
    assert doc["content"] == "hello"
    assert "__vector__" not in doc
    assert "__id__" not in doc
    assert "created_at" in doc

    # SQL not touched for buffered hits.
    storage.db.query.assert_not_called()

    # Now delete and ensure the buffered tombstone wins over SQL.
    await storage.delete(["c1"])
    assert (await storage.get_by_id("c1")) is None


@pytest.mark.asyncio
async def test_get_by_ids_preserves_order_and_uses_any_sql():
    storage = _make_storage()
    await storage.upsert({"c1": _chunk_data(content="a")})
    await storage.delete(["c2"])

    # c3 will fall through to SQL.
    storage.db.query = AsyncMock(
        return_value=[{"id": "c3", "content": "from-pg", "created_at": 0}]
    )

    docs = await storage.get_by_ids(["c1", "c2", "c3"])
    assert docs[0] is not None and docs[0]["id"] == "c1" and docs[0]["content"] == "a"
    assert docs[1] is None  # pending delete
    assert docs[2] is not None and docs[2]["id"] == "c3"

    # SQL fallback used `id = ANY($2)` (not string-built IN).
    sql_used = storage.db.query.await_args.args[0]
    assert "id = ANY($2)" in sql_used
    assert storage.db.query.await_args.args[1] == ["test_ws", ["c3"]]


@pytest.mark.asyncio
async def test_get_vectors_by_ids_returns_cached_and_skips_deletes():
    storage = _make_storage()
    await storage.upsert({"c1": _chunk_data(content="a")})
    await storage.upsert({"c2": _chunk_data(content="b")})
    await storage.delete(["c2"])

    # c3 falls through to SQL.
    storage.db.query = AsyncMock(
        return_value=[{"id": "c3", "content_vector": [0.5, 0.6, 0.7]}]
    )

    vecs = await storage.get_vectors_by_ids(["c1", "c2", "c3"])
    # c1 lazily embedded; c2 skipped; c3 from SQL.
    assert "c1" in vecs and len(vecs["c1"]) == 3
    assert "c2" not in vecs
    assert vecs["c3"] == [0.5, 0.6, 0.7]

    sql_used = storage.db.query.await_args.args[0]
    assert "id = ANY($2)" in sql_used


# ---------------------------------------------------------------------------
# 14. finalize() raises with pending counts if flush failed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_finalize_raises_when_flush_fails_and_releases_client():
    storage = _make_storage(fail_run_with_retry=True)
    await storage.upsert({"c1": _chunk_data()})
    await storage.delete(["c2"])

    # Patch ClientManager.release_client to a no-op so we don't touch real state.
    from lightrag.kg import postgres_impl

    release_mock = AsyncMock()
    original = postgres_impl.ClientManager.release_client
    postgres_impl.ClientManager.release_client = release_mock
    try:
        with pytest.raises(RuntimeError, match="pending upserts"):
            await storage.finalize()
        release_mock.assert_awaited_once()
        assert storage.db is None
    finally:
        postgres_impl.ClientManager.release_client = original


@pytest.mark.asyncio
async def test_finalize_clean_path_flushes_then_releases_client():
    storage = _make_storage()
    await storage.upsert({"c1": _chunk_data()})

    from lightrag.kg import postgres_impl

    release_mock = AsyncMock()
    original = postgres_impl.ClientManager.release_client
    postgres_impl.ClientManager.release_client = release_mock
    try:
        await storage.finalize()
    finally:
        postgres_impl.ClientManager.release_client = original

    release_mock.assert_awaited_once()
    assert storage.db is None
    assert storage._pending_vector_docs == {}


# ---------------------------------------------------------------------------
# 15. Empty input no-ops
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_inputs_are_noops():
    storage = _make_storage()
    await storage.upsert({})
    await storage.delete([])
    await storage.index_done_callback()
    assert storage._retry_call_count["n"] == 0
    assert storage._counting_embed.call_count == 0


# ---------------------------------------------------------------------------
# 16. delete_entity serializes against an in-flight flush via _flush_lock
# ---------------------------------------------------------------------------


class _GatedEmbed:
    """Embedding func that blocks on a gate so a flush can be paused mid-call."""

    def __init__(self, dim: int = 3):
        self.embedding_dim = dim
        self.max_token_size = 512
        self.model_name = "test_model"
        self.gate = asyncio.Event()
        self.entered = asyncio.Event()
        self.call_count = 0

    async def __call__(self, texts, **kwargs):
        self.call_count += 1
        self.entered.set()
        await self.gate.wait()
        return np.array([[1.0, 0.0, 0.0] for _ in texts], dtype=np.float32)


@pytest.mark.asyncio
async def test_delete_entity_serializes_against_in_flight_flush():
    """A `delete_entity` issued while a flush is mid-embedding must wait for
    the flush's lock to release before its SQL predicate runs — otherwise the
    flush could persist the entity row a microsecond after the predicate
    deleted it. This pins the lock-then-SQL contract in the source.
    """
    embed = _GatedEmbed()
    storage = _make_storage(namespace=NameSpace.VECTOR_STORE_ENTITIES, embed=embed)

    entity_id = compute_mdhash_id("Alice", prefix="ent-")
    await storage.upsert({entity_id: _entity_data(name="Alice")})

    flush_task = asyncio.create_task(storage.index_done_callback())

    # Wait until the flush is blocked inside the embedding call (it now holds
    # _flush_lock).
    await asyncio.wait_for(embed.entered.wait(), timeout=1.0)

    # Kick off delete_entity; it must block on the same lock.
    delete_task = asyncio.create_task(storage.delete_entity("Alice"))

    # Give the event loop a few turns to confirm delete_entity is blocked.
    for _ in range(5):
        await asyncio.sleep(0)
    assert (
        not delete_task.done()
    ), "delete_entity should be waiting on _flush_lock while flush holds it"

    # Unblock the flush; both should complete.
    embed.gate.set()
    await asyncio.wait_for(flush_task, timeout=1.0)
    await asyncio.wait_for(delete_task, timeout=1.0)

    # Flush ran its executemany, then delete_entity ran its predicate SQL.
    assert len(storage._captured_executemany) == 1
    storage.db.execute.assert_awaited_once()


# ---------------------------------------------------------------------------
# 17. Deletes-only flush: no executemany, single ANY($2) DELETE
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deletes_only_flush_skips_executemany():
    """A flush that has only buffered deletes (no upserts) must still issue
    the parameterized DELETE under the transaction, and must NOT call
    executemany with an empty batch.
    """
    storage = _make_storage()
    await storage.delete(["c1", "c2", "c3"])
    assert storage._pending_vector_docs == {}
    assert len(storage._pending_vector_deletes) == 3

    await storage.index_done_callback()

    # No embedding was needed.
    assert storage._counting_embed.call_count == 0
    # No upsert executemany ran.
    assert storage._captured_executemany == []
    # Exactly one parameterized DELETE under the transaction.
    assert len(storage._captured_execute) == 1
    sql, args = storage._captured_execute[0]
    assert "DELETE FROM" in sql
    assert "id = ANY($2)" in sql
    assert args[0] == "test_ws"
    assert sorted(args[1]) == ["c1", "c2", "c3"]
    # Buffers cleared on success.
    assert storage._pending_vector_deletes == set()


# ---------------------------------------------------------------------------
# 18. Embedding count mismatch raises and preserves the buffer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_embedding_count_mismatch_raises_and_preserves_buffer():
    """The in-flush ``len(embeddings) != len(docs_to_embed)`` check is
    defense-in-depth against an embedding provider that bypasses the
    ``EmbeddingFunc`` wrapper validation. Bypass the wrapper by replacing
    ``storage.embedding_func`` with a bare async callable that returns
    fewer rows than requested.
    """
    storage = _make_storage(embedding_batch_num=10)

    async def short_embed(texts, **kwargs):
        rows = max(0, len(list(texts)) - 1)
        return np.array([[1.0, 0.0, 0.0] for _ in range(rows)], dtype=np.float32)

    storage.embedding_func = short_embed

    await storage.upsert({"c1": _chunk_data(content="a")})
    await storage.upsert({"c2": _chunk_data(content="b")})

    with pytest.raises(RuntimeError, match="Embedding count mismatch"):
        await storage.index_done_callback()

    # Buffer survives the mismatch; nothing was persisted.
    assert {"c1", "c2"} == set(storage._pending_vector_docs.keys())
    assert storage._retry_call_count["n"] == 0


# ---------------------------------------------------------------------------
# 19. delete_entity discards a matching pending delete for the same hash id
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_entity_discards_matching_pending_delete():
    """If both `delete()` (which buffers a tombstone) and `delete_entity()`
    fire for the same entity, the pending tombstone for the entity's hash id
    must be discarded — the predicate SQL covers it and we don't want a
    redundant ANY-DELETE for the same id in the next flush.
    """
    storage = _make_storage(namespace=NameSpace.VECTOR_STORE_ENTITIES)
    entity_id = compute_mdhash_id("Alice", prefix="ent-")

    await storage.delete([entity_id])
    assert entity_id in storage._pending_vector_deletes

    await storage.delete_entity("Alice")

    # The pending tombstone for the hash id was discarded.
    assert entity_id not in storage._pending_vector_deletes
    # And the predicate SQL ran.
    storage.db.execute.assert_awaited_once()

    # A subsequent flush has nothing to do.
    await storage.index_done_callback()
    assert storage._retry_call_count["n"] == 0


# ---------------------------------------------------------------------------
# 20. delete_entity / delete_entity_relation raise pre-initialize()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_delete_entity_pre_initialize_raises():
    """Calling delete_entity / delete_entity_relation before initialize()
    must raise RuntimeError, not silently drop the destructive intent.

    Silent no-op would defeat the data-loss visibility that finalize() and
    _flush_pending_vector_ops enforce on the symmetric paths.
    """
    db = MagicMock()
    db.execute = AsyncMock(return_value=None)
    embed = CountingEmbed()
    embedding_func = EmbeddingFunc(
        embedding_dim=embed.embedding_dim,
        func=embed,
        model_name=embed.model_name,
    )
    storage = PGVectorStorage(
        namespace=NameSpace.VECTOR_STORE_ENTITIES,
        workspace="test_ws",
        global_config={
            "embedding_batch_num": 10,
            "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.5},
        },
        embedding_func=embedding_func,
    )
    storage.db = db
    # Intentionally do NOT set _flush_lock (mimics pre-initialize state).
    assert storage._flush_lock is None

    with pytest.raises(RuntimeError, match="called before initialize"):
        await storage.delete_entity("Alice")
    with pytest.raises(RuntimeError, match="called before initialize"):
        await storage.delete_entity_relation("Alice")

    # No SQL fired (the methods short-circuited before touching db.execute).
    db.execute.assert_not_called()


# ---------------------------------------------------------------------------
# 21. _flush_pending_vector_ops raises on lifecycle violations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_flush_after_client_release_raises_with_counts():
    """Direct call to _flush_pending_vector_ops after db release with a
    non-empty buffer must raise — silent return would lose data without any
    operator-visible signal (the symmetric path in finalize() already raises).
    """
    storage = _make_storage()
    await storage.upsert({"c1": _chunk_data()})
    await storage.delete(["c2"])

    # Mimic post-finalize state: client released, buffers preserved.
    storage.db = None

    with pytest.raises(RuntimeError, match="after client release"):
        await storage._flush_pending_vector_ops()
    # Buffers untouched — the call must not have eaten the data on its way out.
    assert "c1" in storage._pending_vector_docs
    assert "c2" in storage._pending_vector_deletes


@pytest.mark.asyncio
async def test_flush_pre_initialize_with_pending_raises():
    """Pre-initialize call with a non-empty buffer (programmer error path:
    direct buffer manipulation before initialize) also raises rather than
    silently returning."""
    storage = _make_storage()
    # Reset to pre-initialize state but seed the buffer to simulate the
    # programmer-error scenario.
    storage._flush_lock = None
    storage._pending_vector_docs["c1"] = _PendingPGVectorDoc(
        item={"__id__": "c1", **_chunk_data()},
        created_at=datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None),
    )

    with pytest.raises(RuntimeError, match="called before initialize"):
        await storage._flush_pending_vector_ops()


# ---------------------------------------------------------------------------
# 22. get_vectors_by_ids drops embeddings whose pending record changed
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_vectors_by_ids_drops_when_pending_record_swapped():
    """If a concurrent upsert replaces the pending record while embedding I/O
    is in flight (outside the lock), the freshly-computed vector no longer
    matches the current buffer state and must be dropped from the response
    rather than returned stale."""
    embed = _GatedEmbed()
    storage = _make_storage(embed=embed)

    # Seed the original pending doc.
    await storage.upsert({"c1": _chunk_data(content="original")})
    original_pending = storage._pending_vector_docs["c1"]

    # Kick off get_vectors_by_ids — it will block inside the embedding gate
    # *outside* _flush_lock.
    task = asyncio.create_task(storage.get_vectors_by_ids(["c1"]))
    await asyncio.wait_for(embed.entered.wait(), timeout=1.0)

    # While embedding is in flight, replace the pending record via upsert.
    # The new doc has a different content and a vector=None.
    await storage.upsert({"c1": _chunk_data(content="replaced")})
    assert storage._pending_vector_docs["c1"] is not original_pending

    # Release the gate; the embedding completes and the identity check fires.
    embed.gate.set()
    result = await asyncio.wait_for(task, timeout=1.0)

    # The stale embedding is NOT returned, and is NOT cached on the new
    # pending record (which keeps vector=None for the next flush to embed).
    assert result == {}
    assert storage._pending_vector_docs["c1"].vector is None


@pytest.mark.asyncio
async def test_get_vectors_by_ids_drops_when_pending_record_removed():
    """Same identity-check guard but for the delete-while-embedding race."""
    embed = _GatedEmbed()
    storage = _make_storage(embed=embed)

    await storage.upsert({"c1": _chunk_data(content="original")})

    task = asyncio.create_task(storage.get_vectors_by_ids(["c1"]))
    await asyncio.wait_for(embed.entered.wait(), timeout=1.0)

    # Delete the pending record mid-embedding.
    await storage.delete(["c1"])
    assert "c1" not in storage._pending_vector_docs

    embed.gate.set()
    result = await asyncio.wait_for(task, timeout=1.0)

    # The vector for the removed id is dropped from the response.
    assert result == {}


# ---------------------------------------------------------------------------
# Flush batching: upsert split by record cap, delete split by id cap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_flush_upsert_splits_by_record_cap():
    storage = _make_storage()
    storage._max_upsert_records_per_batch = 2  # byte cap left at default

    await storage.upsert({f"c{i}": _chunk_data(content=f"text {i}") for i in range(5)})
    await storage.index_done_callback()

    # 5 records / cap 2 => 3 executemany calls (one per upsert chunk).
    assert len(storage._captured_executemany) == 3
    assert [len(rows) for _, rows in storage._captured_executemany] == [2, 2, 1]
    # No deletes in this flush.
    assert storage._captured_execute == []
    assert storage._pending_vector_docs == {}


@pytest.mark.asyncio
async def test_flush_upsert_splits_by_payload_bytes():
    storage = _make_storage()
    # Disable record cap; force a byte cap small enough that each big chunk
    # lands in its own batch.
    storage._max_upsert_records_per_batch = 0
    storage._max_upsert_payload_bytes = 200

    await storage.upsert({f"c{i}": _chunk_data(content="X" * 150) for i in range(3)})
    await storage.index_done_callback()

    # Each ~150-byte content row exceeds half the 200-byte budget, so the three
    # rows cannot coalesce -> 3 separate executemany calls.
    assert len(storage._captured_executemany) == 3


@pytest.mark.asyncio
async def test_flush_delete_splits_by_id_cap():
    storage = _make_storage()
    storage._max_delete_records_per_batch = 2

    await storage.delete([f"c{i}" for i in range(5)])
    await storage.index_done_callback()

    # 5 ids / cap 2 => 3 DELETE statements, each a bounded ANY($2) slice.
    assert len(storage._captured_execute) == 3
    slices = [args[1] for _, args in storage._captured_execute]
    assert [len(s) for s in slices] == [2, 2, 1]
    # Every captured statement is a DELETE.
    assert all("DELETE FROM" in sql for sql, _ in storage._captured_execute)
    assert storage._pending_vector_deletes == set()


@pytest.mark.asyncio
async def test_upsert_only_flush_with_delete_splitting_disabled():
    """Regression: delete cap <= 0 + no pending deletes must not raise.

    The disabled-delete fallback used to yield delete_chunk == 0, making the
    empty-delete range() step raise ValueError and fail the upsert.
    """
    storage = _make_storage()
    storage._max_delete_records_per_batch = 0  # documented "disable splitting"

    await storage.upsert({"c1": _chunk_data(content="alpha")})
    await storage.index_done_callback()  # upsert-only flush, no deletes

    assert len(storage._captured_executemany) == 1
    assert storage._captured_execute == []
    assert storage._pending_vector_docs == {}


@pytest.mark.asyncio
async def test_flush_delete_with_splitting_disabled_runs_single_statement():
    """delete cap <= 0 sends every pending id in one ANY($2) statement."""
    storage = _make_storage()
    storage._max_delete_records_per_batch = 0

    await storage.delete([f"c{i}" for i in range(5)])
    await storage.index_done_callback()

    assert len(storage._captured_execute) == 1
    _, args = storage._captured_execute[0]
    assert len(args[1]) == 5


@pytest.mark.asyncio
async def test_flush_upsert_and_delete_are_separate_phases():
    storage = _make_storage()
    storage._max_upsert_records_per_batch = 10
    storage._max_delete_records_per_batch = 10

    await storage.upsert({"keep": _chunk_data(content="v")})
    await storage.delete(["gone1", "gone2"])
    await storage.index_done_callback()

    # One upsert executemany + one delete execute (disjoint buffers).
    assert len(storage._captured_executemany) == 1
    assert len(storage._captured_execute) == 1


@pytest.mark.asyncio
async def test_flush_failure_keeps_pending_buffers():
    storage = _make_storage(fail_run_with_retry=True)
    await storage.upsert({"c1": _chunk_data()})
    await storage.delete(["c2"])

    with pytest.raises(RuntimeError, match="simulated PG failure"):
        await storage.index_done_callback()

    # Fail-fast-retain: nothing cleared, next flush replays everything.
    assert "c1" in storage._pending_vector_docs
    assert "c2" in storage._pending_vector_deletes


@pytest.mark.offline
@pytest.mark.asyncio
async def test_drop_pending_index_ops_clears_buffers():
    """On an internal-error abort the pipeline calls drop_pending_index_ops to
    discard buffered upserts/deletes without flushing them (PR #3187)."""
    storage = _make_storage()
    await storage.upsert({"c1": _chunk_data(content="alpha")})
    storage._pending_vector_deletes.add("old-id")
    assert storage._pending_vector_docs
    await storage.drop_pending_index_ops()
    assert not storage._pending_vector_docs
    assert not storage._pending_vector_deletes
