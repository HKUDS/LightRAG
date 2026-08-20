"""Deferred-delete coverage for ``NanoVectorDBStorage``.

``delete()`` no longer materializes immediately: each ``NanoVectorDB.delete()``
call rebuilds the whole matrix via ``np.delete`` (a full O(N) copy), and the
entity/relation merge stage issues one delete call per relation — on a
multi-GB matrix this dominates ingestion CPU. The storage now queues ids in
``_pending_deletes`` and applies them in ONE batched ``client.delete()`` at
flush time, strictly BEFORE pending upserts materialize.

These tests pin that contract with a counting ``NanoVectorDB.delete`` spy —
no live model or network. They mirror the deferred-embedding protocol tests
in ``test_nano_deferred_embedding.py``.
"""

import numpy as np
import pytest

nano_vectordb = pytest.importorskip("nano_vectordb")

from lightrag.kg.nano_vector_db_impl import NanoVectorDBStorage  # noqa: E402
from lightrag.kg.shared_storage import (  # noqa: E402
    initialize_share_data,
    finalize_share_data,
)
from lightrag.utils import EmbeddingFunc  # noqa: E402

DIM = 8


@pytest.fixture(autouse=True)
def _shared_data():
    finalize_share_data()
    initialize_share_data()
    yield
    finalize_share_data()


class _DeterministicEmbed:
    """Async embedding callable with a per-text deterministic vector."""

    def __init__(self, dim: int = DIM):
        self.dim = dim

    async def __call__(self, texts, **kwargs):
        return np.array(
            [
                np.full(self.dim, (abs(hash(t)) % 97) + 1, dtype=np.float32)
                for t in texts
            ]
        )


def _make_storage(tmp_path) -> NanoVectorDBStorage:
    return NanoVectorDBStorage(
        namespace="test_vectors",
        workspace="ws",
        global_config={
            "working_dir": str(tmp_path),
            "embedding_batch_num": 32,
            "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2},
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=DIM, max_token_size=512, func=_DeterministicEmbed()
        ),
        meta_fields={"content"},
    )


def _spy_client_delete(storage, counter):
    """Wrap the materialized client's ``delete`` to count invocations."""
    original = storage._client.delete

    def spy(ids):
        counter.append(list(ids))
        return original(ids)

    storage._client.delete = spy


async def _seed(storage, items: dict[str, str]):
    await storage.upsert(
        {doc_id: {"content": content} for doc_id, content in items.items()}
    )
    await storage.index_done_callback()


@pytest.mark.offline
@pytest.mark.asyncio
async def test_delete_defers_materialization_to_flush(tmp_path):
    storage = _make_storage(tmp_path)
    await storage.initialize()
    await _seed(storage, {"id1": "alpha", "id2": "beta", "id3": "gamma"})

    calls: list[list[str]] = []
    _spy_client_delete(storage, calls)

    await storage.delete(["id1"])
    await storage.delete(["id2"])
    assert calls == [], "delete() must not touch the materialized client"
    assert len(storage._client) == 3, "rows stay materialized until flush"

    await storage.index_done_callback()
    assert len(calls) == 1, "flush applies all queued deletes in ONE batch"
    assert sorted(calls[0]) == ["id1", "id2"]
    assert len(storage._client) == 1


@pytest.mark.offline
@pytest.mark.asyncio
async def test_deleted_then_reinserted_id_ends_with_new_row(tmp_path):
    """The merge stage deletes an id then re-upserts it: the flush must apply
    the delete BEFORE the upsert so exactly the new row survives."""
    storage = _make_storage(tmp_path)
    await storage.initialize()
    await _seed(storage, {"id1": "old-content", "id2": "keep"})

    await storage.delete(["id1", "missing-id"])
    await storage.upsert({"id1": {"content": "new-content"}})
    await storage.index_done_callback()

    rows = {d["__id__"]: d for d in (await storage.client_storage)["data"]}
    assert sorted(rows) == ["id1", "id2"]
    assert rows["id1"]["content"] == "new-content"
    matrix = (await storage.client_storage)["matrix"]
    assert matrix.shape == (2, DIM), "no duplicate or leftover rows"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_pending_delete_reads_as_absent_on_read_your_writes_paths(tmp_path):
    storage = _make_storage(tmp_path)
    await storage.initialize()
    await _seed(storage, {"id1": "alpha", "id2": "beta"})

    await storage.delete(["id1"])

    assert await storage.get_by_id("id1") is None
    got = await storage.get_by_ids(["id1", "id2"])
    assert got[0] is None and got[1]["content"] == "beta"
    vectors = await storage.get_vectors_by_ids(["id1", "id2"])
    assert "id1" not in vectors and "id2" in vectors


@pytest.mark.offline
@pytest.mark.asyncio
async def test_query_reads_the_materialized_index_only(tmp_path):
    """``query`` ranks the materialized index, so a queued delete is still
    returned until the flush applies it — the same contract the Qdrant and
    PostgreSQL buffers document. Filtering queued ids out of the result
    *after* top-k selection would silently shrink the result set instead.
    """
    storage = _make_storage(tmp_path)
    await storage.initialize()
    await _seed(storage, {f"id{i}": f"doc{i}" for i in range(5)})

    await storage.delete([f"id{i}" for i in range(4)])

    hits = await storage.query("doc0", top_k=3)
    assert len(hits) == 3, "top_k must not be truncated by queued deletes"

    await storage.index_done_callback()
    hits = await storage.query("doc0", top_k=3)
    assert [h["id"] for h in hits] == ["id4"], "flush applies the deletes"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_upsert_supersedes_a_queued_delete_for_the_same_id(tmp_path):
    storage = _make_storage(tmp_path)
    await storage.initialize()
    await _seed(storage, {"id1": "old"})

    await storage.delete(["id1"])
    await storage.upsert({"id1": {"content": "new"}})
    assert storage._pending_deletes == set(), "upsert must cancel the queued delete"

    await storage.index_done_callback()
    assert (await storage.get_by_id("id1"))["content"] == "new"
    assert len(storage._client) == 1, "no duplicate row"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_drop_discards_queued_deletes(tmp_path):
    storage = _make_storage(tmp_path)
    await storage.initialize()
    await _seed(storage, {"id1": "alpha"})

    await storage.delete(["id1"])
    await storage.drop()

    assert storage._pending_deletes == set()
    assert storage._pending_upserts == {}


@pytest.mark.offline
@pytest.mark.asyncio
async def test_finalize_skips_the_save_when_queued_deletes_change_nothing(tmp_path):
    """A queued delete for an id that is not in the index must not trigger a
    full-file rewrite (and the cross-process reload it broadcasts)."""
    storage = _make_storage(tmp_path)
    await storage.initialize()
    await _seed(storage, {"id1": "alpha"})

    saves: list[int] = []
    original = storage._save_to_disk_locked
    storage._save_to_disk_locked = lambda: (saves.append(1), original())[1]

    await storage.delete(["never-inserted"])
    await storage.finalize()

    assert saves == [], "no-op delete must not rewrite the file"
    assert storage._pending_deletes == set()


@pytest.mark.offline
@pytest.mark.asyncio
async def test_delete_cancels_pending_upsert_of_same_id(tmp_path):
    storage = _make_storage(tmp_path)
    await storage.initialize()

    await storage.upsert({"id1": {"content": "never-lands"}})
    await storage.delete(["id1"])
    await storage.index_done_callback()

    assert len(storage._client) == 0
    assert await storage.get_by_id("id1") is None


@pytest.mark.offline
@pytest.mark.asyncio
async def test_delete_only_flush_persists_and_is_idempotent(tmp_path):
    """A flush with ONLY queued deletes (no pending upserts) must still
    materialize, save, and be idempotent on repeat."""
    storage = _make_storage(tmp_path)
    await storage.initialize()
    await _seed(storage, {"id1": "alpha", "id2": "beta"})

    await storage.delete(["id2"])
    await storage.index_done_callback()
    assert len(storage._client) == 1

    await storage.index_done_callback()  # nothing pending — must be a no-op
    assert len(storage._client) == 1

    # A fresh instance must observe the persisted state (save happened).
    reloaded = _make_storage(tmp_path)
    await reloaded.initialize()
    assert await reloaded.get_by_id("id2") is None
    assert (await reloaded.get_by_id("id1"))["content"] == "alpha"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_delete_survives_a_concurrent_commit_by_another_writer(tmp_path):
    """Regression: a delete must not be lost when another writer commits
    between the delete and this writer's flush.

    Eager deletion mutated ``self._client`` in place while the removal was
    still unsaved. ``index_done_callback`` then reloads unconditionally when
    another process has committed, and that reload *replaces* ``self._client``
    with the on-disk snapshot — silently resurrecting the deleted row. With
    the delete queued instead, the flush reloads first and then applies the
    queued id, so the removal survives.
    """
    writer = _make_storage(tmp_path)
    other = _make_storage(tmp_path)
    await writer.initialize()
    await other.initialize()

    await _seed(writer, {"id1": "alpha", "id2": "beta"})
    await writer.delete(["id1"])

    # Another writer commits, flagging `writer` as stale.
    await other.upsert({"id3": {"content": "gamma"}})
    assert await other.index_done_callback() is True
    assert writer.storage_updated.value is True

    assert await writer.index_done_callback() is True

    reader = _make_storage(tmp_path)
    await reader.initialize()
    assert await reader.get_by_id("id1") is None, "delete must not be resurrected"
    assert (await reader.get_by_id("id2"))["content"] == "beta"
    assert (await reader.get_by_id("id3"))["content"] == "gamma"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_unflushed_deletes_roll_back_like_pending_upserts(tmp_path):
    """Queued deletes are in-memory only; a crash before flush drops them,
    matching the pending-upsert recovery semantics."""
    storage = _make_storage(tmp_path)
    await storage.initialize()
    await _seed(storage, {"id1": "alpha"})

    await storage.delete(["id1"])  # never flushed

    reloaded = _make_storage(tmp_path)
    await reloaded.initialize()
    assert (await reloaded.get_by_id("id1"))["content"] == "alpha"
