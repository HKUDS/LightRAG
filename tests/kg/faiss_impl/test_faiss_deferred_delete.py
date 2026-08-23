"""Deferred-delete coverage for ``FaissVectorDBStorage`` (#3681).

``delete`` queues ids in ``_pending_deletes`` instead of rebuilding the
``IndexFlatIP`` per call; ``_flush_pending_locked`` applies the whole queue in
one batched rebuild. The entity/relation merge stage deletes the stale
forward/reverse rows once per relation, so the eager path rebuilt the whole
index per merged relation — quadratic ingestion, the same defect #3679
profiled on Nano (#3680 fixed it there). Deferring also keeps a delete from
being lost when ``index_done_callback`` reloads a newer on-disk snapshot from
another writer.
"""

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")

from lightrag.kg.faiss_impl import FaissVectorDBStorage  # noqa: E402
from lightrag.kg.shared_storage import (  # noqa: E402
    finalize_share_data,
    initialize_share_data,
)
from lightrag.utils import EmbeddingFunc  # noqa: E402

DIM = 8


@pytest.fixture(autouse=True)
def _shared_data():
    finalize_share_data()
    initialize_share_data()
    yield
    finalize_share_data()


class _CountingEmbed:
    def __init__(self, dim: int = DIM):
        self.dim = dim
        self.embedded_texts: list[str] = []

    async def __call__(self, texts, **kwargs):
        self.embedded_texts.extend(texts)
        return np.array(
            [
                np.full(self.dim, (abs(hash(t)) % 97) + 1, dtype=np.float32)
                for t in texts
            ]
        )


async def _make_storage(tmp_path, embed: _CountingEmbed) -> FaissVectorDBStorage:
    storage = FaissVectorDBStorage(
        namespace="test_vectors",
        workspace="ws",
        global_config={
            "working_dir": str(tmp_path),
            "embedding_batch_num": 32,
            "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2},
        },
        embedding_func=EmbeddingFunc(embedding_dim=DIM, max_token_size=512, func=embed),
        meta_fields={"content"},
    )
    await storage.initialize()
    return storage


async def _upsert_and_flush(storage: FaissVectorDBStorage, data: dict) -> None:
    await storage.upsert(data)
    assert await storage.index_done_callback() is True


@pytest.mark.offline
@pytest.mark.asyncio
async def test_delete_is_deferred_until_one_batched_flush(tmp_path, monkeypatch):
    embed = _CountingEmbed()
    storage = await _make_storage(tmp_path, embed)
    await _upsert_and_flush(
        storage,
        {f"rel-{i}": {"content": f"relation {i}"} for i in range(4)},
    )
    assert len(storage._id_to_meta) == 4

    rebuild_calls = []
    original_rebuild = storage._remove_faiss_ids_locked

    def counting_rebuild(fid_list):
        rebuild_calls.append(list(fid_list))
        return original_rebuild(fid_list)

    monkeypatch.setattr(storage, "_remove_faiss_ids_locked", counting_rebuild)

    # The merge-stage shape: one delete call per relation.
    for i in range(4):
        await storage.delete([f"rel-{i}", f"rel-{i}-reverse"])

    assert rebuild_calls == [], "delete must not rebuild the index eagerly"
    assert len(storage._id_to_meta) == 4, "materialized rows must survive until flush"

    assert await storage.index_done_callback() is True

    assert len(rebuild_calls) == 1, "the whole queued batch is one rebuild"
    assert len(storage._id_to_meta) == 0
    assert await storage.get_by_ids([f"rel-{i}" for i in range(4)]) == [None] * 4


@pytest.mark.offline
@pytest.mark.asyncio
async def test_delete_cancels_pending_upsert_without_embedding(tmp_path):
    embed = _CountingEmbed()
    storage = await _make_storage(tmp_path, embed)

    await storage.upsert({"doomed": {"content": "never embedded"}})
    await storage.delete(["doomed"])

    assert await storage.index_done_callback() is True
    assert "doomed" not in embed.embedded_texts, (
        "a queued delete must cancel the buffered upsert before embedding"
    )
    assert await storage.get_by_id("doomed") is None


@pytest.mark.offline
@pytest.mark.asyncio
async def test_fresh_upsert_supersedes_queued_delete(tmp_path):
    embed = _CountingEmbed()
    storage = await _make_storage(tmp_path, embed)
    await _upsert_and_flush(storage, {"row": {"content": "old"}})

    await storage.delete(["row"])
    await storage.upsert({"row": {"content": "new"}})
    assert await storage.index_done_callback() is True

    record = await storage.get_by_id("row")
    assert record is not None
    assert record["content"] == "new"
    # Exactly one materialized row for the custom id.
    matches = [m for m in storage._id_to_meta.values() if m.get("__id__") == "row"]
    assert len(matches) == 1


@pytest.mark.offline
@pytest.mark.asyncio
async def test_read_your_writes_hides_queued_deletes(tmp_path):
    embed = _CountingEmbed()
    storage = await _make_storage(tmp_path, embed)
    await _upsert_and_flush(
        storage, {"a": {"content": "alpha"}, "b": {"content": "beta"}}
    )

    await storage.delete(["a"])

    assert await storage.get_by_id("a") is None
    got = await storage.get_by_ids(["a", "b"])
    assert got[0] is None
    assert got[1] is not None and got[1]["content"] == "beta"
    assert "a" not in await storage.get_vectors_by_ids(["a", "b"])
    assert "b" in await storage.get_vectors_by_ids(["a", "b"])


@pytest.mark.offline
@pytest.mark.asyncio
async def test_finalize_applies_deletes_without_upserts(tmp_path):
    embed = _CountingEmbed()
    storage = await _make_storage(tmp_path, embed)
    await _upsert_and_flush(storage, {"x": {"content": "ex"}})

    await storage.delete(["x"])
    await storage.finalize()

    reloaded = await _make_storage(tmp_path, _CountingEmbed())
    assert await reloaded.get_by_id("x") is None, (
        "finalize alone must persist a queued delete to disk"
    )


@pytest.mark.offline
@pytest.mark.asyncio
async def test_dequeued_delete_survives_reload_from_disk(tmp_path):
    """The lost-delete-across-writers case: a queued id applies after the
    unconditional reload inside index_done_callback, so a concurrent
    writer's commit cannot resurrect the row."""
    embed = _CountingEmbed()
    storage = await _make_storage(tmp_path, embed)
    await _upsert_and_flush(storage, {"shared": {"content": "v1"}})

    # Another process commits a newer snapshot...
    other = await _make_storage(tmp_path, _CountingEmbed())
    await _upsert_and_flush(
        other, {"shared": {"content": "v2"}, "extra": {"content": "e"}}
    )
    # ...and this process's storage_updated flag flips, as set_all_update_flags
    # would have done, so index_done_callback reloads before flushing.
    storage.storage_updated.value = True

    await storage.delete(["shared"])
    assert await storage.index_done_callback() is True

    assert await storage.get_by_id("shared") is None, (
        "the queued delete must win over the reloaded snapshot"
    )
    assert (await storage.get_by_id("extra")) is not None, (
        "the other writer's row must be preserved"
    )


@pytest.mark.offline
@pytest.mark.asyncio
async def test_drop_pending_index_ops_discards_queued_deletes(tmp_path):
    embed = _CountingEmbed()
    storage = await _make_storage(tmp_path, embed)
    await _upsert_and_flush(storage, {"keep": {"content": "kept"}})

    await storage.delete(["keep"])
    await storage.drop_pending_index_ops()
    assert await storage.index_done_callback() is True

    assert (await storage.get_by_id("keep")) is not None, (
        "an aborted batch must discard its queued deletes"
    )


@pytest.mark.offline
@pytest.mark.asyncio
async def test_failed_delete_rebuild_keeps_tombstones_for_retry(tmp_path, monkeypatch):
    """A FAISS rebuild failure during the delete pass must not consume the
    queued tombstones: the next ``index_done_callback`` retry re-applies them
    instead of persisting the stale rows as if they were deleted."""
    embed = _CountingEmbed()
    storage = await _make_storage(tmp_path, embed)
    await _upsert_and_flush(storage, {"x": {"content": "ex"}, "y": {"content": "wy"}})
    assert len(storage._id_to_meta) == 2

    await storage.delete(["x", "y"])

    def failing_rebuild(fid_list):
        raise RuntimeError("faiss rebuild boom")

    monkeypatch.setattr(storage, "_remove_faiss_ids_locked", failing_rebuild)

    with pytest.raises(RuntimeError, match="faiss rebuild boom"):
        await storage.index_done_callback()

    assert storage._pending_deletes == {"x", "y"}, (
        "tombstones must survive a failed rebuild so the retry re-applies them"
    )
    assert len(storage._id_to_meta) == 2, "rows must still be materialized"
    assert storage._index_dirty is False, "nothing landed, so nothing to save"

    monkeypatch.undo()

    # The retry succeeds and the deletes land exactly once.
    assert await storage.index_done_callback() is True
    assert storage._pending_deletes == set()
    assert len(storage._id_to_meta) == 0
    assert await storage.get_by_ids(["x", "y"]) == [None, None]

    reloaded = await _make_storage(tmp_path, _CountingEmbed())
    assert await reloaded.get_by_ids(["x", "y"]) == [None, None], (
        "the retried deletes must be persisted to disk"
    )
