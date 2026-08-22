"""Save-failure replay coverage for ``FaissVectorDBStorage`` (#3688).

``index_done_callback`` used to clear ``_pending_upserts`` inside
``_flush_pending_locked`` — before the save ran. When the save then raised
and another process committed, the next callback's reload-from-disk replaced
the materialized-but-unsaved rows while the (empty) pending buffer left
nothing to replay them from: silent data loss. The pending buffer is now
kept until the save succeeds, so it doubles as the replay record.
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


@pytest.mark.asyncio
async def test_save_failure_keeps_pending_for_replay(tmp_path, monkeypatch):
    """After a failed save the pending buffer must survive, so a reload
    triggered by another writer's commit can replay the lost rows."""
    embed = _CountingEmbed()
    storage = await _make_storage(tmp_path, embed)
    await storage.upsert({"id1": {"content": "alpha"}})

    def failing_save():
        raise OSError("disk full")

    monkeypatch.setattr(storage, "_save_faiss_index", failing_save)

    with pytest.raises(OSError, match="disk full"):
        await storage.index_done_callback()

    assert "id1" in storage._pending_upserts, (
        "pending must survive a failed save — it is the replay record (#3688)"
    )
    assert storage._index_dirty is True

    # Another process commits; the next callback reloads from disk.
    monkeypatch.undo()
    storage.storage_updated.value = True

    assert await storage.index_done_callback() is True
    assert storage._pending_upserts == {}, "successful save drains the buffer"

    reloaded = await _make_storage(tmp_path, _CountingEmbed())
    assert await reloaded.get_by_id("id1") is not None, (
        "the row must survive the save failure + foreign reload cycle"
    )


@pytest.mark.asyncio
async def test_successful_save_clears_pending(tmp_path):
    embed = _CountingEmbed()
    storage = await _make_storage(tmp_path, embed)
    await storage.upsert({"id1": {"content": "alpha"}})

    assert "id1" in storage._pending_upserts
    assert await storage.index_done_callback() is True
    assert storage._pending_upserts == {}


@pytest.mark.asyncio
async def test_finalize_replays_after_save_failure(tmp_path, monkeypatch):
    """The same guarantee through finalize(): a failed save keeps the buffer,
    and the finalize retry persists the rows."""
    embed = _CountingEmbed()
    storage = await _make_storage(tmp_path, embed)
    await storage.upsert({"id1": {"content": "alpha"}})

    calls = {"n": 0}

    def failing_first_save():
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("disk full")
        return original_save()

    original_save = storage._save_faiss_index
    monkeypatch.setattr(storage, "_save_faiss_index", failing_first_save)

    with pytest.raises(OSError, match="disk full"):
        await storage.index_done_callback()

    assert "id1" in storage._pending_upserts

    await storage.finalize()

    assert storage._pending_upserts == {}
    reloaded = await _make_storage(tmp_path, _CountingEmbed())
    assert await reloaded.get_by_id("id1") is not None
