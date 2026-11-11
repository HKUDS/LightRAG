"""NetworkXStorage.index_done_callback must RAISE on a graph-save failure
(PR #3187), not swallow it and return False, and must leave this process's
in-memory graph matching the file it failed to replace.

A swallowed save error would let _insert_done's _flush_one treat the flush as
successful (it only detects failures via exceptions), so the document would be
marked PROCESSED with the graph changes unpersisted — silent data loss. This
locks the raise-on-failure behavior that aligns NetworkX with the other
backends (faiss/nano raise too).

Raising alone is not enough: the mutation is still sitting in ``self._graph``
while the file does not have it, and nothing repairs that divergence — a failed
write never reaches ``_committed``, so ``storage_updated`` stays False and
``_get_graph``'s reload branch never fires. A caller would then be told an
object is absent while it is still on disk. ``utils_graph``'s deletion retry
reads exactly that as a durable removal and sweeps the object's authoritative
tracking row, leaving a live node with no provenance — which a later purge
resolves by deleting an entity other documents still reference.
"""

from __future__ import annotations

import numpy as np
import pytest

from lightrag.kg.networkx_impl import NetworkXStorage
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.utils import EmbeddingFunc

pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def _shared_data():
    finalize_share_data()
    initialize_share_data()
    yield
    finalize_share_data()


async def _embed(texts):
    return np.random.rand(len(texts), 8)


def _make_storage(tmp_path) -> NetworkXStorage:
    return NetworkXStorage(
        namespace="test_graph",
        workspace="ws",
        global_config={
            "working_dir": str(tmp_path),
            "embedding_batch_num": 10,
            "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.5},
        },
        embedding_func=EmbeddingFunc(embedding_dim=8, max_token_size=512, func=_embed),
    )


@pytest.mark.asyncio
async def test_index_done_callback_raises_on_save_failure(tmp_path, monkeypatch):
    storage = _make_storage(tmp_path)
    await storage.initialize()
    try:
        await storage.upsert_node("n1", {"entity_id": "n1", "description": "x"})

        def boom(graph, file_name, workspace):
            raise OSError("save boom")

        # write_nx_graph is invoked as NetworkXStorage.write_nx_graph(...).
        monkeypatch.setattr(NetworkXStorage, "write_nx_graph", staticmethod(boom))

        # Must surface the error (NOT return False / swallow it).
        with pytest.raises(OSError, match="save boom"):
            await storage.index_done_callback()
    finally:
        await storage.finalize()


@pytest.mark.asyncio
async def test_failed_save_restores_the_in_memory_graph(tmp_path, monkeypatch):
    storage = _make_storage(tmp_path)
    await storage.initialize()
    try:
        await storage.upsert_node("n1", {"entity_id": "n1", "description": "x"})
        await storage.index_done_callback()  # n1 is durable

        await storage.delete_node("n1")

        def boom(graph, file_name, workspace):
            raise OSError("save boom")

        monkeypatch.setattr(NetworkXStorage, "write_nx_graph", staticmethod(boom))
        with pytest.raises(OSError, match="save boom"):
            await storage.index_done_callback()

        # The removal never reached the file, so it must not survive in memory
        # either: a reader asking "is n1 there?" has to get the file's answer.
        assert await storage.has_node("n1")
        persisted = NetworkXStorage.load_nx_graph(storage._graphml_xml_file)
        assert persisted is not None and persisted.has_node("n1")
    finally:
        await storage.finalize()


@pytest.mark.asyncio
async def test_failed_save_retries_restore_when_the_first_reload_fails(
    tmp_path, monkeypatch
):
    storage = _make_storage(tmp_path)
    await storage.initialize()
    try:
        await storage.upsert_node("n1", {"entity_id": "n1", "description": "x"})
        await storage.index_done_callback()
        await storage.delete_node("n1")

        original_write = NetworkXStorage.write_nx_graph
        original_load = NetworkXStorage.load_nx_graph
        reload_attempts = 0

        def save_boom(graph, file_name, workspace):
            raise OSError("save boom")

        def first_reload_boom(file_name):
            nonlocal reload_attempts
            reload_attempts += 1
            if reload_attempts == 1:
                raise OSError("reload boom")
            return original_load(file_name)

        monkeypatch.setattr(NetworkXStorage, "write_nx_graph", staticmethod(save_boom))
        monkeypatch.setattr(
            NetworkXStorage, "load_nx_graph", staticmethod(first_reload_boom)
        )
        with pytest.raises(OSError, match="save boom"):
            await storage.index_done_callback()

        monkeypatch.setattr(
            NetworkXStorage, "write_nx_graph", staticmethod(original_write)
        )

        # The next read must retry the restore instead of trusting the deleted
        # process-local graph that never reached disk.
        assert await storage.has_node("n1")
        assert reload_attempts == 2
    finally:
        await storage.finalize()


@pytest.mark.asyncio
async def test_a_failed_save_does_not_publish_the_batch_on_the_next_commit(
    tmp_path, monkeypatch
):
    """The next successful commit must not carry the failed batch's mutations.

    Those documents are marked FAILED and reprocessed from scratch; publishing
    their half-written graph state on an unrelated later flush would resurrect
    work whose document is going to be re-extracted.
    """
    storage = _make_storage(tmp_path)
    await storage.initialize()
    try:

        def boom(graph, file_name, workspace):
            raise OSError("save boom")

        original = NetworkXStorage.write_nx_graph
        monkeypatch.setattr(NetworkXStorage, "write_nx_graph", staticmethod(boom))
        await storage.upsert_node("failed_batch", {"entity_id": "failed_batch"})
        with pytest.raises(OSError, match="save boom"):
            await storage.index_done_callback()

        monkeypatch.setattr(NetworkXStorage, "write_nx_graph", staticmethod(original))
        await storage.upsert_node("later_doc", {"entity_id": "later_doc"})
        await storage.index_done_callback()

        persisted = NetworkXStorage.load_nx_graph(storage._graphml_xml_file)
        assert persisted is not None
        assert persisted.has_node("later_doc")
        assert not persisted.has_node("failed_batch")
    finally:
        await storage.finalize()


@pytest.mark.asyncio
async def test_a_failing_reload_does_not_mask_the_save_error(tmp_path, monkeypatch):
    storage = _make_storage(tmp_path)
    await storage.initialize()
    try:
        await storage.upsert_node("n1", {"entity_id": "n1"})

        def boom(graph, file_name, workspace):
            raise OSError("save boom")

        def unreadable(file_name):
            raise OSError("reload boom")

        monkeypatch.setattr(NetworkXStorage, "write_nx_graph", staticmethod(boom))
        monkeypatch.setattr(NetworkXStorage, "load_nx_graph", staticmethod(unreadable))

        # The save error is what the caller must act on; the reload failure is
        # reported separately in the log, never substituted for it.
        with pytest.raises(OSError, match="save boom"):
            await storage.index_done_callback()
    finally:
        await storage.finalize()
