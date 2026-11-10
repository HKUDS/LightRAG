"""Real Nano persistence through the SDK shutdown path, with local models."""

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.base import StoragesStatus
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.utils import EmbeddingFunc, Tokenizer

pytestmark = [pytest.mark.offline, pytest.mark.asyncio]


class CharacterTokenizer:
    def encode(self, text):
        return list(map(ord, text))

    def decode(self, tokens):
        return "".join(map(chr, tokens))


async def embed(texts):
    return np.ones((len(texts), 4), dtype=np.float32)


async def llm(*args, **kwargs):
    return "local"


def make_rag(root, embedding=embed, model=llm, **kwargs):
    return LightRAG(
        working_dir=str(root),
        workspace="shutdown",
        llm_model_func=model,
        embedding_func=EmbeddingFunc(
            embedding_dim=4, max_token_size=512, func=embedding, model_name="local"
        ),
        tokenizer=Tokenizer("character", CharacterTokenizer()),
        **kwargs,
    )


@pytest.fixture(autouse=True)
def shared():
    finalize_share_data()
    initialize_share_data()
    yield
    finalize_share_data()


async def pending(rag):
    await rag.initialize_storages()
    await rag.relationships_vdb.upsert(
        {"rel-pending": {"src_id": "A", "tgt_id": "B", "content": "pending"}}
    )


def assert_persisted(rag):
    data = json.loads(Path(rag.relationships_vdb._client_file_name).read_text())
    assert "rel-pending" in {row["__id__"] for row in data["data"]}
    assert rag._storages_status == StoragesStatus.FINALIZED


async def test_pending_vectors_embed_before_queue_shutdown(tmp_path):
    rag = make_rag(tmp_path)
    await pending(rag)
    await rag.finalize_storages()
    assert_persisted(rag)
    await rag.finalize_storages()
    assert (await rag.get_embedding_queue_status())["worker_count"] == 0


async def test_llm_callable_finishes_before_vector_commit(tmp_path):
    started, release, flush = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def blocked(*args, **kwargs):
        started.set()
        await release.wait()
        # Work inside the model callable can still use embedding/storage.
        # This does not assert that caller-side pipeline continuations drain.
        await rag.embedding_func(["model finishing"])
        await rag.relationships_vdb.upsert(
            {"rel-pending": {"src_id": "A", "tgt_id": "B", "content": "pending"}}
        )
        return "done"

    rag = make_rag(tmp_path, model=blocked)
    await rag.initialize_storages()
    original = rag.relationships_vdb.finalize

    async def observe():
        flush.set()
        return await original()

    rag.relationships_vdb.finalize = observe
    call = asyncio.create_task(rag.role_llm_funcs["query"]("blocked"))
    await asyncio.wait_for(started.wait(), 5)
    stop = asyncio.create_task(rag.finalize_storages())
    await asyncio.sleep(0)
    assert not flush.is_set()
    assert not stop.done()
    release.set()
    assert await asyncio.wait_for(call, 5) == "done"
    await asyncio.wait_for(stop, 5)
    assert_persisted(rag)
    assert flush.is_set()


async def test_inflight_embedding_is_drained_before_remaining_storage_release(tmp_path):
    started, release, completed = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def blocked(texts):
        if texts == ["inflight"]:
            started.set()
            await release.wait()
            completed.set()
        return await embed(texts)

    rag = make_rag(tmp_path, blocked)
    await pending(rag)
    original_finalize = rag.llm_response_cache.finalize
    cache_released = asyncio.Event()

    async def finalize_cache():
        assert completed.is_set(), "response cache closed before embedding completed"
        await original_finalize()
        cache_released.set()

    rag.llm_response_cache.finalize = finalize_cache
    call = asyncio.create_task(rag.embedding_func(["inflight"]))
    await asyncio.wait_for(started.wait(), 5)
    stop = asyncio.create_task(rag.finalize_storages())
    await asyncio.sleep(0)
    assert rag._storages_status == StoragesStatus.INITIALIZED
    assert not stop.done()
    release.set()
    await asyncio.wait_for(call, 5)
    await asyncio.wait_for(stop, 5)
    assert_persisted(rag)
    assert (await rag.get_embedding_queue_status())["worker_count"] == 0
    assert cache_released.is_set()


async def test_clean_nano_shutdown_needs_no_writable_vector_files(
    tmp_path, monkeypatch
):
    """An already clean namespace must not acquire a new disk-write failure."""
    from unittest.mock import AsyncMock

    rag = make_rag(tmp_path)
    await rag.initialize_storages()
    saves = []
    for storage in (rag.entities_vdb, rag.relationships_vdb, rag.chunks_vdb):
        save = AsyncMock(side_effect=PermissionError("read-only vector storage"))
        monkeypatch.setattr(storage, "_save_to_disk_locked", save)
        saves.append(save)
    await rag.finalize_storages()
    assert rag._storages_status == StoragesStatus.FINALIZED
    for save in saves:
        save.assert_not_awaited()
    assert (await rag.get_embedding_queue_status())["worker_count"] == 0


async def test_cancelled_vector_flush_stops_embedding_workers(tmp_path):
    from unittest.mock import AsyncMock

    started, release = asyncio.Event(), asyncio.Event()

    async def blocked(texts):
        started.set()
        await release.wait()
        return await embed(texts)

    rag = make_rag(tmp_path, blocked)
    await pending(rag)
    cache_finalize = AsyncMock(wraps=rag.llm_response_cache.finalize)
    rag.llm_response_cache.finalize = cache_finalize
    stop = asyncio.create_task(rag.finalize_storages())
    await asyncio.wait_for(started.wait(), 5)
    stop.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(stop, 5)
    # main's teardown remains interruptible, not dev's finish-and-release
    # contract. The new embedding dependency must nevertheless be cleaned up.
    assert rag._storages_status == StoragesStatus.INITIALIZED
    cache_finalize.assert_not_awaited()
    assert (await rag.get_embedding_queue_status())["worker_count"] == 0
    # Explicit test cleanup, not a retry/durability promise.
    await rag.relationships_vdb.drop_pending_index_ops()
    await rag.finalize_storages()


async def test_external_cancel_keeps_embedding_drain_interruptible(tmp_path):
    from unittest.mock import AsyncMock

    started, release, interrupted = asyncio.Event(), asyncio.Event(), asyncio.Event()
    rag = make_rag(tmp_path)
    await pending(rag)
    original = rag._shutdown_embedding_queue

    async def blocked_shutdown():
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            interrupted.set()
            raise

    rag._shutdown_embedding_queue = blocked_shutdown
    cache_finalize = AsyncMock(wraps=rag.llm_response_cache.finalize)
    rag.llm_response_cache.finalize = cache_finalize
    stop = asyncio.create_task(rag.finalize_storages())
    try:
        await asyncio.wait_for(started.wait(), 5)
        stop.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(stop, 5)
        assert interrupted.is_set()
        assert rag._storages_status == StoragesStatus.INITIALIZED
        cache_finalize.assert_not_awaited()
    finally:
        release.set()
        rag._shutdown_embedding_queue = original
        await original()
        await rag.finalize_storages()


async def test_embedding_failure_preserves_best_effort_finalize(tmp_path, caplog):
    async def failed(texts):
        raise RuntimeError("provider unavailable")

    rag = make_rag(tmp_path, failed)
    await pending(rag)
    await rag.finalize_storages()
    assert rag._storages_status == StoragesStatus.FINALIZED
    assert "Failed to finalize relationships_vdb" in caplog.text
    assert rag.relationships_vdb._pending_upserts
    assert (await rag.get_embedding_queue_status())["worker_count"] == 0
    await rag.relationships_vdb.drop_pending_index_ops()


async def test_cancelled_rerank_drain_still_stops_embedding_workers(tmp_path):
    started, release, draining = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def rerank(*args, **kwargs):
        started.set()
        await release.wait()
        return []

    rag = make_rag(tmp_path, rerank_model_func=rerank)
    await rag.initialize_storages()
    await rag.embedding_func(["warmup"])
    call = asyncio.create_task(rag.rerank_model_func("query", ["document"]))
    await asyncio.wait_for(started.wait(), 5)
    original = rag.rerank_model_func.shutdown

    async def observed_shutdown(*args, **kwargs):
        draining.set()
        await original(*args, **kwargs)

    rag.rerank_model_func.shutdown = observed_shutdown
    stop = asyncio.create_task(rag.finalize_storages())
    try:
        await asyncio.wait_for(draining.wait(), 5)
        stop.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(stop, 5)
        assert (await rag.get_embedding_queue_status())["worker_count"] == 0
        assert rag._storages_status == StoragesStatus.INITIALIZED
    finally:
        release.set()
        await asyncio.wait_for(call, 5)
        rag.rerank_model_func.shutdown = original
        await original()
        await rag.finalize_storages()
