"""``NanoVectorDBStorage.drop`` status follows the durable removal (#3855).

The file deletion is the destructive step; ``set_all_update_flags`` afterwards
only tells the other processes to reload. A notification failure must be
logged, not reported as a drop that did not happen — ``/documents/clear``
counts these statuses to decide whether input files are safe to delete.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

nano_vectordb = pytest.importorskip("nano_vectordb")

import lightrag.kg.nano_vector_db_impl as nano_impl  # noqa: E402
from lightrag.kg.nano_vector_db_impl import NanoVectorDBStorage  # noqa: E402
from lightrag.kg.shared_storage import (  # noqa: E402
    finalize_share_data,
    initialize_share_data,
)
from lightrag.utils import EmbeddingFunc  # noqa: E402

pytestmark = pytest.mark.offline

DIM = 8


@pytest.fixture(autouse=True)
def _shared_data():
    finalize_share_data()
    initialize_share_data()
    yield
    finalize_share_data()


async def _embed(texts, **kwargs):
    return np.array(
        [np.full(DIM, (abs(hash(t)) % 97) + 1, dtype=np.float32) for t in texts]
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
            embedding_dim=DIM, max_token_size=512, func=_embed
        ),
        meta_fields={"content"},
    )


async def _seeded_storage(tmp_path) -> NanoVectorDBStorage:
    storage = _make_storage(tmp_path)
    await storage.initialize()
    await storage.upsert({"v1": {"content": "hello"}})
    await storage.index_done_callback()
    assert Path(storage._client_file_name).exists()
    return storage


@pytest.mark.asyncio
async def test_drop_stays_successful_when_peer_notification_fails(
    tmp_path, monkeypatch
):
    storage = await _seeded_storage(tmp_path)
    try:

        async def notification_boom(namespace, workspace=None):
            # Model partial publication before the failure.
            storage.storage_updated.value = True
            raise RuntimeError("notification boom")

        monkeypatch.setattr(nano_impl, "set_all_update_flags", notification_boom)
        logged_errors: list[str] = []
        monkeypatch.setattr(nano_impl.logger, "error", logged_errors.append)

        result = await storage.drop()

        assert result == {"status": "success", "message": "data dropped"}
        assert not Path(storage._client_file_name).exists()
        assert len(storage._client.get(["v1"])) == 0
        assert storage.storage_updated.value is False
        assert any("some processes may not reload" in msg for msg in logged_errors)
        assert any("notification boom" in msg for msg in logged_errors)
    finally:
        await storage.finalize()


@pytest.mark.asyncio
async def test_drop_reports_a_destructive_file_failure(tmp_path, monkeypatch):
    storage = await _seeded_storage(tmp_path)
    try:
        # Unflushed work that a failed drop must not discard: the buffers are
        # cleared only once the file is actually gone.
        await storage.upsert({"v2": {"content": "buffered"}})
        assert "v2" in storage._pending_upserts

        def remove_boom(path):
            raise OSError("delete boom")

        monkeypatch.setattr(nano_impl.os, "remove", remove_boom)

        result = await storage.drop()

        assert result == {"status": "error", "message": "delete boom"}
        assert Path(storage._client_file_name).exists()
        assert len(storage._client.get(["v1"])) == 1
        assert "v2" in storage._pending_upserts
    finally:
        await storage.finalize()


@pytest.mark.asyncio
async def test_drop_stays_successful_when_writer_flag_reset_fails(
    tmp_path, monkeypatch
):
    storage = await _seeded_storage(tmp_path)
    try:

        class BrokenResetFlag:
            @property
            def value(self):
                return True

            @value.setter
            def value(self, value):
                raise RuntimeError("reset boom")

        monkeypatch.setattr(storage, "storage_updated", BrokenResetFlag())
        logged_errors: list[str] = []
        monkeypatch.setattr(nano_impl.logger, "error", logged_errors.append)

        result = await storage.drop()

        assert result == {"status": "success", "message": "data dropped"}
        assert not Path(storage._client_file_name).exists()
        assert any("reset boom" in msg for msg in logged_errors)
    finally:
        await storage.finalize()


@pytest.mark.asyncio
@pytest.mark.parametrize("notification_fails", [False, True])
async def test_cancelled_drop_finishes_notification_and_logs(
    tmp_path, monkeypatch, notification_fails
):
    storage = await _seeded_storage(tmp_path)
    started = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()
    notify = nano_impl.set_all_update_flags
    errors: list[str] = []
    infos: list[str] = []

    async def paused_notification(namespace, workspace=None):
        started.set()
        await release.wait()
        # Model partial publication before a notification failure.
        storage.storage_updated.value = True
        finished.set()
        if notification_fails:
            raise RuntimeError("notification boom")
        await notify(namespace, workspace=workspace)

    monkeypatch.setattr(nano_impl, "set_all_update_flags", paused_notification)
    monkeypatch.setattr(nano_impl.logger, "error", errors.append)
    monkeypatch.setattr(nano_impl.logger, "info", infos.append)
    task = asyncio.create_task(storage.drop())
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        assert not Path(storage._client_file_name).exists()
        task.cancel()
        # Deliver cancellation while publication is still suspended.
        await asyncio.sleep(0)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)
        assert finished.is_set()
        assert storage.storage_updated.value is False
        assert len(storage._client.get(["v1"])) == 0
        assert any("drop test_vectors" in msg for msg in infos)
        if notification_fails:
            assert any("notification boom" in msg for msg in errors)
            assert any("restart" in msg for msg in errors)
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await storage.finalize()
