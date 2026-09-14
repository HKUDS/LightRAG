"""NetworkXStorage drop status follows the durable operation, not notification."""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from lightrag.kg import networkx_impl
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
async def test_drop_stays_successful_when_peer_notification_fails(
    tmp_path, monkeypatch
):
    storage = _make_storage(tmp_path)
    await storage.initialize()
    try:
        await storage.upsert_node("n1", {"entity_id": "n1"})
        await storage.index_done_callback()
        assert Path(storage._graphml_xml_file).exists()

        async def notification_boom(namespace, workspace=None):
            storage.storage_updated.value = True
            raise RuntimeError("notification boom")

        monkeypatch.setattr(networkx_impl, "set_all_update_flags", notification_boom)
        logged_errors = []
        monkeypatch.setattr(networkx_impl.logger, "error", logged_errors.append)
        result = await storage.drop()

        assert result == {"status": "success", "message": "data dropped"}
        assert not Path(storage._graphml_xml_file).exists()
        assert storage._graph.number_of_nodes() == 0
        assert storage.storage_updated.value is False
        assert any("some processes may not reload" in msg for msg in logged_errors)
        assert any("notification boom" in msg for msg in logged_errors)
    finally:
        await storage.finalize()


@pytest.mark.asyncio
async def test_drop_reports_a_destructive_file_failure(tmp_path, monkeypatch):
    storage = _make_storage(tmp_path)
    await storage.initialize()
    try:
        await storage.upsert_node("n1", {"entity_id": "n1"})
        await storage.index_done_callback()

        def remove_boom(path):
            raise OSError("delete boom")

        monkeypatch.setattr(networkx_impl.os, "remove", remove_boom)

        result = await storage.drop()

        assert result == {"status": "error", "message": "delete boom"}
        assert Path(storage._graphml_xml_file).exists()
        assert storage._graph.has_node("n1")
    finally:
        await storage.finalize()


@pytest.mark.asyncio
async def test_drop_blocks_peer_reads_until_notification_completes(
    tmp_path, monkeypatch
):
    writer = _make_storage(tmp_path)
    await writer.initialize()
    await writer.upsert_node("n1", {"entity_id": "n1"})
    await writer.index_done_callback()
    peer = _make_storage(tmp_path)
    await peer.initialize()
    notification_started = asyncio.Event()
    allow_notification = asyncio.Event()
    notify = networkx_impl.set_all_update_flags

    async def paused_notification(namespace, workspace=None):
        notification_started.set()
        await allow_notification.wait()
        await notify(namespace, workspace=workspace)

    monkeypatch.setattr(networkx_impl, "set_all_update_flags", paused_notification)
    drop_task = asyncio.create_task(writer.drop())
    read_task = None
    try:
        await asyncio.wait_for(notification_started.wait(), timeout=2)
        assert not Path(writer._graphml_xml_file).exists()
        read_task = asyncio.create_task(peer.has_node("n1"))
        # A reader arriving after deletion must wait for the reload signal.
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(read_task), timeout=0.1)
        allow_notification.set()
        assert (await asyncio.wait_for(drop_task, timeout=2))["status"] == "success"
        assert await asyncio.wait_for(read_task, timeout=2) is False
        assert writer.storage_updated.value is False
    finally:
        allow_notification.set()
        await asyncio.gather(drop_task, *([read_task] if read_task else []))
        await peer.finalize()
        await writer.finalize()


@pytest.mark.asyncio
async def test_drop_stays_successful_when_writer_flag_reset_fails(
    tmp_path, monkeypatch
):
    storage = _make_storage(tmp_path)
    await storage.initialize()
    try:
        await storage.upsert_node("n1", {"entity_id": "n1"})
        await storage.index_done_callback()

        class BrokenResetFlag:
            @property
            def value(self):
                return True

            @value.setter
            def value(self, value):
                raise RuntimeError("reset boom")

        monkeypatch.setattr(storage, "storage_updated", BrokenResetFlag())
        logged_errors = []
        monkeypatch.setattr(networkx_impl.logger, "error", logged_errors.append)
        result = await storage.drop()

        assert result == {"status": "success", "message": "data dropped"}
        assert not Path(storage._graphml_xml_file).exists()
        assert storage._graph.number_of_nodes() == 0
        assert any("reset boom" in msg for msg in logged_errors)
    finally:
        await storage.finalize()


@pytest.mark.asyncio
@pytest.mark.parametrize("notification_fails", [False, True])
async def test_cancelled_drop_finishes_notification_and_logs(
    tmp_path, monkeypatch, notification_fails
):
    writer = _make_storage(tmp_path)
    await writer.initialize()
    await writer.upsert_node("n1", {"entity_id": "n1"})
    await writer.index_done_callback()
    peer = _make_storage(tmp_path)
    await peer.initialize()
    started = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()
    notify = networkx_impl.set_all_update_flags
    errors = []
    infos = []

    async def paused_notification(namespace, workspace=None):
        started.set()
        await release.wait()
        # Model partial publication before a notification failure.
        writer.storage_updated.value = True
        finished.set()
        if notification_fails:
            raise RuntimeError("notification boom")
        await notify(namespace, workspace=workspace)

    monkeypatch.setattr(networkx_impl, "set_all_update_flags", paused_notification)
    monkeypatch.setattr(networkx_impl.logger, "error", errors.append)
    monkeypatch.setattr(networkx_impl.logger, "info", infos.append)
    task = asyncio.create_task(writer.drop())
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        assert not Path(writer._graphml_xml_file).exists()
        task.cancel()
        # Deliver cancellation while publication is still suspended.
        await asyncio.sleep(0)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)
        assert finished.is_set()
        assert writer.storage_updated.value is False
        assert writer._graph.number_of_nodes() == 0
        assert any("drop graph file:" in msg for msg in infos)
        if notification_fails:
            assert any("notification boom" in msg for msg in errors)
            assert any("restart" in msg for msg in errors)
        else:
            assert await asyncio.wait_for(peer.has_node("n1"), timeout=2) is False
            # Writer ownership may move to this peer after the clear.
            await peer.upsert_node("n2", {"entity_id": "n2"})
            await peer.index_done_callback()
            durable = NetworkXStorage.load_nx_graph(peer._graphml_xml_file)
            assert set(durable.nodes) == {"n2"}
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await peer.finalize()
        await writer.finalize()


@pytest.mark.asyncio
async def test_drop_stays_successful_when_the_success_log_fails(tmp_path, monkeypatch):
    """A broken log sink is not a failed deletion.

    The success log is the last step of the commit hook, so an exception there
    propagates out of ``commit_in_storage_io`` and would be reported as an
    error for a drop that already happened.
    """
    storage = _make_storage(tmp_path)
    await storage.initialize()
    try:
        await storage.upsert_node("n1", {"entity_id": "n1"})
        await storage.index_done_callback()

        def log_boom(msg):
            raise RuntimeError("log sink boom")

        monkeypatch.setattr(networkx_impl.logger, "info", log_boom)

        result = await storage.drop()

        assert result == {"status": "success", "message": "data dropped"}
        assert not Path(storage._graphml_xml_file).exists()
        assert storage._graph.number_of_nodes() == 0
        assert storage.storage_updated.value is False
    finally:
        await storage.finalize()


@pytest.mark.asyncio
async def test_drop_survives_a_broken_sink_reached_through_an_error_path(
    tmp_path, monkeypatch
):
    """Guarding only the success log leaves the error paths through the sink.

    A broken sink and a failing notification together reach ``logger.error``
    inside the notification handler. Unguarded, that raises through
    ``commit_in_storage_io`` and past the outer handler's own ``logger.error``,
    so ``drop`` does not even return its dict — the caller sees an exception
    for a deletion that already landed.
    """
    storage = _make_storage(tmp_path)
    await storage.initialize()
    try:
        await storage.upsert_node("n1", {"entity_id": "n1"})
        await storage.index_done_callback()

        def log_boom(msg):
            raise RuntimeError("log sink boom")

        async def notification_boom(namespace, workspace=None):
            # Model partial publication before the failure, so the reload-flag
            # assignment below is an observable step and not a no-op.
            storage.storage_updated.value = True
            raise RuntimeError("notification boom")

        monkeypatch.setattr(networkx_impl, "set_all_update_flags", notification_boom)
        monkeypatch.setattr(networkx_impl.logger, "info", log_boom)
        monkeypatch.setattr(networkx_impl.logger, "error", log_boom)

        result = await storage.drop()

        assert result == {"status": "success", "message": "data dropped"}
        assert not Path(storage._graphml_xml_file).exists()
        assert storage._graph.number_of_nodes() == 0
        # The step past the unreportable diagnostic still runs. Skipping it
        # would leave the flag set and self-reload a snapshot that is
        # already correct — harmless here, and the same skip on the
        # snapshot-reset path is what serves dropped rows.
        assert storage.storage_updated.value is False
    finally:
        await storage.finalize()


@pytest.mark.asyncio
async def test_destructive_failure_still_reports_error_with_a_broken_sink(
    tmp_path, monkeypatch
):
    """The mirror case: a sink failure must not swallow a real ``"error"``."""
    storage = _make_storage(tmp_path)
    await storage.initialize()
    try:
        await storage.upsert_node("n1", {"entity_id": "n1"})
        await storage.index_done_callback()

        def remove_boom(path):
            raise OSError("delete boom")

        def log_boom(msg):
            raise RuntimeError("log sink boom")

        monkeypatch.setattr(networkx_impl.os, "remove", remove_boom)
        monkeypatch.setattr(networkx_impl.logger, "error", log_boom)

        result = await storage.drop()

        assert result == {"status": "error", "message": "delete boom"}
        assert Path(storage._graphml_xml_file).exists()
    finally:
        await storage.finalize()
