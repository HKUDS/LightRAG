"""The JSON backends must not rewrite their file on the event loop.

``index_done_callback`` serializes the whole store. Done inline in an ``async
def`` that is seconds of a frozen event loop on a large corpus, which is what
made the WebUI freeze while a purge ran: /health, /documents/pipeline_status and
/documents/paginated all hang together because a single uvicorn worker has one
loop.

Each test here runs a heartbeat coroutine and blocks inside the write for a
fixed wall-clock interval. The heartbeat can only advance if the write left the
loop, so the assertion is deterministic rather than timing-sensitive: 200ms of
``asyncio.sleep(0)`` iterations is tens of thousands of beats when the loop is
free, and exactly zero when it is not.

The companion test at the bottom pins the other half of the contract: the write
runs INSIDE the storage lock, so a concurrent writer waits for it rather than
mutating the store while the worker thread serializes it.
"""

import asyncio
import json
import threading
import time

import pytest

from lightrag.kg import json_doc_status_impl, json_kv_impl
from lightrag.kg.json_doc_status_impl import JsonDocStatusStorage
from lightrag.kg.json_kv_impl import JsonKVStorage
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data

pytestmark = pytest.mark.offline

BLOCK_SECONDS = 0.2


class _Heartbeat:
    """Counts event-loop iterations so a blocked loop is directly observable."""

    def __init__(self):
        self.beats = 0
        self._stop = False
        self._task = None

    async def _run(self):
        while not self._stop:
            self.beats += 1
            await asyncio.sleep(0)

    def start(self):
        self._task = asyncio.create_task(self._run())
        return self

    async def stop(self):
        self._stop = True
        if self._task is not None:
            await self._task


@pytest.fixture
def shared_storage():
    initialize_share_data(workers=1)
    yield
    finalize_share_data()


async def _make_kv(tmp_path):
    storage = JsonKVStorage(
        namespace="text_chunks",
        workspace="",
        global_config={"working_dir": str(tmp_path)},
        embedding_func=None,
    )
    await storage.initialize()
    return storage


async def _make_doc_status(tmp_path):
    storage = JsonDocStatusStorage(
        namespace="doc_status",
        workspace="",
        global_config={"working_dir": str(tmp_path)},
        embedding_func=None,
    )
    await storage.initialize()
    return storage


def _blocking_spy(module, heartbeat, observed):
    """Wrap the module's ``write_json`` with a real, GIL-releasing block."""
    real_write_json = module.write_json

    def spy(payload, path):
        before = heartbeat.beats
        time.sleep(BLOCK_SECONDS)
        observed.append((before, heartbeat.beats))
        return real_write_json(payload, path)

    return spy


async def test_kv_commit_keeps_the_event_loop_running(
    tmp_path, monkeypatch, shared_storage
):
    """Fix-proof: called inline, ``time.sleep`` freezes the loop and beats tie."""
    storage = await _make_kv(tmp_path)
    await storage.upsert({"chunk-1": {"content": "hello"}})

    observed = []
    heartbeat = _Heartbeat().start()
    monkeypatch.setattr(
        json_kv_impl, "write_json", _blocking_spy(json_kv_impl, heartbeat, observed)
    )

    await storage.index_done_callback()
    await heartbeat.stop()

    assert observed, "write_json was never called"
    before, after = observed[0]
    assert after > before, "event loop was blocked for the whole write"

    with open(storage._file_name, encoding="utf-8") as handle:
        assert json.load(handle)["chunk-1"]["content"] == "hello"


async def test_doc_status_commit_keeps_the_event_loop_running(
    tmp_path, monkeypatch, shared_storage
):
    storage = await _make_doc_status(tmp_path)

    observed = []
    heartbeat = _Heartbeat().start()
    monkeypatch.setattr(
        json_doc_status_impl,
        "write_json",
        _blocking_spy(json_doc_status_impl, heartbeat, observed),
    )

    # upsert persists on every call for this backend.
    await storage.upsert(
        {
            "doc-1": {
                "status": "processed",
                "content_summary": "s",
                "content_length": 1,
                "file_path": "a.pdf",
                "created_at": "2026-01-01T00:00:00",
                "updated_at": "2026-01-01T00:00:00",
            }
        }
    )
    await heartbeat.stop()

    assert observed, "write_json was never called"
    before, after = observed[0]
    assert after > before, "event loop was blocked for the whole write"

    with open(storage._file_name, encoding="utf-8") as handle:
        assert json.load(handle)["doc-1"]["status"] == "processed"


async def test_a_concurrent_upsert_waits_for_the_in_flight_write(
    tmp_path, monkeypatch, shared_storage
):
    """The offload stays inside ``_storage_lock``, so writers still serialize.

    Freeing the event loop is the point of the change, but it must not free
    other coroutines to mutate the store while the worker thread is serializing
    the snapshot. Move the offload outside the lock and the first assertion
    fails: the upsert completes while the write is still in flight.

    It also checks the resulting ordering is coherent: the file published by the
    in-flight commit predates the concurrent upsert, and a later commit contains
    it.
    """
    storage = await _make_kv(tmp_path)
    await storage.upsert({"chunk-1": {"content": "first"}})

    release = threading.Event()
    entered = threading.Event()
    real_write_json = json_kv_impl.write_json

    def blocking_write(payload, path):
        entered.set()
        assert release.wait(timeout=5), "write was never released"
        return real_write_json(payload, path)

    monkeypatch.setattr(json_kv_impl, "write_json", blocking_write)

    commit = asyncio.create_task(storage.index_done_callback())
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 2
    while not entered.is_set():
        assert loop.time() < deadline, "write never started"
        await asyncio.sleep(0.01)

    upsert = asyncio.create_task(storage.upsert({"chunk-2": {"content": "second"}}))
    for _ in range(20):
        await asyncio.sleep(0)
    assert not upsert.done(), "upsert ran while a commit was in flight"

    release.set()
    await commit
    await upsert

    with open(storage._file_name, encoding="utf-8") as handle:
        published = json.load(handle)
    assert "chunk-1" in published
    assert "chunk-2" not in published, "the in-flight snapshot must not grow"

    monkeypatch.setattr(json_kv_impl, "write_json", real_write_json)
    await storage.index_done_callback()
    with open(storage._file_name, encoding="utf-8") as handle:
        final = json.load(handle)
    assert final["chunk-1"]["content"] == "first"
    assert final["chunk-2"]["content"] == "second"
