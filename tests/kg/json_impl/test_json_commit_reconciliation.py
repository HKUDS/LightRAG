"""A cancelled JSON commit must not skip its sanitize reconciliation.

``write_json`` falls back to a sanitizing encoder when the payload cannot be
encoded, and reports that via ``needs_reload`` so the caller can read the
cleaned file back into the shared dict. Before the write moved off the event
loop, the write and that read-back were one unpreemptable synchronous block —
a cancellation could only be delivered at the ``clear_all_update_flags`` await
that follows. Offloading the write opened a suspension point between them:
the sanitized file gets published while the shared dict keeps the rows that
failed to encode, and the two stay divergent until some later flush happens to
sanitize again.

``commit_in_storage_io`` puts the reconciliation back inside the write's
uncancellable region, which is what these tests pin.
"""

import asyncio
import json
import logging
import threading

import pytest

from lightrag.kg.json_doc_status_impl import JsonDocStatusStorage
from lightrag.kg.json_kv_impl import JsonKVStorage
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data

pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def _shared_data():
    finalize_share_data()
    initialize_share_data()
    yield
    finalize_share_data()


async def _make_kv(tmp_path) -> JsonKVStorage:
    storage = JsonKVStorage(
        namespace="text_chunks",
        workspace="ws",
        global_config={"working_dir": str(tmp_path)},
        embedding_func=None,
    )
    await storage.initialize()
    return storage


async def _make_doc_status(tmp_path) -> JsonDocStatusStorage:
    storage = JsonDocStatusStorage(
        namespace="doc_status",
        workspace="ws",
        global_config={"working_dir": str(tmp_path)},
        embedding_func=None,
    )
    await storage.initialize()
    return storage


def _parked_sanitizing_write(module, inside_write, may_finish, cleaned_payload):
    """Stand in for ``write_json``: park, publish cleaned data, report reload."""

    def _write(data_dict, file_name):
        inside_write.set()
        assert may_finish.wait(timeout=5), "writer was never released"
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(cleaned_payload, f)
        return True  # sanitization happened -> caller must reload

    return _write


@pytest.mark.parametrize(
    "factory, namespace, row",
    [
        (_make_kv, "text_chunks", {"content": "alpha"}),
        (
            _make_doc_status,
            "doc_status",
            {"status": "processed", "file_path": "a.pdf"},
        ),
    ],
    ids=["kv", "doc_status"],
)
async def test_cancelled_commit_still_reconciles_sanitized_data(
    tmp_path, monkeypatch, factory, namespace, row
):
    """Cancel mid-write: the cleaned file must still be read back into memory.

    Fix-proof: route the write through ``run_in_storage_io`` and inline the
    ``needs_reload`` branch after it, and ``_data`` keeps the pre-sanitization
    row — the divergence Codex flagged on #3740.
    """
    storage = await factory(tmp_path)
    await storage.upsert({"id1": row})

    module = type(storage).__module__
    inside_write = threading.Event()
    may_finish = threading.Event()
    cleaned_payload = {"id1": dict(row, content_was="sanitized")}

    monkeypatch.setattr(
        f"{module}.write_json",
        _parked_sanitizing_write(module, inside_write, may_finish, cleaned_payload),
    )
    # doc_status persists on upsert, so mark it dirty again for this commit.
    storage.storage_updated.value = True

    commit = asyncio.create_task(storage.index_done_callback())
    while not inside_write.is_set():
        await asyncio.sleep(0.01)

    commit.cancel()
    may_finish.set()

    with pytest.raises(asyncio.CancelledError):
        await commit

    assert storage._data["id1"].get("content_was") == "sanitized", (
        "a cancelled commit published the sanitized file but left the shared "
        f"dict unreconciled: {dict(storage._data['id1'])}"
    )


@pytest.mark.parametrize(
    "factory, row",
    [
        (_make_kv, {"content": "alpha"}),
        (_make_doc_status, {"status": "processed", "file_path": "a.pdf"}),
    ],
    ids=["kv", "doc_status"],
)
async def test_reconciliation_is_skipped_when_the_write_fails(
    tmp_path, monkeypatch, factory, row
):
    """No write, no reconciliation — and the failure must surface."""
    storage = await factory(tmp_path)
    await storage.upsert({"id1": row})

    module = type(storage).__module__
    reloads: list[str] = []

    def _boom(data_dict, file_name):
        raise OSError("disk full")

    monkeypatch.setattr(f"{module}.write_json", _boom)
    monkeypatch.setattr(
        f"{module}.load_json", lambda *a, **k: reloads.append("read") or {}
    )
    storage.storage_updated.value = True

    with pytest.raises(OSError, match="disk full"):
        await storage.index_done_callback()

    assert reloads == [], "reconciled a write that never landed"


@pytest.mark.parametrize(
    "factory, row",
    [
        (_make_kv, {"content": "alpha"}),
        (_make_doc_status, {"status": "processed", "file_path": "a.pdf"}),
    ],
    ids=["kv", "doc_status"],
)
async def test_a_failed_flag_clear_is_not_reported_as_a_failed_write(
    tmp_path, monkeypatch, caplog, factory, row
):
    """A publication failure must not be raised as a save failure.

    ``on_committed`` runs only after ``write_json`` succeeded, so an exception
    out of ``clear_all_update_flags`` means the file is already on disk. Letting
    it propagate reported a durable write as a lost one, and every caller
    inherited that: ``_insert_done`` marked a document FAILED whose rows are
    persisted, and ``utils_graph``'s deletion paths turned a chunk-tracking
    cleanup they had already completed into ``fail``/500 — stranding, on the
    retry that followed, rows the sweep can no longer reach.

    What failed is only the dirty-flag reset, and it heals: the flags stay set,
    so the next commit rewrites this same snapshot and retries them.
    """
    storage = await factory(tmp_path)
    await storage.upsert({"id1": row})

    module = type(storage).__module__

    async def failing_clear_all_update_flags(namespace, workspace=None):
        raise RuntimeError("shared-storage manager is down")

    monkeypatch.setattr(
        f"{module}.clear_all_update_flags", failing_clear_all_update_flags
    )
    storage.storage_updated.value = True

    # lightrag's logger does not propagate, so caplog cannot see it otherwise.
    logger = logging.getLogger("lightrag")
    monkeypatch.setattr(logger, "propagate", True)

    with caplog.at_level(logging.ERROR, logger="lightrag"):
        await storage.index_done_callback()

    with open(storage._file_name, encoding="utf-8") as f:
        persisted = json.load(f)
    assert "id1" in persisted, "the write did not land, so this proves nothing"
    assert storage.storage_updated.value is True, (
        "the dirty flag was cleared despite the failure, so the next commit "
        "would not retry the publication"
    )
    assert any(
        "post-write bookkeeping failed" in record.getMessage()
        for record in caplog.records
    ), f"the deferred publication was not logged: {caplog.text}"


@pytest.mark.parametrize(
    "factory, row",
    [
        (_make_kv, {"content": "alpha"}),
        (_make_doc_status, {"status": "processed", "file_path": "a.pdf"}),
    ],
    ids=["kv", "doc_status"],
)
async def test_a_broken_sink_cannot_turn_a_landed_write_into_a_failure(
    tmp_path, monkeypatch, factory, row
):
    """The bookkeeping-failure diagnostic is past the point of no return.

    The file is published by the time that handler runs, so a broken logging
    handler, formatter or output target must not escape it — the caller would
    then hear that a durable write never happened. For ``JsonDocStatusStorage``
    that caller is ``upsert``, which flushes synchronously precisely to make the
    recovery anchor durable before it returns.

    Fix-proof: call ``logger.error`` directly in the handler and this raises.
    """
    storage = await factory(tmp_path)
    await storage.upsert({"id1": row})

    module = type(storage).__module__

    async def failing_clear_all_update_flags(namespace, workspace=None):
        raise RuntimeError("shared-storage manager is down")

    def log_boom(msg):
        raise RuntimeError("log sink boom")

    monkeypatch.setattr(
        f"{module}.clear_all_update_flags", failing_clear_all_update_flags
    )
    monkeypatch.setattr(f"{module}.logger.error", log_boom)
    storage.storage_updated.value = True

    await storage.index_done_callback()

    with open(storage._file_name, encoding="utf-8") as f:
        persisted = json.load(f)
    assert "id1" in persisted, "the write did not land, so this proves nothing"


@pytest.mark.parametrize(
    "factory, row",
    [
        (_make_kv, {"content": "alpha"}),
        (_make_doc_status, {"status": "processed", "file_path": "a.pdf"}),
    ],
    ids=["kv", "doc_status"],
)
async def test_a_failed_sanitize_reload_is_not_absorbed(
    tmp_path, monkeypatch, factory, row
):
    """A reconciliation failure is not a deferred publication — it must raise.

    ``self._data.clear()`` and ``self._data.update()`` are two separate RPCs on
    the shared ``Manager().dict()``, so a failure between them leaves the shared
    dict EMPTY while the file on disk holds the correct sanitized snapshot. The
    dirty flags are still set, so the next flush would write that empty dict
    straight over a good file and lose every row in the namespace.

    Absorbing it — as the dirty-flag clear beside it legitimately is — would make
    that loss silent. Nothing about it heals: the shared in-memory view is what a
    later flush PUBLISHES, so the failure has to reach a caller.

    Fix-proof: drop the ``reconcile_failure`` guard from the handler and this
    returns normally with the store emptied.
    """
    storage = await factory(tmp_path)
    await storage.upsert({"id1": row})

    module = type(storage).__module__

    class _BrokenUpdate(dict):
        def update(self, *args, **kwargs):
            raise RuntimeError("shared-dict RPC failed")

    def _sanitizing_write(data_dict, file_name):
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump({"id1": dict(row, content_was="sanitized")}, f)
        return True  # sanitization happened -> caller must reload

    monkeypatch.setattr(f"{module}.write_json", _sanitizing_write)
    storage._data = _BrokenUpdate(storage._data)
    storage.storage_updated.value = True

    with pytest.raises(RuntimeError, match="shared-dict RPC failed"):
        await storage.index_done_callback()

    # The file is the correct sanitized snapshot; it is memory that is now
    # unreliable, which is why the recovery is a restart, not a retry.
    with open(storage._file_name, encoding="utf-8") as f:
        assert json.load(f)["id1"]["content_was"] == "sanitized"
    assert dict(storage._data) == {}, (
        "the scenario under test is the emptied shared dict; if it is intact "
        "this test proves nothing"
    )
