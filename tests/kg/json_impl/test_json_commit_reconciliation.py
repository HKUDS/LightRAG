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
