"""RPC-count regression tests for the JSON storage persistence snapshot.

``index_done_callback`` snapshots the whole shared namespace before writing it
to disk. Under multi-worker mode ``self._data`` is a ``multiprocessing.Manager``
DictProxy, so the snapshot method matters: ``DictProxy.copy()`` is a single
Manager RPC, whereas ``dict(proxy)`` walks the mapping protocol and fetches
every value with its own RPC. These tests pin the storage to exactly one
``copy()`` and zero per-key reads via a counting DictProxy stand-in (the same
technique as ``tests/kg/test_reservation_primitives.py``), plus a real-Manager
end-to-end check that the on-disk content is correct with a genuine proxy.
"""

import json

import pytest

from lightrag.base import DocStatus
from lightrag.kg.json_doc_status_impl import JsonDocStatusStorage
from lightrag.kg.json_kv_impl import JsonKVStorage
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.namespace import NameSpace

pytestmark = pytest.mark.offline


class _CountingDict(dict):
    """DictProxy-shaped fake that counts remote-style method calls.

    ``copy()`` returns a plain ``dict`` (as a real ``DictProxy.copy()`` marshals
    a plain dict back to the client), so any ``len``/iteration the caller does on
    the snapshot is NOT counted against the proxy — only proxy-level access is.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.copy_calls = 0
        self.getitem_calls = 0
        self.keys_calls = 0
        self.contains_calls = 0

    def copy(self):
        self.copy_calls += 1
        return dict(self)

    def __getitem__(self, key):
        self.getitem_calls += 1
        return super().__getitem__(key)

    def keys(self):
        self.keys_calls += 1
        return super().keys()

    def __contains__(self, key):
        self.contains_calls += 1
        return super().__contains__(key)


class _DummyEmbeddingFunc:
    embedding_dim = 1
    max_token_size = 1

    async def __call__(self, texts, **kwargs):
        return [[0.0] for _ in texts]


def _doc(status: str, file_path: str) -> dict:
    return {
        "content_summary": f"{status} summary",
        "content_length": 10,
        "file_path": file_path,
        "status": status,
        "created_at": "2024-01-01T00:00:00+00:00",
        "updated_at": "2024-01-01T00:00:00+00:00",
        "metadata": {},
        "error_msg": None,
        "chunks_list": [],
    }


@pytest.fixture
def single_process_shared_data():
    finalize_share_data()
    initialize_share_data(1)
    yield
    finalize_share_data()


# ---------------------------------------------------------------------------
# Counting-proxy assertions: exactly one copy(), zero per-key reads
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kv_index_done_callback_snapshots_with_single_copy(
    tmp_path, single_process_shared_data
):
    storage = JsonKVStorage(
        namespace=NameSpace.KV_STORE_TEXT_CHUNKS,
        global_config={"working_dir": str(tmp_path)},
        embedding_func=_DummyEmbeddingFunc(),
        workspace="test",
    )
    await storage.initialize()

    counting = _CountingDict({f"chunk-{i}": {"content": str(i)} for i in range(50)})
    storage._data = counting
    storage.storage_updated.value = True

    await storage.index_done_callback()

    assert counting.copy_calls == 1
    assert counting.getitem_calls == 0
    assert counting.keys_calls == 0

    with open(storage._file_name) as f:
        on_disk = json.load(f)
    assert len(on_disk) == 50


@pytest.mark.asyncio
async def test_doc_status_index_done_callback_snapshots_with_single_copy(
    tmp_path, single_process_shared_data
):
    storage = JsonDocStatusStorage(
        namespace=NameSpace.DOC_STATUS,
        global_config={"working_dir": str(tmp_path)},
        embedding_func=_DummyEmbeddingFunc(),
        workspace="test",
    )
    await storage.initialize()

    counting = _CountingDict(
        {f"doc-{i}": _doc("processed", f"{i}.pdf") for i in range(20)}
    )
    storage._data = counting
    storage.storage_updated.value = True

    await storage.index_done_callback()

    assert counting.copy_calls == 1
    assert counting.getitem_calls == 0
    assert counting.keys_calls == 0


# ---------------------------------------------------------------------------
# Real Manager end-to-end: genuine DictProxy.copy() persists correct content
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kv_index_done_callback_real_manager_persists(tmp_path):
    # A real 2-worker Manager makes self._data a genuine DictProxy, exercising
    # the actual copy() RPC path (not the single-process plain-dict shortcut).
    finalize_share_data()
    initialize_share_data(2)
    try:
        storage = JsonKVStorage(
            namespace=NameSpace.KV_STORE_TEXT_CHUNKS,
            global_config={"working_dir": str(tmp_path)},
            embedding_func=_DummyEmbeddingFunc(),
            workspace="test",
        )
        await storage.initialize()

        await storage.upsert(
            {
                "chunk-a": {"content": "alpha"},
                "chunk-b": {"content": "beta"},
            }
        )
        await storage.index_done_callback()

        with open(storage._file_name) as f:
            on_disk = json.load(f)
        assert set(on_disk.keys()) == {"chunk-a", "chunk-b"}
        assert on_disk["chunk-a"]["content"] == "alpha"
        assert on_disk["chunk-b"]["content"] == "beta"
    finally:
        finalize_share_data()


@pytest.mark.asyncio
async def test_doc_status_upsert_real_manager_persists(tmp_path):
    finalize_share_data()
    initialize_share_data(2)
    try:
        storage = JsonDocStatusStorage(
            namespace=NameSpace.DOC_STATUS,
            global_config={"working_dir": str(tmp_path)},
            embedding_func=_DummyEmbeddingFunc(),
            workspace="test",
        )
        await storage.initialize()

        # DocStatus.upsert flushes synchronously via index_done_callback.
        await storage.upsert({"doc-1": _doc("processed", "a.pdf")})

        with open(storage._file_name) as f:
            on_disk = json.load(f)
        assert on_disk["doc-1"]["status"] == DocStatus.PROCESSED.value
        assert on_disk["doc-1"]["file_path"] == "a.pdf"
    finally:
        finalize_share_data()
