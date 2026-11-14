"""Deletion must distinguish absent snapshots from inaccessible snapshots."""

from contextlib import contextmanager
import os
from pathlib import Path
import sys
from unittest.mock import AsyncMock

import numpy as np
import pytest

from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.utils import EmbeddingFunc

pytestmark = pytest.mark.offline


@pytest.fixture(params=["nano", "faiss"])
async def seeded(request, tmp_path):
    if request.param == "nano":
        pytest.importorskip("nano_vectordb")
        import lightrag.kg.nano_vector_db_impl as module

        cls = module.NanoVectorDBStorage
    else:
        pytest.importorskip("faiss")
        import lightrag.kg.faiss_impl as module

        cls = module.FaissVectorDBStorage

    async def embed(texts, **kwargs):
        return np.ones((len(texts), 8), dtype=np.float32)

    initialize_share_data()
    storage = cls(
        namespace="drop_errors",
        workspace="ws",
        global_config={
            "working_dir": str(tmp_path),
            "embedding_batch_num": 32,
            "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2},
        },
        embedding_func=EmbeddingFunc(embedding_dim=8, func=embed),
        meta_fields={"content"},
    )
    try:
        await storage.initialize()
        await storage.upsert({"old": {"content": "existing row"}})
        await storage.index_done_callback()
        paths = (
            [storage._client_file_name]
            if request.param == "nano"
            else [
                storage._faiss_index_file,
                storage._meta_file,
                storage._vector_space_file,
            ]
        )
        yield storage, module, [Path(p) for p in paths]
    finally:
        # Failed-drop tests intentionally retain data. Do not flush it at teardown.
        finalize_share_data()


@pytest.mark.parametrize("denied", [False, True])
async def test_drop_does_not_trust_a_false_exists(seeded, monkeypatch, denied):
    storage, module, paths = seeded
    originals = {p: p.read_bytes() for p in paths}
    await storage.upsert({"pending": {"content": "not committed"}})
    notified = AsyncMock()
    real_exists, real_remove = os.path.exists, os.remove
    names = {str(p) for p in paths}

    def exists(path):
        return False if str(path) in names else real_exists(path)

    attempted = []

    def remove(path):
        attempted.append(str(path))
        if denied and str(path) in names:
            raise PermissionError(13, "deletion denied", str(path))
        return real_remove(path)

    with monkeypatch.context() as m:
        m.setattr(os.path, "exists", exists)
        m.setattr(os, "remove", remove)
        m.setattr(module, "set_all_update_flags", notified)
        result = await storage.drop()
    assert str(paths[0]) in attempted
    if denied:
        assert result["status"] == "error"
        assert {p: p.read_bytes() for p in paths} == originals
        assert "pending" in storage._pending_upserts
        notified.assert_not_awaited()
    else:
        assert result["status"] == "success"
        assert not any(p.exists() for p in paths)
        assert not storage._pending_upserts
        notified.assert_awaited_once()


async def test_drop_accepts_already_missing_files(seeded):
    storage, _, paths = seeded
    for path in paths:
        path.unlink()
    assert (await storage.drop())["status"] == "success"
    assert (await storage.drop())["status"] == "success"


@contextmanager
def deny_shared_delete(path):
    """Hold a native handle that permits reads but not deletion."""
    import ctypes
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.CreateFileW(str(path), 0x80000000, 1, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        yield
    finally:
        if not kernel.CloseHandle(handle):
            raise ctypes.WinError(ctypes.get_last_error())


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows deletion semantics")
async def test_windows_occupied_data_file_refuses_then_retries(seeded, monkeypatch):
    storage, module, paths = seeded
    notified = AsyncMock()
    monkeypatch.setattr(module, "set_all_update_flags", notified)
    with deny_shared_delete(paths[0]):
        assert (await storage.drop())["status"] == "error"
        assert all(p.exists() for p in paths)
        notified.assert_not_awaited()
    assert (await storage.drop())["status"] == "success"
    assert not any(p.exists() for p in paths)


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows deletion semantics")
async def test_windows_readonly_data_file_refuses_then_retries(seeded):
    storage, _, paths = seeded
    paths[0].chmod(0o400)
    try:
        assert (await storage.drop())["status"] == "error"
        assert all(p.exists() for p in paths)
    finally:
        paths[0].chmod(0o600)
    assert (await storage.drop())["status"] == "success"
    assert not any(p.exists() for p in paths)


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows deletion semantics")
async def test_windows_faiss_partial_drop_and_marker_cleanup(seeded, monkeypatch):
    storage, module, paths = seeded
    if len(paths) != 3:
        pytest.skip("Faiss has multiple files")
    notified = AsyncMock()
    monkeypatch.setattr(module, "set_all_update_flags", notified)
    with deny_shared_delete(paths[1]):
        assert (await storage.drop())["status"] == "error"
        assert not paths[0].exists()
        assert paths[1].exists()
        assert storage._index.ntotal == 1
        notified.assert_not_awaited()
    # Once both data files are gone, marker removal failure is only diagnostic.
    with deny_shared_delete(paths[2]):
        assert (await storage.drop())["status"] == "success"
        assert not paths[1].exists()
        assert paths[2].exists()
        notified.assert_awaited_once()
    assert (await storage.drop())["status"] == "success"
    assert not any(p.exists() for p in paths)
