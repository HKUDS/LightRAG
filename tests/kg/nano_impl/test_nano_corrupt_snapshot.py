"""A corrupt NanoVectorDB snapshot fails loud with an actionable error (#2441).

A truncated or binary-garbage ``vdb_*.json`` used to surface as a bare
``json.JSONDecodeError`` ("Unterminated string starting at ...") with no file
path and no recovery path. The storage must instead refuse to attach with a
typed ``CorruptStorageSnapshotError`` that names the corrupt file and explains
how to rebuild — and it must do so both at startup (``initialize``) and when a
reader reloads a snapshot a peer process left corrupt.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

nano_vectordb = pytest.importorskip("nano_vectordb")

from lightrag.exceptions import CorruptStorageSnapshotError  # noqa: E402
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


def _truncate(path: str) -> None:
    original = Path(path).read_bytes()
    Path(path).write_bytes(original[: len(original) // 2])


def _write_binary_garbage(path: str) -> None:
    Path(path).write_bytes(b"\xff\xfe\x00\x01 not json at all \x89\x8a")


@pytest.mark.asyncio
@pytest.mark.parametrize("corrupt", [_truncate, _write_binary_garbage])
async def test_initialize_refuses_a_corrupt_snapshot(tmp_path, corrupt):
    storage = await _seeded_storage(tmp_path)
    client_file = storage._client_file_name
    corrupt(client_file)

    fresh = _make_storage(tmp_path)
    with pytest.raises(CorruptStorageSnapshotError) as exc_info:
        await fresh.initialize()

    message = str(exc_info.value)
    assert client_file in message
    assert "lightrag-rebuild-vdb" in message
    assert isinstance(
        exc_info.value.__cause__, (json.JSONDecodeError, UnicodeDecodeError)
    )


@pytest.mark.asyncio
async def test_reader_reload_refuses_a_snapshot_a_peer_left_corrupt(tmp_path):
    storage = await _seeded_storage(tmp_path)
    client_file = storage._client_file_name

    # A peer process commits a corrupt snapshot: the file changed and the
    # reload notification fired. The next read must fail loud, not serve.
    _truncate(client_file)
    storage.storage_updated.value = True

    with pytest.raises(CorruptStorageSnapshotError) as exc_info:
        await storage.query("hello", top_k=1, query_embedding=[1.0] * DIM)

    assert client_file in str(exc_info.value)


@pytest.mark.asyncio
async def test_offline_recovery_backs_up_and_rebuilds_real_nano(tmp_path):
    from lightrag.tools.rebuild_vdb import RebuildTool

    storage = await _seeded_storage(tmp_path)
    _truncate(storage._client_file_name)
    original = Path(storage._client_file_name).read_bytes()
    refused = _make_storage(tmp_path)
    with pytest.raises(CorruptStorageSnapshotError) as caught:
        await refused.initialize()
    tool = RebuildTool()
    tool.entities_vdb = refused
    tool.corrupt_vdbs = {"entities": caught.value}
    tool.incompatible_vdbs = {"entities": str(caught.value)}
    await tool.recover_incompatible(["entities"])
    backups = list(Path(refused._client_file_name).parent.glob("*.corrupt-*"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == original
    await refused.upsert({"restored": {"content": "hello"}})
    await refused.index_done_callback()
    fresh = _make_storage(tmp_path)
    await fresh.initialize()
    assert (await fresh.get_by_id("restored"))["content"] == "hello"
    assert backups[0].read_bytes() == original
