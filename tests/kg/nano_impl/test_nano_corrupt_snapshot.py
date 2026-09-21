"""A corrupt NanoVectorDB snapshot fails loud with an actionable error (#2441).

A truncated or binary-garbage ``vdb_*.json`` used to surface as a bare
``json.JSONDecodeError`` ("Unterminated string starting at ...") with no file
path and no recovery path. The storage must instead refuse to attach with a
typed ``CorruptStorageSnapshotError`` that names the corrupt file and explains
how to rebuild — and it must do so both at startup (``initialize``) and when a
reader reloads a snapshot a peer process left corrupt.

The refusal covers every way the payload can fail to come back, not only the
JSON layer: ``NanoVectorDB`` decodes a base64 matrix and reshapes it after
``json.load`` returns, and a file that fails there is no more readable than
one that fails to parse. Catching only the JSON errors left those modes
raising the bare library error this test module exists to abolish.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

nano_vectordb = pytest.importorskip("nano_vectordb")

from lightrag.exceptions import (  # noqa: E402
    CorruptStorageSnapshotError,
    VectorSpaceMismatchError,
)
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


# Every payload here is structurally valid JSON that NanoVectorDB still cannot
# turn back into a snapshot, so each one escaped a JSON-only catch. The error
# each raises is named for the record; the test pins the refusal, not the
# library's internals.
UNREADABLE_PAYLOADS = {
    "damaged-base64-matrix": b'{"embedding_dim": 8, "data": [], "matrix": "!!not-b64!"}',
    "matrix-shorter-than-a-row": b'{"embedding_dim": 8, "data": [], "matrix": "AAAAAA=="}',
    "matrix-not-a-string": b'{"embedding_dim": 8, "data": [], "matrix": 5}',
    "no-matrix-at-all": b"{}",
    "not-an-object": b"[1, 2, 3]",
    "json-null": b"null",
}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload", UNREADABLE_PAYLOADS.values(), ids=UNREADABLE_PAYLOADS
)
async def test_initialize_refuses_a_snapshot_it_cannot_reconstitute(tmp_path, payload):
    """Valid JSON is not the bar — reconstituting the snapshot is."""
    storage = await _seeded_storage(tmp_path)
    client_file = storage._client_file_name
    Path(client_file).write_bytes(payload)

    fresh = _make_storage(tmp_path)
    with pytest.raises(CorruptStorageSnapshotError) as exc_info:
        await fresh.initialize()

    assert client_file in str(exc_info.value)
    assert "lightrag-rebuild-vdb" in str(exc_info.value)
    assert exc_info.value.__cause__ is not None
    assert Path(client_file).read_bytes() == payload


@pytest.mark.asyncio
async def test_a_dimension_mismatch_is_not_reported_as_corruption(tmp_path):
    """The widened catch must not swallow the refusal that owns dimensions.

    ``NanoVectorDB`` asserts on the dimension, and the branch below the
    corruption catch turns that into ``VectorSpaceMismatchError``. A catch
    wide enough to take ``ValueError`` must still leave ``AssertionError``
    to it, or a model swap starts reading as a corrupt file and the operator
    is sent to back up and drop a container that is perfectly readable.
    """
    storage = await _seeded_storage(tmp_path)

    wider = NanoVectorDBStorage(
        namespace="test_vectors",
        workspace="ws",
        global_config=storage.global_config,
        embedding_func=EmbeddingFunc(
            embedding_dim=DIM * 2, max_token_size=512, func=_embed
        ),
        meta_fields={"content"},
    )
    with pytest.raises(VectorSpaceMismatchError):
        await wider.initialize()


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="chmod(0) sets the read-only attribute on Windows rather than "
    "revoking read access, so the snapshot would still load",
)
@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root bypasses the file mode this test relies on",
)
@pytest.mark.asyncio
async def test_an_unreadable_snapshot_is_not_labelled_corrupt(tmp_path):
    """A healthy file the OS would not open is not a drop target.

    This backend reads through Python, so a permission failure arrives as
    ``PermissionError`` and falls outside the caught set by construction --
    unlike Faiss, whose C++ layer reports the same fault as ``RuntimeError``
    and needs an explicit probe. Pinned anyway: widening this catch to
    ``Exception`` would register an intact snapshot as recoverable
    corruption, and the tool would back it up, drop it and re-embed.
    """
    storage = await _seeded_storage(tmp_path)
    path = Path(storage._client_file_name)
    healthy = path.read_bytes()
    path.chmod(0o000)

    fresh = _make_storage(tmp_path)
    try:
        with pytest.raises(PermissionError):
            await fresh.initialize()
    finally:
        path.chmod(0o600)

    assert path.read_bytes() == healthy
    recovered = _make_storage(tmp_path)
    await recovered.initialize()
    assert (await recovered.get_by_id("v1"))["content"] == "hello"
