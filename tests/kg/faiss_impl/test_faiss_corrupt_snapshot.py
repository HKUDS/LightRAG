"""A corrupt Faiss pair fails loud instead of degrading to an empty index.

``_load_faiss_index`` used to answer EVERY exception with a fresh
``IndexFlatIP``, an empty ``_id_to_meta`` and ``_vector_space_certified =
True``. That was silent data loss on a delay: the process served the store as
empty, and its next ``index_done_callback`` saved that emptiness over the very
bytes it had failed to read, stamping a fresh embedding baseline on top. These
tests pin the refusal — at startup, on a reader reload, and (the one that
justifies the change) at the save that used to overwrite the corrupt file.

An I/O failure is NOT corruption and must keep propagating as itself: the tool
may back up and drop a corrupt container, and doing that to a pair that was
merely unreadable for a moment would destroy readable data.
"""

from __future__ import annotations

import builtins
import os
import stat
import sys
from pathlib import Path

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")

from lightrag.exceptions import CorruptStorageSnapshotError  # noqa: E402
from lightrag.kg.faiss_impl import FaissVectorDBStorage  # noqa: E402
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


def _make_storage(tmp_path) -> FaissVectorDBStorage:
    return FaissVectorDBStorage(
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


async def _seeded_storage(tmp_path) -> FaissVectorDBStorage:
    storage = _make_storage(tmp_path)
    await storage.initialize()
    await storage.upsert({"v1": {"content": "hello"}})
    await storage.index_done_callback()
    assert Path(storage._faiss_index_file).exists()
    assert Path(storage._meta_file).exists()
    return storage


def _truncate(path: str) -> None:
    payload = Path(path).read_bytes()
    Path(path).write_bytes(payload[: len(payload) // 2])


@pytest.mark.asyncio
@pytest.mark.parametrize("target", ["index", "meta"])
async def test_initialize_refuses_a_corrupt_pair(tmp_path, target):
    """Startup names the unreadable file and the recovery path, and keeps it."""
    storage = await _seeded_storage(tmp_path)
    corrupt_file = (
        storage._faiss_index_file if target == "index" else storage._meta_file
    )
    _truncate(corrupt_file)
    on_disk = Path(corrupt_file).read_bytes()

    fresh = _make_storage(tmp_path)
    with pytest.raises(CorruptStorageSnapshotError) as exc_info:
        await fresh.initialize()

    assert "lightrag-rebuild-vdb" in str(exc_info.value)
    assert exc_info.value.__cause__ is not None
    # The DIAGNOSIS names the file that failed, not the one read first. A
    # truncated `.meta.json` fails in `json.load`, whose `JSONDecodeError`
    # carries no `filename`: reporting the intact `.index` would send the
    # operator to inspect a healthy file.
    assert exc_info.value.container == corrupt_file
    # Every file of the pair is offered to the offline tool, not just the
    # one that failed to parse: a drop removes all three.
    assert set(exc_info.value.artifacts) == {
        storage._faiss_index_file,
        storage._meta_file,
        storage._vector_space_file,
    }
    assert Path(corrupt_file).read_bytes() == on_disk


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload", ["{not json at all", '["a list, not an object"]', '{"x": {}}']
)
async def test_unparsable_metadata_names_the_metadata_file(tmp_path, payload):
    """Three ways the metadata can be unreadable while the index is fine.

    Invalid JSON, valid JSON of the wrong shape, and a row key that is not a
    faiss id. None of the exceptions they raise carries a filename, so the
    file being read is what has to be remembered -- otherwise all three
    report the `.index`, which parsed.
    """
    storage = await _seeded_storage(tmp_path)
    Path(storage._meta_file).write_text(payload, encoding="utf-8")
    index_bytes = Path(storage._faiss_index_file).read_bytes()

    fresh = _make_storage(tmp_path)
    with pytest.raises(CorruptStorageSnapshotError) as exc_info:
        await fresh.initialize()

    assert exc_info.value.container == storage._meta_file
    assert set(exc_info.value.artifacts) == {
        storage._faiss_index_file,
        storage._meta_file,
        storage._vector_space_file,
    }
    # Refusing preserves both files for the offline rebuild.
    assert Path(storage._faiss_index_file).read_bytes() == index_bytes
    assert Path(storage._meta_file).read_text(encoding="utf-8") == payload


@pytest.mark.asyncio
async def test_initialize_refuses_an_index_whose_metadata_vanished(tmp_path):
    """The metadata is the commit marker; its absence contradicts the index."""
    storage = await _seeded_storage(tmp_path)
    Path(storage._meta_file).unlink()

    fresh = _make_storage(tmp_path)
    with pytest.raises(CorruptStorageSnapshotError) as exc_info:
        await fresh.initialize()

    assert exc_info.value.container == storage._meta_file
    assert "missing beside a present index" in str(exc_info.value)
    assert Path(storage._faiss_index_file).exists()


@pytest.mark.asyncio
async def test_a_save_never_publishes_emptiness_over_a_corrupt_pair(tmp_path):
    """The defect this change exists for.

    A peer leaves the pair unreadable, this process reloads on its next
    commit, and the commit must ABORT with the corrupt bytes intact. The old
    degradation reloaded an empty index and then saved it, destroying the rows
    the unreadable file still held.
    """
    storage = await _seeded_storage(tmp_path)
    _truncate(storage._faiss_index_file)
    index_bytes = Path(storage._faiss_index_file).read_bytes()
    meta_bytes = Path(storage._meta_file).read_bytes()
    storage.storage_updated.value = True

    await storage.upsert({"v2": {"content": "world"}})
    with pytest.raises(CorruptStorageSnapshotError):
        await storage.index_done_callback()

    assert Path(storage._faiss_index_file).read_bytes() == index_bytes
    assert Path(storage._meta_file).read_bytes() == meta_bytes
    # Still refused on every later access, so the empty index the failed
    # reload left behind is never served.
    with pytest.raises(CorruptStorageSnapshotError):
        await storage.query("hello", top_k=1, query_embedding=[1.0] * DIM)


@pytest.mark.asyncio
async def test_a_read_error_is_not_reported_as_corruption(tmp_path, monkeypatch):
    """An unreadable pair must not become a droppable one."""
    storage = await _seeded_storage(tmp_path)
    real_open = builtins.open

    def _deny(path, *args, **kwargs):
        if str(path) == storage._meta_file:
            raise PermissionError(13, "Permission denied", str(path))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", _deny)
    fresh = _make_storage(tmp_path)
    with pytest.raises(PermissionError):
        await fresh.initialize()


@pytest.mark.asyncio
async def test_offline_recovery_backs_up_every_file_and_rebuilds(tmp_path):
    """The tool preserves the whole set under one token, then re-provisions."""
    from lightrag.tools.rebuild_vdb import RebuildTool

    storage = await _seeded_storage(tmp_path)
    _truncate(storage._faiss_index_file)
    originals = {
        path: Path(path).read_bytes()
        for path in (
            storage._faiss_index_file,
            storage._meta_file,
            storage._vector_space_file,
        )
    }
    refused = _make_storage(tmp_path)
    with pytest.raises(CorruptStorageSnapshotError) as caught:
        await refused.initialize()

    tool = RebuildTool()
    tool.entities_vdb = refused
    tool.corrupt_vdbs = {"entities": caught.value}
    tool.incompatible_vdbs = {"entities": str(caught.value)}
    await tool.recover_incompatible(["entities"])

    backups = sorted(Path(storage._faiss_index_file).parent.glob("*.corrupt-*"))
    assert len(backups) == 3
    tokens = {path.name.rsplit("-", 1)[1] for path in backups}
    assert len(tokens) == 1
    restored = {path.name.rsplit(".corrupt-", 1)[0] for path in backups}
    assert restored == {Path(path).name for path in originals}
    for backup in backups:
        original = backup.parent / backup.name.rsplit(".corrupt-", 1)[0]
        assert backup.read_bytes() == originals[str(original)]

    await refused.upsert({"restored": {"content": "hello"}})
    await refused.index_done_callback()
    fresh = _make_storage(tmp_path)
    await fresh.initialize()
    assert (await fresh.get_by_id("restored"))["content"] == "hello"


@pytest.mark.asyncio
async def test_offline_recovery_skips_a_file_the_torn_pair_never_had(tmp_path):
    """A missing member is skipped; the ones that exist are still preserved."""
    from lightrag.tools.rebuild_vdb import RebuildTool

    storage = await _seeded_storage(tmp_path)
    Path(storage._meta_file).unlink()
    refused = _make_storage(tmp_path)
    with pytest.raises(CorruptStorageSnapshotError) as caught:
        await refused.initialize()

    tool = RebuildTool()
    tool.entities_vdb = refused
    tool.corrupt_vdbs = {"entities": caught.value}
    tool.incompatible_vdbs = {"entities": str(caught.value)}
    await tool.recover_incompatible(["entities"])

    backups = sorted(Path(storage._faiss_index_file).parent.glob("*.corrupt-*"))
    preserved = {path.name.rsplit(".corrupt-", 1)[0] for path in backups}
    assert preserved == {
        Path(storage._faiss_index_file).name,
        Path(storage._vector_space_file).name,
    }
    assert Path(storage._meta_file).name not in preserved


@pytest.mark.asyncio
async def test_a_marker_that_exists_but_cannot_be_read_is_not_absent(tmp_path):
    """The one file where "unreadable" used to mean "never recorded".

    ``_read_vector_space_file`` answered every failure with ``(None, None)``,
    which is also the answer for a store written before the marker existed --
    and absent evidence never refuses. So a truncated ``.space.json`` let a
    same-dimension model swap attach silently, on the very file recorded to
    catch it, and the next save re-stamped the current model over rows it
    could not vouch for.
    """
    storage = await _seeded_storage(tmp_path)
    marker = Path(storage._vector_space_file)
    assert marker.exists()
    marker.write_bytes(b'{"model": "bge-m3"')

    fresh = _make_storage(tmp_path)
    with pytest.raises(CorruptStorageSnapshotError) as exc_info:
        await fresh.initialize()

    assert exc_info.value.container == storage._vector_space_file
    assert marker.read_bytes() == b'{"model": "bge-m3"'


@pytest.mark.asyncio
async def test_a_marker_that_was_never_written_still_attaches(tmp_path):
    """Absent evidence still never refuses -- the pre-marker store loads."""
    storage = await _seeded_storage(tmp_path)
    Path(storage._vector_space_file).unlink()

    fresh = _make_storage(tmp_path)
    await fresh.initialize()
    assert (await fresh.get_by_id("v1"))["content"] == "hello"


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="chmod(0) cannot revoke read access on Windows; it sets the "
    "read-only attribute, so the index would still load and this test would "
    "fail on a platform whose behaviour it does not describe. The Windows "
    "equivalent needs a denying ACL or a sharing lock.",
)
@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root bypasses the file mode this test relies on",
)
@pytest.mark.asyncio
async def test_an_unreadable_index_is_not_labelled_corrupt(tmp_path):
    """A HEALTHY index the OS would not open must not become a drop target.

    ``faiss.read_index`` runs in C++ and answers a permission failure with the
    same bare ``RuntimeError`` it uses for a truncated file, so classifying on
    the exception type alone registered an intact index as recoverable
    corruption. The tool would then offer to back it up, drop it and
    re-embed — paying for embeddings to replace rows that were never lost, and
    ending with however many rows the sources still hold.
    """
    storage = await _seeded_storage(tmp_path)
    index = Path(storage._faiss_index_file)
    healthy = index.read_bytes()
    index.chmod(0o000)

    fresh = _make_storage(tmp_path)
    try:
        with pytest.raises(PermissionError):
            await fresh.initialize()
    finally:
        index.chmod(0o600)

    # Intact all along, and readable again once the fault clears.
    assert index.read_bytes() == healthy
    recovered = _make_storage(tmp_path)
    await recovered.initialize()
    assert (await recovered.get_by_id("v1"))["content"] == "hello"


@pytest.mark.asyncio
async def test_a_truncated_index_is_still_labelled_corrupt(tmp_path):
    """The probe must not blunt the refusal it guards: readable bytes that
    are not a Faiss index are corruption, exactly as before."""
    storage = await _seeded_storage(tmp_path)
    _truncate(storage._faiss_index_file)

    fresh = _make_storage(tmp_path)
    with pytest.raises(CorruptStorageSnapshotError):
        await fresh.initialize()


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="st_mode carries no ACL on Windows, so a 0600 assertion there "
    "would prove nothing about who can read the backup; the Windows "
    "restriction is icacls and needs its own ACL check",
)
@pytest.mark.asyncio
async def test_a_backup_is_never_more_readable_than_what_it_copies(tmp_path):
    """Backups hold the documents and metadata of the store they copy and are
    kept indefinitely, so their mode must not come from the process umask."""
    from lightrag.tools.rebuild_vdb import RebuildTool

    storage = await _seeded_storage(tmp_path)
    Path(storage._faiss_index_file).chmod(0o600)
    Path(storage._meta_file).chmod(0o600)
    _truncate(storage._faiss_index_file)

    refused = _make_storage(tmp_path)
    with pytest.raises(CorruptStorageSnapshotError) as caught:
        await refused.initialize()
    tool = RebuildTool()
    tool.entities_vdb = refused
    tool.corrupt_vdbs = {"entities": caught.value}
    tool.incompatible_vdbs = {"entities": str(caught.value)}
    # Pin the permissive-but-common umask rather than inherit the runner's:
    # a 077 runner would make a mode derived from the umask look correct and
    # this test pass on an environment instead of on the behaviour. The
    # process-wide set is safe here — pytest runs this serially, and xdist
    # workers are separate processes.
    saved_umask = os.umask(0o022)
    try:
        await tool.recover_incompatible(["entities"])
    finally:
        os.umask(saved_umask)

    backups = list(Path(storage._faiss_index_file).parent.glob("*.corrupt-*"))
    assert backups
    for backup in backups:
        mode = stat.S_IMODE(backup.stat().st_mode)
        assert mode == 0o600, f"{backup.name} is {oct(mode)}"
