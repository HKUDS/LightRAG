"""Embedding-space provenance on the two file-backed vector stores.

FAISS and Nano share a shape no server-backed backend has: the refusal used to
come from the CONSTRUCTOR. FAISS loaded its index in ``__post_init__`` and Nano
built its ``NanoVectorDB`` there, and ``NanoVectorDB.__init__`` asserts on a
dimension mismatch. So ``lightrag-rebuild-vdb`` could not even *create* the
storage object, let alone call ``drop()`` on it, and the operator had to delete
the files by hand — the out-of-band step this work exists to remove.

Both now load in ``initialize()``, after the flag and the lock are in place, so
the object survives its own refusal and stays droppable.

Rules pinned here (see ``docs/design/VectorSpaceProvenance.md``):

- a store written by this backend records the model and dimension;
- attach refuses a foreign model or dimension, with a TYPED exception;
- absent evidence never refuses, and attach is not a backfill;
- a refused instance is constructible, droppable, and drop converges;
- the marker is never a row, so no query can return it.

These run against real files in a tmp directory rather than mocks: the whole
point is what survives on disk between one instance and the next.
"""

import asyncio
import json
import os
from contextlib import asynccontextmanager
from unittest.mock import patch

import numpy as np
import pytest

from lightrag.exceptions import VectorSpaceMismatchError
from lightrag.kg.vector_space import VECTOR_SPACE_DIM_KEY, VECTOR_SPACE_MODEL_KEY

pytestmark = pytest.mark.offline


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class _Embed:
    max_token_size = 512

    def __init__(self, model_name="model-a", embedding_dim=8):
        self.model_name = model_name
        self.embedding_dim = embedding_dim

    async def __call__(self, texts, **kwargs):
        # Deterministic, distinct per text, unit-norm friendly.
        out = np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
        for i, text in enumerate(texts):
            out[i][hash(text) % self.embedding_dim] = 1.0
        return out


@asynccontextmanager
async def _null_lock():
    yield


@pytest.fixture(autouse=True)
def patch_shared_storage():
    """Run initialize() without the multiprocessing manager."""
    locks: dict[tuple, asyncio.Lock] = {}

    class _Flag:
        value = False

    async def get_update_flag(namespace, workspace=None):
        return _Flag()

    def namespace_lock(namespace, workspace=None, enable_logging=False):
        return locks.setdefault((namespace, workspace), asyncio.Lock())

    targets = ("lightrag.kg.faiss_impl", "lightrag.kg.nano_vector_db_impl")
    patches = []
    for module in targets:
        patches.append(patch(f"{module}.get_update_flag", side_effect=get_update_flag))
        patches.append(
            patch(f"{module}.get_namespace_lock", side_effect=namespace_lock)
        )
        patches.append(patch(f"{module}.set_all_update_flags"))
    for p in patches:
        p.start()
    try:
        yield
    finally:
        for p in patches:
            p.stop()


def _global_config(working_dir):
    return {
        "working_dir": str(working_dir),
        "embedding_batch_num": 10,
        "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2},
    }


class _Backend:
    """What each backend needs so one set of tests can drive both."""

    def __init__(self, name):
        self.name = name

    def storage(self, working_dir, embed):
        if self.name == "faiss":
            faiss_impl = pytest.importorskip("lightrag.kg.faiss_impl")
            cls = faiss_impl.FaissVectorDBStorage
        else:
            nano_impl = pytest.importorskip("lightrag.kg.nano_vector_db_impl")
            cls = nano_impl.NanoVectorDBStorage
        return cls(
            namespace="entities",
            workspace="",
            global_config=_global_config(working_dir),
            embedding_func=embed,
            meta_fields={"content"},
        )

    def marker_path(self, working_dir):
        if self.name == "faiss":
            return os.path.join(working_dir, "faiss_index_entities.index.space.json")
        return os.path.join(working_dir, "vdb_entities.json")

    def read_marker(self, working_dir):
        """``(model, dim)`` as recorded on disk, or ``(None, None)``."""
        path = self.marker_path(working_dir)
        if not os.path.exists(path):
            return None, None
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        if self.name == "nano":
            payload = payload.get("additional_data") or {}
        return payload.get(VECTOR_SPACE_MODEL_KEY), payload.get(VECTOR_SPACE_DIM_KEY)


BACKENDS = [
    pytest.param(_Backend("faiss"), id="faiss"),
    pytest.param(_Backend("nano"), id="nano"),
]


async def _seed(backend, working_dir, embed):
    """Write a real store with ``embed``'s embedding space and one row."""
    storage = backend.storage(working_dir, embed)
    await storage.initialize()
    await storage.upsert({"v1": {"content": "hello"}})
    await storage.index_done_callback()
    return storage


# ---------------------------------------------------------------------------
# The marker is recorded, and is not a row
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.asyncio
async def test_a_saved_store_records_its_embedding_space(backend, tmp_path):
    await _seed(backend, tmp_path, _Embed("bge-m3", 8))
    assert backend.read_marker(tmp_path) == ("bge-m3", 8)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.asyncio
async def test_an_unknown_model_records_no_model(backend, tmp_path):
    # A recorded None would be indistinguishable from "written before the
    # marker existed", making the never-refuse rule permanent for this store.
    await _seed(backend, tmp_path, _Embed(None, 8))
    assert backend.read_marker(tmp_path) == (None, 8)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.asyncio
async def test_the_marker_is_not_a_queryable_row(backend, tmp_path):
    """It must never become a vector: a vector in the index can be recalled."""
    storage = await _seed(backend, tmp_path, _Embed("bge-m3", 8))
    hits = await storage.query("hello", top_k=10)
    assert [hit["id"] for hit in hits] == ["v1"]


# ---------------------------------------------------------------------------
# Attach
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.asyncio
async def test_attaching_refuses_a_store_built_by_another_model(backend, tmp_path):
    """The defect: same dimension, different model, same files."""
    await _seed(backend, tmp_path, _Embed("bge-m3", 8))

    storage = backend.storage(tmp_path, _Embed("e5-large", 8))
    with pytest.raises(VectorSpaceMismatchError) as excinfo:
        await storage.initialize()

    message = str(excinfo.value)
    assert "'bge-m3' -> 'e5-large'" in message
    assert "lightrag-rebuild-vdb" in message


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.asyncio
async def test_attaching_refuses_a_foreign_dimension(backend, tmp_path):
    await _seed(backend, tmp_path, _Embed("bge-m3", 8))

    storage = backend.storage(tmp_path, _Embed("bge-m3", 16))
    with pytest.raises(VectorSpaceMismatchError):
        await storage.initialize()


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.asyncio
async def test_attaching_accepts_the_same_space(backend, tmp_path):
    await _seed(backend, tmp_path, _Embed("bge-m3", 8))

    storage = backend.storage(tmp_path, _Embed("bge-m3", 8))
    await storage.initialize()
    assert await storage.get_by_id("v1") is not None


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.asyncio
async def test_a_store_predating_the_marker_is_still_servable(backend, tmp_path):
    """Absent evidence never refuses, or every pre-upgrade store is refused."""
    await _seed(backend, tmp_path, _Embed("bge-m3", 8))
    _strip_marker(backend, tmp_path)

    storage = backend.storage(tmp_path, _Embed("e5-large", 8))
    await storage.initialize()
    assert await storage.get_by_id("v1") is not None


def _strip_marker(backend, working_dir):
    """Make the store look like one written before the marker existed."""
    path = backend.marker_path(working_dir)
    if backend.name == "faiss":
        os.remove(path)
        return
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    payload.pop("additional_data", None)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.asyncio
async def test_writing_to_an_unverified_legacy_store_does_not_certify_it(
    backend, tmp_path
):
    """A save must not stamp a model onto rows this process did not write.

    The sequence: a pre-marker store written by model A, an operator who
    upgrades AND switches to a same-dimension model B in one step, and the
    absent-evidence rule letting startup proceed. If the first flush then
    records B, the lie is permanent -- every later start reads a marker that
    agrees with itself, the gate can never fire, and the adoption probe that
    is supposed to catch this case is defeated too, because it sees no
    conflict.

    Guarding the attach path alone is not enough; the save path is a backfill
    just as much.
    """
    await _seed(backend, tmp_path, _Embed("bge-m3", 8))
    _strip_marker(backend, tmp_path)

    storage = backend.storage(tmp_path, _Embed("e5-large", 8))
    await storage.initialize()
    await storage.upsert({"v2": {"content": "world"}})
    await storage.index_done_callback()

    assert backend.read_marker(tmp_path) == (None, None)

    # And the silence persists, rather than hardening into a false claim.
    again = backend.storage(tmp_path, _Embed("another-model", 8))
    await again.initialize()


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.asyncio
async def test_writing_to_an_already_marked_store_keeps_certifying_it(
    backend, tmp_path
):
    """The healthy path is unaffected: a marked store stays marked."""
    storage = await _seed(backend, tmp_path, _Embed("bge-m3", 8))
    await storage.upsert({"v2": {"content": "world"}})
    await storage.index_done_callback()
    assert backend.read_marker(tmp_path) == ("bge-m3", 8)


# ---------------------------------------------------------------------------
# A refusal must be constructible, droppable, and must converge
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.asyncio
async def test_a_refused_storage_can_be_constructed(backend, tmp_path):
    """The wedge this removes: the refusal used to come from __post_init__.

    The tool cannot drop what it cannot build, so construction must succeed
    even against a store this instance will refuse to attach to.
    """
    await _seed(backend, tmp_path, _Embed("bge-m3", 8))

    storage = backend.storage(tmp_path, _Embed("e5-large", 8))  # must not raise
    assert storage is not None


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.asyncio
async def test_a_refused_storage_can_still_drop(backend, tmp_path):
    await _seed(backend, tmp_path, _Embed("bge-m3", 8))

    storage = backend.storage(tmp_path, _Embed("e5-large", 8))
    with pytest.raises(VectorSpaceMismatchError):
        await storage.initialize()

    result = await storage.drop()
    assert result["status"] == "success"


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.asyncio
async def test_drop_then_initialize_converges(backend, tmp_path):
    """drop() + initialize() is the tool's whole recovery on these backends."""
    await _seed(backend, tmp_path, _Embed("bge-m3", 8))

    storage = backend.storage(tmp_path, _Embed("e5-large", 16))
    with pytest.raises(VectorSpaceMismatchError):
        await storage.initialize()

    assert (await storage.drop())["status"] == "success"
    await storage.initialize()

    # The rebuild's first write re-marks the store in the current space.
    await storage.upsert({"v2": {"content": "world"}})
    await storage.index_done_callback()
    assert backend.read_marker(tmp_path) == ("e5-large", 16)
    assert await storage.get_by_id("v1") is None


@pytest.mark.asyncio
async def test_a_stuck_marker_file_cannot_fail_a_completed_drop(tmp_path):
    """FAISS only: the marker is a third file, and it is past the point of no return.

    Once both authoritative files are gone every persisted vector is gone, and
    this storage's contract is that no step after that may report the completed
    destruction as an error -- ``/documents/clear`` reads that status to decide
    whether the input files are safe to delete. An orphan marker is harmless:
    with the index absent the load returns early and never reads it, and the
    next save replaces it.
    """
    backend = _Backend("faiss")
    storage = await _seed(backend, tmp_path, _Embed("bge-m3", 8))
    marker = backend.marker_path(tmp_path)
    real_remove = os.remove

    def _remove(path, *args, **kwargs):
        if str(path) == marker:
            raise PermissionError("marker file is locked")
        return real_remove(path, *args, **kwargs)

    with patch("lightrag.kg.faiss_impl.os.remove", side_effect=_remove):
        result = await storage.drop()

    assert result["status"] == "success"
    # The bookkeeping that follows the removal really did run.
    assert storage._index.ntotal == 0
    assert storage._id_to_meta == {}
    assert not os.path.exists(os.path.join(tmp_path, "faiss_index_entities.index"))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.asyncio
async def test_drop_leaves_no_marker_behind(backend, tmp_path):
    """A marker outliving its rows would refuse against an empty store."""
    storage = await _seed(backend, tmp_path, _Embed("bge-m3", 8))

    assert (await storage.drop())["status"] == "success"
    assert backend.read_marker(tmp_path) == (None, None)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.asyncio
async def test_an_unnamed_model_does_not_erase_recorded_provenance(backend, tmp_path):
    """An ordinary write must not delete a model name it cannot replace.

    Attach accepts a process with no configured model against a marked store —
    absent evidence never refuses. Writing the marker is a stronger claim: with
    no name to record, the payload is dimension-only, so certifying here would
    ERASE the model that was already established and reopen the same-dimension
    swap it was recorded to catch.
    """
    await _seed(backend, tmp_path, _Embed("bge-m3", 8))

    unnamed = backend.storage(tmp_path, _Embed(None, 8))
    await unnamed.initialize()  # accepted: absent declared evidence
    await unnamed.upsert({"v2": {"content": "world"}})
    await unnamed.index_done_callback()

    assert backend.read_marker(tmp_path)[0] == "bge-m3"

    # And the provenance still bites afterwards.
    with pytest.raises(VectorSpaceMismatchError):
        await backend.storage(tmp_path, _Embed("e5-large", 8)).initialize()


@pytest.mark.asyncio
async def test_nano_rechecks_provenance_when_reloading_a_peer_commit(tmp_path):
    """Nano only: the peer-reload path built a client without the check.

    A peer that rebuilt the namespace under a different same-dimension model
    would have been adopted and served with this process's embedder. A rolling
    embedding change is not supported, so the honest answer to finding one is
    to stop — at runtime if that is when it surfaces.
    """
    backend = _Backend("nano")
    storage = await _seed(backend, tmp_path, _Embed("bge-m3", 8))

    # A peer rebuilds the same files under another model of the same dimension,
    # by the supported route: refused attach, drop, attach again.
    peer = backend.storage(tmp_path, _Embed("e5-large", 8))
    with pytest.raises(VectorSpaceMismatchError):
        await peer.initialize()
    assert (await peer.drop())["status"] == "success"
    await peer.initialize()
    await peer.upsert({"vp": {"content": "peer"}})
    await peer.index_done_callback()

    storage.storage_updated.value = True
    with pytest.raises(VectorSpaceMismatchError, match="'e5-large' -> 'bge-m3'"):
        await storage.get_by_id("vp")


@pytest.mark.asyncio
async def test_faiss_refuses_to_publish_beside_a_contradicting_marker(tmp_path):
    """FAISS only: an unwritable marker must not wedge the rebuild.

    ``drop()`` swallows a marker it cannot remove — correctly, the vectors are
    already gone by then. If the save then also swallowed a marker it cannot
    replace, it would publish the rebuilt pair beside the OLD model's marker,
    every later attach would refuse, and repeated drops could never clear it.
    Failing before publishing is what keeps that from becoming permanent.
    """
    backend = _Backend("faiss")
    await _seed(backend, tmp_path, _Embed("bge-m3", 8))

    storage = backend.storage(tmp_path, _Embed("e5-large", 8))
    with pytest.raises(VectorSpaceMismatchError):
        await storage.initialize()

    marker = backend.marker_path(tmp_path)
    real_remove = os.remove

    def _remove(path, *args, **kwargs):
        if str(path) == marker:
            raise PermissionError("marker file is locked")
        return real_remove(path, *args, **kwargs)

    with patch("lightrag.kg.faiss_impl.os.remove", side_effect=_remove):
        assert (await storage.drop())["status"] == "success"

    await storage.initialize()
    await storage.upsert({"v2": {"content": "world"}})

    # The marker on disk still says bge-m3 and cannot be replaced, so the save
    # must refuse rather than publish e5-large rows next to it.
    with patch(
        "lightrag.kg.faiss_impl.atomic_write",
        side_effect=PermissionError("marker file is locked"),
    ):
        with pytest.raises(VectorSpaceMismatchError, match="'bge-m3' -> 'e5-large'"):
            await storage.index_done_callback()
