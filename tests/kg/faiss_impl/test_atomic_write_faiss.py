"""Atomicity coverage for ``FaissVectorDBStorage._save_faiss_index``.

The save path produces two files (``.index`` and ``.meta.json``). Each one
must land via tmp + rename so a crash during either write preserves the
prior snapshot. Cross-file consistency between the two renames is
intentionally out of scope (declared in the PR).
"""

import json
import os
from unittest.mock import patch

import numpy as np
import pytest

faiss = pytest.importorskip("faiss")  # noqa: F841 — needed before the import below

from lightrag.kg.faiss_impl import FaissVectorDBStorage  # noqa: E402
from lightrag.utils import EmbeddingFunc  # noqa: E402


def _make_storage(tmp_path, namespace: str = "vectors") -> FaissVectorDBStorage:
    """Construct a FaissVectorDBStorage that does not need real embeddings.

    ``_save_faiss_index`` only reads ``self._index`` and ``self._id_to_meta``,
    so a dummy ``EmbeddingFunc`` with the right ``embedding_dim`` is enough.
    """

    def _unused(*_args, **_kwargs):  # pragma: no cover — never called here
        raise AssertionError("embedding_func must not be invoked by save path")

    embedding_func = EmbeddingFunc(embedding_dim=4, func=_unused)
    return FaissVectorDBStorage(
        namespace=namespace,
        workspace="",
        global_config={
            "working_dir": str(tmp_path),
            "embedding_batch_num": 1,
            "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2},
        },
        embedding_func=embedding_func,
    )


def _seed(storage: FaissVectorDBStorage, marker: str) -> None:
    """Push a single vector tagged with ``marker`` into the in-memory state."""
    vec = np.ones((1, 4), dtype=np.float32)
    storage._index.add(vec)
    storage._id_to_meta = {
        storage._index.ntotal - 1: {"__id__": marker, "content": marker}
    }


@pytest.mark.offline
def test_save_faiss_index_publishes_both_files_cleanly(tmp_path):
    storage = _make_storage(tmp_path)
    _seed(storage, "v1")
    storage._save_faiss_index()

    assert os.path.exists(storage._faiss_index_file)
    assert os.path.exists(storage._meta_file)
    meta = json.load(open(storage._meta_file))
    assert next(iter(meta.values()))["__id__"] == "v1"
    leftovers = [p for p in os.listdir(tmp_path) if ".tmp." in p]
    assert leftovers == [], f"Unexpected tmp residue: {leftovers}"


@pytest.mark.offline
def test_save_faiss_index_replace_crash_preserves_prior_index(tmp_path):
    """If ``os.replace`` raises while renaming the ``.index`` tmp, the old
    ``.index`` must remain loadable by ``faiss.read_index``."""
    storage = _make_storage(tmp_path)
    _seed(storage, "v1")
    storage._save_faiss_index()
    assert os.path.exists(storage._faiss_index_file)

    # Bump in-memory state to v2 and then crash the .index rename.
    storage._index.reset()
    _seed(storage, "v2")
    with patch(
        "lightrag.file_atomic.os.replace",
        side_effect=OSError("simulated crash"),
    ):
        with pytest.raises(OSError, match="simulated crash"):
            storage._save_faiss_index()

    # Reload the destination — must still be the v1 single-vector index.
    reloaded = faiss.read_index(storage._faiss_index_file)
    assert reloaded.ntotal == 1
    leftovers = [p for p in os.listdir(tmp_path) if ".tmp." in p]
    assert leftovers == [], f"Python-exception path must clean tmp, got {leftovers}"


@pytest.mark.offline
def test_save_faiss_meta_write_failure_preserves_prior_meta(tmp_path):
    """A failure inside the meta ``write_fn`` (after the index has been
    written) must leave the previous ``.meta.json`` intact."""
    storage = _make_storage(tmp_path)
    _seed(storage, "v1")
    storage._save_faiss_index()
    assert json.load(open(storage._meta_file))

    real_dump = json.dump
    seen: list[bool] = []

    def explode_on_second_dump(*args, **kwargs):
        # The first dump is from the v1 save above — we are past it because
        # this patch is only installed for the v2 attempt. Raise immediately
        # to simulate a serialization failure mid-write.
        seen.append(True)
        raise RuntimeError("simulated meta failure")

    storage._index.reset()
    _seed(storage, "v2")
    with patch("lightrag.kg.faiss_impl.json.dump", side_effect=explode_on_second_dump):
        with pytest.raises(RuntimeError, match="simulated meta failure"):
            storage._save_faiss_index()
    assert seen, "patched json.dump must have been invoked"

    # .meta.json must still parse and still reflect v1.
    meta = json.load(open(storage._meta_file))
    assert any(entry["__id__"] == "v1" for entry in meta.values())
    leftovers = [p for p in os.listdir(tmp_path) if ".tmp." in p]
    assert leftovers == [], f"meta-write failure must clean tmp, got {leftovers}"

    # Restore real json.dump so subsequent tests don't see the patch (defensive).
    assert json.dump is real_dump
