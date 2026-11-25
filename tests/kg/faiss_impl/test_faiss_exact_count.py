"""``FaissVectorDBStorage.get_exact_count``: the maintenance count contract.

The count answers only from a persisted snapshot this process has fully
flushed and whose index agrees with its metadata, and refuses
(``StorageCapabilityError``) rather than guessing. See *Maintenance
consistency capabilities* in ``docs/design/FileBackedSnapshotContract.md``.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

pytest.importorskip("faiss")

from lightrag.exceptions import StorageCapabilityError  # noqa: E402
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
    return np.ones((len(texts), DIM), dtype=np.float32)


@pytest.mark.asyncio
async def test_count_refuses_unflushed_writes_then_counts_persisted_rows(tmp_path):
    storage = FaissVectorDBStorage(
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
    await storage.initialize()
    assert await storage.get_exact_count() == 0

    await storage.upsert({"a": {"content": "alpha"}, "b": {"content": "beta"}})
    with pytest.raises(StorageCapabilityError, match="unflushed_writes"):
        await storage.get_exact_count()

    await storage.index_done_callback()
    assert await storage.get_exact_count() == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("ntotal", "id_to_meta"),
    [
        pytest.param(2, {0: {"__id__": "one"}}, id="fewer-rows-than-slots"),
        pytest.param(1, {2: {"__id__": "one"}}, id="position-out-of-range"),
        pytest.param(2, {0: {"__id__": "x"}, 1: {"__id__": "x"}}, id="duplicate-id"),
        pytest.param(1, {0: {"content": "orphan"}}, id="missing-id"),
    ],
)
async def test_count_refuses_index_metadata_skew(ntotal, id_to_meta):
    storage = object.__new__(FaissVectorDBStorage)
    storage._storage_lock = _NoLock()
    storage._reload_index_from_disk_locked = MagicMock()
    storage._pending_upserts = storage._unsaved_upserts = {}
    storage._pending_deletes = set()
    storage._unsaved_deletes = {}
    storage._index_dirty = False
    storage._index = SimpleNamespace(ntotal=ntotal)
    storage._id_to_meta = id_to_meta

    with pytest.raises(StorageCapabilityError, match="index_meta_skew"):
        await storage.get_exact_count()


class _NoLock:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False
