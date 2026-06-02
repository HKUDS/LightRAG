"""NetworkXStorage.index_done_callback must RAISE on a graph-save failure
(PR #3187), not swallow it and return False.

A swallowed save error would let _insert_done's _flush_one treat the flush as
successful (it only detects failures via exceptions), so the document would be
marked PROCESSED with the graph changes unpersisted — silent data loss. This
locks the raise-on-failure behavior that aligns NetworkX with the other
backends (faiss/nano raise too).
"""

from __future__ import annotations

import numpy as np
import pytest

from lightrag.kg.networkx_impl import NetworkXStorage
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.utils import EmbeddingFunc

pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def _shared_data():
    finalize_share_data()
    initialize_share_data()
    yield
    finalize_share_data()


async def _embed(texts):
    return np.random.rand(len(texts), 8)


def _make_storage(tmp_path) -> NetworkXStorage:
    return NetworkXStorage(
        namespace="test_graph",
        workspace="ws",
        global_config={
            "working_dir": str(tmp_path),
            "embedding_batch_num": 10,
            "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.5},
        },
        embedding_func=EmbeddingFunc(embedding_dim=8, max_token_size=512, func=_embed),
    )


@pytest.mark.asyncio
async def test_index_done_callback_raises_on_save_failure(tmp_path, monkeypatch):
    storage = _make_storage(tmp_path)
    await storage.initialize()
    try:
        await storage.upsert_node("n1", {"entity_id": "n1", "description": "x"})

        def boom(graph, file_name, workspace):
            raise OSError("save boom")

        # write_nx_graph is invoked as NetworkXStorage.write_nx_graph(...).
        monkeypatch.setattr(NetworkXStorage, "write_nx_graph", staticmethod(boom))

        # Must surface the error (NOT return False / swallow it).
        with pytest.raises(OSError, match="save boom"):
            await storage.index_done_callback()
    finally:
        await storage.finalize()
