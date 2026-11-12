"""Qdrant ``drop_sibling_models`` clears this workspace from other models' collections.

Offline: the client is a mock, so nothing here needs a running Qdrant.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip(
    "qdrant_client", reason="qdrant-client is required for Qdrant storage tests"
)

from lightrag.kg.shared_storage import initialize_share_data  # noqa: E402
from lightrag.utils import EmbeddingFunc  # noqa: E402

pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def _shared_storage():
    initialize_share_data(workers=1)


def _embedding_func(model_name: str = "model-b", dim: int = 768) -> EmbeddingFunc:
    async def embed(texts, **kwargs):
        return np.array([[0.1] * dim for _ in texts])

    return EmbeddingFunc(embedding_dim=dim, func=embed, model_name=model_name)


def _storage(client):
    from lightrag.kg.qdrant_impl import QdrantVectorDBStorage

    with patch("lightrag.kg.qdrant_impl.QdrantClient", return_value=client):
        storage = QdrantVectorDBStorage(
            namespace="chunks",
            global_config={
                "embedding_batch_num": 10,
                "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.8},
            },
            embedding_func=_embedding_func(),
            workspace="ws",
        )
    storage._client = client
    return storage


def _client(collections: dict[str, dict]):
    """``collections`` maps name -> payload_schema."""
    client = MagicMock()
    client.get_collections.return_value = SimpleNamespace(
        collections=[SimpleNamespace(name=n) for n in collections]
    )
    client.get_collection.side_effect = lambda n: SimpleNamespace(
        payload_schema=collections[n]
    )
    return client


async def test_clears_workspace_points_from_each_sibling_model_collection():
    client = _client(
        {
            "lightrag_vdb_chunks_model_b_768d": {"workspace_id": object()},
            "lightrag_vdb_chunks_model_a_1536d": {"workspace_id": object()},
            "lightrag_vdb_chunks_model_c_1024d": {"workspace_id": object()},
        }
    )
    storage = _storage(client)

    result = await storage.drop_sibling_models()

    assert result["status"] == "success"
    cleared = {c.kwargs["collection_name"] for c in client.delete.call_args_list}
    assert cleared == {
        "lightrag_vdb_chunks_model_a_1536d",
        "lightrag_vdb_chunks_model_c_1024d",
    }
    # Shared collections are emptied of this workspace, never removed.
    client.delete_collection.assert_not_called()


async def test_leaves_collections_it_cannot_prove_it_owns():
    client = _client(
        {
            # No workspace_id index: not provisioned by setup_collection.
            "lightrag_vdb_chunks_foreign_512d": {},
            # Right prefix, wrong shape: no model/dim suffix.
            "lightrag_vdb_chunks_scratch": {"workspace_id": object()},
            # Another namespace.
            "lightrag_vdb_entities_model_a_1536d": {"workspace_id": object()},
        }
    )
    storage = _storage(client)

    result = await storage.drop_sibling_models()

    assert result["status"] == "success"
    client.delete.assert_not_called()


async def test_reports_a_backend_failure():
    client = _client({"lightrag_vdb_chunks_model_a_1536d": {"workspace_id": 1}})
    client.delete.side_effect = RuntimeError("boom")
    storage = _storage(client)

    result = await storage.drop_sibling_models()

    assert result["status"] == "error"
    assert "boom" in result["message"]
