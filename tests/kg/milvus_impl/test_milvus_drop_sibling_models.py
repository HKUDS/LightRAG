"""Milvus ``drop_sibling_models`` drops only collections it can prove it owns."""

from unittest.mock import MagicMock

import numpy as np
import pytest

pytest.importorskip("pymilvus", reason="pymilvus is required for Milvus storage tests")

from lightrag.kg.milvus_impl import MilvusVectorDBStorage  # noqa: E402
from lightrag.kg.shared_storage import initialize_share_data  # noqa: E402
from lightrag.utils import EmbeddingFunc  # noqa: E402

pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def _shared_storage():
    initialize_share_data(workers=1)


def _storage(descriptions: dict[str, str], workspace="a", namespace="chunks"):
    async def embed(texts, **kwargs):
        return np.array([[0.1] * 768 for _ in texts])

    storage = MilvusVectorDBStorage(
        namespace=namespace,
        workspace=workspace,
        global_config={
            "embedding_batch_num": 10,
            "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2},
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=768, func=embed, model_name="model-b"
        ),
        meta_fields={"content"},
    )
    client = MagicMock()
    client.list_collections.return_value = list(descriptions)
    client.describe_collection.side_effect = lambda n: {"description": descriptions[n]}
    storage._client = client
    return storage


def test_new_collections_carry_the_ownership_marker():
    storage = _storage({})
    schema = storage._create_schema_for_namespace()
    assert storage._ownership_marker() in schema.description


async def test_drops_marked_siblings_only():
    marker = "[lightrag:workspace=a;namespace=chunks]"
    storage = _storage(
        {
            "a_chunks_model_b_768d": f"LightRAG chunks vector storage {marker}",
            "a_chunks_model_a_1536d": f"LightRAG chunks vector storage {marker}",
            # Another workspace whose prefix collides with ours.
            "a_chunks_m_entities_model_a_1536d": (
                "LightRAG entities vector storage "
                "[lightrag:workspace=a_chunks_m;namespace=entities]"
            ),
        }
    )
    result = await storage.drop_sibling_models()

    assert result["status"] == "success"
    storage._client.drop_collection.assert_called_once_with("a_chunks_model_a_1536d")


async def test_unmarked_candidates_are_kept_and_reported():
    storage = _storage({"a_chunks_model_a_1536d": "LightRAG chunks vector storage"})

    result = await storage.drop_sibling_models()

    assert result["status"] == "error"
    assert "a_chunks_model_a_1536d" in result["message"]
    storage._client.drop_collection.assert_not_called()
