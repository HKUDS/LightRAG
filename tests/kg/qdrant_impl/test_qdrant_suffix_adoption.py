"""Pre-digest Qdrant collections stay reachable under the digested name.

Qdrant cannot rename a collection. Adoption records the raw model name and
aliases the digested name onto the physical pre-digest collection. A
collection already owned by a different model is left alone.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

pytest.importorskip(
    "qdrant_client", reason="qdrant-client is required for Qdrant storage tests"
)

from lightrag.kg.qdrant_impl import QdrantVectorDBStorage  # noqa: E402
from lightrag.kg.vector_space import VECTOR_SPACE_MODEL_KEY  # noqa: E402
from lightrag.utils import EmbeddingFunc  # noqa: E402

pytestmark = pytest.mark.offline


def _embedding(model_name, dim=1024):
    async def embed(texts, **kwargs):
        return np.array([[0.1] * dim for _ in texts])

    return EmbeddingFunc(embedding_dim=dim, func=embed, model_name=model_name)


def _storage(model_name):
    return QdrantVectorDBStorage(
        namespace="chunks",
        workspace="ws",
        global_config={
            "embedding_batch_num": 10,
            "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.8},
        },
        embedding_func=_embedding(model_name),
    )


def _client(names, info):
    client = MagicMock()

    def exists(name):
        return name in names

    def update_collection(collection_name, metadata=None, **kwargs):
        info.config.metadata = dict(metadata or {})
        return True

    def update_aliases(change_aliases_operations, **kwargs):
        for operation in change_aliases_operations:
            names.add(operation.create_alias.alias_name)
        return True

    client.collection_exists.side_effect = exists
    client.get_collection.return_value = info
    client.update_collection.side_effect = update_collection
    client.update_collection_aliases.side_effect = update_aliases
    client.count.return_value.count = 4
    return client


@pytest.fixture
def _init_lock():
    with patch("lightrag.kg.qdrant_impl.get_data_init_lock") as mock_lock:
        mock_lock.return_value = AsyncMock()
        yield


@pytest.mark.asyncio
async def test_unowned_pre_digest_collection_is_aliased_and_claimed(_init_lock):
    storage = _storage("vendor/model:v1")
    physical = storage._pre_digest_collection_name()
    info = MagicMock()
    info.config.metadata = None
    names = {physical}
    client = _client(names, info)

    with patch("lightrag.kg.qdrant_impl.QdrantClient", return_value=client):
        await storage.initialize()

    assert info.config.metadata[VECTOR_SPACE_MODEL_KEY] == "vendor/model:v1"
    assert storage.final_namespace in names
    client.create_collection.assert_not_called()
    client.update_collection_aliases.assert_called_once()
    operation = client.update_collection_aliases.call_args.kwargs[
        "change_aliases_operations"
    ][0]
    assert operation.create_alias.collection_name == physical
    assert operation.create_alias.alias_name == storage.final_namespace


@pytest.mark.asyncio
async def test_a_pre_digest_collection_owned_by_another_model_is_not_aliased(
    _init_lock,
):
    storage = _storage("vendor/model:v1")
    physical = storage._pre_digest_collection_name()
    info = MagicMock()
    info.config.metadata = {VECTOR_SPACE_MODEL_KEY: "vendor_model/v1"}
    names = {physical}
    client = _client(names, info)

    with patch("lightrag.kg.qdrant_impl.QdrantClient", return_value=client):
        await storage.initialize()

    client.update_collection_aliases.assert_not_called()
    client.update_collection.assert_not_called()
    created = client.create_collection.call_args.kwargs.get(
        "collection_name", client.create_collection.call_args.args[0]
    )
    assert created == storage.final_namespace
    assert physical in names


@pytest.mark.asyncio
async def test_existing_digested_collection_is_not_realiased(_init_lock):
    storage = _storage("vendor/model:v1")
    physical = storage._pre_digest_collection_name()
    info = MagicMock()
    info.config.metadata = None
    names = {physical, storage.final_namespace}
    client = _client(names, info)

    with patch("lightrag.kg.qdrant_impl.QdrantClient", return_value=client):
        await storage.initialize()

    client.update_collection.assert_not_called()
    client.update_collection_aliases.assert_not_called()
    client.create_collection.assert_not_called()
