"""Pre-digest Qdrant collections stay reachable under the digested name.

Qdrant cannot rename a collection. Adoption records the raw model name and
aliases the digested name onto the physical pre-digest collection. A
collection already owned by a different model is left alone.
"""

import asyncio
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


def _deleted_collections(client):
    return [
        call.kwargs.get("collection_name", call.args[0] if call.args else None)
        for call in client.delete.call_args_list
    ]


def _client_for_drop(storage, metadata, *, include_digested=True):
    """Client whose pre-digest collection exists and records ``metadata``."""
    physical = storage._pre_digest_collection_name()
    info = MagicMock()
    info.config.metadata = metadata
    names = {physical}
    if include_digested:
        names.add(storage.final_namespace)
    client = MagicMock()
    client.collection_exists.side_effect = lambda name: name in names
    client.get_collection.return_value = info
    storage._client = client
    # drop() takes the namespace lock. A real NamespaceLock needs shared
    # storage, which these tests do not start.
    storage._flush_lock = asyncio.Lock()
    return client, physical


@pytest.mark.asyncio
async def test_drop_leaves_points_owned_by_another_model():
    """The model that lost the claim must not wipe the winner's physical collection.

    ``vendor/model:v1`` adopted the pre-digest name and aliased onto it.
    ``vendor_model/v1`` created its own collection. Its ``drop()`` still sees
    the shared physical name, and deleting there removes the first claimant's
    points. Adoption will not run again.
    """
    storage = _storage("vendor_model/v1")
    client, physical = _client_for_drop(
        storage, {VECTOR_SPACE_MODEL_KEY: "vendor/model:v1"}
    )

    result = await storage.drop()

    assert result["status"] == "success"
    deleted = _deleted_collections(client)
    assert storage.final_namespace in deleted
    assert physical not in deleted
    client.delete_collection.assert_not_called()
    client.update_collection_aliases.assert_not_called()


@pytest.mark.asyncio
async def test_drop_clears_an_unowned_pre_digest_collection():
    """A clear before the first adopt must still empty the old container."""
    storage = _storage("vendor/model:v1")
    client, physical = _client_for_drop(storage, None, include_digested=False)

    result = await storage.drop()

    assert result["status"] == "success"
    assert physical in _deleted_collections(client)
    client.delete_collection.assert_not_called()


@pytest.mark.asyncio
async def test_drop_clears_a_pre_digest_collection_owned_by_this_model():
    storage = _storage("vendor/model:v1")
    client, physical = _client_for_drop(
        storage, {VECTOR_SPACE_MODEL_KEY: "vendor/model:v1"}, include_digested=False
    )

    result = await storage.drop()

    assert result["status"] == "success"
    assert physical in _deleted_collections(client)
