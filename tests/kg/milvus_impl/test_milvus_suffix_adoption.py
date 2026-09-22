"""Pre-digest Milvus collections are renamed onto the digested name.

A collection created before the identity digest must not be left behind when
the digested name is allocated. The rename is the claim: a second model that
folds to the same prefix finds the old name gone and creates its own.
"""

from unittest.mock import MagicMock, patch

import pytest

from lightrag.kg.milvus_impl import MilvusVectorDBStorage, PreDigestAdoptionError

pytestmark = pytest.mark.offline


class _EmbeddingFunc:
    def __init__(self, dim=128, model_name="text-embedding-3-small"):
        self.embedding_dim = dim
        self.model_name = model_name


def _storage(model_name, workspace="space1", namespace="chunks", dim=128):
    return MilvusVectorDBStorage(
        namespace=namespace,
        workspace=workspace,
        global_config={
            "embedding_batch_num": 100,
            "working_dir": "/tmp/lightrag",
            "vector_db_storage_cls_kwargs": {
                "cosine_better_than_threshold": 0.3,
            },
        },
        embedding_func=_EmbeddingFunc(dim=dim, model_name=model_name),
        meta_fields=set(),
    )


def _wire(storage, collections):
    client = MagicMock()

    def has_collection(name):
        return name in collections

    def rename_collection(source, target):
        if source not in collections:
            raise RuntimeError(f"missing {source}")
        collections.discard(source)
        collections.add(target)

    client.has_collection.side_effect = has_collection
    client.rename_collection.side_effect = rename_collection
    storage._client = client
    return client


def test_pre_digest_collection_is_renamed_before_a_new_one_is_created():
    storage = _storage("vendor/model:v1")
    pre_digest = storage._pre_digest_collection_name()
    collections = {pre_digest}
    client = _wire(storage, collections)

    with (
        patch.object(storage, "_validate_collection_and_load") as validate,
        patch.object(storage, "_create_collection_with_schema") as create,
    ):
        storage._create_collection_if_not_exist()

    client.rename_collection.assert_called_once_with(
        pre_digest, storage.final_namespace
    )
    assert pre_digest not in collections
    assert storage.final_namespace in collections
    validate.assert_called_once()
    create.assert_not_called()


def test_folded_names_do_not_share_after_the_first_claim():
    first = _storage("vendor/model:v1")
    second = _storage("vendor_model/v1")
    pre_digest = first._pre_digest_collection_name()
    assert pre_digest == second._pre_digest_collection_name()
    assert first.final_namespace != second.final_namespace

    collections = {pre_digest}
    client = _wire(first, collections)
    second._client = client

    def create(name):
        collections.add(name)

    for storage in (first, second):
        storage._validate_collection_and_load = MagicMock()
        storage._ensure_collection_loaded = MagicMock()
        storage._create_collection_with_schema = MagicMock(side_effect=create)

    first._create_collection_if_not_exist()
    second._create_collection_if_not_exist()

    assert pre_digest not in collections
    assert first.final_namespace in collections
    assert second.final_namespace in collections
    client.rename_collection.assert_called_once_with(
        pre_digest, first.final_namespace
    )
    second._create_collection_with_schema.assert_called_once_with(
        second.final_namespace
    )
    first._create_collection_with_schema.assert_not_called()


def test_a_failed_rename_does_not_create_an_empty_shadow():
    storage = _storage("vendor/model:v1")
    pre_digest = storage._pre_digest_collection_name()
    collections = {pre_digest}
    client = _wire(storage, collections)
    client.rename_collection.side_effect = RuntimeError("permission denied")

    with pytest.raises(PreDigestAdoptionError):
        storage._create_collection_if_not_exist()

    assert collections == {pre_digest}
    client.create_collection.assert_not_called()


def test_a_lost_claim_creates_the_digested_collection_empty():
    storage = _storage("vendor_model/v1")
    pre_digest = storage._pre_digest_collection_name()
    collections = {pre_digest}
    client = _wire(storage, collections)

    def steal(source, target):
        collections.discard(source)
        raise RuntimeError("renamed by the other model")

    client.rename_collection.side_effect = steal
    storage._ensure_collection_loaded = MagicMock()
    storage._create_collection_with_schema = MagicMock(
        side_effect=lambda name: collections.add(name)
    )

    storage._create_collection_if_not_exist()

    storage._create_collection_with_schema.assert_called_once_with(
        storage.final_namespace
    )
    assert pre_digest not in collections
