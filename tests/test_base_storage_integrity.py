import pytest
from lightrag.base import BaseVectorStorage
from lightrag.utils import EmbeddingFunc


def test_base_vector_storage_integrity():
    # Just checking if we can import and inspect the class
    assert hasattr(BaseVectorStorage, "_generate_collection_suffix")
    assert hasattr(BaseVectorStorage, "_get_legacy_collection_name")
    assert hasattr(BaseVectorStorage, "_get_new_collection_name")

    # Verify methods raise NotImplementedError
    class ConcreteStorage(BaseVectorStorage):
        async def query(self, *args, **kwargs):
            pass

        async def upsert(self, *args, **kwargs):
            pass

        async def delete_entity(self, *args, **kwargs):
            pass

        async def delete_entity_relation(self, *args, **kwargs):
            pass

        async def get_by_id(self, *args, **kwargs):
            pass

        async def get_by_ids(self, *args, **kwargs):
            pass

        async def delete(self, *args, **kwargs):
            pass

        async def get_vectors_by_ids(self, *args, **kwargs):
            pass

        async def index_done_callback(self):
            pass

        async def drop(self):
            pass

    func = EmbeddingFunc(embedding_dim=128, func=lambda x: x)
    storage = ConcreteStorage(
        namespace="test", workspace="test", global_config={}, embedding_func=func
    )

    assert storage._generate_collection_suffix() == "unknown_128d"

    with pytest.raises(NotImplementedError):
        storage._get_legacy_collection_name()

    with pytest.raises(NotImplementedError):
        storage._get_new_collection_name()
