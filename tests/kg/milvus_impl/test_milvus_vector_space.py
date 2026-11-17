"""Milvus refuses a collection written in another embedding space.

Each test drives the collection-compatibility path against a mocked client --
no running Milvus instance required.

The backend's CONTAINER NAME carries the embedding model (``{workspace}_{namespace}_{folded_model}_{dim}d``), so it
records no provenance marker: a model change lands in a different container by
construction. What it still owes issue #3978 is the other half of the contract:

* the embedding-space refusal is ``VectorSpaceMismatchError`` -- the type
  ``lightrag-rebuild-vdb`` matches on -- and never ``DataMigrationError`` or a
  generic ``RuntimeError``, both of which would make the condition
  indistinguishable from an outage and leave the tool no way to act;
* a refused instance can still serve ``drop()``, because ``drop()`` then
  ``initialize()`` IS the recovery. A refusal raised before the flush lock
  exists wedges the deployment instead of failing closed.

See docs/design/VectorSpaceProvenance.md.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

pytest.importorskip("pymilvus", reason="pymilvus is required for Milvus storage tests")

from lightrag.exceptions import VectorSpaceMismatchError  # noqa: E402
from lightrag.kg.shared_storage import initialize_share_data  # noqa: E402
from lightrag.utils import EmbeddingFunc  # noqa: E402

pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def _shared_storage():
    """The storages take their data-init and flush locks from shared storage."""
    initialize_share_data(workers=1)


def _embedding_func(dim: int = 768, model_name: str = "test-model") -> EmbeddingFunc:
    async def embed(texts, **kwargs):
        return np.array([[0.1] * dim for _ in texts])

    return EmbeddingFunc(embedding_dim=dim, func=embed, model_name=model_name)


_GLOBAL_CONFIG = {
    "embedding_batch_num": 10,
    "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.8},
}


class TestMilvusVectorSpaceRefusal:
    @staticmethod
    def _storage(dim=768):
        from lightrag.kg.milvus_impl import MilvusVectorDBStorage

        storage = MilvusVectorDBStorage.__new__(MilvusVectorDBStorage)
        storage.workspace = "test_ws"
        storage.namespace = "chunks"
        storage.legacy_namespace = "test_ws_chunks"
        storage.final_namespace = f"test_ws_chunks_test_model_{dim}d"
        storage.embedding_func = _embedding_func(dim)
        storage._client = MagicMock()
        return storage

    @staticmethod
    def _collection_info(dim):
        return {
            "fields": [
                {"name": "vector", "type": "FloatVector", "params": {"dim": dim}}
            ]
        }

    def test_dimension_mismatch_raises_the_typed_refusal(self):
        storage = self._storage()

        with pytest.raises(VectorSpaceMismatchError) as excinfo:
            storage._check_vector_dimension(self._collection_info(1536))

        error = excinfo.value
        assert error.container == storage.final_namespace
        assert (error.stored_dim, error.expected_dim) == (1536, 768)

    def test_a_missing_dimension_is_a_schema_error_not_a_refusal(self):
        """A dimension nobody reported is not evidence of a changed space.

        `None != 768` would raise the typed refusal, which
        ``lightrag-rebuild-vdb`` answers by DROPPING the collection -- so a
        malformed describe_collection response would authorise destroying live
        vectors.
        """
        storage = self._storage()
        no_dim = {"fields": [{"name": "vector", "type": "FloatVector", "params": {}}]}

        with pytest.raises(ValueError) as excinfo:
            storage._check_vector_dimension(no_dim)

        assert not isinstance(excinfo.value, VectorSpaceMismatchError)

    def test_an_undeclared_embedding_dimension_is_a_schema_error_not_a_refusal(self):
        """The same rule on the declared side: a process that cannot say what
        dimension it uses contradicts nothing."""
        storage = self._storage()
        storage.embedding_func = _embedding_func()
        storage.embedding_func.embedding_dim = None

        with pytest.raises(ValueError) as excinfo:
            storage._check_vector_dimension(self._collection_info(1536))

        assert not isinstance(excinfo.value, VectorSpaceMismatchError)

    def test_unparseable_dimensions_stay_a_value_error(self):
        """A schema this code cannot read is not an embedding-space change, and
        must never reach the rebuild tool as a droppable condition."""
        storage = self._storage()

        with pytest.raises(ValueError) as excinfo:
            storage._check_vector_dimension(self._collection_info("not-a-number"))

        assert not isinstance(excinfo.value, VectorSpaceMismatchError)

    def test_validation_does_not_reframe_the_refusal_as_a_migration_failure(self):
        storage = self._storage()
        storage._client.describe_collection.return_value = self._collection_info(1536)

        with pytest.raises(VectorSpaceMismatchError):
            storage._validate_collection_and_load()

    def test_an_incompatible_legacy_collection_is_skipped_not_refused(self):
        """The suffixed collection does not exist yet, so nothing is being
        served out of the wrong embedding space: the legacy collection is only
        a migration SOURCE, and an incompatible source is simply not migrated."""
        storage = self._storage()
        storage._client.has_collection.side_effect = lambda name: (
            name == storage.legacy_namespace
        )
        storage._client.describe_collection.return_value = self._collection_info(1536)
        storage._create_collection_with_schema = MagicMock()
        storage._ensure_collection_loaded = MagicMock()
        storage._migrate_collection_schema = MagicMock()

        storage._create_collection_if_not_exist()

        storage._create_collection_with_schema.assert_called_once_with(
            storage.final_namespace
        )
        storage._migrate_collection_schema.assert_not_called()
