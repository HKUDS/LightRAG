"""Qdrant refuses a legacy collection written in another embedding space.

Each test drives ``setup_collection`` / ``initialize`` / ``drop`` against a
mocked client -- no running Qdrant instance required.

The backend's CONTAINER NAME carries the embedding model (``lightrag_vdb_{namespace}_{folded_model}_{dim}d``), so it
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

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip(
    "qdrant_client", reason="qdrant-client is required for Qdrant storage tests"
)

from lightrag.exceptions import (  # noqa: E402
    DataMigrationError,
    VectorSpaceMismatchError,
)
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


class TestQdrantVectorSpaceRefusal:
    @staticmethod
    def _client(*, legacy_name, legacy_dim, legacy_total, workspace_count=None):
        """A Qdrant client holding only ``legacy_name``, at ``legacy_dim``."""
        client = MagicMock()
        client.collection_exists.side_effect = lambda name: name == legacy_name

        legacy_info = MagicMock()
        legacy_info.config.params.vectors.size = legacy_dim
        # Workspace-tagged when a per-workspace count is supplied; otherwise the
        # untagged (pre-isolation) shape, where the whole collection is this
        # workspace's migration source.
        legacy_info.payload_schema = (
            {"workspace_id": MagicMock()} if workspace_count is not None else {}
        )
        client.get_collection.return_value = legacy_info
        client.scroll.return_value = ([], None)

        def count(collection_name, exact=True, count_filter=None):
            result = MagicMock()
            if collection_name != legacy_name:
                result.count = 0
            elif count_filter is not None and workspace_count is not None:
                result.count = workspace_count
            else:
                result.count = legacy_total
            return result

        client.count.side_effect = count
        return client

    @staticmethod
    def _setup(client, *, dim, workspace="test_ws"):
        from qdrant_client import models

        from lightrag.kg.qdrant_impl import QdrantVectorDBStorage

        return QdrantVectorDBStorage.setup_collection(
            client,
            f"lightrag_vdb_chunks_test_model_{dim}d",
            namespace="chunks",
            workspace=workspace,
            vectors_config=models.VectorParams(
                size=dim, distance=models.Distance.COSINE
            ),
            hnsw_config=models.HnswConfigDiff(payload_m=16, m=0),
            model_suffix=f"test_model_{dim}d",
        )

    def test_legacy_at_another_dimension_raises_the_typed_refusal(self):
        client = self._client(
            legacy_name="lightrag_vdb_chunks", legacy_dim=1536, legacy_total=100
        )

        with pytest.raises(VectorSpaceMismatchError) as excinfo:
            self._setup(client, dim=768)

        error = excinfo.value
        assert error.container == "lightrag_vdb_chunks"
        assert (error.stored_dim, error.expected_dim) == (1536, 768)
        # Nothing was mutated: the refusal precedes every write.
        client.create_collection.assert_not_called()
        client.upsert.assert_not_called()

    def test_the_refusal_is_not_a_migration_failure(self):
        """`lightrag-rebuild-vdb` tolerates only VectorSpaceMismatchError.

        Catching DataMigrationError to reach this condition would also catch a
        genuinely failed migration, and the tool answers this condition by
        DROPPING the container.
        """
        client = self._client(
            legacy_name="lightrag_vdb_chunks", legacy_dim=1536, legacy_total=100
        )

        with pytest.raises(VectorSpaceMismatchError) as excinfo:
            self._setup(client, dim=768)

        assert not isinstance(excinfo.value, DataMigrationError)

    def test_a_tagged_legacy_holding_no_rows_for_this_workspace_does_not_refuse(self):
        """The refusal is scoped to what this workspace would migrate.

        A workspace-tagged legacy collection is shared. Refusing over another
        tenant's records would block a workspace with nothing to migrate, and
        the refusal could never be cleared: drop() only removes this
        workspace's legacy points.
        """
        client = self._client(
            legacy_name="lightrag_vdb_chunks",
            legacy_dim=1536,
            legacy_total=100,
            workspace_count=0,
        )

        self._setup(client, dim=768)

        client.create_collection.assert_called_once()

    def test_a_tagged_legacy_holding_rows_for_this_workspace_still_refuses(self):
        client = self._client(
            legacy_name="lightrag_vdb_chunks",
            legacy_dim=1536,
            legacy_total=100,
            workspace_count=7,
        )

        with pytest.raises(VectorSpaceMismatchError) as excinfo:
            self._setup(client, dim=768)

        assert "7 record(s)" in str(excinfo.value)

    async def test_a_refused_instance_can_still_be_dropped(self):
        """drop() then initialize() is the recovery, so the refusal may not
        leave the instance without its flush lock."""
        from lightrag.kg.qdrant_impl import QdrantVectorDBStorage

        client = self._client(
            legacy_name="lightrag_vdb_chunks", legacy_dim=1536, legacy_total=100
        )

        with patch("lightrag.kg.qdrant_impl.QdrantClient", return_value=client):
            storage = QdrantVectorDBStorage(
                namespace="chunks",
                global_config=_GLOBAL_CONFIG,
                embedding_func=_embedding_func(),
                workspace="test_ws",
            )
            with pytest.raises(VectorSpaceMismatchError):
                await storage.initialize()

            assert storage._flush_lock is not None
            result = await storage.drop()

        assert result["status"] == "success"
        # The suffixed collection was never created, so only the legacy
        # collection is cleared -- and clearing it is what lets the next
        # initialize() succeed.
        client.delete_collection.assert_called_once_with(
            collection_name="lightrag_vdb_chunks"
        )
