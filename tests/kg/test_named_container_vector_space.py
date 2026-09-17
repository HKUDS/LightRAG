"""The three backends whose CONTAINER NAME carries the embedding model.

Milvus, Qdrant and PostgreSQL encode ``{folded_model}_{dim}d`` in the
collection / table name, so they record no provenance marker: a model change
lands in a different container by construction. What they still owe issue #3978
is the other half of the contract, and it is what this module pins:

* the embedding-space refusal is ``VectorSpaceMismatchError`` -- the type
  ``lightrag-rebuild-vdb`` matches on -- and never ``DataMigrationError`` or a
  generic ``RuntimeError``, both of which would make the condition
  indistinguishable from an outage and leave the tool no way to act;
* a refused instance can still serve ``drop()``, because ``drop()`` then
  ``initialize()`` IS the recovery. A refusal raised before the flush lock
  exists wedges the deployment instead of failing closed;
* the refusal is scoped to what THIS workspace would actually migrate, so the
  recovery converges: ``drop()`` only ever clears this workspace's legacy
  records, and a refusal counting other tenants' records could not be cleared
  by it.

See docs/design/VectorSpaceProvenance.md.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from lightrag.exceptions import DataMigrationError, VectorSpaceMismatchError
from lightrag.kg.shared_storage import initialize_share_data
from lightrag.utils import EmbeddingFunc

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


# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------

qdrant_client = pytest.importorskip(
    "qdrant_client", reason="qdrant-client is required for Qdrant storage tests"
)


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


# ---------------------------------------------------------------------------
# PostgreSQL
# ---------------------------------------------------------------------------

pytest.importorskip("asyncpg", reason="asyncpg is required for PostgreSQL tests")


class TestPostgresVectorSpaceRefusal:
    @staticmethod
    def _db(*, legacy_table, legacy_dim, legacy_rows):
        db = AsyncMock()
        db.check_table_exists = AsyncMock(
            side_effect=lambda name: name.lower() == legacy_table.lower()
        )

        async def query(sql, params=None, multirows=False, **kwargs):
            if "COUNT(*)" in sql:
                return {"count": legacy_rows}
            if "content_vector" in sql:
                return {"content_vector": [0.1] * legacy_dim}
            return {}

        db.query = AsyncMock(side_effect=query)
        db.execute = AsyncMock()
        db._create_vector_index = AsyncMock()
        return db

    async def test_legacy_at_another_dimension_raises_the_typed_refusal(self):
        from lightrag.kg.postgres_impl import PGVectorStorage

        db = self._db(
            legacy_table="LIGHTRAG_VDB_CHUNKS", legacy_dim=1536, legacy_rows=100
        )

        with pytest.raises(VectorSpaceMismatchError) as excinfo:
            await PGVectorStorage.setup_table(
                db,
                "LIGHTRAG_VDB_CHUNKS_test_model_768d",
                workspace="test_ws",
                embedding_dim=768,
                legacy_table_name="LIGHTRAG_VDB_CHUNKS",
                base_table="LIGHTRAG_VDB_CHUNKS",
            )

        error = excinfo.value
        assert not isinstance(error, DataMigrationError)
        assert error.container == "LIGHTRAG_VDB_CHUNKS"
        assert (error.stored_dim, error.expected_dim) == (1536, 768)

    async def test_a_refused_instance_can_still_be_dropped(self):
        from lightrag.kg.postgres_impl import PGVectorStorage

        db = self._db(
            legacy_table="LIGHTRAG_VDB_CHUNKS", legacy_dim=1536, legacy_rows=100
        )
        db.workspace = None

        storage = PGVectorStorage(
            namespace="chunks",
            global_config=_GLOBAL_CONFIG,
            embedding_func=_embedding_func(),
            workspace="test_ws",
        )
        with patch(
            "lightrag.kg.postgres_impl.ClientManager.get_client",
            AsyncMock(return_value=db),
        ):
            with pytest.raises(VectorSpaceMismatchError):
                await storage.initialize()

            assert storage._flush_lock is not None
            result = await storage.drop()

        assert result["status"] == "success"
        # The suffixed table was never created, so only the legacy table is
        # cleared for this workspace -- which is what lets the next
        # initialize() get past the refusal.
        cleared = [call.args[0] for call in db.execute.await_args_list if call.args]
        assert any("LIGHTRAG_VDB_CHUNKS" in sql for sql in cleared)
        assert all("test_model_768d" not in sql for sql in cleared)


# ---------------------------------------------------------------------------
# Milvus
# ---------------------------------------------------------------------------

pytest.importorskip("pymilvus", reason="pymilvus is required for Milvus tests")


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
