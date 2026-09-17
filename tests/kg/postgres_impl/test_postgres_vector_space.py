"""PostgreSQL refuses a legacy table written in another embedding space.

Each test drives ``setup_table`` / ``initialize`` / ``drop`` against a mocked
``PostgreSQLDB`` -- no running PostgreSQL instance required.

The backend's CONTAINER NAME carries the embedding model (``{base_table}_{folded_model}_{dim}d``), so it
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

from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

pytest.importorskip(
    "asyncpg", reason="asyncpg is required for PostgreSQL storage tests"
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

    async def test_an_undeclared_embedding_dimension_is_a_schema_error_not_a_refusal(
        self,
    ):
        """A dimension nobody declared is not evidence of a changed space.

        `1536 != None` would raise the typed refusal, which
        ``lightrag-rebuild-vdb`` answers by DROPPING the container -- deleting
        this workspace's legacy rows over a fact nobody reported. Same rule
        Milvus applies on both sides of its own comparison; see "Absent
        evidence never refuses" in docs/design/VectorSpaceProvenance.md.
        """
        from lightrag.kg.postgres_impl import PGVectorStorage

        db = self._db(
            legacy_table="LIGHTRAG_VDB_CHUNKS", legacy_dim=1536, legacy_rows=100
        )

        with pytest.raises(ValueError) as excinfo:
            await PGVectorStorage.setup_table(
                db,
                "LIGHTRAG_VDB_CHUNKS_test_model_768d",
                workspace="test_ws",
                embedding_dim=None,
                legacy_table_name="LIGHTRAG_VDB_CHUNKS",
                base_table="LIGHTRAG_VDB_CHUNKS",
            )

        error = excinfo.value
        assert not isinstance(error, VectorSpaceMismatchError)
        assert not isinstance(error, DataMigrationError)
        # Raised before the first storage mutation, so nothing was written and
        # no table was probed for a dimension it could not be compared against.
        db.execute.assert_not_awaited()

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
