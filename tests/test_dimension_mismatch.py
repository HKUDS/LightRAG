"""
Tests for dimension mismatch handling during migration.

This test module verifies that both PostgreSQL and Qdrant storage backends
properly detect and handle vector dimension mismatches when migrating from
legacy collections/tables to new ones with different embedding models.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from lightrag.kg.qdrant_impl import QdrantVectorDBStorage
from lightrag.kg.postgres_impl import PGVectorStorage
from lightrag.exceptions import DataMigrationError


# Note: Tests should use proper table names that have DDL templates
# Valid base tables: LIGHTRAG_VDB_CHUNKS, LIGHTRAG_VDB_ENTITIES, LIGHTRAG_VDB_RELATIONSHIPS,
#                    LIGHTRAG_DOC_CHUNKS, LIGHTRAG_DOC_FULL_DOCS, LIGHTRAG_DOC_TEXT_CHUNKS


class TestQdrantDimensionMismatch:
    """Test suite for Qdrant dimension mismatch handling."""

    def test_qdrant_dimension_mismatch_raises_error(self):
        """
        Test that Qdrant raises DataMigrationError when dimensions don't match.

        Scenario: Legacy collection has 1536d vectors, new model expects 3072d.
        Expected: DataMigrationError is raised to prevent data corruption.
        """
        from qdrant_client import models

        # Setup mock client
        client = MagicMock()

        # Mock legacy collection with 1536d vectors
        legacy_collection_info = MagicMock()
        legacy_collection_info.config.params.vectors.size = 1536

        # Setup collection existence checks
        def collection_exists_side_effect(name):
            if (
                name == "lightrag_vdb_chunks"
            ):  # legacy (matches _find_legacy_collection pattern)
                return True
            elif name == "lightrag_chunks_model_3072d":  # new
                return False
            return False

        client.collection_exists.side_effect = collection_exists_side_effect
        client.get_collection.return_value = legacy_collection_info
        client.count.return_value.count = 100  # Legacy has data

        # Patch _find_legacy_collection to return the legacy collection name
        with patch(
            "lightrag.kg.qdrant_impl._find_legacy_collection",
            return_value="lightrag_vdb_chunks",
        ):
            # Call setup_collection with 3072d (different from legacy 1536d)
            # Should raise DataMigrationError due to dimension mismatch
            with pytest.raises(DataMigrationError) as exc_info:
                QdrantVectorDBStorage.setup_collection(
                    client,
                    "lightrag_chunks_model_3072d",
                    namespace="chunks",
                    workspace="test",
                    vectors_config=models.VectorParams(
                        size=3072, distance=models.Distance.COSINE
                    ),
                    hnsw_config=models.HnswConfigDiff(
                        payload_m=16,
                        m=0,
                    ),
                )

        # Verify error message contains dimension information
        assert "3072" in str(exc_info.value) or "1536" in str(exc_info.value)

        # Verify new collection was NOT created (error raised before creation)
        client.create_collection.assert_not_called()

        # Verify migration was NOT attempted
        client.scroll.assert_not_called()
        client.upsert.assert_not_called()

    def test_qdrant_dimension_match_proceed_migration(self):
        """
        Test that Qdrant proceeds with migration when dimensions match.

        Scenario: Legacy collection has 1536d vectors, new model also expects 1536d.
        Expected: Migration proceeds normally.
        """
        from qdrant_client import models

        client = MagicMock()

        # Mock legacy collection with 1536d vectors (matching new)
        legacy_collection_info = MagicMock()
        legacy_collection_info.config.params.vectors.size = 1536

        def collection_exists_side_effect(name):
            if name == "lightrag_chunks":  # legacy
                return True
            elif name == "lightrag_chunks_model_1536d":  # new
                return False
            return False

        client.collection_exists.side_effect = collection_exists_side_effect
        client.get_collection.return_value = legacy_collection_info
        client.count.return_value.count = 100  # Legacy has data

        # Mock scroll to return sample data
        sample_point = MagicMock()
        sample_point.id = "test_id"
        sample_point.vector = [0.1] * 1536
        sample_point.payload = {"id": "test"}
        client.scroll.return_value = ([sample_point], None)

        # Mock _find_legacy_collection to return the legacy collection name
        with patch(
            "lightrag.kg.qdrant_impl._find_legacy_collection",
            return_value="lightrag_chunks",
        ):
            # Call setup_collection with matching 1536d
            QdrantVectorDBStorage.setup_collection(
                client,
                "lightrag_chunks_model_1536d",
                namespace="chunks",
                workspace="test",
                vectors_config=models.VectorParams(
                    size=1536, distance=models.Distance.COSINE
                ),
                hnsw_config=models.HnswConfigDiff(
                    payload_m=16,
                    m=0,
                ),
            )

        # Verify migration WAS attempted
        client.create_collection.assert_called_once()
        client.scroll.assert_called()
        client.upsert.assert_called()


class TestPostgresDimensionMismatch:
    """Test suite for PostgreSQL dimension mismatch handling."""

    @pytest.mark.asyncio
    async def test_postgres_dimension_mismatch_skip_migration_metadata(self):
        """
        Test that PostgreSQL skips migration when dimensions don't match (via metadata).

        Scenario: Legacy table has 1536d vectors (detected via pg_attribute),
                  new model expects 3072d.
        Expected: Migration skipped, new empty table created, legacy preserved.
        """
        # Setup mock database
        db = AsyncMock()

        # Mock table existence and dimension checks
        async def query_side_effect(query, params, **kwargs):
            if "information_schema.tables" in query:
                if params[0] == "LIGHTRAG_DOC_CHUNKS":  # legacy
                    return {"exists": True}
                elif params[0] == "LIGHTRAG_DOC_CHUNKS_model_3072d":  # new
                    return {"exists": False}
            elif "COUNT(*)" in query:
                return {"count": 100}  # Legacy has data
            elif "pg_attribute" in query:
                return {"vector_dim": 1536}  # Legacy has 1536d vectors
            return {}

        db.query.side_effect = query_side_effect
        db.execute = AsyncMock()
        db._create_vector_index = AsyncMock()

        # Call setup_table with 3072d (different from legacy 1536d)
        await PGVectorStorage.setup_table(
            db,
            "LIGHTRAG_DOC_CHUNKS_model_3072d",
            legacy_table_name="LIGHTRAG_DOC_CHUNKS",
            base_table="LIGHTRAG_DOC_CHUNKS",
            embedding_dim=3072,
            workspace="test",
        )

        # Verify migration was NOT attempted (no INSERT calls)
        # Note: _pg_create_table is mocked, so we check INSERT calls to verify migration was skipped
        insert_calls = [
            call
            for call in db.execute.call_args_list
            if call[0][0] and "INSERT INTO" in call[0][0]
        ]
        assert (
            len(insert_calls) == 0
        ), "Migration should be skipped due to dimension mismatch"

    @pytest.mark.asyncio
    async def test_postgres_dimension_mismatch_skip_migration_sampling(self):
        """
        Test that PostgreSQL skips migration when dimensions don't match (via sampling).

        Scenario: Legacy table dimension detection fails via metadata,
                  falls back to vector sampling, detects 1536d vs expected 3072d.
        Expected: Migration skipped, new empty table created, legacy preserved.
        """
        db = AsyncMock()

        # Mock table existence and dimension checks
        async def query_side_effect(query, params, **kwargs):
            if "information_schema.tables" in query:
                if params[0] == "LIGHTRAG_DOC_CHUNKS":  # legacy
                    return {"exists": True}
                elif params[0] == "LIGHTRAG_DOC_CHUNKS_model_3072d":  # new
                    return {"exists": False}
            elif "COUNT(*)" in query:
                return {"count": 100}  # Legacy has data
            elif "pg_attribute" in query:
                return {"vector_dim": -1}  # Metadata check fails
            elif "SELECT content_vector FROM" in query:
                # Return sample vector with 1536 dimensions
                return {"content_vector": [0.1] * 1536}
            return {}

        db.query.side_effect = query_side_effect
        db.execute = AsyncMock()
        db._create_vector_index = AsyncMock()

        # Call setup_table with 3072d (different from legacy 1536d)
        await PGVectorStorage.setup_table(
            db,
            "LIGHTRAG_DOC_CHUNKS_model_3072d",
            legacy_table_name="LIGHTRAG_DOC_CHUNKS",
            base_table="LIGHTRAG_DOC_CHUNKS",
            embedding_dim=3072,
            workspace="test",
        )

        # Verify new table was created
        create_table_calls = [
            call
            for call in db.execute.call_args_list
            if call[0][0] and "CREATE TABLE" in call[0][0]
        ]
        assert len(create_table_calls) > 0, "New table should be created"

        # Verify migration was NOT attempted
        insert_calls = [
            call
            for call in db.execute.call_args_list
            if call[0][0] and "INSERT INTO" in call[0][0]
        ]
        assert len(insert_calls) == 0, "Migration should be skipped"

    @pytest.mark.asyncio
    async def test_postgres_dimension_match_proceed_migration(self):
        """
        Test that PostgreSQL proceeds with migration when dimensions match.

        Scenario: Legacy table has 1536d vectors, new model also expects 1536d.
        Expected: Migration proceeds normally.
        """
        db = AsyncMock()

        async def query_side_effect(query, params, **kwargs):
            multirows = kwargs.get("multirows", False)

            if "information_schema.tables" in query:
                if params[0] == "LIGHTRAG_DOC_CHUNKS":  # legacy
                    return {"exists": True}
                elif params[0] == "LIGHTRAG_DOC_CHUNKS_model_1536d":  # new
                    return {"exists": False}
            elif "COUNT(*)" in query:
                return {"count": 100}  # Legacy has data
            elif "pg_attribute" in query:
                return {"vector_dim": 1536}  # Legacy has matching 1536d
            elif "SELECT * FROM" in query and multirows:
                # Return sample data for migration (first batch)
                # Handle workspace filtering: params = [workspace, offset, limit]
                if "WHERE workspace" in query:
                    offset = params[1] if len(params) > 1 else 0
                else:
                    offset = params[0] if params else 0

                if offset == 0:  # First batch
                    return [
                        {
                            "id": "test1",
                            "content_vector": [0.1] * 1536,
                            "workspace": "test",
                        },
                        {
                            "id": "test2",
                            "content_vector": [0.2] * 1536,
                            "workspace": "test",
                        },
                    ]
                else:  # offset > 0
                    return []  # No more data
            return {}

        db.query.side_effect = query_side_effect
        db.execute = AsyncMock()
        db._create_vector_index = AsyncMock()

        # Mock _pg_table_exists
        async def mock_table_exists(db_inst, name):
            if name == "LIGHTRAG_DOC_CHUNKS":  # legacy exists
                return True
            elif name == "LIGHTRAG_DOC_CHUNKS_model_1536d":  # new doesn't exist
                return False
            return False

        with patch(
            "lightrag.kg.postgres_impl._pg_table_exists",
            side_effect=mock_table_exists,
        ):
            # Call setup_table with matching 1536d
            await PGVectorStorage.setup_table(
                db,
                "LIGHTRAG_DOC_CHUNKS_model_1536d",
                legacy_table_name="LIGHTRAG_DOC_CHUNKS",
                base_table="LIGHTRAG_DOC_CHUNKS",
                embedding_dim=1536,
                workspace="test",
            )

        # Verify migration WAS attempted (INSERT calls made)
        insert_calls = [
            call
            for call in db.execute.call_args_list
            if call[0][0] and "INSERT INTO" in call[0][0]
        ]
        assert (
            len(insert_calls) > 0
        ), "Migration should proceed with matching dimensions"
