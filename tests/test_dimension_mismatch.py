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


class TestQdrantDimensionMismatch:
    """Test suite for Qdrant dimension mismatch handling."""

    def test_qdrant_dimension_mismatch_skip_migration(self):
        """
        Test that Qdrant skips migration when dimensions don't match.

        Scenario: Legacy collection has 1536d vectors, new model expects 3072d.
        Expected: Migration skipped, new empty collection created, legacy preserved.
        """
        from qdrant_client import models

        # Setup mock client
        client = MagicMock()

        # Mock legacy collection with 1536d vectors
        legacy_collection_info = MagicMock()
        legacy_collection_info.config.params.vectors.size = 1536

        # Setup collection existence checks
        def collection_exists_side_effect(name):
            if name == "lightrag_chunks":  # legacy
                return True
            elif name == "lightrag_chunks_model_3072d":  # new
                return False
            return False

        client.collection_exists.side_effect = collection_exists_side_effect
        client.get_collection.return_value = legacy_collection_info
        client.count.return_value.count = 100  # Legacy has data

        # Call setup_collection with 3072d (different from legacy 1536d)
        QdrantVectorDBStorage.setup_collection(
            client,
            "lightrag_chunks_model_3072d",
            namespace="chunks",
            workspace="test",
            vectors_config=models.VectorParams(
                size=3072, distance=models.Distance.COSINE
            ),
        )

        # Verify new collection was created
        client.create_collection.assert_called_once()

        # Verify migration was NOT attempted (no scroll/upsert calls)
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

        # Call setup_collection with matching 1536d
        QdrantVectorDBStorage.setup_collection(
            client,
            "lightrag_chunks_model_1536d",
            namespace="chunks",
            workspace="test",
            vectors_config=models.VectorParams(
                size=1536, distance=models.Distance.COSINE
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
                if params[0] == "lightrag_doc_chunks":  # legacy
                    return {"exists": True}
                elif params[0] == "lightrag_doc_chunks_model_3072d":  # new
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
            "lightrag_doc_chunks_model_3072d",
            legacy_table_name="lightrag_doc_chunks",
            base_table="lightrag_doc_chunks",
            embedding_dim=3072,
        )

        # Verify new table was created (DDL executed)
        create_table_calls = [
            call
            for call in db.execute.call_args_list
            if call[0][0] and "CREATE TABLE" in call[0][0]
        ]
        assert len(create_table_calls) > 0, "New table should be created"

        # Verify migration was NOT attempted (no INSERT calls)
        insert_calls = [
            call
            for call in db.execute.call_args_list
            if call[0][0] and "INSERT INTO" in call[0][0]
        ]
        assert len(insert_calls) == 0, "Migration should be skipped"

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
                if params[0] == "lightrag_doc_chunks":  # legacy
                    return {"exists": True}
                elif params[0] == "lightrag_doc_chunks_model_3072d":  # new
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
            "lightrag_doc_chunks_model_3072d",
            legacy_table_name="lightrag_doc_chunks",
            base_table="lightrag_doc_chunks",
            embedding_dim=3072,
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
                if params[0] == "lightrag_doc_chunks":  # legacy
                    return {"exists": True}
                elif params[0] == "lightrag_doc_chunks_model_1536d":  # new
                    return {"exists": False}
            elif "COUNT(*)" in query:
                return {"count": 100}  # Legacy has data
            elif "pg_attribute" in query:
                return {"vector_dim": 1536}  # Legacy has matching 1536d
            elif "SELECT * FROM" in query and multirows:
                # Return sample data for migration (first batch)
                if params[0] == 0:  # offset = 0
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

        # Call setup_table with matching 1536d
        await PGVectorStorage.setup_table(
            db,
            "lightrag_doc_chunks_model_1536d",
            legacy_table_name="lightrag_doc_chunks",
            base_table="lightrag_doc_chunks",
            embedding_dim=1536,
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
