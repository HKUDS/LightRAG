"""
Tests for dimension mismatch handling during migration.

This test module verifies that both PostgreSQL and Qdrant storage backends
properly detect and handle vector dimension mismatches when migrating from
legacy collections/tables to new ones with different embedding models.
"""

import json
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
                    model_suffix="model_3072d",
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

        # Track whether upsert has been called (migration occurred)
        migration_done = {"value": False}

        def upsert_side_effect(*args, **kwargs):
            migration_done["value"] = True
            return MagicMock()

        client.upsert.side_effect = upsert_side_effect

        # Mock count to return different values based on collection name and migration state
        # Before migration: new collection has 0 records
        # After migration: new collection has 1 record (matching migrated data)
        def count_side_effect(collection_name, **kwargs):
            result = MagicMock()
            if collection_name == "lightrag_chunks":  # legacy
                result.count = 1  # Legacy has 1 record
            elif collection_name == "lightrag_chunks_model_1536d":  # new
                # Return 0 before migration, 1 after migration
                result.count = 1 if migration_done["value"] else 0
            else:
                result.count = 0
            return result

        client.count.side_effect = count_side_effect

        # Mock scroll to return sample data (1 record for easier verification)
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
                model_suffix="model_1536d",
            )

        # Verify migration WAS attempted
        client.create_collection.assert_called_once()
        client.scroll.assert_called()
        client.upsert.assert_called()


class TestPostgresDimensionMismatch:
    """Test suite for PostgreSQL dimension mismatch handling."""

    async def test_postgres_dimension_mismatch_raises_error_metadata(self):
        """
        Test that PostgreSQL raises DataMigrationError when dimensions don't match.

        Scenario: Legacy table has 1536d vectors, new model expects 3072d.
        Expected: DataMigrationError is raised to prevent data corruption.
        """
        # Setup mock database
        db = AsyncMock()

        # Mock check_table_exists
        async def mock_check_table_exists(table_name):
            if table_name == "LIGHTRAG_DOC_CHUNKS":  # legacy
                return True
            elif table_name == "LIGHTRAG_DOC_CHUNKS_model_3072d":  # new
                return False
            return False

        db.check_table_exists = AsyncMock(side_effect=mock_check_table_exists)

        # Mock table existence and dimension checks
        async def query_side_effect(query, params, **kwargs):
            if "COUNT(*)" in query:
                return {"count": 100}  # Legacy has data
            elif "SELECT content_vector FROM" in query:
                # Return sample vector with 1536 dimensions
                return {"content_vector": [0.1] * 1536}
            return {}

        db.query.side_effect = query_side_effect
        db.execute = AsyncMock()
        db._create_vector_index = AsyncMock()

        # Call setup_table with 3072d (different from legacy 1536d)
        # Should raise DataMigrationError due to dimension mismatch
        with pytest.raises(DataMigrationError) as exc_info:
            await PGVectorStorage.setup_table(
                db,
                "LIGHTRAG_DOC_CHUNKS_model_3072d",
                legacy_table_name="LIGHTRAG_DOC_CHUNKS",
                base_table="LIGHTRAG_DOC_CHUNKS",
                embedding_dim=3072,
                workspace="test",
            )

        # Verify error message contains dimension information
        assert "3072" in str(exc_info.value) or "1536" in str(exc_info.value)

    async def test_postgres_dimension_mismatch_raises_error_sampling(self):
        """
        Test that PostgreSQL raises error when dimensions don't match (via sampling).

        Scenario: Legacy table vector sampling detects 1536d vs expected 3072d.
        Expected: DataMigrationError is raised to prevent data corruption.
        """
        db = AsyncMock()

        # Mock check_table_exists
        async def mock_check_table_exists(table_name):
            if table_name == "LIGHTRAG_DOC_CHUNKS":  # legacy
                return True
            elif table_name == "LIGHTRAG_DOC_CHUNKS_model_3072d":  # new
                return False
            return False

        db.check_table_exists = AsyncMock(side_effect=mock_check_table_exists)

        # Mock table existence and dimension checks
        async def query_side_effect(query, params, **kwargs):
            if "information_schema.tables" in query:
                if params[0] == "LIGHTRAG_DOC_CHUNKS":  # legacy
                    return {"exists": True}
                elif params[0] == "LIGHTRAG_DOC_CHUNKS_model_3072d":  # new
                    return {"exists": False}
            elif "COUNT(*)" in query:
                return {"count": 100}  # Legacy has data
            elif "SELECT content_vector FROM" in query:
                # Return sample vector with 1536 dimensions as a JSON string
                return {"content_vector": json.dumps([0.1] * 1536)}
            return {}

        db.query.side_effect = query_side_effect
        db.execute = AsyncMock()
        db._create_vector_index = AsyncMock()

        # Call setup_table with 3072d (different from legacy 1536d)
        # Should raise DataMigrationError due to dimension mismatch
        with pytest.raises(DataMigrationError) as exc_info:
            await PGVectorStorage.setup_table(
                db,
                "LIGHTRAG_DOC_CHUNKS_model_3072d",
                legacy_table_name="LIGHTRAG_DOC_CHUNKS",
                base_table="LIGHTRAG_DOC_CHUNKS",
                embedding_dim=3072,
                workspace="test",
            )

        # Verify error message contains dimension information
        assert "3072" in str(exc_info.value) or "1536" in str(exc_info.value)

    async def test_postgres_dimension_match_proceed_migration(self):
        """
        Test that PostgreSQL proceeds with migration when dimensions match.

        Scenario: Legacy table has 1536d vectors, new model also expects 1536d.
        Expected: Migration proceeds normally.
        """
        db = AsyncMock()

        # Track migration state
        migration_done = {"value": False}

        # Define exactly 2 records for consistency
        mock_records = [
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

        # Mock check_table_exists
        async def mock_check_table_exists(table_name):
            if table_name == "LIGHTRAG_DOC_CHUNKS":  # legacy exists
                return True
            elif table_name == "LIGHTRAG_DOC_CHUNKS_model_1536d":  # new doesn't exist
                return False
            return False

        db.check_table_exists = AsyncMock(side_effect=mock_check_table_exists)

        async def query_side_effect(query, params, **kwargs):
            multirows = kwargs.get("multirows", False)
            query_upper = query.upper()

            if "information_schema.tables" in query:
                if params[0] == "LIGHTRAG_DOC_CHUNKS":  # legacy
                    return {"exists": True}
                elif params[0] == "LIGHTRAG_DOC_CHUNKS_model_1536d":  # new
                    return {"exists": False}
            elif "COUNT(*)" in query_upper:
                # Return different counts based on table name in query and migration state
                if "LIGHTRAG_DOC_CHUNKS_MODEL_1536D" in query_upper:
                    # After migration: return migrated count, before: return 0
                    return {
                        "count": len(mock_records) if migration_done["value"] else 0
                    }
                # Legacy table always has 2 records (matching mock_records)
                return {"count": len(mock_records)}
            elif "PG_ATTRIBUTE" in query_upper:
                return {"vector_dim": 1536}  # Legacy has matching 1536d
            elif "SELECT" in query_upper and "FROM" in query_upper and multirows:
                # Return sample data for migration using keyset pagination
                # Handle keyset pagination: params = [workspace, limit] or [workspace, last_id, limit]
                if "id >" in query.lower():
                    # Keyset pagination: params = [workspace, last_id, limit]
                    last_id = params[1] if len(params) > 1 else None
                    # Find records after last_id
                    found_idx = -1
                    for i, rec in enumerate(mock_records):
                        if rec["id"] == last_id:
                            found_idx = i
                            break
                    if found_idx >= 0:
                        return mock_records[found_idx + 1 :]
                    return []
                else:
                    # First batch: params = [workspace, limit]
                    return mock_records
            return {}

        db.query.side_effect = query_side_effect

        # Mock _run_with_retry to track when migration happens
        migration_executed = []

        async def mock_run_with_retry(operation, *args, **kwargs):
            migration_executed.append(True)
            migration_done["value"] = True
            return None

        db._run_with_retry = AsyncMock(side_effect=mock_run_with_retry)
        db.execute = AsyncMock()
        db._create_vector_index = AsyncMock()

        # Call setup_table with matching 1536d
        await PGVectorStorage.setup_table(
            db,
            "LIGHTRAG_DOC_CHUNKS_model_1536d",
            legacy_table_name="LIGHTRAG_DOC_CHUNKS",
            base_table="LIGHTRAG_DOC_CHUNKS",
            embedding_dim=1536,
            workspace="test",
        )

        # Verify migration WAS called (via _run_with_retry for batch operations)
        assert len(migration_executed) > 0, "Migration should have been executed"
