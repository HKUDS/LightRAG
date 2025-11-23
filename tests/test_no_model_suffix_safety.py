"""
Tests for safety when model suffix is absent (no model_name provided).

This test module verifies that the system correctly handles the case when
no model_name is provided, preventing accidental deletion of the only table/collection
on restart.

Critical Bug: When model_suffix is empty, table_name == legacy_table_name.
On second startup, Case 1 logic would delete the only table/collection thinking
it's "legacy", causing all subsequent operations to fail.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from lightrag.kg.qdrant_impl import QdrantVectorDBStorage
from lightrag.kg.postgres_impl import PGVectorStorage


class TestNoModelSuffixSafety:
    """Test suite for preventing data loss when model_suffix is absent."""

    def test_qdrant_no_suffix_second_startup(self):
        """
        Test Qdrant doesn't delete collection on second startup when no model_name.

        Scenario:
        1. First startup: Creates collection without suffix
        2. Collection is empty
        3. Second startup: Should NOT delete the collection

        Bug: Without fix, Case 1 would delete the only collection.
        """
        from qdrant_client import models

        client = MagicMock()

        # Simulate second startup: collection already exists and is empty
        # IMPORTANT: Without suffix, collection_name == legacy collection name
        collection_name = "lightrag_vdb_chunks"  # No suffix, same as legacy

        # Both exist (they're the same collection)
        client.collection_exists.return_value = True

        # Collection is empty
        client.count.return_value.count = 0

        # Call setup_collection
        # This should detect that new == legacy and skip deletion
        QdrantVectorDBStorage.setup_collection(
            client,
            collection_name,
            namespace="chunks",
            workspace=None,
            vectors_config=models.VectorParams(
                size=1536, distance=models.Distance.COSINE
            ),
        )

        # CRITICAL: Collection should NOT be deleted
        client.delete_collection.assert_not_called()

        # Verify we returned early (skipped Case 1 cleanup)
        # The collection_exists was checked, but we didn't proceed to count
        # because we detected same name
        assert client.collection_exists.call_count >= 1

    @pytest.mark.asyncio
    async def test_postgres_no_suffix_second_startup(self):
        """
        Test PostgreSQL doesn't delete table on second startup when no model_name.

        Scenario:
        1. First startup: Creates table without suffix
        2. Table is empty
        3. Second startup: Should NOT delete the table

        Bug: Without fix, Case 1 would delete the only table.
        """
        db = AsyncMock()

        # Simulate second startup: table already exists and is empty
        # IMPORTANT: table_name and legacy_table_name are THE SAME
        table_name = "LIGHTRAG_VDB_CHUNKS"  # No suffix
        legacy_table_name = "LIGHTRAG_VDB_CHUNKS"  # Same as new

        # Setup mock responses
        async def table_exists_side_effect(db_instance, name):
            # Both tables exist (they're the same)
            return True

        # Mock _pg_table_exists function
        with patch(
            "lightrag.kg.postgres_impl._pg_table_exists",
            side_effect=table_exists_side_effect,
        ):
            # Call setup_table
            # This should detect that new == legacy and skip deletion
            await PGVectorStorage.setup_table(
                db,
                table_name,
                legacy_table_name=legacy_table_name,
                base_table="LIGHTRAG_VDB_CHUNKS",
                embedding_dim=1536,
            )

        # CRITICAL: Table should NOT be deleted (no DROP TABLE)
        drop_calls = [
            call
            for call in db.execute.call_args_list
            if call[0][0] and "DROP TABLE" in call[0][0]
        ]
        assert len(drop_calls) == 0, (
            "Should not drop table when new and legacy are the same"
        )

        # Also should not try to count (we returned early)
        count_calls = [
            call
            for call in db.query.call_args_list
            if call[0][0] and "COUNT(*)" in call[0][0]
        ]
        assert len(count_calls) == 0, (
            "Should not check count when new and legacy are the same"
        )

    def test_qdrant_with_suffix_case1_still_works(self):
        """
        Test that Case 1 cleanup still works when there IS a suffix.

        This ensures our fix doesn't break the normal Case 1 scenario.
        """
        from qdrant_client import models

        client = MagicMock()

        # Different names (normal case)
        collection_name = "lightrag_vdb_chunks_ada_002_1536d"  # With suffix
        legacy_collection = "lightrag_vdb_chunks"  # Without suffix

        # Setup: both exist
        def collection_exists_side_effect(name):
            return name in [collection_name, legacy_collection]

        client.collection_exists.side_effect = collection_exists_side_effect

        # Legacy is empty
        client.count.return_value.count = 0

        # Call setup_collection
        QdrantVectorDBStorage.setup_collection(
            client,
            collection_name,
            namespace="chunks",
            workspace=None,
            vectors_config=models.VectorParams(
                size=1536, distance=models.Distance.COSINE
            ),
        )

        # SHOULD delete legacy (normal Case 1 behavior)
        client.delete_collection.assert_called_once_with(
            collection_name=legacy_collection
        )

    @pytest.mark.asyncio
    async def test_postgres_with_suffix_case1_still_works(self):
        """
        Test that Case 1 cleanup still works when there IS a suffix.

        This ensures our fix doesn't break the normal Case 1 scenario.
        """
        db = AsyncMock()

        # Different names (normal case)
        table_name = "LIGHTRAG_VDB_CHUNKS_ADA_002_1536D"  # With suffix
        legacy_table_name = "LIGHTRAG_VDB_CHUNKS"  # Without suffix

        # Setup mock responses
        async def table_exists_side_effect(db_instance, name):
            # Both tables exist
            return True

        # Mock empty table
        async def query_side_effect(sql, params, **kwargs):
            if "COUNT(*)" in sql:
                return {"count": 0}
            return {}

        db.query.side_effect = query_side_effect

        # Mock _pg_table_exists function
        with patch(
            "lightrag.kg.postgres_impl._pg_table_exists",
            side_effect=table_exists_side_effect,
        ):
            # Call setup_table
            await PGVectorStorage.setup_table(
                db,
                table_name,
                legacy_table_name=legacy_table_name,
                base_table="LIGHTRAG_VDB_CHUNKS",
                embedding_dim=1536,
            )

        # SHOULD delete legacy (normal Case 1 behavior)
        drop_calls = [
            call
            for call in db.execute.call_args_list
            if call[0][0] and "DROP TABLE" in call[0][0]
        ]
        assert len(drop_calls) == 1, "Should drop legacy table in normal Case 1"
        assert legacy_table_name in drop_calls[0][0][0]
