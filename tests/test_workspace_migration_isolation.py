"""
Tests for workspace isolation during PostgreSQL migration.

This test module verifies that setup_table() properly filters migration data
by workspace, preventing cross-workspace data leakage during legacy table migration.

Critical Bug: Migration copied ALL records from legacy table regardless of workspace,
causing workspace A to receive workspace B's data, violating multi-tenant isolation.
"""

import pytest
from unittest.mock import AsyncMock

from lightrag.kg.postgres_impl import PGVectorStorage


class TestWorkspaceMigrationIsolation:
    """Test suite for workspace-scoped migration in PostgreSQL."""

    @pytest.mark.asyncio
    async def test_migration_filters_by_workspace(self):
        """
        Test that migration only copies data from the specified workspace.

        Scenario: Legacy table contains data from multiple workspaces.
                  Migrate only workspace_a's data to new table.
        Expected: New table contains only workspace_a data, workspace_b data excluded.
        """
        db = AsyncMock()

        # Configure mock return values to avoid unawaited coroutine warnings
        db._create_vector_index.return_value = None

        # Track state for new table count (starts at 0, increases after migration)
        new_table_record_count = {"count": 0}

        # Mock table existence checks
        async def table_exists_side_effect(db_instance, name):
            if name.lower() == "lightrag_doc_chunks":  # legacy
                return True
            elif name.lower() == "lightrag_doc_chunks_model_1536d":  # new
                return False  # New table doesn't exist initially
            return False

        # Mock query responses
        async def query_side_effect(sql, params, **kwargs):
            multirows = kwargs.get("multirows", False)
            sql_lower = sql.lower()

            # Count query for new table workspace data (verification before migration)
            if "count(*)" in sql_lower and "model_1536d" in sql_lower and "where workspace" in sql_lower:
                return new_table_record_count  # Initially 0

            # Count query with workspace filter (legacy table) - for workspace count
            elif "count(*)" in sql_lower and "where workspace" in sql_lower:
                if params and params[0] == "workspace_a":
                    return {"count": 2}  # workspace_a has 2 records
                elif params and params[0] == "workspace_b":
                    return {"count": 3}  # workspace_b has 3 records
                return {"count": 0}

            # Count query for legacy table (total, no workspace filter)
            elif "count(*)" in sql_lower and "lightrag" in sql_lower and "where workspace" not in sql_lower:
                return {"count": 5}  # Total records in legacy

            # SELECT with workspace filter for migration
            elif "select * from" in sql_lower and "where workspace" in sql_lower and multirows:
                workspace = params[0] if params else None
                offset = params[1] if len(params) > 1 else 0
                if workspace == "workspace_a" and offset == 0:
                    # Return only workspace_a data
                    return [
                        {
                            "id": "a1",
                            "workspace": "workspace_a",
                            "content": "content_a1",
                            "content_vector": [0.1] * 1536,
                        },
                        {
                            "id": "a2",
                            "workspace": "workspace_a",
                            "content": "content_a2",
                            "content_vector": [0.2] * 1536,
                        },
                    ]
                else:
                    return []  # No more data

            return {}

        db.query.side_effect = query_side_effect
        db.execute = AsyncMock()

        # Mock _pg_table_exists, _pg_create_table, and _pg_migrate_workspace_data
        from unittest.mock import patch

        async def mock_migrate_workspace_data(db, legacy, new, workspace, expected_count, dim):
            # Simulate migration by updating count
            new_table_record_count["count"] = expected_count
            return expected_count

        with (
            patch(
                "lightrag.kg.postgres_impl._pg_table_exists",
                side_effect=table_exists_side_effect,
            ),
            patch("lightrag.kg.postgres_impl._pg_create_table", new=AsyncMock()),
            patch(
                "lightrag.kg.postgres_impl._pg_migrate_workspace_data",
                side_effect=mock_migrate_workspace_data,
            ),
        ):
            # Migrate for workspace_a only - correct parameter order
            await PGVectorStorage.setup_table(
                db,
                "lightrag_doc_chunks_model_1536d",
                workspace="workspace_a",  # CRITICAL: Only migrate workspace_a
                embedding_dim=1536,
                legacy_table_name="lightrag_doc_chunks",
                base_table="lightrag_doc_chunks",
            )

        # Verify the migration function was called with the correct workspace
        # The mock_migrate_workspace_data tracks that the migration was triggered
        # with workspace_a data (2 records)
        assert new_table_record_count["count"] == 2, "Should have migrated 2 records from workspace_a"

    @pytest.mark.asyncio
    async def test_migration_without_workspace_raises_error(self):
        """
        Test that migration without workspace parameter raises ValueError.

        Scenario: setup_table called without workspace parameter.
        Expected: ValueError is raised because workspace is required.
        """
        db = AsyncMock()

        # workspace is now a required parameter - calling with None should raise ValueError
        with pytest.raises(ValueError, match="workspace must be provided"):
            await PGVectorStorage.setup_table(
                db,
                "lightrag_doc_chunks_model_1536d",
                workspace=None,  # No workspace - should raise ValueError
                embedding_dim=1536,
                legacy_table_name="lightrag_doc_chunks",
                base_table="lightrag_doc_chunks",
            )

    @pytest.mark.asyncio
    async def test_no_cross_workspace_contamination(self):
        """
        Test that workspace B's migration doesn't include workspace A's data.

        Scenario: Migration for workspace_b only.
        Expected: Only workspace_b data is queried, workspace_a data excluded.
        """
        db = AsyncMock()

        # Configure mock return values to avoid unawaited coroutine warnings
        db._create_vector_index.return_value = None

        # Track which workspace is being queried
        queried_workspace = None
        new_table_count = {"count": 0}

        async def table_exists_side_effect(db_instance, name):
            if name.lower() == "lightrag_doc_chunks":  # legacy
                return True
            elif name.lower() == "lightrag_doc_chunks_model_1536d":  # new
                return False
            return False

        async def query_side_effect(sql, params, **kwargs):
            nonlocal queried_workspace
            sql_lower = sql.lower()

            # Count query for new table workspace data (should be 0 initially)
            if "count(*)" in sql_lower and "model_1536d" in sql_lower and "where workspace" in sql_lower:
                return new_table_count

            # Count query with workspace filter (legacy table)
            elif "count(*)" in sql_lower and "where workspace" in sql_lower:
                queried_workspace = params[0] if params else None
                return {"count": 1}  # 1 record for the queried workspace

            # Count query for legacy table total (no workspace filter)
            elif "count(*)" in sql_lower and "lightrag" in sql_lower and "where workspace" not in sql_lower:
                return {"count": 3}  # 3 total records in legacy

            return {}

        db.query.side_effect = query_side_effect
        db.execute = AsyncMock()

        from unittest.mock import patch

        async def mock_migrate_workspace_data(db, legacy, new, workspace, expected_count, dim):
            # Simulate migration by updating count
            new_table_count["count"] = expected_count
            return expected_count

        with (
            patch(
                "lightrag.kg.postgres_impl._pg_table_exists",
                side_effect=table_exists_side_effect,
            ),
            patch("lightrag.kg.postgres_impl._pg_create_table", new=AsyncMock()),
            patch(
                "lightrag.kg.postgres_impl._pg_migrate_workspace_data",
                side_effect=mock_migrate_workspace_data,
            ),
        ):
            # Migrate workspace_b - correct parameter order
            await PGVectorStorage.setup_table(
                db,
                "lightrag_doc_chunks_model_1536d",
                workspace="workspace_b",  # Only migrate workspace_b
                embedding_dim=1536,
                legacy_table_name="lightrag_doc_chunks",
                base_table="lightrag_doc_chunks",
            )

        # Verify only workspace_b was queried
        assert queried_workspace == "workspace_b", "Should only query workspace_b"
