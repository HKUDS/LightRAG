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
from tests._pg_legacy_migration import answer_legacy_migration


pytestmark = pytest.mark.offline


class TestWorkspaceMigrationIsolation:
    """Test suite for workspace-scoped migration in PostgreSQL."""

    async def test_migration_without_workspace_raises_error(self):
        """
        Test that migration without workspace parameter raises ValueError.

        Scenario: setup_table called without workspace parameter.
        Expected: ValueError is raised because workspace is required.
        """
        db = AsyncMock()
        # Real default; an auto-created mock breaks _check_vector_index_dimension.
        db.vector_index_type = None

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

    async def test_no_cross_workspace_contamination(self):
        """
        Test that workspace B's migration doesn't include workspace A's data.

        Scenario: Migration for workspace_b only.
        Expected: Only workspace_b data is queried, workspace_a data excluded.
        """
        db = AsyncMock()
        # Real default; an auto-created mock breaks _check_vector_index_dimension.
        db.vector_index_type = None

        # Configure mock return values to avoid unawaited coroutine warnings
        db._create_vector_index.return_value = None

        # Track which workspace is being queried
        queried_workspace = None
        new_table_count = {"count": 0}

        # Mock data for workspace_b
        mock_records_b = [
            {
                "id": "b1",
                "workspace": "workspace_b",
                "content": "content_b1",
                "content_vector": [0.3] * 1536,
            },
        ]

        async def table_exists_side_effect(db_instance, name):
            if name.lower() == "lightrag_doc_chunks":  # legacy
                return True
            elif name.lower() == "lightrag_doc_chunks_model_1536d":  # new
                return False
            return False

        async def query_side_effect(sql, params, **kwargs):
            nonlocal queried_workspace
            multirows = kwargs.get("multirows", False)
            sql_upper = sql.upper()

            # Count query for new table workspace data (should be 0 initially)
            if (
                "COUNT(*)" in sql_upper
                and "MODEL_1536D" in sql_upper
                and "WHERE WORKSPACE" in sql_upper
            ):
                return new_table_count

            # Count query with workspace filter (legacy table)
            elif "COUNT(*)" in sql_upper and "WHERE WORKSPACE" in sql_upper:
                queried_workspace = params[0] if params else None
                return {"count": 1}  # 1 record for the queried workspace

            # Count query for legacy table total (no workspace filter)
            elif (
                "COUNT(*)" in sql_upper
                and "LIGHTRAG" in sql_upper
                and "WHERE WORKSPACE" not in sql_upper
            ):
                return {"count": 3}  # 3 total records in legacy

            # SELECT with workspace filter for migration (multirows)
            elif "SELECT" in sql_upper and "FROM" in sql_upper and multirows:
                workspace = params[0] if params else None
                if workspace == "workspace_b":
                    # Handle keyset pagination: check for "id >" pattern
                    if "id >" in sql.lower():
                        # Keyset pagination: params = [workspace, last_id, limit]
                        last_id = params[1] if len(params) > 1 else None
                        # Find records after last_id
                        found_idx = -1
                        for i, rec in enumerate(mock_records_b):
                            if rec["id"] == last_id:
                                found_idx = i
                                break
                        if found_idx >= 0:
                            return mock_records_b[found_idx + 1 :]
                        return []
                    else:
                        # First batch: params = [workspace, limit]
                        return mock_records_b
                return []  # No data for other workspaces

            return {}

        db.query.side_effect = query_side_effect
        db.execute = AsyncMock()

        # Mock check_table_exists on db
        async def check_table_exists_side_effect(name):
            if name.lower() == "lightrag_doc_chunks":  # legacy
                return True
            elif name.lower() == "lightrag_doc_chunks_model_1536d":  # new
                return False
            return False

        db.check_table_exists = AsyncMock(side_effect=check_table_exists_side_effect)

        # Track migration through its server-side copy statement
        migration_executed = []

        def record_migration(sql, params):
            migration_executed.append(True)
            new_table_count["count"] = 1  # Simulate migration

        db.query = AsyncMock(
            side_effect=answer_legacy_migration(db.query.side_effect, record_migration)
        )

        # Migrate workspace_b - correct parameter order
        await PGVectorStorage.setup_table(
            db,
            "LIGHTRAG_DOC_CHUNKS_model_1536d",
            workspace="workspace_b",  # Only migrate workspace_b
            embedding_dim=1536,
            legacy_table_name="LIGHTRAG_DOC_CHUNKS",
            base_table="LIGHTRAG_DOC_CHUNKS",
        )

        # The migration ran, and only workspace_b was queried.
        assert len(migration_executed) > 0, (
            "Migration should have been executed for workspace_b"
        )
        assert queried_workspace == "workspace_b", "Should only query workspace_b"
