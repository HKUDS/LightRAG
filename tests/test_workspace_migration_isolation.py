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

        # Mock table existence checks
        async def table_exists_side_effect(db_instance, name):
            if name == "lightrag_doc_chunks":  # legacy
                return True
            elif name == "lightrag_doc_chunks_model_1536d":  # new
                return False
            return False

        # Mock query responses
        async def query_side_effect(sql, params, **kwargs):
            multirows = kwargs.get("multirows", False)

            # Table existence check
            if "information_schema.tables" in sql:
                if params[0] == "lightrag_doc_chunks":
                    return {"exists": True}
                elif params[0] == "lightrag_doc_chunks_model_1536d":
                    return {"exists": False}

            # Count query with workspace filter (legacy table)
            elif "COUNT(*)" in sql and "WHERE workspace" in sql:
                if params[0] == "workspace_a":
                    return {"count": 2}  # workspace_a has 2 records
                elif params[0] == "workspace_b":
                    return {"count": 3}  # workspace_b has 3 records
                return {"count": 0}

            # Count query for new table (verification)
            elif "COUNT(*)" in sql and "lightrag_doc_chunks_model_1536d" in sql:
                return {"count": 2}  # Verification: 2 records migrated

            # Count query for legacy table (no filter)
            elif "COUNT(*)" in sql and "lightrag_doc_chunks" in sql:
                return {"count": 5}  # Total records in legacy

            # Dimension check
            elif "pg_attribute" in sql:
                return {"vector_dim": 1536}

            # SELECT with workspace filter
            elif "SELECT * FROM" in sql and "WHERE workspace" in sql and multirows:
                workspace = params[0]
                if workspace == "workspace_a" and params[1] == 0:  # offset = 0
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
        db._create_vector_index = AsyncMock()

        # Mock _pg_table_exists and _pg_create_table
        from unittest.mock import patch

        with (
            patch(
                "lightrag.kg.postgres_impl._pg_table_exists",
                side_effect=table_exists_side_effect,
            ),
            patch("lightrag.kg.postgres_impl._pg_create_table", new=AsyncMock()),
        ):
            # Migrate for workspace_a only
            await PGVectorStorage.setup_table(
                db,
                "lightrag_doc_chunks_model_1536d",
                legacy_table_name="lightrag_doc_chunks",
                base_table="lightrag_doc_chunks",
                embedding_dim=1536,
                workspace="workspace_a",  # CRITICAL: Only migrate workspace_a
            )

        # Verify workspace filter was used in queries
        count_calls = [
            call
            for call in db.query.call_args_list
            if call[0][0]
            and "COUNT(*)" in call[0][0]
            and "WHERE workspace" in call[0][0]
        ]
        assert len(count_calls) > 0, "Count query should use workspace filter"
        assert (
            count_calls[0][0][1][0] == "workspace_a"
        ), "Count should filter by workspace_a"

        select_calls = [
            call
            for call in db.query.call_args_list
            if call[0][0]
            and "SELECT * FROM" in call[0][0]
            and "WHERE workspace" in call[0][0]
        ]
        assert len(select_calls) > 0, "Select query should use workspace filter"
        assert (
            select_calls[0][0][1][0] == "workspace_a"
        ), "Select should filter by workspace_a"

        # Verify INSERT was called (migration happened)
        insert_calls = [
            call
            for call in db.execute.call_args_list
            if call[0][0] and "INSERT INTO" in call[0][0]
        ]
        assert len(insert_calls) == 2, "Should insert 2 records from workspace_a"

    @pytest.mark.asyncio
    async def test_migration_without_workspace_warns(self):
        """
        Test that migration without workspace parameter logs a warning.

        Scenario: setup_table called without workspace parameter.
        Expected: Warning logged about potential cross-workspace data copying.
        """
        db = AsyncMock()

        async def table_exists_side_effect(db_instance, name):
            if name == "lightrag_doc_chunks":
                return True
            elif name == "lightrag_doc_chunks_model_1536d":
                return False
            return False

        async def query_side_effect(sql, params, **kwargs):
            if "information_schema.tables" in sql:
                return {"exists": params[0] == "lightrag_doc_chunks"}
            elif "COUNT(*)" in sql:
                return {"count": 5}  # 5 records total
            elif "pg_attribute" in sql:
                return {"vector_dim": 1536}
            elif "SELECT * FROM" in sql and kwargs.get("multirows"):
                if params[0] == 0:  # offset = 0
                    return [
                        {
                            "id": "1",
                            "workspace": "workspace_a",
                            "content_vector": [0.1] * 1536,
                        },
                        {
                            "id": "2",
                            "workspace": "workspace_b",
                            "content_vector": [0.2] * 1536,
                        },
                    ]
                else:
                    return []
            return {}

        db.query.side_effect = query_side_effect
        db.execute = AsyncMock()
        db._create_vector_index = AsyncMock()

        from unittest.mock import patch

        with (
            patch(
                "lightrag.kg.postgres_impl._pg_table_exists",
                side_effect=table_exists_side_effect,
            ),
            patch("lightrag.kg.postgres_impl._pg_create_table", new=AsyncMock()),
        ):
            # Migrate WITHOUT workspace parameter (dangerous!)
            await PGVectorStorage.setup_table(
                db,
                "lightrag_doc_chunks_model_1536d",
                legacy_table_name="lightrag_doc_chunks",
                base_table="lightrag_doc_chunks",
                embedding_dim=1536,
                workspace=None,  # No workspace filter!
            )

        # Verify queries do NOT use workspace filter
        count_calls = [
            call
            for call in db.query.call_args_list
            if call[0][0] and "COUNT(*)" in call[0][0]
        ]
        assert len(count_calls) > 0, "Count query should be executed"
        # Check that workspace filter was NOT used
        has_workspace_filter = any(
            "WHERE workspace" in call[0][0] for call in count_calls
        )
        assert (
            not has_workspace_filter
        ), "Count should NOT filter by workspace when workspace=None"

    @pytest.mark.asyncio
    async def test_no_cross_workspace_contamination(self):
        """
        Test that workspace B's migration doesn't include workspace A's data.

        Scenario: Two separate migrations for workspace_a and workspace_b.
        Expected: Each workspace only gets its own data.
        """
        db = AsyncMock()

        # Track which workspace is being queried
        queried_workspace = None

        async def table_exists_side_effect(db_instance, name):
            return "lightrag_doc_chunks" in name and "model" not in name

        async def query_side_effect(sql, params, **kwargs):
            nonlocal queried_workspace
            multirows = kwargs.get("multirows", False)

            if "information_schema.tables" in sql:
                return {"exists": "lightrag_doc_chunks" in params[0]}
            elif "COUNT(*)" in sql and "WHERE workspace" in sql:
                queried_workspace = params[0]
                return {"count": 1}
            elif "COUNT(*)" in sql and "lightrag_doc_chunks_model_1536d" in sql:
                return {"count": 1}  # Verification count
            elif "pg_attribute" in sql:
                return {"vector_dim": 1536}
            elif "SELECT * FROM" in sql and "WHERE workspace" in sql and multirows:
                workspace = params[0]
                if params[1] == 0:  # offset = 0
                    # Return data ONLY for the queried workspace
                    return [
                        {
                            "id": f"{workspace}_1",
                            "workspace": workspace,
                            "content": f"content_{workspace}",
                            "content_vector": [0.1] * 1536,
                        }
                    ]
                else:
                    return []
            return {}

        db.query.side_effect = query_side_effect
        db.execute = AsyncMock()
        db._create_vector_index = AsyncMock()

        from unittest.mock import patch

        with (
            patch(
                "lightrag.kg.postgres_impl._pg_table_exists",
                side_effect=table_exists_side_effect,
            ),
            patch("lightrag.kg.postgres_impl._pg_create_table", new=AsyncMock()),
        ):
            # Migrate workspace_b
            await PGVectorStorage.setup_table(
                db,
                "lightrag_doc_chunks_model_1536d",
                legacy_table_name="lightrag_doc_chunks",
                base_table="lightrag_doc_chunks",
                embedding_dim=1536,
                workspace="workspace_b",
            )

        # Verify only workspace_b was queried
        assert queried_workspace == "workspace_b", "Should only query workspace_b"

        # Verify INSERT contains workspace_b data only
        insert_calls = [
            call
            for call in db.execute.call_args_list
            if call[0][0] and "INSERT INTO" in call[0][0]
        ]
        assert len(insert_calls) > 0, "Should have INSERT calls"
