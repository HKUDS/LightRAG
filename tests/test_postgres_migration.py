import pytest
from unittest.mock import patch, AsyncMock
import numpy as np
from lightrag.utils import EmbeddingFunc
from lightrag.kg.postgres_impl import (
    PGVectorStorage,
)
from lightrag.namespace import NameSpace


# Mock PostgreSQLDB
@pytest.fixture
def mock_pg_db():
    """Mock PostgreSQL database connection"""
    db = AsyncMock()
    db.workspace = "test_workspace"

    # Mock query responses with multirows support
    async def mock_query(sql, params=None, multirows=False, **kwargs):
        # Default return value
        if multirows:
            return []  # Return empty list for multirows
        return {"exists": False, "count": 0}

    # Mock for execute that mimics PostgreSQLDB.execute() behavior
    async def mock_execute(sql, data=None, **kwargs):
        """
        Mock that mimics PostgreSQLDB.execute() behavior:
        - Accepts data as dict[str, Any] | None (second parameter)
        - Internally converts dict.values() to tuple for AsyncPG
        """
        # Mimic real execute() which accepts dict and converts to tuple
        if data is not None and not isinstance(data, dict):
            raise TypeError(
                f"PostgreSQLDB.execute() expects data as dict, got {type(data).__name__}"
            )
        return None

    db.query = AsyncMock(side_effect=mock_query)
    db.execute = AsyncMock(side_effect=mock_execute)

    return db


# Mock get_data_init_lock to avoid async lock issues in tests
@pytest.fixture(autouse=True)
def mock_data_init_lock():
    with patch("lightrag.kg.postgres_impl.get_data_init_lock") as mock_lock:
        mock_lock_ctx = AsyncMock()
        mock_lock.return_value = mock_lock_ctx
        yield mock_lock


# Mock ClientManager
@pytest.fixture
def mock_client_manager(mock_pg_db):
    with patch("lightrag.kg.postgres_impl.ClientManager") as mock_manager:
        mock_manager.get_client = AsyncMock(return_value=mock_pg_db)
        mock_manager.release_client = AsyncMock()
        yield mock_manager


# Mock Embedding function
@pytest.fixture
def mock_embedding_func():
    async def embed_func(texts, **kwargs):
        return np.array([[0.1] * 768 for _ in texts])

    func = EmbeddingFunc(embedding_dim=768, func=embed_func, model_name="test_model")
    return func


@pytest.mark.asyncio
async def test_postgres_table_naming(
    mock_client_manager, mock_pg_db, mock_embedding_func
):
    """Test if table name is correctly generated with model suffix"""
    config = {
        "embedding_batch_num": 10,
        "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.8},
    }

    storage = PGVectorStorage(
        namespace=NameSpace.VECTOR_STORE_CHUNKS,
        global_config=config,
        embedding_func=mock_embedding_func,
        workspace="test_ws",
    )

    # Verify table name contains model suffix
    expected_suffix = "test_model_768d"
    assert expected_suffix in storage.table_name
    assert storage.table_name == f"LIGHTRAG_VDB_CHUNKS_{expected_suffix}"

    # Verify legacy table name
    assert storage.legacy_table_name == "LIGHTRAG_VDB_CHUNKS"


@pytest.mark.asyncio
async def test_postgres_migration_trigger(
    mock_client_manager, mock_pg_db, mock_embedding_func
):
    """Test if migration logic is triggered correctly"""
    config = {
        "embedding_batch_num": 10,
        "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.8},
    }

    storage = PGVectorStorage(
        namespace=NameSpace.VECTOR_STORE_CHUNKS,
        global_config=config,
        embedding_func=mock_embedding_func,
        workspace="test_ws",
    )

    # Setup mocks for migration scenario
    # 1. New table does not exist, legacy table exists
    async def mock_table_exists(db, table_name):
        return table_name == storage.legacy_table_name

    # 2. Legacy table has 100 records
    mock_rows = [
        {"id": f"test_id_{i}", "content": f"content_{i}", "workspace": "test_ws"}
        for i in range(100)
    ]

    async def mock_query(sql, params=None, multirows=False, **kwargs):
        if "COUNT(*)" in sql:
            return {"count": 100}
        elif multirows and "SELECT *" in sql:
            # Mock batch fetch for migration
            # Handle workspace filtering: params = [workspace, offset, limit] or [offset, limit]
            if "WHERE workspace" in sql:
                # With workspace filter: params[0]=workspace, params[1]=offset, params[2]=limit
                offset = params[1] if len(params) > 1 else 0
                limit = params[2] if len(params) > 2 else 500
            else:
                # No workspace filter: params[0]=offset, params[1]=limit
                offset = params[0] if params else 0
                limit = params[1] if len(params) > 1 else 500
            start = offset
            end = min(offset + limit, len(mock_rows))
            return mock_rows[start:end]
        return {}

    mock_pg_db.query = AsyncMock(side_effect=mock_query)

    with (
        patch(
            "lightrag.kg.postgres_impl._pg_table_exists", side_effect=mock_table_exists
        ),
        patch("lightrag.kg.postgres_impl._pg_create_table", AsyncMock()),
    ):
        # Initialize storage (should trigger migration)
        await storage.initialize()

        # Verify migration was executed
        # Check that execute was called for inserting rows
        assert mock_pg_db.execute.call_count > 0


@pytest.mark.asyncio
async def test_postgres_no_migration_needed(
    mock_client_manager, mock_pg_db, mock_embedding_func
):
    """Test scenario where new table already exists (no migration needed)"""
    config = {
        "embedding_batch_num": 10,
        "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.8},
    }

    storage = PGVectorStorage(
        namespace=NameSpace.VECTOR_STORE_CHUNKS,
        global_config=config,
        embedding_func=mock_embedding_func,
        workspace="test_ws",
    )

    # Mock: new table already exists
    async def mock_table_exists(db, table_name):
        return table_name == storage.table_name

    with (
        patch(
            "lightrag.kg.postgres_impl._pg_table_exists", side_effect=mock_table_exists
        ),
        patch("lightrag.kg.postgres_impl._pg_create_table", AsyncMock()) as mock_create,
    ):
        await storage.initialize()

        # Verify no table creation was attempted
        mock_create.assert_not_called()


@pytest.mark.asyncio
async def test_scenario_1_new_workspace_creation(
    mock_client_manager, mock_pg_db, mock_embedding_func
):
    """
    Scenario 1: New workspace creation

    Expected behavior:
    - No legacy table exists
    - Directly create new table with model suffix
    - No migration needed
    """
    config = {
        "embedding_batch_num": 10,
        "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.8},
    }

    embedding_func = EmbeddingFunc(
        embedding_dim=3072,
        func=mock_embedding_func.func,
        model_name="text-embedding-3-large",
    )

    storage = PGVectorStorage(
        namespace=NameSpace.VECTOR_STORE_CHUNKS,
        global_config=config,
        embedding_func=embedding_func,
        workspace="new_workspace",
    )

    # Mock: neither table exists
    async def mock_table_exists(db, table_name):
        return False

    with (
        patch(
            "lightrag.kg.postgres_impl._pg_table_exists", side_effect=mock_table_exists
        ),
        patch("lightrag.kg.postgres_impl._pg_create_table", AsyncMock()) as mock_create,
    ):
        await storage.initialize()

        # Verify table name format
        assert "text_embedding_3_large_3072d" in storage.table_name

        # Verify new table creation was called
        mock_create.assert_called_once()
        call_args = mock_create.call_args
        assert (
            call_args[0][1] == storage.table_name
        )  # table_name is second positional arg


@pytest.mark.asyncio
async def test_scenario_2_legacy_upgrade_migration(
    mock_client_manager, mock_pg_db, mock_embedding_func
):
    """
    Scenario 2: Upgrade from legacy version

    Expected behavior:
    - Legacy table exists (without model suffix)
    - New table doesn't exist
    - Automatically migrate data to new table with suffix
    """
    config = {
        "embedding_batch_num": 10,
        "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.8},
    }

    embedding_func = EmbeddingFunc(
        embedding_dim=1536,
        func=mock_embedding_func.func,
        model_name="text-embedding-ada-002",
    )

    storage = PGVectorStorage(
        namespace=NameSpace.VECTOR_STORE_CHUNKS,
        global_config=config,
        embedding_func=embedding_func,
        workspace="legacy_workspace",
    )

    # Mock: only legacy table exists
    async def mock_table_exists(db, table_name):
        return table_name == storage.legacy_table_name

    # Mock: legacy table has 50 records
    mock_rows = [
        {
            "id": f"legacy_id_{i}",
            "content": f"legacy_content_{i}",
            "workspace": "legacy_workspace",
        }
        for i in range(50)
    ]

    # Track which queries have been made for proper response
    query_history = []

    async def mock_query(sql, params=None, multirows=False, **kwargs):
        query_history.append(sql)

        if "COUNT(*)" in sql:
            # Determine table type:
            # - Legacy: contains base name but NOT model suffix
            # - New: contains model suffix (e.g., text_embedding_ada_002_1536d)
            sql_upper = sql.upper()
            base_name = storage.legacy_table_name.upper()

            # Check if this is querying the new table (has model suffix)
            has_model_suffix = any(
                suffix in sql_upper
                for suffix in ["TEXT_EMBEDDING", "_1536D", "_768D", "_1024D", "_3072D"]
            )

            is_legacy_table = base_name in sql_upper and not has_model_suffix
            is_new_table = has_model_suffix
            has_workspace_filter = "WHERE workspace" in sql

            if is_legacy_table and has_workspace_filter:
                # Count for legacy table with workspace filter (before migration)
                return {"count": 50}
            elif is_legacy_table and not has_workspace_filter:
                # Total count for legacy table (after deletion, checking remaining)
                return {"count": 0}
            elif is_new_table:
                # Count for new table (verification after migration)
                return {"count": 50}
            else:
                # Fallback
                return {"count": 0}
        elif multirows and "SELECT *" in sql:
            # Mock batch fetch for migration
            # Handle workspace filtering: params = [workspace, offset, limit] or [offset, limit]
            if "WHERE workspace" in sql:
                # With workspace filter: params[0]=workspace, params[1]=offset, params[2]=limit
                offset = params[1] if len(params) > 1 else 0
                limit = params[2] if len(params) > 2 else 500
            else:
                # No workspace filter: params[0]=offset, params[1]=limit
                offset = params[0] if params else 0
                limit = params[1] if len(params) > 1 else 500
            start = offset
            end = min(offset + limit, len(mock_rows))
            return mock_rows[start:end]
        return {}

    mock_pg_db.query = AsyncMock(side_effect=mock_query)

    with (
        patch(
            "lightrag.kg.postgres_impl._pg_table_exists", side_effect=mock_table_exists
        ),
        patch("lightrag.kg.postgres_impl._pg_create_table", AsyncMock()) as mock_create,
    ):
        await storage.initialize()

        # Verify table name contains ada-002
        assert "text_embedding_ada_002_1536d" in storage.table_name

        # Verify migration was executed
        assert mock_pg_db.execute.call_count >= 50  # At least one execute per row
        mock_create.assert_called_once()

        # Verify legacy table was automatically deleted after successful migration
        # This prevents Case 1 warnings on next startup
        delete_calls = [
            call
            for call in mock_pg_db.execute.call_args_list
            if call[0][0] and "DROP TABLE" in call[0][0]
        ]
        assert (
            len(delete_calls) >= 1
        ), "Legacy table should be deleted after successful migration"
        # Check if legacy table was dropped
        dropped_table = storage.legacy_table_name
        assert any(
            dropped_table in str(call) for call in delete_calls
        ), f"Expected to drop '{dropped_table}'"


@pytest.mark.asyncio
async def test_scenario_3_multi_model_coexistence(
    mock_client_manager, mock_pg_db, mock_embedding_func
):
    """
    Scenario 3: Multiple embedding models coexist

    Expected behavior:
    - Different embedding models create separate tables
    - Tables are isolated by model suffix
    - No interference between different models
    """
    config = {
        "embedding_batch_num": 10,
        "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.8},
    }

    # Workspace A: uses bge-small (768d)
    embedding_func_a = EmbeddingFunc(
        embedding_dim=768, func=mock_embedding_func.func, model_name="bge-small"
    )

    storage_a = PGVectorStorage(
        namespace=NameSpace.VECTOR_STORE_CHUNKS,
        global_config=config,
        embedding_func=embedding_func_a,
        workspace="workspace_a",
    )

    # Workspace B: uses bge-large (1024d)
    async def embed_func_b(texts, **kwargs):
        return np.array([[0.1] * 1024 for _ in texts])

    embedding_func_b = EmbeddingFunc(
        embedding_dim=1024, func=embed_func_b, model_name="bge-large"
    )

    storage_b = PGVectorStorage(
        namespace=NameSpace.VECTOR_STORE_CHUNKS,
        global_config=config,
        embedding_func=embedding_func_b,
        workspace="workspace_b",
    )

    # Verify different table names
    assert storage_a.table_name != storage_b.table_name
    assert "bge_small_768d" in storage_a.table_name
    assert "bge_large_1024d" in storage_b.table_name

    # Mock: both tables don't exist yet
    async def mock_table_exists(db, table_name):
        return False

    with (
        patch(
            "lightrag.kg.postgres_impl._pg_table_exists", side_effect=mock_table_exists
        ),
        patch("lightrag.kg.postgres_impl._pg_create_table", AsyncMock()) as mock_create,
    ):
        # Initialize both storages
        await storage_a.initialize()
        await storage_b.initialize()

        # Verify two separate tables were created
        assert mock_create.call_count == 2

        # Verify table names are different
        call_args_list = mock_create.call_args_list
        table_names = [call[0][1] for call in call_args_list]  # Second positional arg
        assert len(set(table_names)) == 2  # Two unique table names
        assert storage_a.table_name in table_names
        assert storage_b.table_name in table_names


@pytest.mark.asyncio
async def test_case1_empty_legacy_auto_cleanup(
    mock_client_manager, mock_pg_db, mock_embedding_func
):
    """
    Case 1a: Both new and legacy tables exist, but legacy is EMPTY
    Expected: Automatically delete empty legacy table (safe cleanup)
    """
    config = {
        "embedding_batch_num": 10,
        "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.8},
    }

    embedding_func = EmbeddingFunc(
        embedding_dim=1536,
        func=mock_embedding_func.func,
        model_name="test-model",
    )

    storage = PGVectorStorage(
        namespace=NameSpace.VECTOR_STORE_CHUNKS,
        global_config=config,
        embedding_func=embedding_func,
        workspace="test_ws",
    )

    # Mock: Both tables exist
    async def mock_table_exists(db, table_name):
        return True  # Both new and legacy exist

    # Mock: Legacy table is empty (0 records)
    async def mock_query(sql, params=None, multirows=False, **kwargs):
        if "COUNT(*)" in sql:
            if storage.legacy_table_name in sql:
                return {"count": 0}  # Empty legacy table
            else:
                return {"count": 100}  # New table has data
        return {}

    mock_pg_db.query = AsyncMock(side_effect=mock_query)

    with patch(
        "lightrag.kg.postgres_impl._pg_table_exists", side_effect=mock_table_exists
    ):
        await storage.initialize()

        # Verify: Empty legacy table should be automatically cleaned up
        # Empty tables are safe to delete without data loss risk
        delete_calls = [
            call
            for call in mock_pg_db.execute.call_args_list
            if call[0][0] and "DROP TABLE" in call[0][0]
        ]
        assert len(delete_calls) >= 1, "Empty legacy table should be auto-deleted"
        # Check if legacy table was dropped
        dropped_table = storage.legacy_table_name
        assert any(
            dropped_table in str(call) for call in delete_calls
        ), f"Expected to drop empty legacy table '{dropped_table}'"

        print(
            f"✅ Case 1a: Empty legacy table '{dropped_table}' auto-deleted successfully"
        )


@pytest.mark.asyncio
async def test_case1_nonempty_legacy_warning(
    mock_client_manager, mock_pg_db, mock_embedding_func
):
    """
    Case 1b: Both new and legacy tables exist, and legacy HAS DATA
    Expected: Log warning, do not delete legacy (preserve data)
    """
    config = {
        "embedding_batch_num": 10,
        "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.8},
    }

    embedding_func = EmbeddingFunc(
        embedding_dim=1536,
        func=mock_embedding_func.func,
        model_name="test-model",
    )

    storage = PGVectorStorage(
        namespace=NameSpace.VECTOR_STORE_CHUNKS,
        global_config=config,
        embedding_func=embedding_func,
        workspace="test_ws",
    )

    # Mock: Both tables exist
    async def mock_table_exists(db, table_name):
        return True  # Both new and legacy exist

    # Mock: Legacy table has data (50 records)
    async def mock_query(sql, params=None, multirows=False, **kwargs):
        if "COUNT(*)" in sql:
            if storage.legacy_table_name in sql:
                return {"count": 50}  # Legacy has data
            else:
                return {"count": 100}  # New table has data
        return {}

    mock_pg_db.query = AsyncMock(side_effect=mock_query)

    with patch(
        "lightrag.kg.postgres_impl._pg_table_exists", side_effect=mock_table_exists
    ):
        await storage.initialize()

        # Verify: Legacy table with data should be preserved
        # We never auto-delete tables that contain data to prevent accidental data loss
        delete_calls = [
            call
            for call in mock_pg_db.execute.call_args_list
            if call[0][0] and "DROP TABLE" in call[0][0]
        ]
        # Check if legacy table was deleted (it should not be)
        dropped_table = storage.legacy_table_name
        legacy_deleted = any(dropped_table in str(call) for call in delete_calls)
        assert not legacy_deleted, "Legacy table with data should NOT be auto-deleted"

        print(
            f"✅ Case 1b: Legacy table '{dropped_table}' with data preserved (warning only)"
        )


@pytest.mark.asyncio
async def test_case1_sequential_workspace_migration(
    mock_client_manager, mock_pg_db, mock_embedding_func
):
    """
    Case 1c: Sequential workspace migration (Multi-tenant scenario)

    Critical bug fix verification:
    Timeline:
    1. Legacy table has workspace_a (3 records) + workspace_b (3 records)
    2. Workspace A initializes first → Case 4 (only legacy exists) → migrates A's data
    3. Workspace B initializes later → Case 1 (both tables exist) → should migrate B's data
    4. Verify workspace B's data is correctly migrated to new table
    5. Verify legacy table is cleaned up after both workspaces migrate

    This test verifies the fix where Case 1 now checks and migrates current
    workspace's data instead of just checking if legacy table is empty globally.
    """
    config = {
        "embedding_batch_num": 10,
        "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.8},
    }

    embedding_func = EmbeddingFunc(
        embedding_dim=1536,
        func=mock_embedding_func.func,
        model_name="test-model",
    )

    # Mock data: Legacy table has 6 records total (3 from workspace_a, 3 from workspace_b)
    mock_rows_a = [
        {"id": f"a_{i}", "content": f"A content {i}", "workspace": "workspace_a"}
        for i in range(3)
    ]
    mock_rows_b = [
        {"id": f"b_{i}", "content": f"B content {i}", "workspace": "workspace_b"}
        for i in range(3)
    ]

    # Track migration state
    migration_state = {"new_table_exists": False, "workspace_a_migrated": False}

    # Step 1: Simulate workspace_a initialization (Case 4)
    # CRITICAL: Set db.workspace to workspace_a
    mock_pg_db.workspace = "workspace_a"

    storage_a = PGVectorStorage(
        namespace=NameSpace.VECTOR_STORE_CHUNKS,
        global_config=config,
        embedding_func=embedding_func,
        workspace="workspace_a",
    )

    # Mock table_exists for workspace_a
    async def mock_table_exists_a(db, table_name):
        if table_name == storage_a.legacy_table_name:
            return True
        if table_name == storage_a.table_name:
            return migration_state["new_table_exists"]
        return False

    # Track inserted records count for verification
    inserted_count = {"workspace_a": 0}

    # Mock execute to track inserts
    async def mock_execute_a(sql, data=None, **kwargs):
        if sql and "INSERT INTO" in sql.upper():
            inserted_count["workspace_a"] += 1
        return None

    # Mock query for workspace_a (Case 4)
    async def mock_query_a(sql, params=None, multirows=False, **kwargs):
        sql_upper = sql.upper()
        base_name = storage_a.legacy_table_name.upper()

        if "COUNT(*)" in sql:
            has_model_suffix = "TEST_MODEL_1536D" in sql_upper
            is_legacy = base_name in sql_upper and not has_model_suffix
            has_workspace_filter = "WHERE workspace" in sql

            if is_legacy and has_workspace_filter:
                workspace = params[0] if params and len(params) > 0 else None
                if workspace == "workspace_a":
                    # After migration starts, pretend legacy is empty for this workspace
                    return {"count": 3 - inserted_count["workspace_a"]}
                elif workspace == "workspace_b":
                    return {"count": 3}
            elif is_legacy and not has_workspace_filter:
                # Global count in legacy table
                remaining = 6 - inserted_count["workspace_a"]
                return {"count": remaining}
            elif has_model_suffix:
                # New table count (for verification)
                return {"count": inserted_count["workspace_a"]}
        elif multirows and "SELECT *" in sql:
            if "WHERE workspace" in sql:
                workspace = params[0] if params and len(params) > 0 else None
                if workspace == "workspace_a":
                    offset = params[1] if len(params) > 1 else 0
                    limit = params[2] if len(params) > 2 else 500
                    return mock_rows_a[offset : offset + limit]
        return {}

    mock_pg_db.query = AsyncMock(side_effect=mock_query_a)
    mock_pg_db.execute = AsyncMock(side_effect=mock_execute_a)

    # Initialize workspace_a (Case 4)
    with (
        patch(
            "lightrag.kg.postgres_impl._pg_table_exists",
            side_effect=mock_table_exists_a,
        ),
        patch("lightrag.kg.postgres_impl._pg_create_table", AsyncMock()),
    ):
        await storage_a.initialize()
        migration_state["new_table_exists"] = True
        migration_state["workspace_a_migrated"] = True

    print("✅ Step 1: Workspace A initialized (Case 4)")
    assert mock_pg_db.execute.call_count >= 3
    print(f"✅ Step 1: {mock_pg_db.execute.call_count} execute calls")

    # Step 2: Simulate workspace_b initialization (Case 1)
    # CRITICAL: Set db.workspace to workspace_b
    mock_pg_db.workspace = "workspace_b"

    storage_b = PGVectorStorage(
        namespace=NameSpace.VECTOR_STORE_CHUNKS,
        global_config=config,
        embedding_func=embedding_func,
        workspace="workspace_b",
    )

    mock_pg_db.reset_mock()
    migration_state["workspace_b_migrated"] = False

    # Mock table_exists for workspace_b (both exist)
    async def mock_table_exists_b(db, table_name):
        return True

    # Track inserted records count for workspace_b
    inserted_count["workspace_b"] = 0

    # Mock execute for workspace_b to track inserts
    async def mock_execute_b(sql, data=None, **kwargs):
        if sql and "INSERT INTO" in sql.upper():
            inserted_count["workspace_b"] += 1
        return None

    # Mock query for workspace_b (Case 1)
    async def mock_query_b(sql, params=None, multirows=False, **kwargs):
        sql_upper = sql.upper()
        base_name = storage_b.legacy_table_name.upper()

        if "COUNT(*)" in sql:
            has_model_suffix = "TEST_MODEL_1536D" in sql_upper
            is_legacy = base_name in sql_upper and not has_model_suffix
            has_workspace_filter = "WHERE workspace" in sql

            if is_legacy and has_workspace_filter:
                workspace = params[0] if params and len(params) > 0 else None
                if workspace == "workspace_b":
                    # After migration starts, pretend legacy is empty for this workspace
                    return {"count": 3 - inserted_count["workspace_b"]}
                elif workspace == "workspace_a":
                    return {"count": 0}  # Already migrated
            elif is_legacy and not has_workspace_filter:
                # Global count: only workspace_b data remains
                return {"count": 3 - inserted_count["workspace_b"]}
            elif has_model_suffix:
                # New table total count (workspace_a: 3 + workspace_b: inserted)
                if has_workspace_filter:
                    workspace = params[0] if params and len(params) > 0 else None
                    if workspace == "workspace_b":
                        return {"count": inserted_count["workspace_b"]}
                    elif workspace == "workspace_a":
                        return {"count": 3}
                else:
                    # Total count in new table (for verification)
                    return {"count": 3 + inserted_count["workspace_b"]}
        elif multirows and "SELECT *" in sql:
            if "WHERE workspace" in sql:
                workspace = params[0] if params and len(params) > 0 else None
                if workspace == "workspace_b":
                    offset = params[1] if len(params) > 1 else 0
                    limit = params[2] if len(params) > 2 else 500
                    return mock_rows_b[offset : offset + limit]
        return {}

    mock_pg_db.query = AsyncMock(side_effect=mock_query_b)
    mock_pg_db.execute = AsyncMock(side_effect=mock_execute_b)

    # Initialize workspace_b (Case 1)
    with patch(
        "lightrag.kg.postgres_impl._pg_table_exists", side_effect=mock_table_exists_b
    ):
        await storage_b.initialize()
        migration_state["workspace_b_migrated"] = True

    print("✅ Step 2: Workspace B initialized (Case 1)")

    # Verify workspace_b migration happened
    execute_calls = mock_pg_db.execute.call_args_list
    insert_calls = [
        call for call in execute_calls if call[0][0] and "INSERT INTO" in call[0][0]
    ]
    assert len(insert_calls) >= 3, f"Expected >= 3 inserts, got {len(insert_calls)}"
    print(f"✅ Step 2: {len(insert_calls)} insert calls")

    # Verify DELETE and DROP TABLE
    delete_calls = [
        call
        for call in execute_calls
        if call[0][0]
        and "DELETE FROM" in call[0][0]
        and "WHERE workspace" in call[0][0]
    ]
    assert len(delete_calls) >= 1, "Expected DELETE workspace_b data"
    print("✅ Step 2: DELETE workspace_b from legacy")

    drop_calls = [
        call for call in execute_calls if call[0][0] and "DROP TABLE" in call[0][0]
    ]
    assert len(drop_calls) >= 1, "Expected DROP TABLE"
    print("✅ Step 2: Legacy table dropped")

    print("\n🎉 Case 1c: Sequential workspace migration verified!")
