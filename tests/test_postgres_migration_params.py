"""
Strict test to verify PostgreSQL migration parameter passing.

This test specifically validates that the migration code passes parameters
to AsyncPG execute() in the correct format (positional args, not dict).
"""

import pytest
from unittest.mock import patch, AsyncMock
from lightrag.utils import EmbeddingFunc
from lightrag.kg.postgres_impl import PGVectorStorage
from lightrag.namespace import NameSpace


@pytest.mark.asyncio
async def test_migration_parameter_passing():
    """
    Verify that migration passes positional parameters correctly to execute().

    This test specifically checks that execute() is called with:
    - SQL query as first argument
    - Values as separate positional arguments (*values)
    NOT as a dictionary or list
    """

    # Track all execute calls
    execute_calls = []

    async def strict_execute(sql, *args, **kwargs):
        """Record all execute calls with their arguments"""
        execute_calls.append(
            {
                "sql": sql,
                "args": args,  # Should be tuple of values
                "kwargs": kwargs,
            }
        )

        # Validate: if args has only one element and it's a dict/list, that's wrong
        if args and len(args) == 1 and isinstance(args[0], (dict, list)):
            raise TypeError(
                f"BUG DETECTED: execute() called with {type(args[0]).__name__} "
                "instead of positional parameters! "
                f"Got: execute(sql, {args[0]!r})"
            )
        return None

    # Create mocks
    mock_db = AsyncMock()
    mock_db.workspace = "test_workspace"
    mock_db.execute = AsyncMock(side_effect=strict_execute)

    # Mock query to simulate legacy table with data
    mock_rows = [
        {
            "id": "row1",
            "content": "content1",
            "workspace": "test",
            "vector": [0.1] * 1536,
        },
        {
            "id": "row2",
            "content": "content2",
            "workspace": "test",
            "vector": [0.2] * 1536,
        },
    ]

    async def mock_query(sql, params=None, multirows=False, **kwargs):
        if "COUNT(*)" in sql:
            return {"count": len(mock_rows)}
        elif multirows and "SELECT *" in sql:
            return mock_rows
        return {}

    mock_db.query = AsyncMock(side_effect=mock_query)

    # Mock table existence: only legacy table exists
    async def mock_table_exists(db, table_name):
        return "test_model_1536d" not in table_name  # Legacy exists, new doesn't

    # Setup embedding function
    async def embed_func(texts, **kwargs):
        import numpy as np

        return np.array([[0.1] * 1536 for _ in texts])

    embedding_func = EmbeddingFunc(
        embedding_dim=1536, func=embed_func, model_name="test-model"
    )

    config = {
        "embedding_batch_num": 10,
        "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.8},
    }

    storage = PGVectorStorage(
        namespace=NameSpace.VECTOR_STORE_CHUNKS,
        global_config=config,
        embedding_func=embedding_func,
        workspace="test",
    )

    with (
        patch("lightrag.kg.postgres_impl.get_data_init_lock") as mock_lock,
        patch("lightrag.kg.postgres_impl.ClientManager") as mock_manager,
        patch(
            "lightrag.kg.postgres_impl._pg_table_exists", side_effect=mock_table_exists
        ),
        patch("lightrag.kg.postgres_impl._pg_create_table", AsyncMock()),
    ):
        mock_lock_ctx = AsyncMock()
        mock_lock.return_value = mock_lock_ctx
        mock_manager.get_client = AsyncMock(return_value=mock_db)
        mock_manager.release_client = AsyncMock()

        # This should trigger migration
        await storage.initialize()

    # Verify execute was called (migration happened)
    assert len(execute_calls) > 0, "Migration should have called execute()"

    # Verify parameter format for INSERT statements
    insert_calls = [c for c in execute_calls if "INSERT INTO" in c["sql"]]
    assert len(insert_calls) > 0, "Should have INSERT statements from migration"

    print(f"\n✓ Migration executed {len(insert_calls)} INSERT statements")

    # Check each INSERT call
    for i, call_info in enumerate(insert_calls):
        args = call_info["args"]
        sql = call_info["sql"]

        print(f"\n  INSERT #{i+1}:")
        print(f"    SQL: {sql[:100]}...")
        print(f"    Args count: {len(args)}")
        print(f"    Args types: {[type(arg).__name__ for arg in args]}")

        # Key validation: args should be a tuple of values, not a single dict/list
        if args:
            # Check if first (and only) arg is a dict or list - that's the bug!
            if len(args) == 1 and isinstance(args[0], (dict, list)):
                pytest.fail(
                    f"BUG: execute() called with {type(args[0]).__name__} instead of "
                    f"positional parameters!\n"
                    f"  SQL: {sql}\n"
                    f"  Args: {args[0]}\n"
                    f"Expected: execute(sql, val1, val2, val3, ...)\n"
                    f"Got: execute(sql, {type(args[0]).__name__})"
                )

            # Validate all args are primitive types (not collections)
            for j, arg in enumerate(args):
                if isinstance(arg, (dict, list)) and not isinstance(arg, (str, bytes)):
                    # Exception: vector columns might be lists, that's OK
                    if "vector" not in sql:
                        pytest.fail(
                            f"BUG: Parameter #{j} is {type(arg).__name__}, "
                            f"expected primitive type"
                        )

    print(
        f"\n✅ All {len(insert_calls)} INSERT statements use correct parameter format"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
