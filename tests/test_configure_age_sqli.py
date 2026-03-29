"""
PoC / regression test for CWE-89: SQL injection in configure_age() via graph_name.

The configure_age() static method on PostgreSQLDB previously interpolated
``graph_name`` directly into an f-string SQL statement:

    f"select create_graph('{graph_name}')"

A malicious graph_name (e.g. containing a single-quote) allows SQL injection.

After the fix, configure_age() must use a parameterized query ($1) so that
the graph_name is safely bound.

This test does NOT need a running PostgreSQL instance; it mocks asyncpg to
capture the exact SQL and parameters that would be sent.
"""

import pytest
from unittest.mock import AsyncMock


@pytest.fixture
def mock_connection():
    """Create a mock asyncpg connection that records execute() calls."""
    conn = AsyncMock()
    conn.execute = AsyncMock(return_value=None)
    return conn


@pytest.mark.asyncio
async def test_configure_age_uses_parameterized_query(mock_connection):
    """configure_age must NOT embed graph_name literally into the SQL string."""
    from lightrag.kg.postgres_impl import PostgreSQLDB

    malicious_name = "test'); DROP SCHEMA public CASCADE; --"

    await PostgreSQLDB.configure_age(mock_connection, malicious_name)

    calls = mock_connection.execute.call_args_list

    # We expect two calls:
    # 1. SET search_path = ...
    # 2. SELECT create_graph($1) with the graph_name as a bound parameter
    assert len(calls) == 2, f"Expected 2 execute calls, got {len(calls)}: {calls}"

    # Second call should use a parameterized query, NOT string interpolation
    create_graph_call = calls[1]
    sql_arg = create_graph_call.args[0] if create_graph_call.args else ""

    # The SQL must contain a placeholder ($1) rather than the literal name
    assert "$1" in sql_arg, (
        f"create_graph SQL should use parameterized query ($1), "
        f"but got: {sql_arg!r}"
    )

    # The literal malicious string must NOT appear in the SQL
    assert malicious_name not in sql_arg, (
        f"graph_name was interpolated directly into SQL (injection!): {sql_arg!r}"
    )

    # The malicious name should be passed as a separate parameter
    bound_params = create_graph_call.args[1:] if len(create_graph_call.args) > 1 else ()
    assert malicious_name in bound_params, (
        f"graph_name should be passed as a bound parameter, "
        f"but params were: {bound_params}"
    )


@pytest.mark.asyncio
async def test_configure_age_safe_name_still_works(mock_connection):
    """Even with a safe name, parameterized query should be used."""
    from lightrag.kg.postgres_impl import PostgreSQLDB

    safe_name = "my_workspace_graph"
    await PostgreSQLDB.configure_age(mock_connection, safe_name)

    calls = mock_connection.execute.call_args_list
    assert len(calls) == 2

    create_graph_call = calls[1]
    sql_arg = create_graph_call.args[0]
    assert "$1" in sql_arg, f"Expected parameterized query, got: {sql_arg!r}"
