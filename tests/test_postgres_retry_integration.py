"""
Integration test suite for PostgreSQL retry mechanism using real database.

This test suite connects to a real PostgreSQL database using credentials from .env
and tests the retry mechanism with actual network failures.

Prerequisites:
1. PostgreSQL server running and accessible
2. .env file with POSTGRES_* configuration
3. asyncpg installed: pip install asyncpg
"""

import pytest
import asyncio
import os
import time
from dotenv import load_dotenv
from unittest.mock import patch
import asyncpg
from lightrag.kg.postgres_impl import PostgreSQLDB

# Load environment variables
load_dotenv(dotenv_path=".env", override=False)


class TestPostgresRetryIntegration:
    """Integration tests for PostgreSQL retry mechanism with real database."""

    @pytest.fixture
    def db_config(self):
        """Load database configuration from environment variables."""
        return {
            "host": os.getenv("POSTGRES_HOST", "localhost"),
            "port": int(os.getenv("POSTGRES_PORT", "5432")),
            "user": os.getenv("POSTGRES_USER", "postgres"),
            "password": os.getenv("POSTGRES_PASSWORD", ""),
            "database": os.getenv("POSTGRES_DATABASE", "postgres"),
            "workspace": os.getenv("POSTGRES_WORKSPACE", "test_retry"),
            "max_connections": int(os.getenv("POSTGRES_MAX_CONNECTIONS", "10")),
        }

    @pytest.fixture
    def test_env(self, monkeypatch):
        """Set up test environment variables for retry configuration."""
        monkeypatch.setenv("POSTGRES_CONNECTION_RETRIES", "3")
        monkeypatch.setenv("POSTGRES_CONNECTION_RETRY_BACKOFF", "0.5")
        monkeypatch.setenv("POSTGRES_CONNECTION_RETRY_BACKOFF_MAX", "2.0")
        monkeypatch.setenv("POSTGRES_POOL_CLOSE_TIMEOUT", "3.0")

    @pytest.mark.asyncio
    async def test_real_connection_success(self, db_config, test_env):
        """
        Test successful connection to real PostgreSQL database.

        This validates that:
        1. Database credentials are correct
        2. Connection pool initializes properly
        3. Basic query works
        """
        print("\n" + "=" * 80)
        print("INTEGRATION TEST 1: Real Database Connection")
        print("=" * 80)
        print(
            f"  → Connecting to {db_config['host']}:{db_config['port']}/{db_config['database']}"
        )

        db = PostgreSQLDB(db_config)

        try:
            # Initialize database connection
            await db.initdb()
            print("  ✓ Connection successful")

            # Test simple query
            result = await db.query("SELECT 1 as test", multirows=False)
            assert result is not None
            assert result.get("test") == 1
            print("  ✓ Query executed successfully")

            print("\n✅ Test passed: Real database connection works")
            print("=" * 80)
        finally:
            if db.pool:
                await db.pool.close()

    @pytest.mark.asyncio
    async def test_simulated_transient_error_with_real_db(self, db_config, test_env):
        """
        Test retry mechanism with simulated transient errors on real database.

        Simulates connection failures on first 2 attempts, then succeeds.
        """
        print("\n" + "=" * 80)
        print("INTEGRATION TEST 2: Simulated Transient Errors")
        print("=" * 80)

        db = PostgreSQLDB(db_config)
        attempt_count = {"value": 0}

        # Original create_pool function
        original_create_pool = asyncpg.create_pool

        async def mock_create_pool_with_failures(*args, **kwargs):
            """Mock that fails first 2 times, then calls real create_pool."""
            attempt_count["value"] += 1
            print(f"  → Connection attempt {attempt_count['value']}")

            if attempt_count["value"] <= 2:
                print("    ✗ Simulating connection failure")
                raise asyncpg.exceptions.ConnectionFailureError(
                    f"Simulated failure on attempt {attempt_count['value']}"
                )

            print("    ✓ Allowing real connection")
            return await original_create_pool(*args, **kwargs)

        try:
            # Patch create_pool to simulate failures
            with patch(
                "asyncpg.create_pool", side_effect=mock_create_pool_with_failures
            ):
                await db.initdb()

            assert (
                attempt_count["value"] == 3
            ), f"Expected 3 attempts, got {attempt_count['value']}"
            assert db.pool is not None, "Pool should be initialized after retries"

            # Verify database is actually working
            result = await db.query("SELECT 1 as test", multirows=False)
            assert result.get("test") == 1

            print(
                f"\n✅ Test passed: Retry mechanism worked, connected after {attempt_count['value']} attempts"
            )
            print("=" * 80)
        finally:
            if db.pool:
                await db.pool.close()

    @pytest.mark.asyncio
    async def test_query_retry_with_real_db(self, db_config, test_env):
        """
        Test query-level retry with simulated connection issues.

        Tests that queries retry on transient failures by simulating
        a temporary database unavailability.
        """
        print("\n" + "=" * 80)
        print("INTEGRATION TEST 3: Query-Level Retry")
        print("=" * 80)

        db = PostgreSQLDB(db_config)

        try:
            # First initialize normally
            await db.initdb()
            print("  ✓ Database initialized")

            # Close the pool to simulate connection loss
            print("  → Simulating connection loss (closing pool)...")
            await db.pool.close()
            db.pool = None

            # Now query should trigger pool recreation and retry
            print("  → Attempting query (should auto-reconnect)...")
            result = await db.query("SELECT 1 as test", multirows=False)

            assert result.get("test") == 1, "Query should succeed after reconnection"
            assert db.pool is not None, "Pool should be recreated"

            print("  ✓ Query succeeded after automatic reconnection")
            print("\n✅ Test passed: Auto-reconnection works correctly")
            print("=" * 80)
        finally:
            if db.pool:
                await db.pool.close()

    @pytest.mark.asyncio
    async def test_concurrent_queries_with_real_db(self, db_config, test_env):
        """
        Test concurrent queries to validate thread safety and connection pooling.

        Runs multiple concurrent queries to ensure no deadlocks or race conditions.
        """
        print("\n" + "=" * 80)
        print("INTEGRATION TEST 4: Concurrent Queries")
        print("=" * 80)

        db = PostgreSQLDB(db_config)

        try:
            await db.initdb()
            print("  ✓ Database initialized")

            # Launch 10 concurrent queries
            num_queries = 10
            print(f"  → Launching {num_queries} concurrent queries...")

            async def run_query(query_id):
                result = await db.query(
                    f"SELECT {query_id} as id, pg_sleep(0.1)", multirows=False
                )
                return result.get("id")

            start_time = time.time()
            tasks = [run_query(i) for i in range(num_queries)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            elapsed = time.time() - start_time

            # Check results
            successful = sum(1 for r in results if not isinstance(r, Exception))
            failed = sum(1 for r in results if isinstance(r, Exception))

            print(f"  → Completed in {elapsed:.2f}s")
            print(f"  → Results: {successful} successful, {failed} failed")

            assert (
                successful == num_queries
            ), f"All {num_queries} queries should succeed"
            assert failed == 0, "No queries should fail"

            print("\n✅ Test passed: All concurrent queries succeeded, no deadlocks")
            print("=" * 80)
        finally:
            if db.pool:
                await db.pool.close()

    @pytest.mark.asyncio
    async def test_pool_close_timeout_real(self, db_config, test_env):
        """
        Test pool close timeout protection with real database.
        """
        print("\n" + "=" * 80)
        print("INTEGRATION TEST 5: Pool Close Timeout")
        print("=" * 80)

        db = PostgreSQLDB(db_config)

        try:
            await db.initdb()
            print("  ✓ Database initialized")

            # Trigger pool reset (which includes close)
            print("  → Triggering pool reset...")
            start_time = time.time()
            await db._reset_pool()
            elapsed = time.time() - start_time

            print(f"  ✓ Pool reset completed in {elapsed:.2f}s")
            assert db.pool is None, "Pool should be None after reset"
            assert (
                elapsed < db.pool_close_timeout + 1
            ), "Reset should complete within timeout"

            print("\n✅ Test passed: Pool reset handled correctly")
            print("=" * 80)
        finally:
            # Already closed in test
            pass

    @pytest.mark.asyncio
    async def test_configuration_from_env(self, db_config):
        """
        Test that configuration is correctly loaded from environment variables.
        """
        print("\n" + "=" * 80)
        print("INTEGRATION TEST 6: Environment Configuration")
        print("=" * 80)

        db = PostgreSQLDB(db_config)

        print("  → Configuration loaded:")
        print(f"    • Host: {db.host}")
        print(f"    • Port: {db.port}")
        print(f"    • Database: {db.database}")
        print(f"    • User: {db.user}")
        print(f"    • Workspace: {db.workspace}")
        print(f"    • Max Connections: {db.max}")
        print(f"    • Retry Attempts: {db.connection_retry_attempts}")
        print(f"    • Retry Backoff: {db.connection_retry_backoff}s")
        print(f"    • Max Backoff: {db.connection_retry_backoff_max}s")
        print(f"    • Pool Close Timeout: {db.pool_close_timeout}s")

        # Verify required fields are present
        assert db.host, "Host should be configured"
        assert db.port, "Port should be configured"
        assert db.user, "User should be configured"
        assert db.database, "Database should be configured"

        print("\n✅ Test passed: All configuration loaded correctly from .env")
        print("=" * 80)


def run_integration_tests():
    """Run all integration tests with detailed output."""
    print("\n" + "=" * 80)
    print("POSTGRESQL RETRY MECHANISM - INTEGRATION TESTS")
    print("Testing with REAL database from .env configuration")
    print("=" * 80)

    # Check if database configuration exists
    if not os.getenv("POSTGRES_HOST"):
        print("\n⚠️  WARNING: No POSTGRES_HOST in .env file")
        print("Please ensure .env file exists with PostgreSQL configuration.")
        return

    print("\nRunning integration tests...\n")

    # Run pytest with verbose output
    pytest.main(
        [
            __file__,
            "-v",
            "-s",  # Don't capture output
            "--tb=short",  # Short traceback format
            "--color=yes",
            "-x",  # Stop on first failure
        ]
    )


if __name__ == "__main__":
    run_integration_tests()
