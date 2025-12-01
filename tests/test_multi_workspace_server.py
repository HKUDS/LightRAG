"""
Tests for multi-workspace server support.

This module tests the server-level multi-workspace functionality including:
- Workspace identifier validation
- WorkspacePool management and LRU eviction
- Header-based workspace routing
- Workspace isolation (documents, queries, graphs)
- Backward compatibility with single-workspace mode
- Strict multi-tenant mode

Tests are organized by user story to match the implementation plan.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from lightrag.api.workspace_manager import (
    WorkspaceConfig,
    WorkspacePool,
    WorkspaceInstance,
    validate_workspace_id,
    get_workspace_from_request,
    WORKSPACE_ID_PATTERN,
)


# =============================================================================
# Phase 2: Foundational - Unit Tests
# =============================================================================


class TestWorkspaceValidation:
    """T013: Unit tests for workspace identifier validation."""

    def test_valid_workspace_ids(self):
        """Valid workspace identifiers should pass validation."""
        valid_ids = [
            "tenant1",
            "tenant-a",
            "tenant_b",
            "Workspace123",
            "a",
            "A1b2C3",
            "workspace-with-dashes",
            "workspace_with_underscores",
            "a" * 64,  # Max length
        ]
        for workspace_id in valid_ids:
            validate_workspace_id(workspace_id)  # Should not raise

    def test_invalid_workspace_ids(self):
        """Invalid workspace identifiers should raise ValueError."""
        invalid_ids = [
            "",  # Empty
            "-starts-with-dash",
            "_starts_with_underscore",
            "has spaces",
            "has/slashes",
            "has\\backslashes",
            "has.dots",
            "a" * 65,  # Too long
            "../path-traversal",
            "has:colons",
        ]
        for workspace_id in invalid_ids:
            with pytest.raises(ValueError):
                validate_workspace_id(workspace_id)

    def test_workspace_id_pattern(self):
        """Verify the regex pattern matches expected identifiers."""
        assert WORKSPACE_ID_PATTERN.match("tenant1")
        assert WORKSPACE_ID_PATTERN.match("tenant-a")
        assert WORKSPACE_ID_PATTERN.match("tenant_b")
        assert not WORKSPACE_ID_PATTERN.match("")
        assert not WORKSPACE_ID_PATTERN.match("-invalid")
        assert not WORKSPACE_ID_PATTERN.match("_invalid")


class TestWorkspacePool:
    """T014: Unit tests for WorkspacePool."""

    @pytest.fixture
    def mock_rag_factory(self):
        """Create a mock RAG factory."""
        async def factory(workspace_id: str):
            mock_rag = MagicMock()
            mock_rag.workspace = workspace_id
            mock_rag.finalize_storages = AsyncMock()
            return mock_rag
        return factory

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return WorkspaceConfig(
            default_workspace="default",
            allow_default_workspace=True,
            max_workspaces_in_pool=3,
        )

    @pytest.fixture
    def pool(self, config, mock_rag_factory):
        """Create a workspace pool for testing."""
        return WorkspacePool(config, mock_rag_factory)

    async def test_get_creates_new_instance(self, pool):
        """First request for a workspace should create a new instance."""
        rag = await pool.get("tenant1")
        assert rag is not None
        assert rag.workspace == "tenant1"
        assert pool.size == 1

    async def test_get_returns_cached_instance(self, pool):
        """Subsequent requests should return the cached instance."""
        rag1 = await pool.get("tenant1")
        rag2 = await pool.get("tenant1")
        assert rag1 is rag2
        assert pool.size == 1

    async def test_lru_eviction(self, pool):
        """When pool is full, LRU instance should be evicted."""
        # Fill the pool (max 3)
        await pool.get("tenant1")
        await pool.get("tenant2")
        await pool.get("tenant3")
        assert pool.size == 3

        # Access tenant1 to make it most recently used
        await pool.get("tenant1")

        # Add a new tenant, should evict tenant2 (LRU)
        await pool.get("tenant4")
        assert pool.size == 3
        assert "tenant2" not in pool._instances
        assert "tenant1" in pool._instances
        assert "tenant3" in pool._instances
        assert "tenant4" in pool._instances

    async def test_invalid_workspace_id_rejected(self, pool):
        """Invalid workspace identifiers should be rejected."""
        with pytest.raises(ValueError):
            await pool.get("")
        with pytest.raises(ValueError):
            await pool.get("-invalid")

    async def test_finalize_all(self, pool):
        """finalize_all should clean up all instances."""
        await pool.get("tenant1")
        await pool.get("tenant2")
        assert pool.size == 2

        await pool.finalize_all()
        assert pool.size == 0


class TestGetWorkspaceFromRequest:
    """Tests for header extraction from requests."""

    def test_primary_header(self):
        """LIGHTRAG-WORKSPACE header should be used as primary."""
        request = MagicMock()
        request.headers = {"LIGHTRAG-WORKSPACE": "tenant1"}
        assert get_workspace_from_request(request) == "tenant1"

    def test_fallback_header(self):
        """X-Workspace-ID should be used as fallback."""
        request = MagicMock()
        request.headers = {"X-Workspace-ID": "tenant2"}
        assert get_workspace_from_request(request) == "tenant2"

    def test_primary_takes_precedence(self):
        """LIGHTRAG-WORKSPACE should take precedence over X-Workspace-ID."""
        request = MagicMock()
        request.headers = {
            "LIGHTRAG-WORKSPACE": "primary",
            "X-Workspace-ID": "fallback",
        }
        assert get_workspace_from_request(request) == "primary"

    def test_no_header_returns_none(self):
        """Missing headers should return None."""
        request = MagicMock()
        request.headers = {}
        assert get_workspace_from_request(request) is None

    def test_empty_header_returns_none(self):
        """Empty header values should return None."""
        request = MagicMock()
        request.headers = {"LIGHTRAG-WORKSPACE": "  "}
        assert get_workspace_from_request(request) is None


# =============================================================================
# Phase 3: User Story 1+2 - Isolation & Routing Tests
# =============================================================================


@pytest.mark.integration
class TestWorkspaceIsolation:
    """T015-T016: Tests for workspace data isolation."""

    async def test_ingest_in_workspace_a_query_from_workspace_b_returns_nothing(self):
        """Documents ingested in workspace A should not be visible in workspace B."""
        # TODO: Implement with actual server integration
        pytest.skip("Integration test - requires running server")

    async def test_query_from_workspace_a_returns_own_documents(self):
        """Queries should return documents from the same workspace."""
        # TODO: Implement with actual server integration
        pytest.skip("Integration test - requires running server")


@pytest.mark.integration
class TestWorkspaceRouting:
    """T017-T019: Tests for header-based workspace routing."""

    async def test_lightrag_workspace_header_routes_correctly(self):
        """LIGHTRAG-WORKSPACE header should route to correct workspace."""
        # TODO: Implement with actual server integration
        pytest.skip("Integration test - requires running server")

    async def test_x_workspace_id_fallback_works(self):
        """X-Workspace-ID should work as fallback header."""
        # TODO: Implement with actual server integration
        pytest.skip("Integration test - requires running server")

    async def test_lightrag_workspace_takes_precedence(self):
        """LIGHTRAG-WORKSPACE should take precedence over X-Workspace-ID."""
        # TODO: Implement with actual server integration
        pytest.skip("Integration test - requires running server")


# =============================================================================
# Phase 4: User Story 3 - Backward Compatibility Tests
# =============================================================================


@pytest.mark.integration
class TestBackwardCompatibility:
    """T031-T033: Tests for backward compatibility."""

    async def test_no_header_uses_workspace_env_var(self):
        """Requests without headers should use WORKSPACE env var."""
        # TODO: Implement with actual server integration
        pytest.skip("Integration test - requires running server")

    async def test_existing_routes_unchanged(self):
        """Existing route paths should remain unchanged."""
        # TODO: Implement with actual server integration
        pytest.skip("Integration test - requires running server")

    async def test_response_formats_unchanged(self):
        """Response formats should remain unchanged."""
        # TODO: Implement with actual server integration
        pytest.skip("Integration test - requires running server")


# =============================================================================
# Phase 5: User Story 4 - Strict Mode Tests
# =============================================================================


@pytest.mark.integration
class TestStrictMode:
    """T038-T040: Tests for strict multi-tenant mode."""

    async def test_missing_header_returns_400_when_default_disabled(self):
        """Missing header should return 400 when default workspace disabled."""
        # TODO: Implement with actual server integration
        pytest.skip("Integration test - requires running server")

    async def test_error_message_indicates_missing_header(self):
        """Error message should clearly indicate missing header."""
        # TODO: Implement with actual server integration
        pytest.skip("Integration test - requires running server")

    async def test_missing_header_uses_default_when_enabled(self):
        """Missing header should use default when enabled."""
        # TODO: Implement with actual server integration
        pytest.skip("Integration test - requires running server")


# =============================================================================
# Phase 6: User Story 5 - Pool Management Tests
# =============================================================================


class TestPoolManagement:
    """T045-T048: Tests for workspace pool management."""

    @pytest.fixture
    def mock_rag_factory(self):
        """Create a mock RAG factory with initialization tracking."""
        init_count = {"value": 0}

        async def factory(workspace_id: str):
            init_count["value"] += 1
            mock_rag = MagicMock()
            mock_rag.workspace = workspace_id
            mock_rag.init_order = init_count["value"]
            mock_rag.finalize_storages = AsyncMock()
            return mock_rag

        factory.init_count = init_count
        return factory

    async def test_new_workspace_initializes_on_first_request(self, mock_rag_factory):
        """New workspace should initialize on first request."""
        config = WorkspaceConfig(max_workspaces_in_pool=5)
        pool = WorkspacePool(config, mock_rag_factory)

        rag = await pool.get("new-workspace")
        assert rag.workspace == "new-workspace"
        assert mock_rag_factory.init_count["value"] == 1

    async def test_lru_eviction_when_pool_full(self, mock_rag_factory):
        """LRU workspace should be evicted when pool is full."""
        config = WorkspaceConfig(max_workspaces_in_pool=2)
        pool = WorkspacePool(config, mock_rag_factory)

        await pool.get("workspace1")
        await pool.get("workspace2")
        assert pool.size == 2

        await pool.get("workspace3")
        assert pool.size == 2
        assert "workspace1" not in pool._instances

    async def test_concurrent_requests_share_initialization(self, mock_rag_factory):
        """Concurrent requests for same workspace should share initialization."""
        config = WorkspaceConfig(max_workspaces_in_pool=5)
        pool = WorkspacePool(config, mock_rag_factory)

        # Start multiple concurrent requests
        results = await asyncio.gather(
            pool.get("shared-workspace"),
            pool.get("shared-workspace"),
            pool.get("shared-workspace"),
        )

        # All should return the same instance
        assert results[0] is results[1] is results[2]
        # Only one initialization should have occurred
        assert mock_rag_factory.init_count["value"] == 1

    async def test_max_workspaces_config_respected(self, mock_rag_factory):
        """Pool should respect max workspaces configuration."""
        config = WorkspaceConfig(max_workspaces_in_pool=3)
        pool = WorkspacePool(config, mock_rag_factory)

        for i in range(5):
            await pool.get(f"workspace{i}")

        assert pool.size == 3
        assert pool.max_size == 3
