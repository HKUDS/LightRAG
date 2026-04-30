"""Unit tests for WorkspaceManager and sanitize_workspace_name."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from lightrag.api.utils import sanitize_workspace_name, WorkspaceNameError
from lightrag.api.workspace_manager import WorkspaceManager, WorkspaceCapacityError


class MockLightRAG:
    """Mock LightRAG instance for testing."""

    def __init__(self, workspace: str) -> None:
        self.workspace = workspace
        self.finalize_called = False
        self._finalize_task = None

    async def finalize_storages(self) -> None:
        """Mock finalize method that sets a flag."""
        self.finalize_called = True


async def mock_factory(workspace: str) -> MockLightRAG:
    """Factory function that creates MockLightRAG instances."""
    return MockLightRAG(workspace)


# =============================================================================
# sanitize_workspace_name Tests
# =============================================================================


class TestSanitizeWorkspaceName:
    """Tests for the sanitize_workspace_name function."""

    def test_none_returns_empty(self) -> None:
        """Test that None input returns empty string."""
        assert sanitize_workspace_name(None) == ""

    def test_empty_returns_empty(self) -> None:
        """Test that empty string input returns empty string."""
        assert sanitize_workspace_name("") == ""

    def test_whitespace_stripped(self) -> None:
        """Test that leading/trailing whitespace is stripped."""
        assert sanitize_workspace_name("  my-workspace  ") == "my-workspace"

    def test_lowercased(self) -> None:
        """Test that uppercase letters are converted to lowercase."""
        assert sanitize_workspace_name("MyWorkspace") == "myworkspace"

    def test_valid_name(self) -> None:
        """Test that valid names pass through unchanged (after lowercasing)."""
        assert sanitize_workspace_name("ws-1_test") == "ws-1_test"

    def test_path_traversal_rejected(self) -> None:
        """Test that path traversal attempts are rejected."""
        with pytest.raises(WorkspaceNameError, match="path traversal detected"):
            sanitize_workspace_name("../etc")

    def test_slash_rejected(self) -> None:
        """Test that forward slashes are rejected."""
        with pytest.raises(WorkspaceNameError, match="path traversal detected"):
            sanitize_workspace_name("ws/name")

    def test_backslash_rejected(self) -> None:
        """Test that backslashes are rejected."""
        with pytest.raises(WorkspaceNameError, match="path traversal detected"):
            sanitize_workspace_name("ws\\name")

    def test_too_long_rejected(self) -> None:
        """Test that names exceeding 64 characters are rejected."""
        long_name = "a" * 65
        with pytest.raises(WorkspaceNameError, match="max 64 characters"):
            sanitize_workspace_name(long_name)

    def test_max_length_accepted(self) -> None:
        """Test that 64-character names are accepted."""
        max_name = "a" * 64
        assert sanitize_workspace_name(max_name) == max_name

    def test_special_chars_rejected(self) -> None:
        """Test that special characters are rejected."""
        with pytest.raises(
            WorkspaceNameError, match="only lowercase letters.*allowed"
        ):
            sanitize_workspace_name("ws!@#")

    # -------------------------------------------------------------------------
    # Additional Edge Cases
    # -------------------------------------------------------------------------

    def test_null_bytes_rejected(self) -> None:
        """Test that null bytes in name are rejected."""
        with pytest.raises(WorkspaceNameError, match="only lowercase letters.*allowed"):
            sanitize_workspace_name("ws\x00name")

    def test_unicode_chars_rejected(self) -> None:
        """Test that unicode characters are rejected."""
        with pytest.raises(
            WorkspaceNameError, match="only lowercase letters.*allowed"
        ):
            sanitize_workspace_name("workspace日本語")

    def test_mixed_valid_invalid_space_rejected(self) -> None:
        """Test that space in name is rejected."""
        with pytest.raises(
            WorkspaceNameError, match="only lowercase letters.*allowed"
        ):
            sanitize_workspace_name("ws-1 test")

    def test_exactly_64_chars_valid(self) -> None:
        """Test that exactly 64-character names are valid."""
        name_64 = "a" * 64
        assert sanitize_workspace_name(name_64) == name_64

    def test_65_chars_rejected(self) -> None:
        """Test that 65-character names are rejected."""
        name_65 = "a" * 65
        with pytest.raises(WorkspaceNameError, match="max 64 characters"):
            sanitize_workspace_name(name_65)

    def test_leading_trailing_spaces_trimmed_and_lower(self) -> None:
        """Test that leading/trailing spaces are trimmed and lowercased."""
        assert sanitize_workspace_name("  MyWorkspace  ") == "myworkspace"


# =============================================================================
# WorkspaceManager Tests
# =============================================================================


class TestWorkspaceManager:
    """Tests for the WorkspaceManager class."""

    @pytest.fixture
    def fresh_manager(self) -> WorkspaceManager:
        """Create a fresh WorkspaceManager instance for each test."""
        return WorkspaceManager(factory=mock_factory, max_instances=10)

    @pytest.fixture
    def small_manager(self) -> WorkspaceManager:
        """Create a WorkspaceManager with small capacity for eviction tests."""
        return WorkspaceManager(factory=mock_factory, max_instances=3)

    # -------------------------------------------------------------------------
    # Basic get_or_create Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_or_create_creates_instance(self, fresh_manager) -> None:
        """Test that first call to get_or_create creates a new instance."""
        instance = await fresh_manager.get_or_create("ws-a")
        assert isinstance(instance, MockLightRAG)
        assert instance.workspace == "ws-a"
        assert fresh_manager.get_stats()["active_instances"] == 1

    @pytest.mark.asyncio
    async def test_get_or_create_returns_cached(self, fresh_manager) -> None:
        """Test that second call returns the same cached instance."""
        instance1 = await fresh_manager.get_or_create("ws-a")
        instance2 = await fresh_manager.get_or_create("ws-a")
        assert instance1 is instance2
        assert fresh_manager.get_stats()["active_instances"] == 1
        assert fresh_manager.get_stats()["cache_hits"] == 1

    # -------------------------------------------------------------------------
    # LRU Eviction Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_lru_eviction_at_capacity(self, small_manager) -> None:
        """Test LRU eviction when cache reaches capacity."""
        # Create 3 workspaces (fills cache)
        ws_a = await small_manager.get_or_create("ws-a")
        ws_b = await small_manager.get_or_create("ws-b")
        ws_c = await small_manager.get_or_create("ws-c")

        # All should be in cache
        assert small_manager.get_stats()["active_instances"] == 3

        # Release ws-a so it can be evicted
        small_manager.release("ws-a")

        # Create ws-d - should evict ws-a (LRU with ref_count=0)
        ws_d = await small_manager.get_or_create("ws-d")

        # ws-d should be in cache
        assert small_manager.get_stats()["active_instances"] == 3
        # ws-a should have been evicted
        assert ws_a.finalize_called is True

    @pytest.mark.asyncio
    async def test_eviction_skips_in_flight(self, small_manager) -> None:
        """Test that in-flight workspaces (ref_count > 0) are not evicted."""
        # Acquire ws-1 (don't release)
        ws1 = await small_manager.get_or_create("ws-1")

        # Fill cache with others (all will have ref_count > 0)
        ws2 = await small_manager.get_or_create("ws-2")
        ws3 = await small_manager.get_or_create("ws-3")

        # Cache is full, all have ref_count > 0
        assert small_manager.get_stats()["active_instances"] == 3

        # Try to create ws-4 - should evict ws-2 (LRU with ref_count=1, but will be decremented)
        # Actually, ws-2 should have ref_count=1 since we got it, so it won't be evicted
        # The eviction will evict the LRU with ref_count == 0
        # Since ws-1, ws-2, ws-3 all have ref_count > 0, no eviction should happen
        # Wait, the code checks len >= max_instances first, then all(ref > 0)
        # If all have ref > 0, it raises WorkspaceCapacityError
        # But the per-workspace lock prevents the same workspace from being created twice

        # Let's release ws-2 to make it evictable
        small_manager.release("ws-2")

        # Now ws-4 should evict ws-2
        ws4 = await small_manager.get_or_create("ws-4")
        assert small_manager.get_stats()["active_instances"] == 3

        # ws-1 should NOT have been evicted (ref_count > 0)
        assert ws1.finalize_called is False
        assert small_manager.get_stats()["ref_counts"]["ws-1"] == 1

    @pytest.mark.asyncio
    async def test_lru_eviction_order_verification(self, small_manager) -> None:
        """Test LRU eviction picks LEAST recently used, not just oldest created.

        Scenario:
        - Create ws-1, ws-2, ws-3 (in that order)
        - Access ws-1 again (moves it to end of LRU)
        - Release ws-2 (ref_count=0)
        - Create ws-4 → should evict ws-2 (least recently used with ref_count=0),
          NOT ws-3 (oldest created but not LRU anymore)
        """
        # Create 3 workspaces (fills cache)
        ws_1 = await small_manager.get_or_create("ws-1")
        ws_2 = await small_manager.get_or_create("ws-2")
        ws_3 = await small_manager.get_or_create("ws-3")

        # Now access order is: ws-1, ws-2, ws-3 (ws-1 was created first)

        # Access ws-1 again - this moves it to end of LRU order
        # Now LRU order is: ws-2 (oldest accessed), ws-3, ws-1 (most recent)
        await small_manager.get_or_create("ws-1")

        # Release ws-2 (ref_count=0, but ws-1 is most recent, ws-3 is middle)
        small_manager.release("ws-2")

        # Create ws-4 - should evict ws-2 (LRU with ref_count=0)
        ws_4 = await small_manager.get_or_create("ws-4")

        # Verify ws-4 is in cache
        assert small_manager.get_stats()["active_instances"] == 3

        # ws-2 should have been evicted (least recently used with ref_count=0)
        assert ws_2.finalize_called is True

        # ws-1 should NOT have been evicted (ref_count > 0 and was recently accessed)
        assert ws_1.finalize_called is False

        # ws-3 should NOT have been evicted
        assert ws_3.finalize_called is False

        # Verify ws-4 is usable
        assert ws_4.workspace == "ws-4"

    @pytest.mark.asyncio
    async def test_eviction_of_most_recently_accessed(self, small_manager) -> None:
        """Test eviction respects access order, not creation order.

        Scenario:
        - Create ws-1, ws-2, ws-3 (in that order)
        - Access ws-1 again (move to end of LRU)
        - Release ws-2
        - Create ws-4
        - Verify ws-2 was evicted (not ws-1, which was recently accessed)
        """
        # Create ws-1, ws-2, ws-3 (fills capacity)
        ws_1 = await small_manager.get_or_create("ws-1")
        ws_2 = await small_manager.get_or_create("ws-2")
        ws_3 = await small_manager.get_or_create("ws-3")

        # Access ws-1 again to move it to end (most recently accessed)
        await small_manager.get_or_create("ws-1")

        # Release ws-2 (now ref_count=0, LRU candidate)
        small_manager.release("ws-2")

        # Create ws-4 - should evict ws-2
        ws_4 = await small_manager.get_or_create("ws-4")

        # Verify state
        assert small_manager.get_stats()["active_instances"] == 3

        # ws-2 should be evicted (oldest accessed with ref_count=0)
        assert ws_2.finalize_called is True

        # ws-1 should NOT be evicted (was most recently accessed)
        assert ws_1.finalize_called is False
        assert small_manager.get_stats()["ref_counts"]["ws-1"] == 2  # created + accessed

        # ws-3 should NOT be evicted
        assert ws_3.finalize_called is False

    @pytest.mark.asyncio
    async def test_reference_counting_exact_tracking(self, fresh_manager) -> None:
        """Test reference counting with exact tracking through multiple operations.

        Scenario:
        - get_or_create 3 times for same workspace → ref_count=3
        - release 2 times → ref_count=1
        - get_or_create again → ref_count=2
        """
        # Get ws-a 3 times → ref_count should be 3
        await fresh_manager.get_or_create("ws-a")
        await fresh_manager.get_or_create("ws-a")
        ws_a = await fresh_manager.get_or_create("ws-a")

        assert fresh_manager.get_stats()["ref_counts"]["ws-a"] == 3

        # Release twice → ref_count should be 1
        fresh_manager.release("ws-a")
        fresh_manager.release("ws-a")

        assert fresh_manager.get_stats()["ref_counts"]["ws-a"] == 1

        # Get again → ref_count should be 2
        await fresh_manager.get_or_create("ws-a")

        assert fresh_manager.get_stats()["ref_counts"]["ws-a"] == 2

        # Verify active instances is still 1 (same workspace)
        assert fresh_manager.get_stats()["active_instances"] == 1

    @pytest.mark.asyncio
    async def test_workspace_capacity_error_recovery_after_release(
        self, small_manager
    ) -> None:
        """Test recovery from WorkspaceCapacityError after releasing a workspace.

        Scenario:
        - Fill capacity (max=3), all with ref_count > 0
        - WorkspaceCapacityError on 4th attempt
        - Release one workspace
        - Retry → should succeed
        - Verify the newly created workspace is in cache
        """
        # Fill cache with 3 workspaces (all with ref_count > 0)
        ws_1 = await small_manager.get_or_create("ws-1")
        ws_2 = await small_manager.get_or_create("ws-2")
        ws_3 = await small_manager.get_or_create("ws-3")

        assert small_manager.get_stats()["active_instances"] == 3

        # Try to create 4th workspace - should raise WorkspaceCapacityError
        with pytest.raises(WorkspaceCapacityError):
            await small_manager.get_or_create("ws-4")

        # Release ws-2 to free up a slot
        small_manager.release("ws-2")

        # Retry - should succeed now
        ws_4 = await small_manager.get_or_create("ws-4")

        assert ws_4 is not None
        assert ws_4.workspace == "ws-4"

        # Verify ws-4 is in cache
        stats = small_manager.get_stats()
        assert stats["active_instances"] == 3
        assert "ws-4" in stats["workspaces"]

        # ws-2 should have been evicted
        assert ws_2.finalize_called is True

        # ws-1 and ws-3 should still be active
        assert ws_1.finalize_called is False
        assert ws_3.finalize_called is False

    @pytest.mark.asyncio
    async def test_capacity_error_when_all_busy(self, small_manager) -> None:
        """Test WorkspaceCapacityError when all slots have in-flight requests."""
        # Fill cache with all in-flight workspaces
        await small_manager.get_or_create("ws-1")
        await small_manager.get_or_create("ws-2")
        await small_manager.get_or_create("ws-3")

        # All have ref_count > 0, next request should raise error
        with pytest.raises(WorkspaceCapacityError):
            await small_manager.get_or_create("ws-4")

    @pytest.mark.asyncio
    async def test_release_decrements_ref_count(self, fresh_manager) -> None:
        """Test that release decrements the reference count."""
        instance1 = await fresh_manager.get_or_create("ws-a")
        instance2 = await fresh_manager.get_or_create("ws-a")

        assert fresh_manager.get_stats()["ref_counts"]["ws-a"] == 2

        fresh_manager.release("ws-a")

        assert fresh_manager.get_stats()["ref_counts"]["ws-a"] == 1

    @pytest.mark.asyncio
    async def test_finalize_called_on_eviction(self, small_manager) -> None:
        """Test that finalize_storages is called when workspace is evicted."""
        # Create and release ws-a
        ws_a = await small_manager.get_or_create("ws-a")
        small_manager.release("ws-a")

        # Create ws-b, ws-c to fill cache
        await small_manager.get_or_create("ws-b")
        await small_manager.get_or_create("ws-c")

        # Create ws-d to trigger eviction of ws-a
        await small_manager.get_or_create("ws-d")

        # ws-a should have been finalized
        assert ws_a.finalize_called is True

    @pytest.mark.asyncio
    async def test_finalize_called_on_shutdown(self, fresh_manager) -> None:
        """Test that all instances are finalized on shutdown."""
        ws1 = await fresh_manager.get_or_create("ws-1")
        ws2 = await fresh_manager.get_or_create("ws-2")

        await fresh_manager.shutdown()

        assert ws1.finalize_called is True
        assert ws2.finalize_called is True

    @pytest.mark.asyncio
    async def test_shutdown_with_active_instances(self, fresh_manager) -> None:
        """Test shutdown finalizes ALL instances including those with ref_count > 0.

        After shutdown:
        - get_stats should show 0 active instances
        - workspaces list should be empty
        - All finalize_called flags should be True
        """
        # Create 3 workspaces with ref_count > 0 (don't release)
        ws1 = await fresh_manager.get_or_create("ws-1")
        ws2 = await fresh_manager.get_or_create("ws-2")
        ws3 = await fresh_manager.get_or_create("ws-3")

        # All have ref_count > 0
        assert fresh_manager.get_stats()["active_instances"] == 3
        assert fresh_manager.get_stats()["ref_counts"]["ws-1"] == 1
        assert fresh_manager.get_stats()["ref_counts"]["ws-2"] == 1
        assert fresh_manager.get_stats()["ref_counts"]["ws-3"] == 1

        # Shutdown should finalize ALL instances
        await fresh_manager.shutdown()

        # All instances should be finalized
        assert ws1.finalize_called is True
        assert ws2.finalize_called is True
        assert ws3.finalize_called is True

        # Stats should show 0 active instances
        stats = fresh_manager.get_stats()
        assert stats["active_instances"] == 0
        assert stats["workspaces"] == []

    @pytest.mark.asyncio
    async def test_shutdown_clears_cache(self, fresh_manager) -> None:
        """Test that shutdown clears the cache."""
        await fresh_manager.get_or_create("ws-1")
        await fresh_manager.get_or_create("ws-2")

        assert fresh_manager.get_stats()["active_instances"] == 2

        await fresh_manager.shutdown()

        assert fresh_manager.get_stats()["active_instances"] == 0
        assert fresh_manager.get_stats()["workspaces"] == []

    @pytest.mark.asyncio
    async def test_get_stats_accuracy(self, fresh_manager) -> None:
        """Test that get_stats returns accurate information."""
        # Create ws-a (cache miss)
        ws_a = await fresh_manager.get_or_create("ws-a")

        # Get ws-a again (cache hit)
        await fresh_manager.get_or_create("ws-a")

        # Create ws-b (cache miss)
        ws_b = await fresh_manager.get_or_create("ws-b")

        # Get ws-b again (cache hit)
        await fresh_manager.get_or_create("ws-b")

        # Release ws-a
        fresh_manager.release("ws-a")

        stats = fresh_manager.get_stats()

        assert stats["active_instances"] == 2
        assert stats["max_instances"] == 10
        assert stats["cache_hits"] == 2
        assert stats["cache_misses"] == 2
        assert stats["evictions"] == 0
        assert set(stats["workspaces"]) == {"ws-a", "ws-b"}
        assert stats["ref_counts"]["ws-a"] == 1
        assert stats["ref_counts"]["ws-b"] == 2

    @pytest.mark.asyncio
    async def test_stats_reporting_accuracy(self, small_manager) -> None:
        """Test stats reporting with cache_hits, cache_misses, evictions, and ref_counts.

        Extended version that verifies:
        - cache_hits and cache_misses are correctly tracked
        - evictions counter increments on each eviction
        - workspaces list matches actual cache contents
        - ref_counts dictionary is accurate
        """
        # Create 3 workspaces (3 cache misses)
        ws_1 = await small_manager.get_or_create("ws-1")
        ws_2 = await small_manager.get_or_create("ws-2")
        ws_3 = await small_manager.get_or_create("ws-3")

        # Get ws-1 twice (2 cache hits)
        await small_manager.get_or_create("ws-1")
        await small_manager.get_or_create("ws-1")

        stats = small_manager.get_stats()
        assert stats["cache_misses"] == 3
        assert stats["cache_hits"] == 2
        assert stats["evictions"] == 0
        assert stats["ref_counts"]["ws-1"] == 3
        assert stats["ref_counts"]["ws-2"] == 1
        assert stats["ref_counts"]["ws-3"] == 1

        # Release ws-1 twice (ref_count goes from 3 to 1)
        small_manager.release("ws-1")
        small_manager.release("ws-1")

        # Release ws-2 (ref_count goes from 1 to 0)
        small_manager.release("ws-2")

        # Create ws-4 → should evict ws-2 (eviction #1)
        ws_4 = await small_manager.get_or_create("ws-4")

        stats = small_manager.get_stats()
        assert stats["cache_misses"] == 4
        assert stats["cache_hits"] == 2
        assert stats["evictions"] == 1
        assert set(stats["workspaces"]) == {"ws-1", "ws-3", "ws-4"}
        assert stats["ref_counts"]["ws-1"] == 1
        # ws-2 was evicted and is no longer in ref_counts
        assert "ws-2" not in stats["ref_counts"]
        assert stats["ref_counts"]["ws-3"] == 1
        assert stats["ref_counts"]["ws-4"] == 1

        # Release ws-3 (ref_count goes from 1 to 0)
        small_manager.release("ws-3")

        # Create ws-5 → should evict ws-3 (eviction #2)
        ws_5 = await small_manager.get_or_create("ws-5")

        stats = small_manager.get_stats()
        assert stats["evictions"] == 2
        assert set(stats["workspaces"]) == {"ws-1", "ws-4", "ws-5"}
        # ws-3 was evicted and is no longer in ref_counts
        assert "ws-3" not in stats["ref_counts"]

    @pytest.mark.asyncio
    async def test_none_workspace_maps_to_empty(self, fresh_manager) -> None:
        """Test that None workspace uses empty string as key."""
        instance1 = await fresh_manager.get_or_create(None)
        instance2 = await fresh_manager.get_or_create("")

        # Both should return the same instance (empty string key)
        assert instance1 is instance2
        assert instance1.workspace == ""
        assert fresh_manager.get_stats()["active_instances"] == 1

    @pytest.mark.asyncio
    async def test_release_nonexistent_logs_warning(self, fresh_manager) -> None:
        """Test that releasing unknown workspace doesn't crash."""
        # Should not raise, just log a warning
        fresh_manager.release("nonexistent-workspace")
        fresh_manager.release(None)  # Should be fine since no workspaces exist

    @pytest.mark.asyncio
    async def test_async_lock_prevents_double_creation(self, fresh_manager) -> None:
        """Test that concurrent get_or_create for same workspace creates only one instance."""
        workspace_name = "concurrent-ws"

        # Create multiple concurrent requests for the same workspace
        async def get_workspace() -> MockLightRAG:
            return await fresh_manager.get_or_create(workspace_name)

        results = await asyncio.gather(
            get_workspace(),
            get_workspace(),
            get_workspace(),
        )

        # All should return the same instance
        assert all(r is results[0] for r in results)
        assert fresh_manager.get_stats()["active_instances"] == 1
        assert fresh_manager.get_stats()["cache_hits"] == 2  # 2nd and 3rd are hits
        assert fresh_manager.get_stats()["cache_misses"] == 1  # 1st is miss

    @pytest.mark.asyncio
    async def test_concurrent_get_or_create_deduplication(self, fresh_manager) -> None:
        """Test factory is called exactly ONCE even with 10 concurrent requests.

        Uses a side-effect counter to track factory calls.
        """
        factory_call_count = 0

        async def counting_factory(workspace: str) -> MockLightRAG:
            nonlocal factory_call_count
            factory_call_count += 1
            # Small delay to increase chance of race condition
            await asyncio.sleep(0.01)
            return MockLightRAG(workspace)

        manager = WorkspaceManager(factory=counting_factory, max_instances=10)
        workspace_name = "concurrent-dedup-ws"

        async def get_workspace() -> MockLightRAG:
            return await manager.get_or_create(workspace_name)

        # Launch 10 concurrent requests
        results = await asyncio.gather(
            *[get_workspace() for _ in range(10)]
        )

        # Factory should be called exactly once
        assert factory_call_count == 1

        # All results should be the same instance
        assert all(r is results[0] for r in results)

        # Stats should show 1 cache miss and 9 cache hits
        stats = manager.get_stats()
        assert stats["cache_misses"] == 1
        assert stats["cache_hits"] == 9
        assert stats["active_instances"] == 1

    # -------------------------------------------------------------------------
    # Edge Cases
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_multiple_releases_no_underflow(self, fresh_manager) -> None:
        """Test that multiple releases don't cause ref_count underflow."""
        await fresh_manager.get_or_create("ws-a")

        fresh_manager.release("ws-a")
        fresh_manager.release("ws-a")
        fresh_manager.release("ws-a")

        # Should not crash, ref_count should be 0 (clamped)
        assert fresh_manager.get_stats()["ref_counts"]["ws-a"] == 0

    @pytest.mark.asyncio
    async def test_different_workspaces_independent(self, fresh_manager) -> None:
        """Test that different workspaces are managed independently."""
        ws1 = await fresh_manager.get_or_create("ws-1")
        ws2 = await fresh_manager.get_or_create("ws-2")

        assert ws1 is not ws2
        assert ws1.workspace == "ws-1"
        assert ws2.workspace == "ws-2"

        fresh_manager.release("ws-1")
        fresh_manager.release("ws-1")  # Release twice

        # ws-2 should still have ref_count > 0
        assert fresh_manager.get_stats()["ref_counts"]["ws-2"] == 1
