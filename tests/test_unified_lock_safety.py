"""
Tests for UnifiedLock safety when lock is None.

This test module verifies that get_internal_lock() and get_data_init_lock()
raise RuntimeError when shared data is not initialized, preventing false
security and potential race conditions.

Design: The None check has been moved from UnifiedLock.__aenter__/__enter__
to the lock factory functions (get_internal_lock, get_data_init_lock) for
early failure detection.

Critical Bug 1 (Fixed): When self._lock is None, the code would fail with
AttributeError. Now the check is in factory functions for clearer errors.

Critical Bug 2: In __aexit__, when async_lock.release() fails, the error
recovery logic would attempt to release it again, causing double-release issues.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock
from lightrag.kg.shared_storage import (
    UnifiedLock,
    get_internal_lock,
    get_data_init_lock,
    finalize_share_data,
)


class TestUnifiedLockSafety:
    """Test suite for UnifiedLock None safety checks."""

    def setup_method(self):
        """Ensure shared data is finalized before each test."""
        finalize_share_data()

    def teardown_method(self):
        """Clean up after each test."""
        finalize_share_data()

    def test_get_internal_lock_raises_when_not_initialized(self):
        """
        Test that get_internal_lock() raises RuntimeError when shared data is not initialized.

        Scenario: Call get_internal_lock() before initialize_share_data() is called.
        Expected: RuntimeError raised with clear error message.

        This test verifies the None check has been moved to the factory function.
        """
        with pytest.raises(
            RuntimeError, match="Shared data not initialized.*initialize_share_data"
        ):
            get_internal_lock()

    def test_get_data_init_lock_raises_when_not_initialized(self):
        """
        Test that get_data_init_lock() raises RuntimeError when shared data is not initialized.

        Scenario: Call get_data_init_lock() before initialize_share_data() is called.
        Expected: RuntimeError raised with clear error message.

        This test verifies the None check has been moved to the factory function.
        """
        with pytest.raises(
            RuntimeError, match="Shared data not initialized.*initialize_share_data"
        ):
            get_data_init_lock()

    @pytest.mark.asyncio
    async def test_aexit_no_double_release_on_async_lock_failure(self):
        """
        Test that __aexit__ doesn't attempt to release async_lock twice when it fails.

        Scenario: async_lock.release() fails during normal release.
        Expected: Recovery logic should NOT attempt to release async_lock again,
                  preventing double-release issues.

        This tests Bug 2 fix: async_lock_released tracking prevents double release.
        """
        # Create mock locks
        main_lock = MagicMock()
        main_lock.acquire = MagicMock()
        main_lock.release = MagicMock()

        async_lock = AsyncMock()
        async_lock.acquire = AsyncMock()

        # Make async_lock.release() fail
        release_call_count = 0

        def mock_release_fail():
            nonlocal release_call_count
            release_call_count += 1
            raise RuntimeError("Async lock release failed")

        async_lock.release = MagicMock(side_effect=mock_release_fail)

        # Create UnifiedLock with both locks (sync mode with async_lock)
        lock = UnifiedLock(
            lock=main_lock,
            is_async=False,
            name="test_double_release",
            enable_logging=False,
        )
        lock._async_lock = async_lock

        # Try to use the lock - should fail during __aexit__
        try:
            async with lock:
                pass
        except RuntimeError as e:
            # Should get the async lock release error
            assert "Async lock release failed" in str(e)

        # Verify async_lock.release() was called only ONCE, not twice
        assert (
            release_call_count == 1
        ), f"async_lock.release() should be called only once, but was called {release_call_count} times"

        # Main lock should have been released successfully
        main_lock.release.assert_called_once()
