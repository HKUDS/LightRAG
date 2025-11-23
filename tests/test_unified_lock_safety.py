"""
Tests for UnifiedLock safety when lock is None.

This test module verifies that UnifiedLock raises RuntimeError instead of
allowing unprotected execution when the underlying lock is None, preventing
false security and potential race conditions.

Critical Bug 1: When self._lock is None, __aenter__ used to log WARNING but
still return successfully, allowing critical sections to run without lock
protection, causing race conditions and data corruption.

Critical Bug 2: In __aexit__, when async_lock.release() fails, the error
recovery logic would attempt to release it again, causing double-release issues.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock
from lightrag.kg.shared_storage import UnifiedLock


class TestUnifiedLockSafety:
    """Test suite for UnifiedLock None safety checks."""

    @pytest.mark.asyncio
    async def test_unified_lock_raises_on_none_async(self):
        """
        Test that UnifiedLock raises RuntimeError when lock is None (async mode).

        Scenario: Attempt to use UnifiedLock before initialize_share_data() is called.
        Expected: RuntimeError raised, preventing unprotected critical section execution.
        """
        lock = UnifiedLock(
            lock=None, is_async=True, name="test_async_lock", enable_logging=False
        )

        with pytest.raises(
            RuntimeError, match="shared data not initialized|Lock.*is None"
        ):
            async with lock:
                # This code should NEVER execute
                pytest.fail(
                    "Code inside lock context should not execute when lock is None"
                )

    @pytest.mark.asyncio
    async def test_unified_lock_raises_on_none_sync(self):
        """
        Test that UnifiedLock raises RuntimeError when lock is None (sync mode).

        Scenario: Attempt to use UnifiedLock with None lock in sync mode.
        Expected: RuntimeError raised with clear error message.
        """
        lock = UnifiedLock(
            lock=None, is_async=False, name="test_sync_lock", enable_logging=False
        )

        with pytest.raises(
            RuntimeError, match="shared data not initialized|Lock.*is None"
        ):
            async with lock:
                # This code should NEVER execute
                pytest.fail(
                    "Code inside lock context should not execute when lock is None"
                )

    @pytest.mark.asyncio
    async def test_error_message_clarity(self):
        """
        Test that the error message clearly indicates the problem and solution.

        Scenario: Lock is None and user tries to acquire it.
        Expected: Error message mentions 'shared data not initialized' and
                  'initialize_share_data()'.
        """
        lock = UnifiedLock(
            lock=None,
            is_async=True,
            name="test_error_message",
            enable_logging=False,
        )

        with pytest.raises(RuntimeError) as exc_info:
            async with lock:
                pass

        error_message = str(exc_info.value)
        # Verify error message contains helpful information
        assert (
            "shared data not initialized" in error_message.lower()
            or "lock" in error_message.lower()
        )
        assert "initialize_share_data" in error_message or "None" in error_message

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

    @pytest.mark.asyncio
    async def test_aexit_recovery_on_main_lock_failure(self):
        """
        Test that __aexit__ recovery logic works when main lock release fails.

        Scenario: main_lock.release() fails before async_lock is attempted.
        Expected: Recovery logic should attempt to release async_lock to prevent
                  resource leaks.

        This verifies the recovery logic still works correctly with async_lock_released tracking.
        """
        # Create mock locks
        main_lock = MagicMock()
        main_lock.acquire = MagicMock()

        # Make main_lock.release() fail
        def mock_main_release_fail():
            raise RuntimeError("Main lock release failed")

        main_lock.release = MagicMock(side_effect=mock_main_release_fail)

        async_lock = AsyncMock()
        async_lock.acquire = AsyncMock()
        async_lock.release = MagicMock()

        # Create UnifiedLock with both locks (sync mode with async_lock)
        lock = UnifiedLock(
            lock=main_lock, is_async=False, name="test_recovery", enable_logging=False
        )
        lock._async_lock = async_lock

        # Try to use the lock - should fail during __aexit__
        try:
            async with lock:
                pass
        except RuntimeError as e:
            # Should get the main lock release error
            assert "Main lock release failed" in str(e)

        # Main lock release should have been attempted
        main_lock.release.assert_called_once()

        # Recovery logic should have attempted to release async_lock
        async_lock.release.assert_called_once()
