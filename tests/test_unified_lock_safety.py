"""
Tests for UnifiedLock safety when lock is None.

This test module verifies that UnifiedLock raises RuntimeError instead of
allowing unprotected execution when the underlying lock is None, preventing
false security and potential race conditions.

Critical Bug: When self._lock is None, __aenter__ used to log WARNING but
still return successfully, allowing critical sections to run without lock
protection, causing race conditions and data corruption.
"""

import pytest
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
