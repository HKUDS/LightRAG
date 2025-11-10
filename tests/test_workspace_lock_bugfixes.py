"""
Test bug fixes for workspace lock implementation.

This test module specifically verifies the bug fixes for:
1. RuntimeError when _registry_guard is None
2. Per-process workspace async locks in multiprocess mode
"""

import pytest
from unittest.mock import patch
from lightrag.kg.shared_storage import (
    _get_workspace_lock,
    initialize_share_data,
)


def test_error_when_not_initialized():
    """Test that _get_workspace_lock raises RuntimeError when called before initialize_share_data().

    Bug Fix: Previously, calling _get_workspace_lock() before initialize_share_data()
    would cause TypeError: 'NoneType' object does not support the context manager protocol
    because _registry_guard was None.

    Now it should raise a clear RuntimeError with helpful message.
    """
    # Mock _is_multiprocess to be True to trigger the bug path
    with patch("lightrag.kg.shared_storage._is_multiprocess", True):
        with patch("lightrag.kg.shared_storage._registry_guard", None):
            with pytest.raises(RuntimeError, match="Shared data not initialized"):
                # This should raise RuntimeError, not TypeError
                _get_workspace_lock(
                    "test_lock", None, workspace="tenant1", enable_logging=False
                )


def test_workspace_async_locks_per_process():
    """Test that workspace async locks are stored per-process.

    Bug Fix: Previously, workspace async locks were added to _async_locks dict,
    which is a regular Python dict in multiprocess mode. This meant modifications
    in one process were not visible to other processes.

    Now workspace async locks are stored in _workspace_async_locks, which is
    per-process by design (since asyncio.Lock cannot be shared across processes).
    """
    from lightrag.kg import shared_storage

    # Initialize in multiprocess mode
    initialize_share_data(workers=2)

    # Get a workspace lock
    lock1 = shared_storage.get_storage_lock(workspace="tenant1")

    # Verify the workspace async lock exists in _workspace_async_locks
    assert "tenant1:storage_lock" in shared_storage._workspace_async_locks

    # Get the same workspace lock again
    lock2 = shared_storage.get_storage_lock(workspace="tenant1")

    # Both locks should reference the same async_lock instance
    # (within the same process)
    assert lock1._async_lock is lock2._async_lock

    # Verify it's the same as the one stored in _workspace_async_locks
    assert lock1._async_lock is shared_storage._workspace_async_locks["tenant1:storage_lock"]


def test_multiple_workspace_locks_different_async_locks():
    """Test that different workspaces have different async locks."""
    from lightrag.kg import shared_storage

    # Initialize in multiprocess mode
    initialize_share_data(workers=2)

    # Get locks for different workspaces
    lock1 = shared_storage.get_storage_lock(workspace="tenant1")
    lock2 = shared_storage.get_storage_lock(workspace="tenant2")

    # They should have different async_lock instances
    assert lock1._async_lock is not lock2._async_lock

    # Verify both are in _workspace_async_locks
    assert "tenant1:storage_lock" in shared_storage._workspace_async_locks
    assert "tenant2:storage_lock" in shared_storage._workspace_async_locks
