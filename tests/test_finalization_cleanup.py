"""
Test finalization cleanup for workspace locks.

This test module verifies that finalize_share_data() properly cleans up
all lock-related global variables, including:
- _sync_locks (Manager.dict in multiprocess mode)
- _workspace_async_locks (per-process dict)
- _lock_registry, _lock_registry_count, _lock_cleanup_data
- _storage_keyed_lock

Bug: Previously, these weren't properly cleaned up, causing EOFError/BrokenPipeError
when re-initializing after finalization.
"""

import pytest
from lightrag.kg import shared_storage


@pytest.fixture(autouse=True)
def cleanup_shared_storage():
    """Ensure shared storage is cleaned up after each test."""
    yield
    shared_storage.finalize_share_data()


def test_finalization_clears_workspace_locks():
    """Test that finalize_share_data() clears workspace lock dictionaries.

    Bug Fix: Previously, _sync_locks and _workspace_async_locks were not
    cleared during finalization, causing stale references to shut-down Manager.
    """
    # Initialize in multiprocess mode
    shared_storage.initialize_share_data(workers=2)

    # Create some workspace locks
    lock1 = shared_storage.get_storage_lock(workspace="tenant1")
    lock2 = shared_storage.get_pipeline_status_lock(workspace="tenant2")

    # Verify locks were created
    assert "tenant1:storage_lock" in shared_storage._sync_locks
    assert "tenant2:pipeline_status_lock" in shared_storage._sync_locks
    assert "tenant1:storage_lock" in shared_storage._workspace_async_locks

    # Finalize
    shared_storage.finalize_share_data()

    # Verify all lock-related globals are None
    assert shared_storage._sync_locks is None
    assert shared_storage._workspace_async_locks is None
    assert shared_storage._lock_registry is None
    assert shared_storage._lock_registry_count is None
    assert shared_storage._lock_cleanup_data is None
    assert shared_storage._registry_guard is None
    assert shared_storage._storage_keyed_lock is None
    assert shared_storage._manager is None


def test_reinitialize_after_finalization():
    """Test that re-initialization works after finalization.

    Bug Fix: Previously, stale references to shut-down Manager caused
    EOFError/BrokenPipeError when creating locks after re-initialization.
    """
    # First initialization
    shared_storage.initialize_share_data(workers=2)
    lock1 = shared_storage.get_storage_lock(workspace="tenant1")
    assert "tenant1:storage_lock" in shared_storage._sync_locks

    # Finalize
    shared_storage.finalize_share_data()
    assert shared_storage._manager is None

    # Re-initialize
    shared_storage.initialize_share_data(workers=2)

    # Should work without EOFError/BrokenPipeError
    lock2 = shared_storage.get_storage_lock(workspace="tenant2")
    assert "tenant2:storage_lock" in shared_storage._sync_locks

    # Clean up
    shared_storage.finalize_share_data()


def test_single_process_finalization():
    """Test finalization in single-process mode.

    Ensures finalization works correctly when not using multiprocess Manager.
    """
    # Initialize in single-process mode
    shared_storage.initialize_share_data(workers=1)

    # Create some workspace locks
    lock1 = shared_storage.get_storage_lock(workspace="tenant1")
    assert "tenant1:storage_lock" in shared_storage._sync_locks

    # Finalize
    shared_storage.finalize_share_data()

    # Verify globals are None
    assert shared_storage._sync_locks is None
    assert shared_storage._workspace_async_locks is None
    assert shared_storage._manager is None  # Should be None even in single-process

    # Re-initialize should work
    shared_storage.initialize_share_data(workers=1)
    lock2 = shared_storage.get_storage_lock(workspace="tenant2")
    assert "tenant2:storage_lock" in shared_storage._sync_locks

    # Clean up
    shared_storage.finalize_share_data()
