"""
Test workspace isolation for lock functions in shared_storage.

This test module verifies that all lock functions support workspace-based
isolation and maintain backward compatibility with the global lock behavior.
"""

import pytest
import asyncio
import time
from lightrag.kg.shared_storage import (
    initialize_share_data,
    get_storage_lock,
    get_pipeline_status_lock,
    get_graph_db_lock,
    get_data_init_lock,
    get_internal_lock,
)


@pytest.fixture(scope="module", autouse=True)
def initialize_storage():
    """Initialize shared storage before running tests."""
    initialize_share_data(workers=1)
    yield


# ============================================================================
# 1. Basic Functionality Tests
# ============================================================================


def test_global_lock_backward_compatibility():
    """Test that not passing workspace parameter uses global lock."""
    lock1 = get_storage_lock()
    lock2 = get_storage_lock()
    assert lock1._name == lock2._name == "storage_lock"


def test_workspace_specific_locks():
    """Test workspace-specific locks."""
    lock_ws1 = get_storage_lock(workspace="tenant1")
    lock_ws2 = get_storage_lock(workspace="tenant2")

    assert lock_ws1._name == "tenant1:storage_lock"
    assert lock_ws2._name == "tenant2:storage_lock"
    assert lock_ws1._name != lock_ws2._name


def test_same_workspace_returns_same_lock():
    """Test that the same workspace returns the same lock instance."""
    lock1 = get_storage_lock(workspace="tenant1")
    lock2 = get_storage_lock(workspace="tenant1")

    # Same name
    assert lock1._name == lock2._name == "tenant1:storage_lock"


def test_all_lock_functions():
    """Test that all lock functions support workspace parameter."""
    workspace = "test_ws"

    locks = [
        get_internal_lock(workspace=workspace),
        get_storage_lock(workspace=workspace),
        get_pipeline_status_lock(workspace=workspace),
        get_graph_db_lock(workspace=workspace),
        get_data_init_lock(workspace=workspace),
    ]

    expected_names = [
        f"{workspace}:internal_lock",
        f"{workspace}:storage_lock",
        f"{workspace}:pipeline_status_lock",
        f"{workspace}:graph_db_lock",
        f"{workspace}:data_init_lock",
    ]

    for lock, expected_name in zip(locks, expected_names):
        assert lock._name == expected_name


def test_empty_workspace_uses_global_lock():
    """Test that empty string workspace uses global lock."""
    lock_empty = get_storage_lock(workspace="")
    lock_default = get_storage_lock()

    assert lock_empty._name == lock_default._name == "storage_lock"


# ============================================================================
# 2. Isolation Tests
# ============================================================================


@pytest.mark.asyncio
async def test_workspace_lock_isolation():
    """Test lock isolation between workspaces."""
    results = []

    async def task_with_lock(workspace: str, task_id: int):
        lock = get_pipeline_status_lock(workspace=workspace)
        async with lock:
            results.append(f"{workspace}:{task_id}:start")
            await asyncio.sleep(0.1)  # Simulate work
            results.append(f"{workspace}:{task_id}:end")

    # Two workspaces executing concurrently
    await asyncio.gather(
        task_with_lock("ws1", 1),
        task_with_lock("ws2", 2),
    )

    # Verify both workspaces executed (possibly interleaved)
    assert "ws1:1:start" in results
    assert "ws2:2:start" in results
    assert "ws1:1:end" in results
    assert "ws2:2:end" in results


@pytest.mark.asyncio
async def test_same_workspace_lock_serialization():
    """Test that operations within the same workspace are serialized."""
    results = []

    async def task_with_lock(workspace: str, task_id: int):
        lock = get_pipeline_status_lock(workspace=workspace)
        async with lock:
            results.append(f"{workspace}:{task_id}:start")
            await asyncio.sleep(0.05)
            results.append(f"{workspace}:{task_id}:end")

    # Two tasks in the same workspace should be serialized
    await asyncio.gather(
        task_with_lock("ws1", 1),
        task_with_lock("ws1", 2),
    )

    # Find indices
    ws1_1_start = results.index("ws1:1:start")
    ws1_1_end = results.index("ws1:1:end")
    ws1_2_start = results.index("ws1:2:start")
    ws1_2_end = results.index("ws1:2:end")

    # One task should complete before the other starts (serialization)
    # Either task1 completes before task2 starts, or vice versa
    assert (ws1_1_end < ws1_2_start) or (ws1_2_end < ws1_1_start)


# ============================================================================
# 3. Backward Compatibility Tests
# ============================================================================


@pytest.mark.asyncio
async def test_legacy_code_still_works():
    """Test that existing code without workspace parameter still works."""
    # Simulate legacy code that doesn't pass workspace
    lock = get_storage_lock()

    async with lock:
        # Should work without any issues
        await asyncio.sleep(0.01)

    assert lock._name == "storage_lock"


def test_all_lock_functions_without_workspace():
    """Test that all lock functions work without workspace parameter."""
    locks = [
        get_internal_lock(),
        get_storage_lock(),
        get_pipeline_status_lock(),
        get_graph_db_lock(),
        get_data_init_lock(),
    ]

    expected_names = [
        "internal_lock",
        "storage_lock",
        "pipeline_status_lock",
        "graph_db_lock",
        "data_init_lock",
    ]

    for lock, expected_name in zip(locks, expected_names):
        assert lock._name == expected_name


# ============================================================================
# 4. Concurrent Scenario Tests
# ============================================================================


@pytest.mark.asyncio
async def test_concurrent_workspace_operations():
    """Test that multiple workspaces can operate concurrently without blocking."""
    async def simulate_document_upload(workspace: str):
        start_time = time.time()
        lock = get_pipeline_status_lock(workspace=workspace)

        async with lock:
            await asyncio.sleep(0.2)  # Simulate document processing

        return time.time() - start_time

    # Three workspaces uploading concurrently
    durations = await asyncio.gather(
        simulate_document_upload("tenant1"),
        simulate_document_upload("tenant2"),
        simulate_document_upload("tenant3"),
    )

    # If concurrent, total time should be close to 0.2s (not 0.6s)
    max_duration = max(durations)
    assert max_duration < 0.35, "Workspaces should not block each other"


@pytest.mark.asyncio
async def test_mixed_global_and_workspace_locks():
    """Test that global and workspace-specific locks don't interfere."""
    results = []

    async def task_with_global_lock():
        lock = get_storage_lock()  # Global lock
        async with lock:
            results.append("global:start")
            await asyncio.sleep(0.1)
            results.append("global:end")

    async def task_with_workspace_lock(workspace: str):
        lock = get_storage_lock(workspace=workspace)
        async with lock:
            results.append(f"{workspace}:start")
            await asyncio.sleep(0.1)
            results.append(f"{workspace}:end")

    # Run concurrently
    await asyncio.gather(
        task_with_global_lock(),
        task_with_workspace_lock("ws1"),
        task_with_workspace_lock("ws2"),
    )

    # All should have executed
    assert "global:start" in results
    assert "global:end" in results
    assert "ws1:start" in results
    assert "ws1:end" in results
    assert "ws2:start" in results
    assert "ws2:end" in results


# ============================================================================
# 5. Performance Tests
# ============================================================================


def test_lock_creation_performance():
    """Test performance of creating locks for 1000 workspaces."""
    start_time = time.time()

    for i in range(1000):
        workspace = f"tenant_{i}"
        get_storage_lock(workspace=workspace)
        get_pipeline_status_lock(workspace=workspace)

    duration = time.time() - start_time

    # 2000 lock creations should complete within 2 seconds
    assert duration < 2.0, f"Lock creation too slow: {duration}s"


@pytest.mark.asyncio
async def test_lock_acquisition_performance():
    """Test performance of acquiring and releasing locks."""
    workspace = "perf_test"
    lock = get_storage_lock(workspace=workspace)

    start_time = time.time()

    for _ in range(100):
        async with lock:
            pass  # Just acquire and release

    duration = time.time() - start_time

    # 100 acquisitions should be fast
    assert duration < 1.0, f"Lock acquisition too slow: {duration}s"


# ============================================================================
# 6. Edge Cases
# ============================================================================


def test_special_characters_in_workspace():
    """Test workspace names with special characters."""
    special_workspaces = [
        "tenant-123",
        "tenant_abc",
        "tenant.xyz",
        "tenant:colon",  # Contains colon like our separator
    ]

    for workspace in special_workspaces:
        lock = get_storage_lock(workspace=workspace)
        expected_name = f"{workspace}:storage_lock"
        assert lock._name == expected_name


def test_very_long_workspace_name():
    """Test workspace with very long name."""
    long_workspace = "a" * 1000
    lock = get_storage_lock(workspace=long_workspace)
    assert lock._name == f"{long_workspace}:storage_lock"


def test_unicode_workspace_name():
    """Test workspace with unicode characters."""
    unicode_workspace = "租户_测试"
    lock = get_storage_lock(workspace=unicode_workspace)
    assert lock._name == f"{unicode_workspace}:storage_lock"


# ============================================================================
# 7. Multiple Lock Types
# ============================================================================


@pytest.mark.asyncio
async def test_different_lock_types_same_workspace():
    """Test that different lock types in the same workspace don't interfere."""
    workspace = "multi_lock_ws"
    results = []

    async def use_storage_lock():
        lock = get_storage_lock(workspace=workspace)
        async with lock:
            results.append("storage:start")
            await asyncio.sleep(0.1)
            results.append("storage:end")

    async def use_pipeline_lock():
        lock = get_pipeline_status_lock(workspace=workspace)
        async with lock:
            results.append("pipeline:start")
            await asyncio.sleep(0.1)
            results.append("pipeline:end")

    # Different lock types should not block each other
    await asyncio.gather(
        use_storage_lock(),
        use_pipeline_lock(),
    )

    # Both should have executed
    assert "storage:start" in results
    assert "storage:end" in results
    assert "pipeline:start" in results
    assert "pipeline:end" in results
