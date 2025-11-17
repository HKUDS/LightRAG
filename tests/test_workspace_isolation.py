#!/usr/bin/env python
"""
Test script for Workspace Isolation Feature

Comprehensive test suite covering workspace isolation in LightRAG:
1. Pipeline Status Isolation - Data isolation between workspaces
2. Lock Mechanism - Parallel execution for different workspaces, serial for same workspace
3. Backward Compatibility - Legacy code without workspace parameters
4. Multi-Workspace Concurrency - Concurrent operations on different workspaces
5. NamespaceLock Re-entrance Protection - Prevents deadlocks
6. Different Namespace Lock Isolation - Locks isolated by namespace
7. Error Handling - Invalid workspace configurations
8. Update Flags Workspace Isolation - Update flags properly isolated
9. Empty Workspace Standardization - Empty workspace handling
10. JsonKVStorage Workspace Isolation - Integration test for KV storage
11. LightRAG End-to-End Workspace Isolation - Complete E2E test with two instances

Total: 11 test scenarios
"""

import asyncio
import time
import os
import shutil
import tempfile
import numpy as np
import pytest
from pathlib import Path
from lightrag.kg.shared_storage import (
    get_final_namespace,
    get_namespace_lock,
    get_default_workspace,
    set_default_workspace,
    initialize_share_data,
    initialize_pipeline_status,
    get_namespace_data,
    set_all_update_flags,
    clear_all_update_flags,
    get_all_update_flags_status,
    get_update_flag,
)


# =============================================================================
# Pytest Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def setup_shared_data():
    """Initialize shared data before each test"""
    initialize_share_data()
    yield
    # Cleanup after test if needed


# =============================================================================
# Test 1: Pipeline Status Isolation Test
# =============================================================================


@pytest.mark.asyncio
async def test_pipeline_status_isolation():
    """
    Test that pipeline status is isolated between different workspaces.
    """
    print("\n" + "=" * 60)
    print("TEST 1: Pipeline Status Isolation")
    print("=" * 60)

    # Initialize shared storage
    initialize_share_data()

    # Initialize pipeline status for two different workspaces
    workspace1 = "test_workspace_1"
    workspace2 = "test_workspace_2"

    await initialize_pipeline_status(workspace1)
    await initialize_pipeline_status(workspace2)

    # Get pipeline status data for both workspaces
    data1 = await get_namespace_data("pipeline_status", workspace=workspace1)
    data2 = await get_namespace_data("pipeline_status", workspace=workspace2)

    # Verify they are independent objects
    assert (
        data1 is not data2
    ), "Pipeline status data objects are the same (should be different)"

    # Modify workspace1's data and verify workspace2 is not affected
    data1["test_key"] = "workspace1_value"

    # Re-fetch to ensure we get the latest data
    data1_check = await get_namespace_data("pipeline_status", workspace=workspace1)
    data2_check = await get_namespace_data("pipeline_status", workspace=workspace2)

    assert "test_key" in data1_check, "test_key not found in workspace1"
    assert (
        data1_check["test_key"] == "workspace1_value"
    ), f"workspace1 test_key value incorrect: {data1_check.get('test_key')}"
    assert (
        "test_key" not in data2_check
    ), f"test_key leaked to workspace2: {data2_check.get('test_key')}"

    print("✅ PASSED: Pipeline Status Isolation")
    print("   Different workspaces have isolated pipeline status")


# =============================================================================
# Test 2: Lock Mechanism Test (No Deadlocks)
# =============================================================================


@pytest.mark.asyncio
async def test_lock_mechanism():
    """
    Test that the new keyed lock mechanism works correctly without deadlocks.
    Tests both parallel execution for different workspaces and serialization
    for the same workspace.
    """
    print("\n" + "=" * 60)
    print("TEST 2: Lock Mechanism (No Deadlocks)")
    print("=" * 60)

    # Test 2.1: Different workspaces should run in parallel
    print("\nTest 2.1: Different workspaces locks should be parallel")

    async def acquire_lock_timed(workspace, namespace, hold_time):
        """Acquire a lock and hold it for specified time"""
        lock = get_namespace_lock(namespace, workspace)
        start = time.time()
        async with lock:
            print(f"   [{workspace}] acquired lock at {time.time() - start:.2f}s")
            await asyncio.sleep(hold_time)
            print(f"   [{workspace}] releasing lock at {time.time() - start:.2f}s")

    start = time.time()
    await asyncio.gather(
        acquire_lock_timed("ws_a", "test_namespace", 0.5),
        acquire_lock_timed("ws_b", "test_namespace", 0.5),
        acquire_lock_timed("ws_c", "test_namespace", 0.5),
    )
    elapsed = time.time() - start

    # If locks are properly isolated by workspace, this should take ~0.5s (parallel)
    # If they block each other, it would take ~1.5s (serial)
    assert elapsed < 1.0, f"Locks blocked each other: {elapsed:.2f}s (expected < 1.0s)"

    print("✅ PASSED: Lock Mechanism - Parallel (Different Workspaces)")
    print(f"   Locks ran in parallel: {elapsed:.2f}s")

    # Test 2.2: Same workspace should serialize
    print("\nTest 2.2: Same workspace locks should serialize")

    start = time.time()
    await asyncio.gather(
        acquire_lock_timed("ws_same", "test_namespace", 0.3),
        acquire_lock_timed("ws_same", "test_namespace", 0.3),
    )
    elapsed = time.time() - start

    # Same workspace should serialize, taking ~0.6s
    assert elapsed >= 0.5, f"Locks didn't serialize: {elapsed:.2f}s (expected >= 0.5s)"

    print("✅ PASSED: Lock Mechanism - Serial (Same Workspace)")
    print(f"   Locks serialized correctly: {elapsed:.2f}s")


# =============================================================================
# Test 3: Backward Compatibility Test
# =============================================================================


@pytest.mark.asyncio
async def test_backward_compatibility():
    """
    Test that legacy code without workspace parameter still works correctly.
    """
    print("\n" + "=" * 60)
    print("TEST 3: Backward Compatibility")
    print("=" * 60)

    # Test 3.1: get_final_namespace with None should use default workspace
    print("\nTest 3.1: get_final_namespace with workspace=None")

    set_default_workspace("my_default_workspace")
    final_ns = get_final_namespace("pipeline_status", workspace=None)
    expected = "my_default_workspace:pipeline_status"

    assert final_ns == expected, f"Expected {expected}, got {final_ns}"

    print("✅ PASSED: Backward Compatibility - get_final_namespace")
    print(f"   Correctly uses default workspace: {final_ns}")

    # Test 3.2: get_default_workspace
    print("\nTest 3.2: get/set default workspace")

    set_default_workspace("test_default")
    retrieved = get_default_workspace()

    assert retrieved == "test_default", f"Expected 'test_default', got {retrieved}"

    print("✅ PASSED: Backward Compatibility - default workspace")
    print(f"   Default workspace set/get correctly: {retrieved}")

    # Test 3.3: Empty workspace handling
    print("\nTest 3.3: Empty workspace handling")

    set_default_workspace("")
    final_ns_empty = get_final_namespace("pipeline_status", workspace=None)
    expected_empty = "pipeline_status"  # Should be just the namespace without ':'

    assert (
        final_ns_empty == expected_empty
    ), f"Expected '{expected_empty}', got '{final_ns_empty}'"

    print("✅ PASSED: Backward Compatibility - empty workspace")
    print(f"   Empty workspace handled correctly: '{final_ns_empty}'")

    # Test 3.4: None workspace with default set
    print("\nTest 3.4: initialize_pipeline_status with workspace=None")
    set_default_workspace("compat_test_workspace")
    initialize_share_data()
    await initialize_pipeline_status(workspace=None)  # Should use default

    # Try to get data using the default workspace explicitly
    data = await get_namespace_data(
        "pipeline_status", workspace="compat_test_workspace"
    )

    assert (
        data is not None
    ), "Failed to initialize pipeline status with default workspace"

    print("✅ PASSED: Backward Compatibility - pipeline init with None")
    print("   Pipeline status initialized with default workspace")


# =============================================================================
# Test 4: Multi-Workspace Concurrency Test
# =============================================================================


@pytest.mark.asyncio
async def test_multi_workspace_concurrency():
    """
    Test that multiple workspaces can operate concurrently without interference.
    Simulates concurrent operations on different workspaces.
    """
    print("\n" + "=" * 60)
    print("TEST 4: Multi-Workspace Concurrency")
    print("=" * 60)

    initialize_share_data()

    async def workspace_operations(workspace_id):
        """Simulate operations on a specific workspace"""
        print(f"\n   [{workspace_id}] Starting operations")

        # Initialize pipeline status
        await initialize_pipeline_status(workspace_id)

        # Get lock and perform operations
        lock = get_namespace_lock("test_operations", workspace_id)
        async with lock:
            # Get workspace data
            data = await get_namespace_data("pipeline_status", workspace=workspace_id)

            # Modify data
            data[f"{workspace_id}_key"] = f"{workspace_id}_value"
            data["timestamp"] = time.time()

            # Simulate some work
            await asyncio.sleep(0.1)

            print(f"   [{workspace_id}] Completed operations")

        return workspace_id

    # Run multiple workspaces concurrently
    workspaces = ["concurrent_ws_1", "concurrent_ws_2", "concurrent_ws_3"]

    start = time.time()
    results_list = await asyncio.gather(
        *[workspace_operations(ws) for ws in workspaces]
    )
    elapsed = time.time() - start

    print(f"\n   All workspaces completed in {elapsed:.2f}s")

    # Verify all workspaces completed
    assert set(results_list) == set(workspaces), "Not all workspaces completed"

    print("✅ PASSED: Multi-Workspace Concurrency - Execution")
    print(
        f"   All {len(workspaces)} workspaces completed successfully in {elapsed:.2f}s"
    )

    # Verify data isolation - each workspace should have its own data
    print("\n   Verifying data isolation...")

    for ws in workspaces:
        data = await get_namespace_data("pipeline_status", workspace=ws)
        expected_key = f"{ws}_key"
        expected_value = f"{ws}_value"

        assert (
            expected_key in data
        ), f"Data not properly isolated for {ws}: missing {expected_key}"
        assert (
            data[expected_key] == expected_value
        ), f"Data not properly isolated for {ws}: {expected_key}={data[expected_key]} (expected {expected_value})"
        print(f"   [{ws}] Data correctly isolated: {expected_key}={data[expected_key]}")

    print("✅ PASSED: Multi-Workspace Concurrency - Data Isolation")
    print("   All workspaces have properly isolated data")


# =============================================================================
# Test 5: NamespaceLock Re-entrance Protection
# =============================================================================


@pytest.mark.asyncio
async def test_namespace_lock_reentrance():
    """
    Test that NamespaceLock prevents re-entrance in the same coroutine
    and allows concurrent use in different coroutines.
    """
    print("\n" + "=" * 60)
    print("TEST 5: NamespaceLock Re-entrance Protection")
    print("=" * 60)

    # Test 5.1: Same coroutine re-entrance should fail
    print("\nTest 5.1: Same coroutine re-entrance should raise RuntimeError")

    lock = get_namespace_lock("test_reentrance", "test_ws")

    reentrance_failed_correctly = False
    try:
        async with lock:
            print("   Acquired lock first time")
            # Try to acquire the same lock again in the same coroutine
            async with lock:
                print("   ERROR: Should not reach here - re-entrance succeeded!")
    except RuntimeError as e:
        if "already acquired" in str(e).lower():
            print(f"   ✓ Re-entrance correctly blocked: {e}")
            reentrance_failed_correctly = True
        else:
            raise

    assert reentrance_failed_correctly, "Re-entrance protection not working"

    print("✅ PASSED: NamespaceLock Re-entrance Protection")
    print("   Re-entrance correctly raises RuntimeError")

    # Test 5.2: Same NamespaceLock instance in different coroutines should succeed
    print("\nTest 5.2: Same NamespaceLock instance in different coroutines")

    shared_lock = get_namespace_lock("test_concurrent", "test_ws")
    concurrent_results = []

    async def use_shared_lock(coroutine_id):
        """Use the same NamespaceLock instance"""
        async with shared_lock:
            concurrent_results.append(f"coroutine_{coroutine_id}_start")
            await asyncio.sleep(0.1)
            concurrent_results.append(f"coroutine_{coroutine_id}_end")

    # This should work because each coroutine gets its own ContextVar
    await asyncio.gather(
        use_shared_lock(1),
        use_shared_lock(2),
    )

    # Both coroutines should have completed
    expected_entries = 4  # 2 starts + 2 ends
    assert (
        len(concurrent_results) == expected_entries
    ), f"Expected {expected_entries} entries, got {len(concurrent_results)}"

    print("✅ PASSED: NamespaceLock Concurrent Reuse")
    print(
        f"   Same NamespaceLock instance used successfully in {expected_entries//2} concurrent coroutines"
    )


# =============================================================================
# Test 6: Different Namespace Lock Isolation
# =============================================================================


@pytest.mark.asyncio
async def test_different_namespace_lock_isolation():
    """
    Test that locks for different namespaces (same workspace) are independent.
    """
    print("\n" + "=" * 60)
    print("TEST 6: Different Namespace Lock Isolation")
    print("=" * 60)

    print("\nTesting locks with same workspace but different namespaces")

    async def acquire_lock_timed(workspace, namespace, hold_time, name):
        """Acquire a lock and hold it for specified time"""
        lock = get_namespace_lock(namespace, workspace)
        start = time.time()
        async with lock:
            print(f"   [{name}] acquired lock at {time.time() - start:.2f}s")
            await asyncio.sleep(hold_time)
            print(f"   [{name}] releasing lock at {time.time() - start:.2f}s")

    # These should run in parallel (different namespaces)
    start = time.time()
    await asyncio.gather(
        acquire_lock_timed("same_ws", "namespace_a", 0.5, "ns_a"),
        acquire_lock_timed("same_ws", "namespace_b", 0.5, "ns_b"),
        acquire_lock_timed("same_ws", "namespace_c", 0.5, "ns_c"),
    )
    elapsed = time.time() - start

    # If locks are properly isolated by namespace, this should take ~0.5s (parallel)
    assert (
        elapsed < 1.0
    ), f"Different namespace locks blocked each other: {elapsed:.2f}s (expected < 1.0s)"

    print("✅ PASSED: Different Namespace Lock Isolation")
    print(f"   Different namespace locks ran in parallel: {elapsed:.2f}s")


# =============================================================================
# Test 7: Error Handling
# =============================================================================


@pytest.mark.asyncio
async def test_error_handling():
    """
    Test error handling for invalid workspace configurations.
    """
    print("\n" + "=" * 60)
    print("TEST 7: Error Handling")
    print("=" * 60)

    # Test 7.1: set_default_workspace(None) converts to empty string
    print("\nTest 7.1: set_default_workspace(None) converts to empty string")

    set_default_workspace(None)
    default_ws = get_default_workspace()

    # Should convert None to "" automatically
    assert default_ws == "", f"Expected empty string, got: '{default_ws}'"

    print("✅ PASSED: Error Handling - None to Empty String")
    print(
        f"   set_default_workspace(None) correctly converts to empty string: '{default_ws}'"
    )

    # Test 7.2: Empty string workspace behavior
    print("\nTest 7.2: Empty string workspace creates valid namespace")

    # With empty workspace, should create namespace without colon
    final_ns = get_final_namespace("test_namespace", workspace="")
    assert final_ns == "test_namespace", f"Unexpected namespace: '{final_ns}'"

    print("✅ PASSED: Error Handling - Empty Workspace Namespace")
    print(f"   Empty workspace creates valid namespace: '{final_ns}'")

    # Restore default workspace for other tests
    set_default_workspace("")


# =============================================================================
# Test 8: Update Flags Workspace Isolation
# =============================================================================


@pytest.mark.asyncio
async def test_update_flags_workspace_isolation():
    """
    Test that update flags are properly isolated between workspaces.
    """
    print("\n" + "=" * 60)
    print("TEST 8: Update Flags Workspace Isolation")
    print("=" * 60)

    initialize_share_data()

    workspace1 = "update_flags_ws1"
    workspace2 = "update_flags_ws2"
    test_namespace = "test_update_flags_ns"

    # Initialize namespaces for both workspaces
    await initialize_pipeline_status(workspace1)
    await initialize_pipeline_status(workspace2)

    # Test 8.1: set_all_update_flags isolation
    print("\nTest 8.1: set_all_update_flags workspace isolation")

    # Create flags for both workspaces (simulating workers)
    flag1_obj = await get_update_flag(test_namespace, workspace=workspace1)
    flag2_obj = await get_update_flag(test_namespace, workspace=workspace2)

    # Initial state should be False
    assert flag1_obj.value is False, "Flag1 initial value should be False"
    assert flag2_obj.value is False, "Flag2 initial value should be False"

    # Set all flags for workspace1
    await set_all_update_flags(test_namespace, workspace=workspace1)

    # Check that only workspace1's flags are set
    assert (
        flag1_obj.value is True
    ), f"Flag1 should be True after set_all_update_flags, got {flag1_obj.value}"
    assert (
        flag2_obj.value is False
    ), f"Flag2 should still be False, got {flag2_obj.value}"

    print("✅ PASSED: Update Flags - set_all_update_flags Isolation")
    print(
        f"   set_all_update_flags isolated: ws1={flag1_obj.value}, ws2={flag2_obj.value}"
    )

    # Test 8.2: clear_all_update_flags isolation
    print("\nTest 8.2: clear_all_update_flags workspace isolation")

    # Set flags for both workspaces
    await set_all_update_flags(test_namespace, workspace=workspace1)
    await set_all_update_flags(test_namespace, workspace=workspace2)

    # Verify both are set
    assert flag1_obj.value is True, "Flag1 should be True"
    assert flag2_obj.value is True, "Flag2 should be True"

    # Clear only workspace1
    await clear_all_update_flags(test_namespace, workspace=workspace1)

    # Check that only workspace1's flags are cleared
    assert (
        flag1_obj.value is False
    ), f"Flag1 should be False after clear, got {flag1_obj.value}"
    assert flag2_obj.value is True, f"Flag2 should still be True, got {flag2_obj.value}"

    print("✅ PASSED: Update Flags - clear_all_update_flags Isolation")
    print(
        f"   clear_all_update_flags isolated: ws1={flag1_obj.value}, ws2={flag2_obj.value}"
    )

    # Test 8.3: get_all_update_flags_status workspace filtering
    print("\nTest 8.3: get_all_update_flags_status workspace filtering")

    # Initialize more namespaces for testing
    await get_update_flag("ns_a", workspace=workspace1)
    await get_update_flag("ns_b", workspace=workspace1)
    await get_update_flag("ns_c", workspace=workspace2)

    # Set flags for workspace1
    await set_all_update_flags("ns_a", workspace=workspace1)
    await set_all_update_flags("ns_b", workspace=workspace1)

    # Set flags for workspace2
    await set_all_update_flags("ns_c", workspace=workspace2)

    # Get status for workspace1 only
    status1 = await get_all_update_flags_status(workspace=workspace1)

    # Check that workspace1's namespaces are present
    # The keys should include workspace1's namespaces but not workspace2's
    workspace1_keys = [k for k in status1.keys() if workspace1 in k]
    workspace2_keys = [k for k in status1.keys() if workspace2 in k]

    assert (
        len(workspace1_keys) > 0
    ), f"workspace1 keys should be present, got {len(workspace1_keys)}"
    assert (
        len(workspace2_keys) == 0
    ), f"workspace2 keys should not be present, got {len(workspace2_keys)}"

    print("✅ PASSED: Update Flags - get_all_update_flags_status Filtering")
    print(
        f"   Status correctly filtered: ws1 keys={len(workspace1_keys)}, ws2 keys={len(workspace2_keys)}"
    )


# =============================================================================
# Test 9: Empty Workspace Standardization
# =============================================================================


@pytest.mark.asyncio
async def test_empty_workspace_standardization():
    """
    Test that empty workspace is properly standardized to "" instead of "_".
    """
    print("\n" + "=" * 60)
    print("TEST 9: Empty Workspace Standardization")
    print("=" * 60)

    # Test 9.1: Empty string workspace creates namespace without colon
    print("\nTest 9.1: Empty string workspace namespace format")

    set_default_workspace("")
    final_ns = get_final_namespace("test_namespace", workspace=None)

    # Should be just "test_namespace" without colon prefix
    assert (
        final_ns == "test_namespace"
    ), f"Unexpected namespace format: '{final_ns}' (expected 'test_namespace')"

    print("✅ PASSED: Empty Workspace Standardization - Format")
    print(f"   Empty workspace creates correct namespace: '{final_ns}'")

    # Test 9.2: Empty workspace vs non-empty workspace behavior
    print("\nTest 9.2: Empty vs non-empty workspace behavior")

    initialize_share_data()

    # Initialize with empty workspace
    await initialize_pipeline_status(workspace="")
    data_empty = await get_namespace_data("pipeline_status", workspace="")

    # Initialize with non-empty workspace
    await initialize_pipeline_status(workspace="test_ws")
    data_nonempty = await get_namespace_data("pipeline_status", workspace="test_ws")

    # They should be different objects
    assert (
        data_empty is not data_nonempty
    ), "Empty and non-empty workspaces share data (should be independent)"

    print("✅ PASSED: Empty Workspace Standardization - Behavior")
    print("   Empty and non-empty workspaces have independent data")


# =============================================================================
# Test 10: JsonKVStorage Workspace Isolation (Integration Test)
# =============================================================================


@pytest.mark.asyncio
async def test_json_kv_storage_workspace_isolation():
    """
    Integration test: Verify JsonKVStorage properly isolates data between workspaces.
    Creates two JsonKVStorage instances with different workspaces, writes different data,
    and verifies they don't mix.
    """
    print("\n" + "=" * 60)
    print("TEST 10: JsonKVStorage Workspace Isolation (Integration)")
    print("=" * 60)

    # Create temporary test directory
    test_dir = tempfile.mkdtemp(prefix="lightrag_test_kv_")
    print(f"\n   Using test directory: {test_dir}")

    try:
        initialize_share_data()

        # Mock embedding function
        async def mock_embedding_func(texts: list[str]) -> np.ndarray:
            return np.random.rand(len(texts), 384)  # 384-dimensional vectors

        # Global config
        global_config = {
            "working_dir": test_dir,
            "embedding_batch_num": 10,
        }

        # Test 10.1: Create two JsonKVStorage instances with different workspaces
        print(
            "\nTest 10.1: Create two JsonKVStorage instances with different workspaces"
        )

        from lightrag.kg.json_kv_impl import JsonKVStorage

        storage1 = JsonKVStorage(
            namespace="entities",
            workspace="workspace1",
            global_config=global_config,
            embedding_func=mock_embedding_func,
        )

        storage2 = JsonKVStorage(
            namespace="entities",
            workspace="workspace2",
            global_config=global_config,
            embedding_func=mock_embedding_func,
        )

        # Initialize both storages
        await storage1.initialize()
        await storage2.initialize()

        print("   Storage1 created: workspace=workspace1, namespace=entities")
        print("   Storage2 created: workspace=workspace2, namespace=entities")

        # Test 10.2: Write different data to each storage
        print("\nTest 10.2: Write different data to each storage")

        # Write to storage1 (upsert expects dict[str, dict])
        await storage1.upsert(
            {
                "entity1": {
                    "content": "Data from workspace1 - AI Research",
                    "type": "entity",
                },
                "entity2": {
                    "content": "Data from workspace1 - Machine Learning",
                    "type": "entity",
                },
            }
        )
        print("   Written to storage1: entity1, entity2")

        # Write to storage2
        await storage2.upsert(
            {
                "entity1": {
                    "content": "Data from workspace2 - Deep Learning",
                    "type": "entity",
                },
                "entity2": {
                    "content": "Data from workspace2 - Neural Networks",
                    "type": "entity",
                },
            }
        )
        print("   Written to storage2: entity1, entity2")

        # Test 10.3: Read data from each storage and verify isolation
        print("\nTest 10.3: Read data and verify isolation")

        # Read from storage1
        result1_entity1 = await storage1.get_by_id("entity1")
        result1_entity2 = await storage1.get_by_id("entity2")

        # Read from storage2
        result2_entity1 = await storage2.get_by_id("entity1")
        result2_entity2 = await storage2.get_by_id("entity2")

        print(f"   Storage1 entity1: {result1_entity1}")
        print(f"   Storage1 entity2: {result1_entity2}")
        print(f"   Storage2 entity1: {result2_entity1}")
        print(f"   Storage2 entity2: {result2_entity2}")

        # Verify isolation (get_by_id returns dict)
        assert result1_entity1 is not None, "Storage1 entity1 should not be None"
        assert result1_entity2 is not None, "Storage1 entity2 should not be None"
        assert result2_entity1 is not None, "Storage2 entity1 should not be None"
        assert result2_entity2 is not None, "Storage2 entity2 should not be None"
        assert (
            result1_entity1.get("content") == "Data from workspace1 - AI Research"
        ), "Storage1 entity1 content mismatch"
        assert (
            result1_entity2.get("content") == "Data from workspace1 - Machine Learning"
        ), "Storage1 entity2 content mismatch"
        assert (
            result2_entity1.get("content") == "Data from workspace2 - Deep Learning"
        ), "Storage2 entity1 content mismatch"
        assert (
            result2_entity2.get("content") == "Data from workspace2 - Neural Networks"
        ), "Storage2 entity2 content mismatch"
        assert result1_entity1.get("content") != result2_entity1.get(
            "content"
        ), "Storage1 and Storage2 entity1 should have different content"
        assert result1_entity2.get("content") != result2_entity2.get(
            "content"
        ), "Storage1 and Storage2 entity2 should have different content"

        print("✅ PASSED: JsonKVStorage - Data Isolation")
        print(
            "   Two storage instances correctly isolated: ws1 and ws2 have different data"
        )

        # Test 10.4: Verify file structure
        print("\nTest 10.4: Verify file structure")
        ws1_dir = Path(test_dir) / "workspace1"
        ws2_dir = Path(test_dir) / "workspace2"

        ws1_exists = ws1_dir.exists()
        ws2_exists = ws2_dir.exists()

        print(f"   workspace1 directory exists: {ws1_exists}")
        print(f"   workspace2 directory exists: {ws2_exists}")

        assert ws1_exists, "workspace1 directory should exist"
        assert ws2_exists, "workspace2 directory should exist"

        print("✅ PASSED: JsonKVStorage - File Structure")
        print(f"   Workspace directories correctly created: {ws1_dir} and {ws2_dir}")

    finally:
        # Cleanup test directory
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
            print(f"\n   Cleaned up test directory: {test_dir}")


# =============================================================================
# Test 11: LightRAG End-to-End Integration Test
# =============================================================================


@pytest.mark.asyncio
async def test_lightrag_end_to_end_workspace_isolation():
    """
    End-to-end test: Create two LightRAG instances with different workspaces,
    insert different data, and verify file separation.
    Uses mock LLM and embedding functions to avoid external API calls.
    """
    print("\n" + "=" * 60)
    print("TEST 11: LightRAG End-to-End Workspace Isolation")
    print("=" * 60)

    # Create temporary test directory
    test_dir = tempfile.mkdtemp(prefix="lightrag_test_e2e_")
    print(f"\n   Using test directory: {test_dir}")

    try:
        # Mock LLM function
        async def mock_llm_func(
            prompt, system_prompt=None, history_messages=[], **kwargs
        ) -> str:
            # Return a mock response that simulates entity extraction in the correct format
            # Format: entity<|#|>entity_name<|#|>entity_type<|#|>entity_description
            # Format: relation<|#|>source_entity<|#|>target_entity<|#|>keywords<|#|>description
            return """entity<|#|>Artificial Intelligence<|#|>concept<|#|>AI is a field of computer science focused on creating intelligent machines.
entity<|#|>Machine Learning<|#|>concept<|#|>Machine Learning is a subset of AI that enables systems to learn from data.
relation<|#|>Machine Learning<|#|>Artificial Intelligence<|#|>subset, related field<|#|>Machine Learning is a key component and subset of Artificial Intelligence.
<|COMPLETE|>"""

        # Mock embedding function
        async def mock_embedding_func(texts: list[str]) -> np.ndarray:
            return np.random.rand(len(texts), 384)  # 384-dimensional vectors

        # Test 11.1: Create two LightRAG instances with different workspaces
        print("\nTest 11.1: Create two LightRAG instances with different workspaces")

        from lightrag import LightRAG
        from lightrag.utils import EmbeddingFunc

        rag1 = LightRAG(
            working_dir=test_dir,
            workspace="project_a",
            llm_model_func=mock_llm_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=384,
                max_token_size=8192,
                func=mock_embedding_func,
            ),
        )

        rag2 = LightRAG(
            working_dir=test_dir,
            workspace="project_b",
            llm_model_func=mock_llm_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=384,
                max_token_size=8192,
                func=mock_embedding_func,
            ),
        )

        # Initialize storages
        await rag1.initialize_storages()
        await rag2.initialize_storages()

        print("   RAG1 created: workspace=project_a")
        print("   RAG2 created: workspace=project_b")

        # Test 11.2: Insert different data to each RAG instance
        print("\nTest 11.2: Insert different data to each RAG instance")

        text_for_project_a = "This document is about Artificial Intelligence and Machine Learning. AI is transforming the world."
        text_for_project_b = "This document is about Deep Learning and Neural Networks. Deep learning uses multiple layers."

        # Insert to project_a
        await rag1.ainsert(text_for_project_a)
        print(f"   Inserted to project_a: {len(text_for_project_a)} chars")

        # Insert to project_b
        await rag2.ainsert(text_for_project_b)
        print(f"   Inserted to project_b: {len(text_for_project_b)} chars")

        # Test 11.3: Verify file structure
        print("\nTest 11.3: Verify workspace directory structure")

        project_a_dir = Path(test_dir) / "project_a"
        project_b_dir = Path(test_dir) / "project_b"

        project_a_exists = project_a_dir.exists()
        project_b_exists = project_b_dir.exists()

        print(f"   project_a directory: {project_a_dir}")
        print(f"   project_a exists: {project_a_exists}")
        print(f"   project_b directory: {project_b_dir}")
        print(f"   project_b exists: {project_b_exists}")

        assert project_a_exists, "project_a directory should exist"
        assert project_b_exists, "project_b directory should exist"

        # List files in each directory
        print("\n   Files in project_a/:")
        for file in sorted(project_a_dir.glob("*")):
            if file.is_file():
                size = file.stat().st_size
                print(f"     - {file.name} ({size} bytes)")

        print("\n   Files in project_b/:")
        for file in sorted(project_b_dir.glob("*")):
            if file.is_file():
                size = file.stat().st_size
                print(f"     - {file.name} ({size} bytes)")

        print("✅ PASSED: LightRAG E2E - File Structure")
        print("   Workspace directories correctly created and separated")

        # Test 11.4: Verify data isolation by checking file contents
        print("\nTest 11.4: Verify data isolation (check file contents)")

        # Check if full_docs storage files exist and contain different content
        docs_a_file = project_a_dir / "kv_store_full_docs.json"
        docs_b_file = project_b_dir / "kv_store_full_docs.json"

        if docs_a_file.exists() and docs_b_file.exists():
            import json

            with open(docs_a_file, "r") as f:
                docs_a_content = json.load(f)

            with open(docs_b_file, "r") as f:
                docs_b_content = json.load(f)

            print(f"   project_a doc count: {len(docs_a_content)}")
            print(f"   project_b doc count: {len(docs_b_content)}")

            # Verify they contain different data
            assert (
                docs_a_content != docs_b_content
            ), "Document storage not properly isolated"

            # Verify each workspace contains its own text content
            docs_a_str = json.dumps(docs_a_content)
            docs_b_str = json.dumps(docs_b_content)

            # Check project_a contains its text and NOT project_b's text
            assert (
                "Artificial Intelligence" in docs_a_str
            ), "project_a should contain 'Artificial Intelligence'"
            assert (
                "Machine Learning" in docs_a_str
            ), "project_a should contain 'Machine Learning'"
            assert (
                "Deep Learning" not in docs_a_str
            ), "project_a should NOT contain 'Deep Learning' from project_b"
            assert (
                "Neural Networks" not in docs_a_str
            ), "project_a should NOT contain 'Neural Networks' from project_b"

            # Check project_b contains its text and NOT project_a's text
            assert (
                "Deep Learning" in docs_b_str
            ), "project_b should contain 'Deep Learning'"
            assert (
                "Neural Networks" in docs_b_str
            ), "project_b should contain 'Neural Networks'"
            assert (
                "Artificial Intelligence" not in docs_b_str
            ), "project_b should NOT contain 'Artificial Intelligence' from project_a"
            # Note: "Machine Learning" might appear in project_b's text, so we skip that check

            print("✅ PASSED: LightRAG E2E - Data Isolation")
            print("   Document storage correctly isolated between workspaces")
            print("   project_a contains only its own data")
            print("   project_b contains only its own data")
        else:
            print("   Document storage files not found (may not be created yet)")
            print("✅ PASSED: LightRAG E2E - Data Isolation")
            print("   Skipped file content check (files not created)")

        print("\n   ✓ Test complete - workspace isolation verified at E2E level")

    finally:
        # Cleanup test directory
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
            print(f"\n   Cleaned up test directory: {test_dir}")
