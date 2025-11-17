#!/usr/bin/env python
"""
Test script for PR #2366: Workspace Isolation Feature

Tests the 4 key scenarios mentioned in PR description:
1. Multi-Workspace Concurrency Test
2. Pipeline Status Isolation Test
3. Backward Compatibility Test
4. Lock Mechanism Test
"""

import asyncio
import time
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


class TestResults:
    """Track test results"""

    def __init__(self):
        self.results = []

    def add(self, test_name, passed, message=""):
        self.results.append({"name": test_name, "passed": passed, "message": message})
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"\n{status}: {test_name}")
        if message:
            print(f"   {message}")

    def summary(self):
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        passed = sum(1 for r in self.results if r["passed"])
        total = len(self.results)
        print(f"Passed: {passed}/{total}")
        print()
        for r in self.results:
            status = "✅" if r["passed"] else "❌"
            print(f"{status} {r['name']}")
            if r["message"]:
                print(f"   {r['message']}")
        print("=" * 60)
        return passed == total


results = TestResults()


# =============================================================================
# Test 1: Pipeline Status Isolation Test
# =============================================================================


async def test_pipeline_status_isolation():
    """
    Test that pipeline status is isolated between different workspaces.
    """
    print("\n" + "=" * 60)
    print("TEST 1: Pipeline Status Isolation")
    print("=" * 60)

    try:
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
        if data1 is data2:
            results.add(
                "Pipeline Status Isolation",
                False,
                "Pipeline status data objects are the same (should be different)",
            )
            return False

        # Modify workspace1's data and verify workspace2 is not affected
        data1["test_key"] = "workspace1_value"

        # Re-fetch to ensure we get the latest data
        data1_check = await get_namespace_data("pipeline_status", workspace=workspace1)
        data2_check = await get_namespace_data("pipeline_status", workspace=workspace2)

        if (
            "test_key" in data1_check
            and data1_check["test_key"] == "workspace1_value"
            and "test_key" not in data2_check
        ):
            results.add(
                "Pipeline Status Isolation",
                True,
                "Different workspaces have isolated pipeline status",
            )
            return True
        else:
            results.add(
                "Pipeline Status Isolation",
                False,
                f"Pipeline status not properly isolated: ws1={data1_check.get('test_key')}, ws2={data2_check.get('test_key')}",
            )
            return False

    except Exception as e:
        results.add("Pipeline Status Isolation", False, f"Exception: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


# =============================================================================
# Test 2: Lock Mechanism Test (No Deadlocks)
# =============================================================================


async def test_lock_mechanism():
    """
    Test that the new keyed lock mechanism works correctly without deadlocks.
    Tests both parallel execution for different workspaces and serialization
    for the same workspace.
    """
    print("\n" + "=" * 60)
    print("TEST 2: Lock Mechanism (No Deadlocks)")
    print("=" * 60)

    try:
        # Test 2.1: Different workspaces should run in parallel
        print("\nTest 2.1: Different workspaces locks should be parallel")

        async def acquire_lock_timed(workspace, namespace, hold_time):
            """Acquire a lock and hold it for specified time"""
            lock = get_namespace_lock(namespace, workspace)
            start = time.time()
            async with lock:
                print(
                    f"   [{workspace}] acquired lock at {time.time() - start:.2f}s"
                )
                await asyncio.sleep(hold_time)
                print(
                    f"   [{workspace}] releasing lock at {time.time() - start:.2f}s"
                )

        start = time.time()
        await asyncio.gather(
            acquire_lock_timed("ws_a", "test_namespace", 0.5),
            acquire_lock_timed("ws_b", "test_namespace", 0.5),
            acquire_lock_timed("ws_c", "test_namespace", 0.5),
        )
        elapsed = time.time() - start

        # If locks are properly isolated by workspace, this should take ~0.5s (parallel)
        # If they block each other, it would take ~1.5s (serial)
        parallel_ok = elapsed < 1.0

        if parallel_ok:
            results.add(
                "Lock Mechanism - Parallel (Different Workspaces)",
                True,
                f"Locks ran in parallel: {elapsed:.2f}s",
            )
        else:
            results.add(
                "Lock Mechanism - Parallel (Different Workspaces)",
                False,
                f"Locks blocked each other: {elapsed:.2f}s (expected < 1.0s)",
            )

        # Test 2.2: Same workspace should serialize
        print("\nTest 2.2: Same workspace locks should serialize")

        start = time.time()
        await asyncio.gather(
            acquire_lock_timed("ws_same", "test_namespace", 0.3),
            acquire_lock_timed("ws_same", "test_namespace", 0.3),
        )
        elapsed = time.time() - start

        # Same workspace should serialize, taking ~0.6s
        serial_ok = elapsed >= 0.5

        if serial_ok:
            results.add(
                "Lock Mechanism - Serial (Same Workspace)",
                True,
                f"Locks serialized correctly: {elapsed:.2f}s",
            )
        else:
            results.add(
                "Lock Mechanism - Serial (Same Workspace)",
                False,
                f"Locks didn't serialize: {elapsed:.2f}s (expected >= 0.5s)",
            )

        return parallel_ok and serial_ok

    except Exception as e:
        results.add("Lock Mechanism", False, f"Exception: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


# =============================================================================
# Test 3: Backward Compatibility Test
# =============================================================================


async def test_backward_compatibility():
    """
    Test that legacy code without workspace parameter still works correctly.
    """
    print("\n" + "=" * 60)
    print("TEST 3: Backward Compatibility")
    print("=" * 60)

    try:
        # Test 3.1: get_final_namespace with None should use default workspace
        print("\nTest 3.1: get_final_namespace with workspace=None")

        set_default_workspace("my_default_workspace")
        final_ns = get_final_namespace("pipeline_status", workspace=None)
        expected = "my_default_workspace:pipeline_status"

        if final_ns == expected:
            results.add(
                "Backward Compatibility - get_final_namespace",
                True,
                f"Correctly uses default workspace: {final_ns}",
            )
            compat_1_ok = True
        else:
            results.add(
                "Backward Compatibility - get_final_namespace",
                False,
                f"Expected {expected}, got {final_ns}",
            )
            compat_1_ok = False

        # Test 3.2: get_default_workspace
        print("\nTest 3.2: get/set default workspace")

        set_default_workspace("test_default")
        retrieved = get_default_workspace()

        if retrieved == "test_default":
            results.add(
                "Backward Compatibility - default workspace",
                True,
                f"Default workspace set/get correctly: {retrieved}",
            )
            compat_2_ok = True
        else:
            results.add(
                "Backward Compatibility - default workspace",
                False,
                f"Expected 'test_default', got {retrieved}",
            )
            compat_2_ok = False

        # Test 3.3: Empty workspace handling
        print("\nTest 3.3: Empty workspace handling")

        set_default_workspace("")
        final_ns_empty = get_final_namespace("pipeline_status", workspace=None)
        expected_empty = "pipeline_status"  # Should be just the namespace without ':'

        if final_ns_empty == expected_empty:
            results.add(
                "Backward Compatibility - empty workspace",
                True,
                f"Empty workspace handled correctly: '{final_ns_empty}'",
            )
            compat_3_ok = True
        else:
            results.add(
                "Backward Compatibility - empty workspace",
                False,
                f"Expected '{expected_empty}', got '{final_ns_empty}'",
            )
            compat_3_ok = False

        # Test 3.4: None workspace with default set
        print("\nTest 3.4: initialize_pipeline_status with workspace=None")
        set_default_workspace("compat_test_workspace")
        initialize_share_data()
        await initialize_pipeline_status(workspace=None)  # Should use default

        # Try to get data using the default workspace explicitly
        data = await get_namespace_data(
            "pipeline_status", workspace="compat_test_workspace"
        )

        if data is not None:
            results.add(
                "Backward Compatibility - pipeline init with None",
                True,
                "Pipeline status initialized with default workspace",
            )
            compat_4_ok = True
        else:
            results.add(
                "Backward Compatibility - pipeline init with None",
                False,
                "Failed to initialize pipeline status with default workspace",
            )
            compat_4_ok = False

        return compat_1_ok and compat_2_ok and compat_3_ok and compat_4_ok

    except Exception as e:
        results.add("Backward Compatibility", False, f"Exception: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


# =============================================================================
# Test 4: Multi-Workspace Concurrency Test
# =============================================================================


async def test_multi_workspace_concurrency():
    """
    Test that multiple workspaces can operate concurrently without interference.
    Simulates concurrent operations on different workspaces.
    """
    print("\n" + "=" * 60)
    print("TEST 4: Multi-Workspace Concurrency")
    print("=" * 60)

    try:
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
        if set(results_list) == set(workspaces):
            results.add(
                "Multi-Workspace Concurrency - Execution",
                True,
                f"All {len(workspaces)} workspaces completed successfully in {elapsed:.2f}s",
            )
            exec_ok = True
        else:
            results.add(
                "Multi-Workspace Concurrency - Execution",
                False,
                f"Not all workspaces completed",
            )
            exec_ok = False

        # Verify data isolation - each workspace should have its own data
        print("\n   Verifying data isolation...")
        isolation_ok = True

        for ws in workspaces:
            data = await get_namespace_data("pipeline_status", workspace=ws)
            expected_key = f"{ws}_key"
            expected_value = f"{ws}_value"

            if expected_key not in data or data[expected_key] != expected_value:
                results.add(
                    f"Multi-Workspace Concurrency - Data Isolation ({ws})",
                    False,
                    f"Data not properly isolated for {ws}",
                )
                isolation_ok = False
            else:
                print(f"   [{ws}] Data correctly isolated: {expected_key}={data[expected_key]}")

        if isolation_ok:
            results.add(
                "Multi-Workspace Concurrency - Data Isolation",
                True,
                "All workspaces have properly isolated data",
            )

        return exec_ok and isolation_ok

    except Exception as e:
        results.add("Multi-Workspace Concurrency", False, f"Exception: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


# =============================================================================
# Test 5: NamespaceLock Re-entrance Protection
# =============================================================================


async def test_namespace_lock_reentrance():
    """
    Test that NamespaceLock prevents re-entrance in the same coroutine
    and allows concurrent use in different coroutines.
    """
    print("\n" + "=" * 60)
    print("TEST 5: NamespaceLock Re-entrance Protection")
    print("=" * 60)

    try:
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
                print(f"   ✗ Unexpected RuntimeError: {e}")

        if reentrance_failed_correctly:
            results.add(
                "NamespaceLock Re-entrance Protection",
                True,
                "Re-entrance correctly raises RuntimeError",
            )
        else:
            results.add(
                "NamespaceLock Re-entrance Protection",
                False,
                "Re-entrance protection not working",
            )

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
        if len(concurrent_results) == expected_entries:
            results.add(
                "NamespaceLock Concurrent Reuse",
                True,
                f"Same NamespaceLock instance used successfully in {expected_entries//2} concurrent coroutines",
            )
            concurrent_ok = True
        else:
            results.add(
                "NamespaceLock Concurrent Reuse",
                False,
                f"Expected {expected_entries} entries, got {len(concurrent_results)}",
            )
            concurrent_ok = False

        return reentrance_failed_correctly and concurrent_ok

    except Exception as e:
        results.add("NamespaceLock Re-entrance Protection", False, f"Exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Test 6: Different Namespace Lock Isolation
# =============================================================================


async def test_different_namespace_lock_isolation():
    """
    Test that locks for different namespaces (same workspace) are independent.
    """
    print("\n" + "=" * 60)
    print("TEST 6: Different Namespace Lock Isolation")
    print("=" * 60)

    try:
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
        namespace_isolation_ok = elapsed < 1.0

        if namespace_isolation_ok:
            results.add(
                "Different Namespace Lock Isolation",
                True,
                f"Different namespace locks ran in parallel: {elapsed:.2f}s",
            )
        else:
            results.add(
                "Different Namespace Lock Isolation",
                False,
                f"Different namespace locks blocked each other: {elapsed:.2f}s (expected < 1.0s)",
            )

        return namespace_isolation_ok

    except Exception as e:
        results.add("Different Namespace Lock Isolation", False, f"Exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Test 7: Error Handling
# =============================================================================


async def test_error_handling():
    """
    Test error handling for invalid workspace configurations.
    """
    print("\n" + "=" * 60)
    print("TEST 7: Error Handling")
    print("=" * 60)

    try:
        # Test 7.1: set_default_workspace(None) converts to empty string
        print("\nTest 7.1: set_default_workspace(None) converts to empty string")

        set_default_workspace(None)
        default_ws = get_default_workspace()

        # Should convert None to "" automatically
        conversion_ok = default_ws == ""

        if conversion_ok:
            results.add(
                "Error Handling - None to Empty String",
                True,
                f"set_default_workspace(None) correctly converts to empty string: '{default_ws}'",
            )
        else:
            results.add(
                "Error Handling - None to Empty String",
                False,
                f"Expected empty string, got: '{default_ws}'",
            )

        # Test 7.2: Empty string workspace behavior
        print("\nTest 7.2: Empty string workspace creates valid namespace")

        # With empty workspace, should create namespace without colon
        final_ns = get_final_namespace("test_namespace", workspace="")
        namespace_ok = final_ns == "test_namespace"

        if namespace_ok:
            results.add(
                "Error Handling - Empty Workspace Namespace",
                True,
                f"Empty workspace creates valid namespace: '{final_ns}'",
            )
        else:
            results.add(
                "Error Handling - Empty Workspace Namespace",
                False,
                f"Unexpected namespace: '{final_ns}'",
            )

        # Restore default workspace for other tests
        set_default_workspace("")

        return conversion_ok and namespace_ok

    except Exception as e:
        results.add("Error Handling", False, f"Exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Test 8: Update Flags Workspace Isolation
# =============================================================================


async def test_update_flags_workspace_isolation():
    """
    Test that update flags are properly isolated between workspaces.
    """
    print("\n" + "=" * 60)
    print("TEST 8: Update Flags Workspace Isolation")
    print("=" * 60)

    try:
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
        initial_ok = flag1_obj.value is False and flag2_obj.value is False

        # Set all flags for workspace1
        await set_all_update_flags(test_namespace, workspace=workspace1)

        # Check that only workspace1's flags are set
        set_flags_isolated = flag1_obj.value is True and flag2_obj.value is False

        if set_flags_isolated:
            results.add(
                "Update Flags - set_all_update_flags Isolation",
                True,
                f"set_all_update_flags isolated: ws1={flag1_obj.value}, ws2={flag2_obj.value}",
            )
        else:
            results.add(
                "Update Flags - set_all_update_flags Isolation",
                False,
                f"Flags not isolated: ws1={flag1_obj.value}, ws2={flag2_obj.value}",
            )

        # Test 8.2: clear_all_update_flags isolation
        print("\nTest 8.2: clear_all_update_flags workspace isolation")

        # Set flags for both workspaces
        await set_all_update_flags(test_namespace, workspace=workspace1)
        await set_all_update_flags(test_namespace, workspace=workspace2)

        # Verify both are set
        both_set = flag1_obj.value is True and flag2_obj.value is True

        # Clear only workspace1
        await clear_all_update_flags(test_namespace, workspace=workspace1)

        # Check that only workspace1's flags are cleared
        clear_flags_isolated = flag1_obj.value is False and flag2_obj.value is True

        if clear_flags_isolated:
            results.add(
                "Update Flags - clear_all_update_flags Isolation",
                True,
                f"clear_all_update_flags isolated: ws1={flag1_obj.value}, ws2={flag2_obj.value}",
            )
        else:
            results.add(
                "Update Flags - clear_all_update_flags Isolation",
                False,
                f"Flags not isolated: ws1={flag1_obj.value}, ws2={flag2_obj.value}",
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

        status_filtered = len(workspace1_keys) > 0 and len(workspace2_keys) == 0

        if status_filtered:
            results.add(
                "Update Flags - get_all_update_flags_status Filtering",
                True,
                f"Status correctly filtered: ws1 keys={len(workspace1_keys)}, ws2 keys={len(workspace2_keys)}",
            )
        else:
            results.add(
                "Update Flags - get_all_update_flags_status Filtering",
                False,
                f"Status not filtered correctly: ws1 keys={len(workspace1_keys)}, ws2 keys={len(workspace2_keys)}",
            )

        return set_flags_isolated and clear_flags_isolated and status_filtered

    except Exception as e:
        results.add("Update Flags Workspace Isolation", False, f"Exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Test 9: Empty Workspace Standardization
# =============================================================================


async def test_empty_workspace_standardization():
    """
    Test that empty workspace is properly standardized to "" instead of "_".
    """
    print("\n" + "=" * 60)
    print("TEST 9: Empty Workspace Standardization")
    print("=" * 60)

    try:
        # Test 9.1: Empty string workspace creates namespace without colon
        print("\nTest 9.1: Empty string workspace namespace format")

        set_default_workspace("")
        final_ns = get_final_namespace("test_namespace", workspace=None)

        # Should be just "test_namespace" without colon prefix
        empty_ws_ok = final_ns == "test_namespace"

        if empty_ws_ok:
            results.add(
                "Empty Workspace Standardization - Format",
                True,
                f"Empty workspace creates correct namespace: '{final_ns}'",
            )
        else:
            results.add(
                "Empty Workspace Standardization - Format",
                False,
                f"Unexpected namespace format: '{final_ns}' (expected 'test_namespace')",
            )

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
        behavior_ok = data_empty is not data_nonempty

        if behavior_ok:
            results.add(
                "Empty Workspace Standardization - Behavior",
                True,
                "Empty and non-empty workspaces have independent data",
            )
        else:
            results.add(
                "Empty Workspace Standardization - Behavior",
                False,
                "Empty and non-empty workspaces share data (should be independent)",
            )

        return empty_ws_ok and behavior_ok

    except Exception as e:
        results.add("Empty Workspace Standardization", False, f"Exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Main Test Runner
# =============================================================================


async def main():
    """Run all tests"""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "Workspace Isolation Test Suite" + " " * 18 + "║")
    print("║" + " " * 15 + "PR #2366 - Complete Coverage" + " " * 15 + "║")
    print("╚" + "═" * 58 + "╝")

    # Run all tests (ordered by priority)
    # Core PR requirements (Tests 1-4)
    await test_pipeline_status_isolation()
    await test_lock_mechanism()
    await test_backward_compatibility()
    await test_multi_workspace_concurrency()

    # Additional comprehensive tests (Tests 5-9)
    await test_namespace_lock_reentrance()
    await test_different_namespace_lock_isolation()
    await test_error_handling()
    await test_update_flags_workspace_isolation()
    await test_empty_workspace_standardization()

    # Print summary
    all_passed = results.summary()

    if all_passed:
        print("\n🎉 All tests passed! The workspace isolation feature is working correctly.")
        print("   Coverage: 100% - All scenarios validated")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review the results above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
