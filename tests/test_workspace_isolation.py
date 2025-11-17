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
# Main Test Runner
# =============================================================================


async def main():
    """Run all tests"""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "Workspace Isolation Test Suite" + " " * 18 + "║")
    print("║" + " " * 18 + "PR #2366" + " " * 32 + "║")
    print("╚" + "═" * 58 + "╝")

    # Run all tests
    await test_pipeline_status_isolation()
    await test_lock_mechanism()
    await test_backward_compatibility()
    await test_multi_workspace_concurrency()

    # Print summary
    all_passed = results.summary()

    if all_passed:
        print("\n🎉 All tests passed! The workspace isolation feature is working correctly.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review the results above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
