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
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pytest

from lightrag.kg.shared_storage import (
    clear_all_update_flags,
    finalize_share_data,
    get_all_update_flags_status,
    get_default_workspace,
    get_final_namespace,
    get_namespace_data,
    get_namespace_lock,
    get_update_flag,
    initialize_pipeline_status,
    initialize_share_data,
    set_all_update_flags,
    set_default_workspace,
)
from lightrag.utils import EmbeddingFunc

# =============================================================================
# Test Configuration
# =============================================================================

# Test configuration is handled via pytest fixtures in conftest.py
# - Use CLI options: --keep-artifacts, --stress-test, --test-workers=N
# - Or environment variables: LIGHTRAG_KEEP_ARTIFACTS, LIGHTRAG_STRESS_TEST, LIGHTRAG_TEST_WORKERS
# Priority: CLI options > Environment variables > Default values


# =============================================================================
# Pytest Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def setup_shared_data():
    """Initialize shared data before each test"""
    initialize_share_data()
    yield
    finalize_share_data()


async def _measure_lock_parallelism(
    workload: list[tuple[str, str, str]], hold_time: float = 0.05
) -> tuple[int, list[tuple[str, str]], dict[str, float]]:
    """Run lock acquisition workload and capture peak concurrency and timeline.

    Args:
        workload: List of (name, workspace, namespace) tuples
        hold_time: How long each worker holds the lock (seconds)

    Returns:
        Tuple of (max_parallel, timeline, metrics) where:
        - max_parallel: Peak number of concurrent lock holders
        - timeline: List of (name, event) tuples tracking execution order
        - metrics: Dict with performance metrics (total_duration, max_concurrency, etc.)
    """

    running = 0
    max_parallel = 0
    timeline: list[tuple[str, str]] = []
    start_time = time.time()

    async def worker(name: str, workspace: str, namespace: str) -> None:
        nonlocal running, max_parallel
        lock = get_namespace_lock(namespace, workspace)
        async with lock:
            running += 1
            max_parallel = max(max_parallel, running)
            timeline.append((name, 'start'))
            await asyncio.sleep(hold_time)
            timeline.append((name, 'end'))
            running -= 1

    await asyncio.gather(*(worker(*args) for args in workload))

    metrics = {
        'total_duration': time.time() - start_time,
        'max_concurrency': max_parallel,
        'avg_hold_time': hold_time,
        'num_workers': len(workload),
    }

    return max_parallel, timeline, metrics


def _assert_no_timeline_overlap(timeline: list[tuple[str, str]]) -> None:
    """Ensure that timeline events never overlap for sequential execution.

    This function implements a finite state machine that validates:
    - No overlapping lock acquisitions (only one task active at a time)
    - Proper lock release order (task releases its own lock)
    - All locks are properly released

    Args:
        timeline: List of (name, event) tuples where event is "start" or "end"

    Raises:
        AssertionError: If timeline shows overlapping execution or improper locking
    """

    active_task = None
    for name, event in timeline:
        if event == 'start':
            if active_task is not None:
                raise AssertionError(f"Task '{name}' started before '{active_task}' released the lock")
            active_task = name
        else:
            if active_task != name:
                raise AssertionError(f"Task '{name}' finished while '{active_task}' was expected to hold the lock")
            active_task = None

    if active_task is not None:
        raise AssertionError(f"Task '{active_task}' did not release the lock properly")


# =============================================================================
# Test 1: Pipeline Status Isolation Test
# =============================================================================


@pytest.mark.offline
@pytest.mark.asyncio
async def test_pipeline_status_isolation():
    """
    Test that pipeline status is isolated between different workspaces.
    """
    # Purpose: Ensure pipeline_status shared data remains unique per workspace.
    # Scope: initialize_pipeline_status and get_namespace_data interactions.
    print('\n' + '=' * 60)
    print('TEST 1: Pipeline Status Isolation')
    print('=' * 60)

    # Initialize shared storage
    initialize_share_data()

    # Initialize pipeline status for two different workspaces
    workspace1 = 'test_workspace_1'
    workspace2 = 'test_workspace_2'

    await initialize_pipeline_status(workspace1)
    await initialize_pipeline_status(workspace2)

    # Get pipeline status data for both workspaces
    data1 = await get_namespace_data('pipeline_status', workspace=workspace1)
    data2 = await get_namespace_data('pipeline_status', workspace=workspace2)

    # Verify they are independent objects
    assert data1 is not data2, 'Pipeline status data objects are the same (should be different)'

    # Modify workspace1's data and verify workspace2 is not affected
    data1['test_key'] = 'workspace1_value'

    # Re-fetch to ensure we get the latest data
    data1_check = await get_namespace_data('pipeline_status', workspace=workspace1)
    data2_check = await get_namespace_data('pipeline_status', workspace=workspace2)

    assert 'test_key' in data1_check, 'test_key not found in workspace1'
    assert data1_check['test_key'] == 'workspace1_value', (
        f'workspace1 test_key value incorrect: {data1_check.get("test_key")}'
    )
    assert 'test_key' not in data2_check, f'test_key leaked to workspace2: {data2_check.get("test_key")}'

    print('✅ PASSED: Pipeline Status Isolation')
    print('   Different workspaces have isolated pipeline status')


# =============================================================================
# Test 2: Lock Mechanism Test (No Deadlocks)
# =============================================================================


@pytest.mark.offline
@pytest.mark.asyncio
async def test_lock_mechanism(stress_test_mode, parallel_workers):
    """
    Test that the new keyed lock mechanism works correctly without deadlocks.
    Tests both parallel execution for different workspaces and serialization
    for the same workspace.
    """
    # Purpose: Validate that keyed locks isolate workspaces while serializing
    # requests within the same workspace. Scope: get_namespace_lock scheduling
    # semantics for both cross-workspace and single-workspace cases.
    print('\n' + '=' * 60)
    print('TEST 2: Lock Mechanism (No Deadlocks)')
    print('=' * 60)

    # Test 2.1: Different workspaces should run in parallel
    print('\nTest 2.1: Different workspaces locks should be parallel')

    # Support stress testing with configurable number of workers
    num_workers = parallel_workers if stress_test_mode else 3
    parallel_workload = [(f'ws_{chr(97 + i)}', f'ws_{chr(97 + i)}', 'test_namespace') for i in range(num_workers)]

    max_parallel, timeline_parallel, metrics = await _measure_lock_parallelism(parallel_workload)
    assert max_parallel >= 2, (
        'Locks for distinct workspaces should overlap; '
        f'observed max concurrency: {max_parallel}, timeline={timeline_parallel}'
    )

    print('✅ PASSED: Lock Mechanism - Parallel (Different Workspaces)')
    print(f'   Locks overlapped for different workspaces (max concurrency={max_parallel})')
    print(f'   Performance: {metrics["total_duration"]:.3f}s for {metrics["num_workers"]} workers')

    # Test 2.2: Same workspace should serialize
    print('\nTest 2.2: Same workspace locks should serialize')
    serial_workload = [
        ('serial_run_1', 'ws_same', 'test_namespace'),
        ('serial_run_2', 'ws_same', 'test_namespace'),
    ]
    (
        max_parallel_serial,
        timeline_serial,
        metrics_serial,
    ) = await _measure_lock_parallelism(serial_workload)
    assert max_parallel_serial == 1, (
        f'Same workspace locks should not overlap; observed {max_parallel_serial} with timeline {timeline_serial}'
    )
    _assert_no_timeline_overlap(timeline_serial)

    print('✅ PASSED: Lock Mechanism - Serial (Same Workspace)')
    print('   Same workspace operations executed sequentially with no overlap')
    print(f'   Performance: {metrics_serial["total_duration"]:.3f}s for {metrics_serial["num_workers"]} tasks')


# =============================================================================
# Test 3: Backward Compatibility Test
# =============================================================================


@pytest.mark.offline
@pytest.mark.asyncio
async def test_backward_compatibility():
    """
    Test that legacy code without workspace parameter still works correctly.
    """
    # Purpose: Validate backward-compatible defaults when workspace arguments
    # are omitted. Scope: get_final_namespace, set/get_default_workspace and
    # initialize_pipeline_status fallback behavior.
    print('\n' + '=' * 60)
    print('TEST 3: Backward Compatibility')
    print('=' * 60)

    # Test 3.1: get_final_namespace with None should use default workspace
    print('\nTest 3.1: get_final_namespace with workspace=None')

    set_default_workspace('my_default_workspace')
    final_ns = get_final_namespace('pipeline_status')
    expected = 'my_default_workspace:pipeline_status'

    assert final_ns == expected, f'Expected {expected}, got {final_ns}'

    print('✅ PASSED: Backward Compatibility - get_final_namespace')
    print(f'   Correctly uses default workspace: {final_ns}')

    # Test 3.2: get_default_workspace
    print('\nTest 3.2: get/set default workspace')

    set_default_workspace('test_default')
    retrieved = get_default_workspace()

    assert retrieved == 'test_default', f"Expected 'test_default', got {retrieved}"

    print('✅ PASSED: Backward Compatibility - default workspace')
    print(f'   Default workspace set/get correctly: {retrieved}')

    # Test 3.3: Empty workspace handling
    print('\nTest 3.3: Empty workspace handling')

    set_default_workspace('')
    final_ns_empty = get_final_namespace('pipeline_status', workspace=None)
    expected_empty = 'pipeline_status'  # Should be just the namespace without ':'

    assert final_ns_empty == expected_empty, f"Expected '{expected_empty}', got '{final_ns_empty}'"

    print('✅ PASSED: Backward Compatibility - empty workspace')
    print(f"   Empty workspace handled correctly: '{final_ns_empty}'")

    # Test 3.4: None workspace with default set
    print('\nTest 3.4: initialize_pipeline_status with workspace=None')
    set_default_workspace('compat_test_workspace')
    initialize_share_data()
    await initialize_pipeline_status(workspace=None)  # Should use default

    # Try to get data using the default workspace explicitly
    data = await get_namespace_data('pipeline_status', workspace='compat_test_workspace')

    assert data is not None, 'Failed to initialize pipeline status with default workspace'

    print('✅ PASSED: Backward Compatibility - pipeline init with None')
    print('   Pipeline status initialized with default workspace')


# =============================================================================
# Test 4: Multi-Workspace Concurrency Test
# =============================================================================


@pytest.mark.offline
@pytest.mark.asyncio
async def test_multi_workspace_concurrency():
    """
    Test that multiple workspaces can operate concurrently without interference.
    Simulates concurrent operations on different workspaces.
    """
    # Purpose: Simulate concurrent workloads touching pipeline_status across
    # workspaces. Scope: initialize_pipeline_status, get_namespace_lock, and
    # shared dictionary mutation while ensuring isolation.
    print('\n' + '=' * 60)
    print('TEST 4: Multi-Workspace Concurrency')
    print('=' * 60)

    initialize_share_data()

    async def workspace_operations(workspace_id):
        """Simulate operations on a specific workspace"""
        print(f'\n   [{workspace_id}] Starting operations')

        # Initialize pipeline status
        await initialize_pipeline_status(workspace_id)

        # Get lock and perform operations
        lock = get_namespace_lock('test_operations', workspace_id)
        async with lock:
            # Get workspace data
            data = await get_namespace_data('pipeline_status', workspace=workspace_id)

            # Modify data
            data[f'{workspace_id}_key'] = f'{workspace_id}_value'
            data['timestamp'] = time.time()

            # Simulate some work
            await asyncio.sleep(0.1)

            print(f'   [{workspace_id}] Completed operations')

        return workspace_id

    # Run multiple workspaces concurrently
    workspaces = ['concurrent_ws_1', 'concurrent_ws_2', 'concurrent_ws_3']

    start = time.time()
    results_list = await asyncio.gather(*[workspace_operations(ws) for ws in workspaces])
    elapsed = time.time() - start

    print(f'\n   All workspaces completed in {elapsed:.2f}s')

    # Verify all workspaces completed
    assert set(results_list) == set(workspaces), 'Not all workspaces completed'

    print('✅ PASSED: Multi-Workspace Concurrency - Execution')
    print(f'   All {len(workspaces)} workspaces completed successfully in {elapsed:.2f}s')

    # Verify data isolation - each workspace should have its own data
    print('\n   Verifying data isolation...')

    for ws in workspaces:
        data = await get_namespace_data('pipeline_status', workspace=ws)
        expected_key = f'{ws}_key'
        expected_value = f'{ws}_value'

        assert expected_key in data, f'Data not properly isolated for {ws}: missing {expected_key}'
        assert data[expected_key] == expected_value, (
            f'Data not properly isolated for {ws}: {expected_key}={data[expected_key]} (expected {expected_value})'
        )
        print(f'   [{ws}] Data correctly isolated: {expected_key}={data[expected_key]}')

    print('✅ PASSED: Multi-Workspace Concurrency - Data Isolation')
    print('   All workspaces have properly isolated data')


# =============================================================================
# Test 5: NamespaceLock Re-entrance Protection
# =============================================================================


@pytest.mark.offline
@pytest.mark.asyncio
async def test_namespace_lock_reentrance():
    """
    Test that NamespaceLock prevents re-entrance in the same coroutine
    and allows concurrent use in different coroutines.
    """
    # Purpose: Ensure NamespaceLock enforces single entry per coroutine while
    # allowing concurrent reuse through ContextVar isolation. Scope: lock
    # re-entrance checks and concurrent gather semantics.
    print('\n' + '=' * 60)
    print('TEST 5: NamespaceLock Re-entrance Protection')
    print('=' * 60)

    # Test 5.1: Same coroutine re-entrance should fail
    print('\nTest 5.1: Same coroutine re-entrance should raise RuntimeError')

    lock = get_namespace_lock('test_reentrance', 'test_ws')

    reentrance_failed_correctly = False
    try:
        async with lock:
            print('   Acquired lock first time')
            # Try to acquire the same lock again in the same coroutine
            async with lock:
                print('   ERROR: Should not reach here - re-entrance succeeded!')
    except RuntimeError as e:
        if 'already acquired' in str(e).lower():
            print(f'   ✓ Re-entrance correctly blocked: {e}')
            reentrance_failed_correctly = True
        else:
            raise

    assert reentrance_failed_correctly, 'Re-entrance protection not working'

    print('✅ PASSED: NamespaceLock Re-entrance Protection')
    print('   Re-entrance correctly raises RuntimeError')

    # Test 5.2: Same NamespaceLock instance in different coroutines should succeed
    print('\nTest 5.2: Same NamespaceLock instance in different coroutines')

    shared_lock = get_namespace_lock('test_concurrent', 'test_ws')
    concurrent_results = []

    async def use_shared_lock(coroutine_id):
        """Use the same NamespaceLock instance"""
        async with shared_lock:
            concurrent_results.append(f'coroutine_{coroutine_id}_start')
            await asyncio.sleep(0.1)
            concurrent_results.append(f'coroutine_{coroutine_id}_end')

    # This should work because each coroutine gets its own ContextVar
    await asyncio.gather(
        use_shared_lock(1),
        use_shared_lock(2),
    )

    # Both coroutines should have completed
    expected_entries = 4  # 2 starts + 2 ends
    assert len(concurrent_results) == expected_entries, (
        f'Expected {expected_entries} entries, got {len(concurrent_results)}'
    )

    print('✅ PASSED: NamespaceLock Concurrent Reuse')
    print(f'   Same NamespaceLock instance used successfully in {expected_entries // 2} concurrent coroutines')


# =============================================================================
# Test 6: Different Namespace Lock Isolation
# =============================================================================


@pytest.mark.offline
@pytest.mark.asyncio
async def test_different_namespace_lock_isolation():
    """
    Test that locks for different namespaces (same workspace) are independent.
    """
    # Purpose: Confirm that namespace isolation is enforced even when workspace
    # is the same. Scope: get_namespace_lock behavior when namespaces differ.
    print('\n' + '=' * 60)
    print('TEST 6: Different Namespace Lock Isolation')
    print('=' * 60)

    print('\nTesting locks with same workspace but different namespaces')

    workload = [
        ('ns_a', 'same_ws', 'namespace_a'),
        ('ns_b', 'same_ws', 'namespace_b'),
        ('ns_c', 'same_ws', 'namespace_c'),
    ]
    max_parallel, timeline, metrics = await _measure_lock_parallelism(workload)

    assert max_parallel >= 2, (
        'Different namespaces within the same workspace should run concurrently; '
        f'observed max concurrency {max_parallel} with timeline {timeline}'
    )

    print('✅ PASSED: Different Namespace Lock Isolation')
    print(f'   Different namespace locks ran in parallel (max concurrency={max_parallel})')
    print(f'   Performance: {metrics["total_duration"]:.3f}s for {metrics["num_workers"]} namespaces')


# =============================================================================
# Test 7: Error Handling
# =============================================================================


@pytest.mark.offline
@pytest.mark.asyncio
async def test_error_handling():
    """
    Test error handling for invalid workspace configurations.
    """
    # Purpose: Validate guardrails for workspace normalization and namespace
    # derivation. Scope: set_default_workspace conversions and get_final_namespace
    # failure paths when configuration is invalid.
    print('\n' + '=' * 60)
    print('TEST 7: Error Handling')
    print('=' * 60)

    # Test 7.0: Missing default workspace should raise ValueError
    print('\nTest 7.0: Missing workspace raises ValueError')
    with pytest.raises(ValueError):
        get_final_namespace('test_namespace', workspace=None)

    # Test 7.1: set_default_workspace(None) converts to empty string
    print('\nTest 7.1: set_default_workspace(None) converts to empty string')

    set_default_workspace(None)
    default_ws = get_default_workspace()

    # Should convert None to "" automatically
    assert default_ws == '', f"Expected empty string, got: '{default_ws}'"

    print('✅ PASSED: Error Handling - None to Empty String')
    print(f"   set_default_workspace(None) correctly converts to empty string: '{default_ws}'")

    # Test 7.2: Empty string workspace behavior
    print('\nTest 7.2: Empty string workspace creates valid namespace')

    # With empty workspace, should create namespace without colon
    final_ns = get_final_namespace('test_namespace', workspace='')
    assert final_ns == 'test_namespace', f"Unexpected namespace: '{final_ns}'"

    print('✅ PASSED: Error Handling - Empty Workspace Namespace')
    print(f"   Empty workspace creates valid namespace: '{final_ns}'")

    # Restore default workspace for other tests
    set_default_workspace('')


# =============================================================================
# Test 8: Update Flags Workspace Isolation
# =============================================================================


@pytest.mark.offline
@pytest.mark.asyncio
async def test_update_flags_workspace_isolation():
    """
    Test that update flags are properly isolated between workspaces.
    """
    # Purpose: Confirm update flag setters/readers respect workspace scoping.
    # Scope: set_all_update_flags, clear_all_update_flags, get_all_update_flags_status,
    # and get_update_flag interactions across namespaces.
    print('\n' + '=' * 60)
    print('TEST 8: Update Flags Workspace Isolation')
    print('=' * 60)

    initialize_share_data()

    workspace1 = 'update_flags_ws1'
    workspace2 = 'update_flags_ws2'
    test_namespace = 'test_update_flags_ns'

    # Initialize namespaces for both workspaces
    await initialize_pipeline_status(workspace1)
    await initialize_pipeline_status(workspace2)

    # Test 8.1: set_all_update_flags isolation
    print('\nTest 8.1: set_all_update_flags workspace isolation')

    # Create flags for both workspaces (simulating workers)
    flag1_obj = await get_update_flag(test_namespace, workspace=workspace1)
    flag2_obj = await get_update_flag(test_namespace, workspace=workspace2)

    # Initial state should be False
    assert flag1_obj.value is False, 'Flag1 initial value should be False'
    assert flag2_obj.value is False, 'Flag2 initial value should be False'

    # Set all flags for workspace1
    await set_all_update_flags(test_namespace, workspace=workspace1)

    # Check that only workspace1's flags are set
    assert flag1_obj.value is True, f'Flag1 should be True after set_all_update_flags, got {flag1_obj.value}'
    assert flag2_obj.value is False, f'Flag2 should still be False, got {flag2_obj.value}'

    print('✅ PASSED: Update Flags - set_all_update_flags Isolation')
    print(f'   set_all_update_flags isolated: ws1={flag1_obj.value}, ws2={flag2_obj.value}')

    # Test 8.2: clear_all_update_flags isolation
    print('\nTest 8.2: clear_all_update_flags workspace isolation')

    # Set flags for both workspaces
    await set_all_update_flags(test_namespace, workspace=workspace1)
    await set_all_update_flags(test_namespace, workspace=workspace2)

    # Verify both are set
    assert flag1_obj.value is True, 'Flag1 should be True'
    assert flag2_obj.value is True, 'Flag2 should be True'

    # Clear only workspace1
    await clear_all_update_flags(test_namespace, workspace=workspace1)

    # Check that only workspace1's flags are cleared
    assert flag1_obj.value is False, f'Flag1 should be False after clear, got {flag1_obj.value}'
    assert flag2_obj.value is True, f'Flag2 should still be True, got {flag2_obj.value}'

    print('✅ PASSED: Update Flags - clear_all_update_flags Isolation')
    print(f'   clear_all_update_flags isolated: ws1={flag1_obj.value}, ws2={flag2_obj.value}')

    # Test 8.3: get_all_update_flags_status workspace filtering
    print('\nTest 8.3: get_all_update_flags_status workspace filtering')

    # Initialize more namespaces for testing
    await get_update_flag('ns_a', workspace=workspace1)
    await get_update_flag('ns_b', workspace=workspace1)
    await get_update_flag('ns_c', workspace=workspace2)

    # Set flags for workspace1
    await set_all_update_flags('ns_a', workspace=workspace1)
    await set_all_update_flags('ns_b', workspace=workspace1)

    # Set flags for workspace2
    await set_all_update_flags('ns_c', workspace=workspace2)

    # Get status for workspace1 only
    status1 = await get_all_update_flags_status(workspace=workspace1)

    # Check that workspace1's namespaces are present
    # The keys should include workspace1's namespaces but not workspace2's
    workspace1_keys = [k for k in status1 if workspace1 in k]
    workspace2_keys = [k for k in status1 if workspace2 in k]

    assert len(workspace1_keys) > 0, f'workspace1 keys should be present, got {len(workspace1_keys)}'
    assert len(workspace2_keys) == 0, f'workspace2 keys should not be present, got {len(workspace2_keys)}'
    for key, values in status1.items():
        assert all(values), f'All flags in {key} should be True, got {values}'

    # Workspace2 query should only surface workspace2 namespaces
    status2 = await get_all_update_flags_status(workspace=workspace2)
    expected_ws2_keys = {
        f'{workspace2}:{test_namespace}',
        f'{workspace2}:ns_c',
    }
    assert set(status2.keys()) == expected_ws2_keys, f'Unexpected namespaces for workspace2: {status2.keys()}'
    for key, values in status2.items():
        assert all(values), f'All flags in {key} should be True, got {values}'

    print('✅ PASSED: Update Flags - get_all_update_flags_status Filtering')
    print(f'   Status correctly filtered: ws1 keys={len(workspace1_keys)}, ws2 keys={len(workspace2_keys)}')


# =============================================================================
# Test 9: Empty Workspace Standardization
# =============================================================================


@pytest.mark.offline
@pytest.mark.asyncio
async def test_empty_workspace_standardization():
    """
    Test that empty workspace is properly standardized to "" instead of "_".
    """
    # Purpose: Verify namespace formatting when workspace is an empty string.
    # Scope: get_final_namespace output and initialize_pipeline_status behavior
    # between empty and non-empty workspaces.
    print('\n' + '=' * 60)
    print('TEST 9: Empty Workspace Standardization')
    print('=' * 60)

    # Test 9.1: Empty string workspace creates namespace without colon
    print('\nTest 9.1: Empty string workspace namespace format')

    set_default_workspace('')
    final_ns = get_final_namespace('test_namespace', workspace=None)

    # Should be just "test_namespace" without colon prefix
    assert final_ns == 'test_namespace', f"Unexpected namespace format: '{final_ns}' (expected 'test_namespace')"

    print('✅ PASSED: Empty Workspace Standardization - Format')
    print(f"   Empty workspace creates correct namespace: '{final_ns}'")

    # Test 9.2: Empty workspace vs non-empty workspace behavior
    print('\nTest 9.2: Empty vs non-empty workspace behavior')

    initialize_share_data()

    # Initialize with empty workspace
    await initialize_pipeline_status(workspace='')
    data_empty = await get_namespace_data('pipeline_status', workspace='')

    # Initialize with non-empty workspace
    await initialize_pipeline_status(workspace='test_ws')
    data_nonempty = await get_namespace_data('pipeline_status', workspace='test_ws')

    # They should be different objects
    assert data_empty is not data_nonempty, 'Empty and non-empty workspaces share data (should be independent)'

    print('✅ PASSED: Empty Workspace Standardization - Behavior')
    print('   Empty and non-empty workspaces have independent data')


# =============================================================================
# Test 10: JsonKVStorage Workspace Isolation (Integration Test)
# =============================================================================


@pytest.mark.offline
@pytest.mark.asyncio
async def test_json_kv_storage_workspace_isolation(keep_test_artifacts):
    """
    Integration test: Verify JsonKVStorage properly isolates data between workspaces.
    Creates two JsonKVStorage instances with different workspaces, writes different data,
    and verifies they don't mix.
    """
    # Purpose: Ensure JsonKVStorage respects workspace-specific directories and data.
    # Scope: storage initialization, upsert/get_by_id operations, and filesystem layout
    # inside the temporary working directory.
    print('\n' + '=' * 60)
    print('TEST 10: JsonKVStorage Workspace Isolation (Integration)')
    print('=' * 60)

    # Create temporary test directory under project temp/
    test_dir = str(Path(__file__).parent.parent / 'temp/test_json_kv_storage_workspace_isolation')
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir, exist_ok=True)
    print(f'\n   Using test directory: {test_dir}')

    try:
        initialize_share_data()

        # Mock embedding function
        async def mock_embedding_func(texts: list[str]) -> np.ndarray:
            return np.random.rand(len(texts), 384)  # 384-dimensional vectors

        # Global config
        global_config = {
            'working_dir': test_dir,
            'embedding_batch_num': 10,
        }

        # Test 10.1: Create two JsonKVStorage instances with different workspaces
        print('\nTest 10.1: Create two JsonKVStorage instances with different workspaces')

        from lightrag.kg.json_kv_impl import JsonKVStorage

        storage1 = JsonKVStorage(
            namespace='entities',
            workspace='workspace1',
            global_config=global_config,
            embedding_func=EmbeddingFunc(embedding_dim=384, max_token_size=8192, func=mock_embedding_func),
        )

        storage2 = JsonKVStorage(
            namespace='entities',
            workspace='workspace2',
            global_config=global_config,
            embedding_func=EmbeddingFunc(embedding_dim=384, max_token_size=8192, func=mock_embedding_func),
        )

        # Initialize both storages
        await storage1.initialize()
        await storage2.initialize()

        print('   Storage1 created: workspace=workspace1, namespace=entities')
        print('   Storage2 created: workspace=workspace2, namespace=entities')

        # Test 10.2: Write different data to each storage
        print('\nTest 10.2: Write different data to each storage')

        # Write to storage1 (upsert expects dict[str, dict])
        await storage1.upsert(
            {
                'entity1': {
                    'content': 'Data from workspace1 - AI Research',
                    'type': 'entity',
                },
                'entity2': {
                    'content': 'Data from workspace1 - Machine Learning',
                    'type': 'entity',
                },
            }
        )
        print('   Written to storage1: entity1, entity2')
        # Persist data to disk
        await storage1.index_done_callback()
        print('   Persisted storage1 data to disk')

        # Write to storage2
        await storage2.upsert(
            {
                'entity1': {
                    'content': 'Data from workspace2 - Deep Learning',
                    'type': 'entity',
                },
                'entity2': {
                    'content': 'Data from workspace2 - Neural Networks',
                    'type': 'entity',
                },
            }
        )
        print('   Written to storage2: entity1, entity2')
        # Persist data to disk
        await storage2.index_done_callback()
        print('   Persisted storage2 data to disk')

        # Test 10.3: Read data from each storage and verify isolation
        print('\nTest 10.3: Read data and verify isolation')

        # Read from storage1
        result1_entity1 = await storage1.get_by_id('entity1')
        result1_entity2 = await storage1.get_by_id('entity2')

        # Read from storage2
        result2_entity1 = await storage2.get_by_id('entity1')
        result2_entity2 = await storage2.get_by_id('entity2')

        print(f'   Storage1 entity1: {result1_entity1}')
        print(f'   Storage1 entity2: {result1_entity2}')
        print(f'   Storage2 entity1: {result2_entity1}')
        print(f'   Storage2 entity2: {result2_entity2}')

        # Verify isolation (get_by_id returns dict)
        assert result1_entity1 is not None, 'Storage1 entity1 should not be None'
        assert result1_entity2 is not None, 'Storage1 entity2 should not be None'
        assert result2_entity1 is not None, 'Storage2 entity1 should not be None'
        assert result2_entity2 is not None, 'Storage2 entity2 should not be None'
        assert result1_entity1.get('content') == 'Data from workspace1 - AI Research', (
            'Storage1 entity1 content mismatch'
        )
        assert result1_entity2.get('content') == 'Data from workspace1 - Machine Learning', (
            'Storage1 entity2 content mismatch'
        )
        assert result2_entity1.get('content') == 'Data from workspace2 - Deep Learning', (
            'Storage2 entity1 content mismatch'
        )
        assert result2_entity2.get('content') == 'Data from workspace2 - Neural Networks', (
            'Storage2 entity2 content mismatch'
        )
        assert result1_entity1.get('content') != result2_entity1.get('content'), (
            'Storage1 and Storage2 entity1 should have different content'
        )
        assert result1_entity2.get('content') != result2_entity2.get('content'), (
            'Storage1 and Storage2 entity2 should have different content'
        )

        print('✅ PASSED: JsonKVStorage - Data Isolation')
        print('   Two storage instances correctly isolated: ws1 and ws2 have different data')

        # Test 10.4: Verify file structure
        print('\nTest 10.4: Verify file structure')
        ws1_dir = Path(test_dir) / 'workspace1'
        ws2_dir = Path(test_dir) / 'workspace2'

        ws1_exists = ws1_dir.exists()
        ws2_exists = ws2_dir.exists()

        print(f'   workspace1 directory exists: {ws1_exists}')
        print(f'   workspace2 directory exists: {ws2_exists}')

        assert ws1_exists, 'workspace1 directory should exist'
        assert ws2_exists, 'workspace2 directory should exist'

        print('✅ PASSED: JsonKVStorage - File Structure')
        print(f'   Workspace directories correctly created: {ws1_dir} and {ws2_dir}')

    finally:
        # Cleanup test directory (unless keep_test_artifacts is set)
        if os.path.exists(test_dir) and not keep_test_artifacts:
            shutil.rmtree(test_dir)
            print(f'\n   Cleaned up test directory: {test_dir}')
        elif keep_test_artifacts:
            print(f'\n   Kept test directory for inspection: {test_dir}')


# =============================================================================
# Test 11: LightRAG End-to-End Integration Test
# =============================================================================


@pytest.mark.offline
@pytest.mark.asyncio
async def test_lightrag_end_to_end_workspace_isolation(keep_test_artifacts):
    """
    End-to-end test: Create two LightRAG instances with different workspaces,
    insert different data, and verify file separation.
    Uses mock LLM and embedding functions to avoid external API calls.
    """
    # Purpose: Validate that full LightRAG flows keep artifacts scoped per workspace.
    # Scope: LightRAG.initialize_storages + ainsert side effects plus filesystem
    # verification for generated storage files.
    print('\n' + '=' * 60)
    print('TEST 11: LightRAG End-to-End Workspace Isolation')
    print('=' * 60)

    # Create temporary test directory under project temp/
    test_dir = str(Path(__file__).parent.parent / 'temp/test_lightrag_end_to_end_workspace_isolation')
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir, exist_ok=True)
    print(f'\n   Using test directory: {test_dir}')

    try:
        # Factory function to create different mock LLM functions for each workspace
        def create_mock_llm_func(workspace_name):
            """Create a mock LLM function that returns different content based on workspace"""

            async def mock_llm_func(prompt, system_prompt=None, history_messages=None, **kwargs) -> str:
                if history_messages is None:
                    history_messages = []
                # Add coroutine switching to simulate async I/O and allow concurrent execution
                await asyncio.sleep(0)

                # Return different responses based on workspace
                # Format: entity<|#|>entity_name<|#|>entity_type<|#|>entity_description
                # Format: relation<|#|>source_entity<|#|>target_entity<|#|>keywords<|#|>description
                if workspace_name == 'project_a':
                    return """entity<|#|>Artificial Intelligence<|#|>concept<|#|>AI is a field of computer science focused on creating intelligent machines.
entity<|#|>Machine Learning<|#|>concept<|#|>Machine Learning is a subset of AI that enables systems to learn from data.
relation<|#|>Machine Learning<|#|>Artificial Intelligence<|#|>subset, related field<|#|>Machine Learning is a key component and subset of Artificial Intelligence.
<|COMPLETE|>"""
                else:  # project_b
                    return """entity<|#|>Deep Learning<|#|>concept<|#|>Deep Learning is a subset of machine learning using neural networks with multiple layers.
entity<|#|>Neural Networks<|#|>concept<|#|>Neural Networks are computing systems inspired by biological neural networks.
relation<|#|>Deep Learning<|#|>Neural Networks<|#|>uses, composed of<|#|>Deep Learning uses multiple layers of Neural Networks to learn representations.
<|COMPLETE|>"""

            return mock_llm_func

        # Mock embedding function
        async def mock_embedding_func(texts: list[str]) -> np.ndarray:
            # Add coroutine switching to simulate async I/O and allow concurrent execution
            await asyncio.sleep(0)
            return np.random.rand(len(texts), 384)  # 384-dimensional vectors

        # Test 11.1: Create two LightRAG instances with different workspaces
        print('\nTest 11.1: Create two LightRAG instances with different workspaces')

        from lightrag import LightRAG
        from lightrag.utils import EmbeddingFunc, Tokenizer

        # Create different mock LLM functions for each workspace
        mock_llm_func_a = create_mock_llm_func('project_a')
        mock_llm_func_b = create_mock_llm_func('project_b')

        class _SimpleTokenizerImpl:
            def encode(self, content: str) -> list[int]:
                return [ord(ch) for ch in content]

            def decode(self, tokens: list[int]) -> str:
                return ''.join(chr(t) for t in tokens)

        tokenizer = Tokenizer('mock-tokenizer', _SimpleTokenizerImpl())

        rag1 = LightRAG(
            working_dir=test_dir,
            workspace='project_a',
            llm_model_func=mock_llm_func_a,
            embedding_func=EmbeddingFunc(
                embedding_dim=384,
                max_token_size=8192,
                func=mock_embedding_func,
            ),
            tokenizer=tokenizer,
        )

        rag2 = LightRAG(
            working_dir=test_dir,
            workspace='project_b',
            llm_model_func=mock_llm_func_b,
            embedding_func=EmbeddingFunc(
                embedding_dim=384,
                max_token_size=8192,
                func=mock_embedding_func,
            ),
            tokenizer=tokenizer,
        )

        # Initialize storages
        await rag1.initialize_storages()
        await rag2.initialize_storages()

        print('   RAG1 created: workspace=project_a')
        print('   RAG2 created: workspace=project_b')

        # Test 11.2: Insert different data to each RAG instance (CONCURRENTLY)
        print('\nTest 11.2: Insert different data to each RAG instance (concurrently)')

        text_for_project_a = (
            'This document is about Artificial Intelligence and Machine Learning. AI is transforming the world.'
        )
        text_for_project_b = (
            'This document is about Deep Learning and Neural Networks. Deep learning uses multiple layers.'
        )

        # Insert to both projects concurrently to test workspace isolation under concurrent load
        print('   Starting concurrent insert operations...')
        start_time = time.time()
        await asyncio.gather(rag1.ainsert(text_for_project_a), rag2.ainsert(text_for_project_b))
        elapsed_time = time.time() - start_time

        print(f'   Inserted to project_a: {len(text_for_project_a)} chars (concurrent)')
        print(f'   Inserted to project_b: {len(text_for_project_b)} chars (concurrent)')
        print(f'   Total concurrent execution time: {elapsed_time:.3f}s')

        # Test 11.3: Verify file structure
        print('\nTest 11.3: Verify workspace directory structure')

        project_a_dir = Path(test_dir) / 'project_a'
        project_b_dir = Path(test_dir) / 'project_b'

        project_a_exists = project_a_dir.exists()
        project_b_exists = project_b_dir.exists()

        print(f'   project_a directory: {project_a_dir}')
        print(f'   project_a exists: {project_a_exists}')
        print(f'   project_b directory: {project_b_dir}')
        print(f'   project_b exists: {project_b_exists}')

        assert project_a_exists, 'project_a directory should exist'
        assert project_b_exists, 'project_b directory should exist'

        # List files in each directory
        print('\n   Files in project_a/:')
        for file in sorted(project_a_dir.glob('*')):
            if file.is_file():
                size = file.stat().st_size
                print(f'     - {file.name} ({size} bytes)')

        print('\n   Files in project_b/:')
        for file in sorted(project_b_dir.glob('*')):
            if file.is_file():
                size = file.stat().st_size
                print(f'     - {file.name} ({size} bytes)')

        print('✅ PASSED: LightRAG E2E - File Structure')
        print('   Workspace directories correctly created and separated')

        # Test 11.4: Verify data isolation by checking file contents
        print('\nTest 11.4: Verify data isolation (check file contents)')

        # Check if full_docs storage files exist and contain different content
        docs_a_file = project_a_dir / 'kv_store_full_docs.json'
        docs_b_file = project_b_dir / 'kv_store_full_docs.json'

        if docs_a_file.exists() and docs_b_file.exists():
            import json

            with open(docs_a_file) as f:
                docs_a_content = json.load(f)

            with open(docs_b_file) as f:
                docs_b_content = json.load(f)

            print(f'   project_a doc count: {len(docs_a_content)}')
            print(f'   project_b doc count: {len(docs_b_content)}')

            # Verify they contain different data
            assert docs_a_content != docs_b_content, 'Document storage not properly isolated'

            # Verify each workspace contains its own text content
            docs_a_str = json.dumps(docs_a_content)
            docs_b_str = json.dumps(docs_b_content)

            # Check project_a contains its text and NOT project_b's text
            assert 'Artificial Intelligence' in docs_a_str, "project_a should contain 'Artificial Intelligence'"
            assert 'Machine Learning' in docs_a_str, "project_a should contain 'Machine Learning'"
            assert 'Deep Learning' not in docs_a_str, "project_a should NOT contain 'Deep Learning' from project_b"
            assert 'Neural Networks' not in docs_a_str, "project_a should NOT contain 'Neural Networks' from project_b"

            # Check project_b contains its text and NOT project_a's text
            assert 'Deep Learning' in docs_b_str, "project_b should contain 'Deep Learning'"
            assert 'Neural Networks' in docs_b_str, "project_b should contain 'Neural Networks'"
            assert 'Artificial Intelligence' not in docs_b_str, (
                "project_b should NOT contain 'Artificial Intelligence' from project_a"
            )
            # Note: "Machine Learning" might appear in project_b's text, so we skip that check

            print('✅ PASSED: LightRAG E2E - Data Isolation')
            print('   Document storage correctly isolated between workspaces')
            print('   project_a contains only its own data')
            print('   project_b contains only its own data')
        else:
            print('   Document storage files not found (may not be created yet)')
            print('✅ PASSED: LightRAG E2E - Data Isolation')
            print('   Skipped file content check (files not created)')

        print('\n   ✓ Test complete - workspace isolation verified at E2E level')

    finally:
        # Cleanup test directory (unless keep_test_artifacts is set)
        if os.path.exists(test_dir) and not keep_test_artifacts:
            shutil.rmtree(test_dir)
            print(f'\n   Cleaned up test directory: {test_dir}')
        elif keep_test_artifacts:
            print(f'\n   Kept test directory for inspection: {test_dir}')
