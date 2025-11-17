# Workspace Isolation Test Suite

## Overview
Comprehensive test coverage for LightRAG's workspace isolation feature, ensuring that different workspaces (projects) can coexist independently without data contamination or resource conflicts.

## Test Architecture

### Design Principles
1. **Concurrency-Based Assertions**: Instead of timing-based tests (which are flaky), we measure actual concurrent lock holders
2. **Timeline Validation**: Finite state machine validates proper sequential execution
3. **Performance Metrics**: Each test reports execution metrics for debugging and optimization
4. **Configurable Stress Testing**: Environment variables control test intensity

## Test Categories

### 1. Data Isolation Tests
**Tests:** 1, 4, 8, 9, 10  
**Purpose:** Verify that data in one workspace doesn't leak into another

- **Test 1: Pipeline Status Isolation** - Core shared data structures remain separate
- **Test 4: Multi-Workspace Concurrency** - Concurrent operations don't interfere
- **Test 8: Update Flags Isolation** - Flag management respects workspace boundaries
- **Test 9: Empty Workspace Standardization** - Edge case handling for empty workspace strings
- **Test 10: JsonKVStorage Integration** - Storage layer properly isolates data

### 2. Lock Mechanism Tests
**Tests:** 2, 5, 6  
**Purpose:** Validate that locking mechanisms allow parallelism across workspaces while enforcing serialization within workspaces

- **Test 2: Lock Mechanism** - Different workspaces run in parallel, same workspace serializes
- **Test 5: Re-entrance Protection** - Prevent deadlocks from re-entrant lock acquisition
- **Test 6: Namespace Lock Isolation** - Different namespaces within same workspace are independent

### 3. Backward Compatibility Tests
**Test:** 3  
**Purpose:** Ensure legacy code without workspace parameters still functions correctly

- Default workspace fallback behavior
- Empty workspace handling
- None vs empty string normalization

### 4. Error Handling Tests
**Test:** 7  
**Purpose:** Validate guardrails for invalid configurations

- Missing workspace validation
- Workspace normalization
- Edge case handling

### 5. End-to-End Integration Tests
**Test:** 11  
**Purpose:** Validate complete LightRAG workflows maintain isolation

- Full document insertion pipeline
- File system separation
- Data content verification

## Running Tests

### Basic Usage
```bash
# Run all workspace isolation tests
pytest tests/test_workspace_isolation.py -v

# Run specific test
pytest tests/test_workspace_isolation.py::test_lock_mechanism -v

# Run with detailed output
pytest tests/test_workspace_isolation.py -v -s
```

### Environment Configuration

#### Stress Testing
Enable stress testing with configurable number of workers:
```bash
# Enable stress mode with default 3 workers
LIGHTRAG_STRESS_TEST=true pytest tests/test_workspace_isolation.py -v

# Custom number of workers (e.g., 10)
LIGHTRAG_STRESS_TEST=true LIGHTRAG_TEST_WORKERS=10 pytest tests/test_workspace_isolation.py -v
```

#### Keep Test Artifacts
Preserve temporary directories for manual inspection:
```bash
# Keep test artifacts (useful for debugging)
LIGHTRAG_KEEP_ARTIFACTS=true pytest tests/test_workspace_isolation.py -v
```

#### Combined Example
```bash
# Stress test with 20 workers and keep artifacts
LIGHTRAG_STRESS_TEST=true \
LIGHTRAG_TEST_WORKERS=20 \
LIGHTRAG_KEEP_ARTIFACTS=true \
pytest tests/test_workspace_isolation.py::test_lock_mechanism -v -s
```

### CI/CD Integration
```bash
# Recommended CI/CD command (no artifacts, default workers)
pytest tests/test_workspace_isolation.py -v --tb=short
```

## Test Implementation Details

### Helper Functions

#### `_measure_lock_parallelism`
Measures actual concurrency rather than wall-clock time.

**Returns:**
- `max_parallel`: Peak number of concurrent lock holders
- `timeline`: Ordered list of (task_name, event) tuples
- `metrics`: Dict with performance data (duration, concurrency, workers)

**Example:**
```python
workload = [
    ("task1", "workspace1", "namespace"),
    ("task2", "workspace2", "namespace"),
]
max_parallel, timeline, metrics = await _measure_lock_parallelism(workload)

# Assert on actual behavior, not timing
assert max_parallel >= 2  # Two different workspaces should run concurrently
```

#### `_assert_no_timeline_overlap`
Validates sequential execution using finite state machine.

**Validates:**
- No overlapping lock acquisitions
- Proper lock release ordering
- All locks properly released

**Example:**
```python
timeline = [
    ("task1", "start"),
    ("task1", "end"),
    ("task2", "start"),
    ("task2", "end"),
]
_assert_no_timeline_overlap(timeline)  # Passes - no overlap

timeline_bad = [
    ("task1", "start"),
    ("task2", "start"),  # ERROR: task2 started before task1 ended
    ("task1", "end"),
]
_assert_no_timeline_overlap(timeline_bad)  # Raises AssertionError
```

## Configuration Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `LIGHTRAG_STRESS_TEST` | bool | `false` | Enable stress testing mode |
| `LIGHTRAG_TEST_WORKERS` | int | `3` | Number of parallel workers in stress mode |
| `LIGHTRAG_KEEP_ARTIFACTS` | bool | `false` | Keep temporary test directories |

## Performance Benchmarks

### Expected Performance (Reference System)
- **Test 1-9**: < 1s each
- **Test 10**: < 2s (includes file I/O)
- **Test 11**: < 5s (includes full RAG pipeline)
- **Total Suite**: < 15s

### Stress Test Performance
With `LIGHTRAG_TEST_WORKERS=10`:
- **Test 2 (Parallel)**: ~0.05s (10 workers, all concurrent)
- **Test 2 (Serial)**: ~0.10s (2 workers, serialized)

## Troubleshooting

### Common Issues

#### Flaky Test Failures
**Symptom:** Tests pass locally but fail in CI/CD
**Cause:** System under heavy load, timing-based assertions
**Solution:** Our tests use concurrency-based assertions, not timing. If failures persist, check the `timeline` output in error messages.

#### Resource Cleanup Errors
**Symptom:** "Directory not empty" or "Cannot remove directory"
**Cause:** Concurrent test execution or OS file locking
**Solution:** Run tests serially (`pytest -n 1`) or use `LIGHTRAG_KEEP_ARTIFACTS=true` to inspect state

#### Lock Timeout Errors
**Symptom:** "Lock acquisition timeout"
**Cause:** Deadlock or resource starvation
**Solution:** Check test output for deadlock patterns, review lock acquisition order

### Debug Tips

1. **Enable verbose output:**
   ```bash
   pytest tests/test_workspace_isolation.py -v -s
   ```

2. **Run single test with artifacts:**
   ```bash
   LIGHTRAG_KEEP_ARTIFACTS=true pytest tests/test_workspace_isolation.py::test_json_kv_storage_workspace_isolation -v -s
   ```

3. **Check performance metrics:**
   Look for the "Performance:" lines in test output showing duration and concurrency.

4. **Inspect timeline on failure:**
   Timeline data is included in assertion error messages.

## Contributing

### Adding New Tests

1. **Follow naming convention:** `test_<feature>_<aspect>`
2. **Add purpose/scope comments:** Explain what and why
3. **Use helper functions:** `_measure_lock_parallelism`, `_assert_no_timeline_overlap`
4. **Document assertions:** Explain expected behavior in assertions
5. **Update this README:** Add test to appropriate category

### Test Template
```python
@pytest.mark.asyncio
async def test_new_feature():
    """
    Brief description of what this test validates.
    """
    # Purpose: Why this test exists
    # Scope: What functions/classes this tests
    print("\n" + "=" * 60)
    print("TEST N: Feature Name")
    print("=" * 60)
    
    # Test implementation
    # ...
    
    print("✅ PASSED: Feature Name")
    print(f"   Validation details")
```

## Related Documentation

- [Workspace Isolation Design Doc](../docs/LightRAG_concurrent_explain.md)
- [Project Intelligence](.clinerules/01-basic.md)
- [Memory Bank](../.memory-bank/)

## Test Coverage Matrix

| Component | Data Isolation | Lock Mechanism | Backward Compat | Error Handling | E2E |
|-----------|:--------------:|:--------------:|:---------------:|:--------------:|:---:|
| shared_storage | ✅ T1, T4 | ✅ T2, T5, T6 | ✅ T3 | ✅ T7 | ✅ T11 |
| update_flags | ✅ T8 | - | - | - | - |
| JsonKVStorage | ✅ T10 | - | - | - | ✅ T11 |
| LightRAG Core | - | - | - | - | ✅ T11 |
| Namespace | ✅ T9 | - | ✅ T3 | ✅ T7 | - |

**Legend:** T# = Test number

## Version History

- **v2.0** (2025-01-18): Added performance metrics, stress testing, configurable cleanup
- **v1.0** (Initial): Basic workspace isolation tests with timing-based assertions
