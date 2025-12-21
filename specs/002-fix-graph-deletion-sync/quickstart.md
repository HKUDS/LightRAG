# Quickstart: Verifying Graph Deletion Sync Fix

**Feature**: 002-fix-graph-deletion-sync
**Date**: 2025-12-21

## Prerequisites

- Python 3.10+
- pytest installed
- LightRAG development environment set up
- Test data (any markdown or text document)

## Quick Verification

### Step 1: Run Existing Tests

Verify no regressions in existing functionality:

```bash
cd src
pytest tests/ -v -k "delete"
```

Expected: All existing deletion tests pass.

### Step 2: Run New Regression Tests

```bash
pytest tests/test_graph_deletion_sync.py -v
```

Expected: All new tests pass, including:
- `test_graph_node_count_decreases_after_deletion`
- `test_cache_invalidated_before_rebuild`
- `test_no_graph_reload_during_deletion`

## Manual Verification Steps

### Step 1: Setup Test Environment

```bash
# Start local server
cd src
python -m lightrag.api.lightrag_server
```

### Step 2: Create Test Document

```bash
# Upload a test document
curl -X POST "http://localhost:9621/documents/upload_file" \
  -H "X-API-Key: your-api-key" \
  -H "LIGHTRAG-WORKSPACE: test-deletion" \
  -F "file=@test_document.md"
```

Note the document ID from the response.

### Step 3: Get Initial Graph Stats

```bash
# Get graph node/edge counts
curl "http://localhost:9621/graphs?label=entity" \
  -H "X-API-Key: your-api-key" \
  -H "LIGHTRAG-WORKSPACE: test-deletion" | jq '.nodes | length'
```

Record: `INITIAL_NODE_COUNT = ___`

### Step 4: Delete Document

```bash
# Delete the document
curl -X DELETE "http://localhost:9621/documents/{doc_id}" \
  -H "X-API-Key: your-api-key" \
  -H "LIGHTRAG-WORKSPACE: test-deletion"
```

### Step 5: Verify Graph Updated

```bash
# Get graph node/edge counts again
curl "http://localhost:9621/graphs?label=entity" \
  -H "X-API-Key: your-api-key" \
  -H "LIGHTRAG-WORKSPACE: test-deletion" | jq '.nodes | length'
```

Record: `FINAL_NODE_COUNT = ___`

### Step 6: Verify Fix

**PASS if**: `FINAL_NODE_COUNT < INITIAL_NODE_COUNT`

**FAIL if**: `FINAL_NODE_COUNT == INITIAL_NODE_COUNT` (original bug)

## Log Verification

Check server logs for the new logging output:

```
# Expected log sequence after fix:
INFO: [test-deletion] Graph before deletion: 50 nodes, 30 edges
INFO: [test-deletion] Graph after remove_nodes: 38 nodes (removed 12)
INFO: [test-deletion] Writing graph with 38 nodes, 22 edges
```

**PASS if**: Node counts decrease between "before" and "Writing graph" logs.

**FAIL if**: Node counts remain the same or increase.

## Reproducing the Original Bug

To verify the bug existed before the fix (on main branch):

```bash
# Checkout main branch
git checkout main

# Start server and run steps 1-5 above
# Expected: FINAL_NODE_COUNT == INITIAL_NODE_COUNT (bug present)

# Checkout fix branch
git checkout 002-fix-graph-deletion-sync

# Run steps 1-5 again
# Expected: FINAL_NODE_COUNT < INITIAL_NODE_COUNT (bug fixed)
```

## Test Scenarios

### Scenario 1: Single Document Deletion

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Upload document with unique entities | Document processed, entities in graph |
| 2 | Record graph node count | N nodes |
| 3 | Delete document | Success response |
| 4 | Record graph node count | < N nodes |

### Scenario 2: Shared Entity Preservation

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Upload doc A with entity "Paris" | "Paris" in graph |
| 2 | Upload doc B with entity "Paris" | "Paris" still in graph (shared) |
| 3 | Delete doc A | Success |
| 4 | Check "Paris" | Still in graph (from doc B) |
| 5 | Delete doc B | Success |
| 6 | Check "Paris" | Removed from graph |

### Scenario 3: Cache Invalidation Verification

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Upload document | Document processed |
| 2 | Note chunk count | N chunks |
| 3 | Delete with delete_llm_cache=true | Success |
| 4 | Check server logs | "Deleted N LLM cache entries" BEFORE rebuild log |

### Scenario 4: Multi-Workspace Isolation

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Upload doc to workspace A | Processed |
| 2 | Upload doc to workspace B | Processed |
| 3 | Delete doc from workspace A | Success |
| 4 | Check workspace A graph | Reduced node count |
| 5 | Check workspace B graph | Unchanged node count |

## Troubleshooting

### Graph Count Unchanged After Deletion

1. Check server logs for "reloading graph" during deletion
   - If present: Reload prevention not working
2. Check logs for "Rebuilding knowledge from X cached chunk extractions"
   - If X > 0 for deleted chunks: Cache not invalidated before rebuild

### Tests Failing

1. Ensure test database is clean: `pytest --setup-show`
2. Check for concurrent test interference
3. Verify NetworkX storage is being used (not PostgreSQL graph)

### Performance Issues

1. Large documents may take longer to delete
2. Check `MAX_PARALLEL_INSERT` setting
3. Monitor memory usage during deletion

## Success Criteria Checklist

- [ ] All existing deletion tests pass
- [ ] New regression tests pass
- [ ] Manual verification shows node count decreases
- [ ] Logs show correct before/after sequence
- [ ] Multi-workspace isolation verified
- [ ] No performance regression (deletion < 30s)
