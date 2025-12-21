# Research: Fix Graph Not Updating After Document Deletion

**Feature**: 002-fix-graph-deletion-sync
**Date**: 2025-12-21
**Status**: Complete

## Executive Summary

Root cause analysis of the bug where knowledge graph doesn't update after document deletion. Two primary issues identified with clear solutions.

---

## Research Area 1: Cache Invalidation Timing

### Question
Why do deleted entities get restored during the rebuild phase?

### Investigation

Analyzed `adelete_by_doc_id()` in `lightrag/lightrag.py`:

1. **Current flow** (problematic):
   ```
   Line 3131-3180: Collect LLM cache IDs
   Line 3391-3406: Delete chunks from storage
   Line 3408-3443: Delete relationships from graph
   Line 3445-3539: Delete entities from graph
   Line 3541-3542: Persist graph (_insert_done)
   Line 3544-3564: Rebuild from remaining chunks ← READS STALE CACHE
   Line 3584-3599: Delete LLM cache ← TOO LATE!
   ```

2. **Problem**: Cache deletion happens AFTER rebuild. The rebuild function `_get_cached_extraction_results()` reads cache entries for chunks that were deleted but whose cache entries still exist.

### Decision
Move LLM cache deletion to BEFORE the rebuild phase.

### Rationale
- Cache entries for deleted chunks should not exist when rebuild runs
- This matches user expectation: deleted document = deleted cache
- Simple code reordering, minimal risk

### Alternatives Considered

| Alternative | Why Rejected |
|------------|--------------|
| Filter cache entries in rebuild | More complex, requires passing deleted chunk IDs through call chain |
| Skip rebuild entirely | Would break legitimate entity rebuilding for shared entities |
| Delete cache in separate pass | Additional complexity, same timing issue |

---

## Research Area 2: Graph Reload During Deletion

### Question
Why does the graph show the same node count (1244) before and after deletion?

### Investigation

Analyzed `NetworkXStorage._get_graph()` in `lightrag/kg/networkx_impl.py`:

```python
async def _get_graph(self):
    async with self._storage_lock:
        if self.storage_updated.value:  # ← Problem trigger
            # Reload from disk, OVERWRITING in-memory changes!
            self._graph = NetworkXStorage.load_nx_graph(self._graphml_xml_file) or nx.Graph()
            self.storage_updated.value = False
        return self._graph
```

**Root cause sequence**:
1. Deletion starts, calls `remove_nodes()` → modifies in-memory graph
2. Something sets `storage_updated.value = True`
3. Another operation calls `_get_graph()`
4. Graph reloaded from disk → all in-memory deletions lost
5. `index_done_callback()` writes the reloaded (unmodified) graph

**What sets storage_updated?**
- `set_all_update_flags()` in `index_done_callback()` (line 527)
- Called by another process/workspace completing its operations

### Decision
Add deletion-in-progress flag to prevent graph reload during active deletion sequence.

### Rationale
- Deletion is already protected by pipeline lock (single process at a time)
- Adding a simple flag prevents reload without changing locking semantics
- Minimal code change, easily testable

### Alternatives Considered

| Alternative | Why Rejected |
|------------|--------------|
| Hold graph reference directly | Requires changing multiple function signatures |
| Disable storage_updated during all operations | Too broad, could cause stale reads |
| Use transaction-like pattern | Overkill for this use case |

---

## Research Area 3: NetworkX Graph Operations Behavior

### Question
Are NetworkX remove_node/remove_edge operations synchronous and in-place?

### Investigation

From NetworkX documentation and code analysis:
- `graph.remove_node(node_id)` - Synchronous, immediately modifies graph in-place
- `graph.remove_edge(u, v)` - Synchronous, immediately modifies graph in-place
- Both operations affect the same graph object referenced by `self._graph`

**Verified in code** (`networkx_impl.py` lines 170-200):
```python
async def remove_nodes(self, nodes: list[str]):
    graph = await self._get_graph()  # ← This is the problem call
    for node in nodes:
        if graph.has_node(node):
            graph.remove_node(node)  # ← Synchronous, works correctly
```

### Decision
The NetworkX operations themselves are correct. The issue is `_get_graph()` being called, which may trigger a reload.

### Rationale
- No need to change NetworkX usage
- Fix should be in the reload prevention, not the removal logic

---

## Research Area 4: Logging Best Practices

### Question
What logging should be added to verify the fix?

### Investigation

Current logging (`networkx_impl.py` line 35):
```python
f"[{workspace}] Writing graph with {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
```

Missing:
- Log BEFORE deletion starts
- Log AFTER `remove_nodes()` completes
- Log verification that counts decreased

### Decision
Add explicit before/after logging with workspace context.

### Rationale
- Enables verification of fix
- Helps debugging future issues
- Low overhead (only during deletion operations)

### Logging Format
```python
# Before deletion
logger.info(f"[{workspace}] Graph before deletion: {node_count} nodes, {edge_count} edges")

# After remove_nodes
logger.info(f"[{workspace}] Graph after remove_nodes: {node_count} nodes (removed {removed_count})")

# At persistence
logger.info(f"[{workspace}] Persisting graph: {node_count} nodes, {edge_count} edges")
```

---

## Summary of Decisions

| Area | Decision | Confidence |
|------|----------|------------|
| Cache invalidation | Move before rebuild | High |
| Graph reload | Add deletion flag | High |
| NetworkX operations | No changes needed | High |
| Logging | Add before/after counts | High |

## Implementation Order

1. **Cache invalidation fix** - Highest impact, simplest change
2. **Graph reload prevention** - Addresses race condition
3. **Logging additions** - Verification and debugging
4. **Tests** - Regression tests to prevent recurrence

## Open Questions

None - all clarifications resolved through code analysis.
