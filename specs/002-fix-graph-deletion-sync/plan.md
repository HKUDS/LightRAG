# Implementation Plan: Fix Graph Not Updating After Document Deletion

**Branch**: `002-fix-graph-deletion-sync` | **Date**: 2025-12-21 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/002-fix-graph-deletion-sync/spec.md`

## Summary

Fix a bug where the knowledge graph does not update after document deletion. The graph retains the same entities/relations (1244 nodes) even after successfully deleting 112 entities. Root causes: (1) LLM cache not invalidated before rebuild, causing stale data to restore deleted entities, (2) graph may reload from disk during deletion sequence, losing in-memory changes.

**Technical Approach**:
1. Invalidate LLM cache entries for deleted document chunks BEFORE rebuild
2. Add deletion lock to prevent graph reload during deletion operation
3. Ensure atomic graph operations complete before persistence
4. Add logging for before/after node counts to verify fix

## Technical Context

**Language/Version**: Python 3.10+
**Primary Dependencies**: FastAPI, Pydantic, asyncio, NetworkX
**Storage**: PostgreSQL (Supabase) for KV/Vector/DocStatus, NetworkX files for graph
**Testing**: pytest with async support
**Target Platform**: Linux server (Docker/Render)
**Project Type**: Single project (library + API server)
**Performance Goals**: Deletion must complete within 30 seconds, graph endpoint response < 5 seconds
**Constraints**: Must maintain backward compatibility with existing API, multi-workspace isolation
**Scale/Scope**: Multi-tenant deployment with 50+ concurrent workspaces

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Evidence |
|-----------|--------|----------|
| **I. API Backward Compatibility** | PASS | No changes to public API signatures. Internal bug fix only. Existing `delete_llm_cache` parameter behavior preserved. |
| **II. Workspace/Tenant Isolation** | PASS | Fix is workspace-scoped. Graph operations already isolated by workspace path. No cross-workspace impact. |
| **III. Explicit Server Configuration** | PASS | No new configuration required. Fix uses existing `delete_llm_cache` parameter. |
| **IV. Multi-Workspace Test Coverage** | REQUIRES | Must add tests verifying graph count decreases after deletion in multi-workspace context. |

**Security Requirements**:
- PASS: No changes to workspace identifier handling
- PASS: No new cross-workspace operations introduced

**Performance Standards**:
- PASS: Fix should improve performance (fewer unnecessary rebuilds)
- PASS: No additional latency to workspace resolution

**Gate Result**: PASS - Proceed to Phase 0

## Project Structure

### Documentation (this feature)

```text
specs/002-fix-graph-deletion-sync/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output (N/A - internal fix)
└── tasks.md             # Phase 2 output (/speckit.tasks command)
```

### Source Code (repository root)

```text
lightrag/
├── lightrag.py              # Main class - adelete_by_doc_id() method
├── operate.py               # rebuild_knowledge_from_chunks() function
└── kg/
    └── networkx_impl.py     # NetworkXStorage class - remove_nodes, _get_graph, index_done_callback

tests/
└── test_graph_deletion_sync.py  # New regression tests
```

**Structure Decision**: Existing LightRAG structure. Changes limited to 3 files in `lightrag/` module plus new test file.

## Complexity Tracking

No complexity violations. This is a targeted bug fix within existing architecture.

## Implementation Strategy

### Phase 1: Cache Invalidation Fix (P1)

**Problem**: `rebuild_knowledge_from_chunks()` reads stale LLM cache entries for chunks that were just deleted.

**Solution**: Move cache invalidation BEFORE rebuild in `adelete_by_doc_id()`:
1. Collect LLM cache IDs for deleted chunks (already done, line 3131-3180)
2. Delete cache entries BEFORE calling `rebuild_knowledge_from_chunks()` (currently done AFTER)
3. Ensure `_get_cached_extraction_results()` returns empty for deleted chunk IDs

### Phase 2: Graph Reload Prevention (P1)

**Problem**: `_get_graph()` may reload graph from disk if `storage_updated.value` is True, overwriting in-memory deletions.

**Solution**: Add deletion context to prevent reload:
1. Introduce deletion lock/flag in NetworkXStorage
2. Modify `_get_graph()` to skip reload when deletion is in progress
3. Clear flag after `index_done_callback()` completes

### Phase 3: Atomic Operation Assurance (P2)

**Problem**: Multiple calls to `_get_graph()` during deletion may cause inconsistent state.

**Solution**: Ensure single graph reference throughout deletion:
1. Get graph reference once at start of deletion
2. Pass reference through all operations
3. Only call `_get_graph()` at final persistence

### Phase 4: Verification Logging (P2)

**Problem**: Cannot verify fix without clear before/after logging.

**Solution**: Add explicit node count logging:
1. Log node count before any deletion operations
2. Log node count after `remove_nodes()` completes
3. Log node count at persistence time
4. Verify logs show decreasing counts

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Concurrent deletion race conditions | Medium | High | Use existing pipeline lock mechanism |
| Breaking other graph operations | Low | High | Comprehensive test coverage |
| Performance regression | Low | Medium | Benchmark before/after |
| Multi-process graph conflicts | Medium | Medium | Existing `storage_updated` flag mechanism |

## Test Strategy

1. **Unit tests**: Mock graph storage, verify node counts decrease
2. **Integration tests**: Full deletion flow with actual NetworkX storage
3. **Regression tests**: Reproduce original bug scenario (1244 nodes → verify < 1244 after deletion)
4. **Multi-workspace tests**: Verify isolation during concurrent deletions
