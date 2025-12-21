# Tasks: Fix Graph Not Updating After Document Deletion

**Input**: Design documents from `/specs/002-fix-graph-deletion-sync/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, quickstart.md
**Branch**: `002-fix-graph-deletion-sync`

**Tests**: Included as required by Constitution Principle IV (Multi-Workspace Test Coverage)

**Organization**: Tasks grouped by user story for independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Repository root**: `lightrag/` for source, `tests/` for tests
- Changes limited to 3 existing files + 1 new test file

---

## Phase 1: Setup (Verification)

**Purpose**: Verify current state and understand the bug

- [x] T001 Run existing deletion tests to establish baseline in tests/
- [x] T002 [P] Review adelete_by_doc_id() flow in lightrag/lightrag.py (lines 2943-3637)
- [x] T003 [P] Review NetworkXStorage._get_graph() in lightrag/kg/networkx_impl.py (lines 80-96)

---

## Phase 2: Foundational (Test Infrastructure)

**Purpose**: Create test infrastructure for validating the fix

**⚠️ CRITICAL**: Tests must exist before implementation to verify fix works

- [x] T004 Create test file tests/test_graph_deletion_sync.py with pytest async fixtures
- [x] T005 [P] Add helper function to count graph nodes before/after operations in tests/test_graph_deletion_sync.py
- [x] T006 [P] Add mock NetworkXStorage for unit testing in tests/test_graph_deletion_sync.py

**Checkpoint**: Test infrastructure ready - implementation can begin

---

## Phase 3: User Story 2 - Cache Invalidation Before Rebuild (Priority: P1) 🎯 MVP

**Goal**: Move LLM cache deletion BEFORE rebuild to prevent stale data from restoring deleted entities

**Independent Test**: Delete document with cached extraction, verify rebuild doesn't restore deleted entities

**Why US2 First**: This is the root cause fix. Must be done before US1 can be properly tested.

### Tests for User Story 2

- [x] T007 [P] [US2] Add test_cache_deleted_before_rebuild() in tests/test_graph_deletion_sync.py
- [x] T008 [P] [US2] Add test_rebuild_skips_deleted_chunk_cache() in tests/test_graph_deletion_sync.py

### Implementation for User Story 2

- [x] T009 [US2] Move cache deletion block (lines 3584-3599) to before rebuild (line 3544) in lightrag/lightrag.py
- [x] T010 [US2] Add log message "Invalidating LLM cache before rebuild" in lightrag/lightrag.py
- [x] T011 [US2] Verify _get_cached_extraction_results() handles missing cache entries gracefully in lightrag/operate.py

**Checkpoint**: Cache invalidation now occurs before rebuild - root cause addressed

---

## Phase 4: User Story 3 - Atomic Graph Operations (Priority: P2)

**Goal**: Prevent graph reload during deletion sequence to ensure in-memory changes persist

**Independent Test**: Simulate storage_updated flag during deletion, verify graph not reloaded

**Why US3 Before US1**: Graph reload prevention must work before observable fix can be verified.

### Tests for User Story 3

- [x] T012 [P] [US3] Add test_no_graph_reload_during_deletion() in tests/test_graph_deletion_sync.py
- [x] T013 [P] [US3] Add test_deletion_flag_cleared_after_persist() in tests/test_graph_deletion_sync.py

### Implementation for User Story 3

- [x] T014 [US3] Add _deletion_in_progress flag to NetworkXStorage.__post_init__() in lightrag/kg/networkx_impl.py
- [x] T015 [US3] Modify _get_graph() to skip reload when _deletion_in_progress is True in lightrag/kg/networkx_impl.py
- [x] T016 [US3] Add set_deletion_mode(enabled: bool) method to NetworkXStorage in lightrag/kg/networkx_impl.py
- [x] T017 [US3] Call set_deletion_mode(True) at start of entity/relation deletion in lightrag/lightrag.py
- [x] T018 [US3] Call set_deletion_mode(False) after index_done_callback() in lightrag/lightrag.py

**Checkpoint**: Graph operations are now atomic - no reload during deletion

---

## Phase 5: User Story 1 - Document Deletion Reflects in Graph (Priority: P1)

**Goal**: Verify and ensure graph node count decreases after document deletion

**Independent Test**: Upload document, count nodes, delete document, verify nodes decreased

**Why US1 Last**: This is the observable outcome that depends on US2 and US3 fixes being in place.

### Tests for User Story 1

- [x] T019 [P] [US1] Add test_graph_node_count_decreases_after_deletion() in tests/test_graph_deletion_sync.py
- [x] T020 [P] [US1] Add test_shared_entities_preserved_after_deletion() in tests/test_graph_deletion_sync.py
- [x] T021 [P] [US1] Add test_graphs_endpoint_reflects_deletion() in tests/test_graph_deletion_sync.py

### Implementation for User Story 1

- [x] T022 [US1] Add before-deletion node count logging in lightrag/lightrag.py (before line 3408)
- [x] T023 [US1] Add after-remove_nodes() count logging in lightrag/lightrag.py (after line 3516)
- [x] T024 [US1] Verify remove_nodes() operates on same graph reference throughout in lightrag/kg/networkx_impl.py
- [x] T025 [US1] Add assertion in index_done_callback() that node count matches in-memory in lightrag/kg/networkx_impl.py

**Checkpoint**: All user stories complete - graph now properly reflects deletions

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final verification, documentation, and cleanup

- [x] T026 [P] Run all existing deletion tests to verify no regressions in tests/
- [x] T027 [P] Run new test_graph_deletion_sync.py tests in tests/
- [x] T028 [P] Execute manual verification steps from quickstart.md
- [x] T029 Update env.example if any new configuration added (N/A - no new config)
- [x] T030 Add multi-workspace deletion test in tests/test_graph_deletion_sync.py
- [x] T031 Code review: verify no API breaking changes per Constitution Principle I

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies - start immediately
- **Phase 2 (Foundational)**: Depends on Phase 1 - creates test infrastructure
- **Phase 3 (US2)**: Depends on Phase 2 - implements root cause fix
- **Phase 4 (US3)**: Depends on Phase 2 - can run parallel to US2
- **Phase 5 (US1)**: Depends on Phase 3 AND Phase 4 - requires both fixes
- **Phase 6 (Polish)**: Depends on all user stories

### User Story Dependencies

```
Phase 2 (Tests) ──┬──> Phase 3 (US2: Cache) ──┬──> Phase 5 (US1: Observable Fix)
                  │                            │
                  └──> Phase 4 (US3: Atomic) ──┘
```

- **US2 (Cache Invalidation)**: Can start after Phase 2 - root cause fix
- **US3 (Atomic Operations)**: Can start after Phase 2 - parallel to US2
- **US1 (Observable Fix)**: Requires US2 AND US3 complete - final verification

### Within Each User Story

- Tests MUST be written and FAIL before implementation
- Implementation tasks in dependency order
- Story complete before checkpoint

### Parallel Opportunities

**Phase 2 (Test Infrastructure)**:
```
T005 [P] Helper function
T006 [P] Mock storage
```

**Phase 3 + Phase 4 (Can run in parallel)**:
```
US2: T007, T008 (tests) → T009, T010, T011 (impl)
US3: T012, T013 (tests) → T014, T015, T016, T017, T018 (impl)
```

**Phase 5 (All tests can run parallel)**:
```
T019 [P], T020 [P], T021 [P] (all tests)
```

**Phase 6 (Verification in parallel)**:
```
T026 [P], T027 [P], T028 [P] (all verification)
```

---

## Parallel Example: User Stories 2 and 3

```bash
# US2 and US3 can run in parallel after Phase 2:

# Developer A - US2 Cache Invalidation:
Task: "Add test_cache_deleted_before_rebuild() in tests/test_graph_deletion_sync.py"
Task: "Move cache deletion block to before rebuild in lightrag/lightrag.py"

# Developer B - US3 Atomic Operations:
Task: "Add test_no_graph_reload_during_deletion() in tests/test_graph_deletion_sync.py"
Task: "Add _deletion_in_progress flag to NetworkXStorage in lightrag/kg/networkx_impl.py"
```

---

## Implementation Strategy

### MVP First (US2 Only)

1. Complete Phase 1: Setup (verification)
2. Complete Phase 2: Test infrastructure
3. Complete Phase 3: US2 Cache Invalidation
4. **STOP and VALIDATE**: Test that rebuild no longer restores deleted entities
5. This alone may fix the bug in many cases

### Full Fix (US2 + US3 + US1)

1. Complete Setup + Test infrastructure
2. Complete US2 (Cache) AND US3 (Atomic) in parallel
3. Complete US1 (Observable verification)
4. Run full test suite
5. Deploy

### Incremental Delivery

1. US2 alone → Partial fix (cache invalidation)
2. US2 + US3 → Complete fix (cache + atomic)
3. US2 + US3 + US1 → Verified fix with logging

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story
- US2 and US3 are both P1 but US2 is the root cause
- US1 is the observable outcome, depends on US2 + US3
- All tests should FAIL before implementation (TDD)
- Commit after each task or logical group
- Stop at any checkpoint to validate independently

---

## Task Summary

| Phase | Story | Task Count | Parallel |
|-------|-------|------------|----------|
| Setup | - | 3 | 2 |
| Foundational | - | 3 | 2 |
| US2 (Cache) | P1 | 5 | 2 |
| US3 (Atomic) | P2 | 7 | 2 |
| US1 (Observable) | P1 | 7 | 3 |
| Polish | - | 6 | 3 |
| **Total** | | **31** | **14** |

**MVP Scope**: Phases 1-3 (11 tasks) - fixes root cause
**Full Scope**: All phases (31 tasks) - complete fix with verification
