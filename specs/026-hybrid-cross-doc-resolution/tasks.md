# Tasks: Hybrid Cross-Document Entity Resolution

**Input**: Design documents from `/specs/026-hybrid-cross-doc-resolution/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: Included (unit tests for new functionality)

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3, US4)
- Include exact file paths in descriptions

## Path Conventions

- **Source**: `lightrag/` at repository root
- **Tests**: `tests/` at repository root

---

## Phase 1: Setup

**Purpose**: Add constants and configuration infrastructure

- [ ] T001 [P] Add CROSS_DOC_RESOLUTION_MODE constant in lightrag/constants.py
- [ ] T002 [P] Add CROSS_DOC_THRESHOLD_ENTITIES constant in lightrag/constants.py
- [ ] T003 [P] Add CROSS_DOC_VDB_TOP_K constant in lightrag/constants.py

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**CRITICAL**: No user story work can begin until this phase is complete

- [ ] T004 Add `get_node_count()` abstract method to BaseGraphStorage in lightrag/base.py
- [ ] T005 [P] Implement `get_node_count()` in PostgresStorage in lightrag/kg/postgres_impl.py
- [ ] T006 [P] Implement `get_node_count()` in MongoStorage in lightrag/kg/mongo_impl.py
- [ ] T007 [P] Implement `get_node_count()` in NetworkXStorage in lightrag/kg/networkx_impl.py
- [ ] T008 [P] Implement `get_node_count()` in Neo4jStorage in lightrag/kg/neo4j_impl.py
- [ ] T009 [P] Implement `get_node_count()` in MemgraphStorage in lightrag/kg/memgraph_impl.py (if exists)
- [ ] T010 Add configuration parameters to LightRAG class in lightrag/lightrag.py (cross_doc_resolution_mode, cross_doc_threshold_entities, cross_doc_vdb_top_k)

**Checkpoint**: Foundation ready - user story implementation can now begin

---

## Phase 3: User Story 1 - Fast Document Indexing at Scale (Priority: P1)

**Goal**: Implement VDB-assisted entity resolution for O(n × log m) complexity with large graphs

**Independent Test**: Index documents into a graph with >5000 entities and verify resolution completes in <5 seconds per document

### Tests for User Story 1

- [ ] T011 [P] [US1] Create test file tests/test_cross_doc_resolution.py with test class structure
- [ ] T012 [P] [US1] Add test_vdb_resolution_finds_similar_entities() in tests/test_cross_doc_resolution.py
- [ ] T013 [P] [US1] Add test_vdb_resolution_respects_entity_type_filter() in tests/test_cross_doc_resolution.py
- [ ] T014 [P] [US1] Add test_vdb_resolution_handles_no_candidates() in tests/test_cross_doc_resolution.py

### Implementation for User Story 1

- [ ] T015 [US1] Implement `_resolve_cross_document_entities_vdb()` function in lightrag/operate.py
- [ ] T016 [US1] Add VDB query with entity_type filter support in _resolve_cross_document_entities_vdb()
- [ ] T017 [US1] Add fuzzy matching against VDB candidates using compute_entity_similarity() in lightrag/operate.py
- [ ] T018 [US1] Integrate VDB resolution into main cross-doc resolution flow in lightrag/operate.py

**Checkpoint**: VDB-assisted resolution working for large graphs

---

## Phase 4: User Story 2 - Maximum Quality for Small Graphs (Priority: P2)

**Goal**: Implement hybrid mode that uses full matching for small graphs and VDB for large graphs

**Independent Test**: Index documents into graph with <5000 entities, verify full fuzzy matching is used

### Tests for User Story 2

- [ ] T019 [P] [US2] Add test_hybrid_uses_full_mode_below_threshold() in tests/test_cross_doc_resolution.py
- [ ] T020 [P] [US2] Add test_hybrid_uses_vdb_mode_above_threshold() in tests/test_cross_doc_resolution.py
- [ ] T021 [P] [US2] Add test_hybrid_mode_boundary_at_threshold() in tests/test_cross_doc_resolution.py

### Implementation for User Story 2

- [ ] T022 [US2] Implement `_resolve_cross_document_entities_hybrid()` wrapper function in lightrag/operate.py
- [ ] T023 [US2] Add get_node_count() call for mode determination in hybrid function
- [ ] T024 [US2] Implement mode switching logic (count < threshold → full, count >= threshold → vdb)
- [ ] T025 [US2] Update main resolution entry point to use hybrid mode by default in lightrag/operate.py

**Checkpoint**: Hybrid mode automatically selects best algorithm based on graph size

---

## Phase 5: User Story 3 - Configurable Resolution Behavior (Priority: P3)

**Goal**: Allow configuration of resolution mode and thresholds via environment variables

**Independent Test**: Set CROSS_DOC_RESOLUTION_MODE=full, verify system uses full matching regardless of graph size

### Tests for User Story 3

- [ ] T026 [P] [US3] Add test_config_mode_full_always_uses_full() in tests/test_cross_doc_resolution.py
- [ ] T027 [P] [US3] Add test_config_mode_vdb_always_uses_vdb() in tests/test_cross_doc_resolution.py
- [ ] T028 [P] [US3] Add test_config_mode_disabled_skips_resolution() in tests/test_cross_doc_resolution.py
- [ ] T029 [P] [US3] Add test_config_custom_threshold_respected() in tests/test_cross_doc_resolution.py

### Implementation for User Story 3

- [ ] T030 [US3] Add mode selection logic based on CROSS_DOC_RESOLUTION_MODE in lightrag/operate.py
- [ ] T031 [US3] Implement "disabled" mode that skips cross-doc resolution entirely
- [ ] T032 [US3] Add threshold parameter reading from global_config in hybrid function
- [ ] T033 [US3] Add top_k parameter reading from global_config in VDB function

**Checkpoint**: All configuration options working via environment variables

---

## Phase 6: User Story 4 - Observability and Metrics (Priority: P4)

**Goal**: Add logging for resolution mode, performance, and deduplication metrics

**Independent Test**: Index a document, verify logs contain mode, entities, duplicates, time_ms

### Tests for User Story 4

- [ ] T034 [P] [US4] Add test_resolution_logs_metrics() in tests/test_cross_doc_resolution.py
- [ ] T035 [P] [US4] Add test_resolution_logs_mode_switch() in tests/test_cross_doc_resolution.py

### Implementation for User Story 4

- [ ] T036 [US4] Add PERF logging at end of resolution functions in lightrag/operate.py
- [ ] T037 [US4] Track and log: mode_used, entities_checked, duplicates_found, time_ms
- [ ] T038 [US4] Add INFO log when hybrid mode switches from full to vdb
- [ ] T039 [US4] Add WARNING log on VDB query failures with fallback

**Checkpoint**: All resolution operations logged with performance metrics

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Final integration, edge cases, and documentation

- [ ] T040 [P] Add workspace isolation test for get_node_count() in tests/test_cross_doc_resolution.py
- [ ] T041 [P] Add edge case tests (embedding failure, VDB unavailable) in tests/test_cross_doc_resolution.py
- [ ] T042 Verify no regression in existing entity resolution tests in tests/test_entity_resolution.py
- [ ] T043 Run full test suite and verify all tests pass
- [ ] T044 Update env.example with new CROSS_DOC_* variables
- [ ] T045 Validate quickstart.md scenarios work as documented

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-6)**: All depend on Foundational phase completion
  - Can proceed sequentially in priority order (P1 → P2 → P3 → P4)
  - US2 builds on US1 (hybrid wraps VDB)
  - US3 adds configuration to US1+US2
  - US4 adds logging to all modes
- **Polish (Phase 7)**: Depends on all user stories being complete

### User Story Dependencies

- **US1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **US2 (P2)**: Depends on US1 (uses _resolve_cross_document_entities_vdb)
- **US3 (P3)**: Depends on US2 (configures hybrid mode behavior)
- **US4 (P4)**: Can run in parallel with US3 (adds logging to existing functions)

### Within Each User Story

- Tests MUST be written and FAIL before implementation
- Implementation tasks are sequential within each story
- Complete story before moving to next priority

### Parallel Opportunities

**Phase 1 (Setup)**:
```
T001, T002, T003 - All constants can be added in parallel
```

**Phase 2 (Foundational)**:
```
T005, T006, T007, T008, T009 - All backend implementations can run in parallel
```

**Each User Story Tests**:
```
# US1 tests can run in parallel:
T011, T012, T013, T014

# US2 tests can run in parallel:
T019, T020, T021

# US3 tests can run in parallel:
T026, T027, T028, T029

# US4 tests can run in parallel:
T034, T035
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001-T003)
2. Complete Phase 2: Foundational (T004-T010)
3. Complete Phase 3: User Story 1 (T011-T018)
4. **STOP and VALIDATE**: Test VDB resolution with large graph
5. Deploy if performance improvement confirmed

### Incremental Delivery

1. Setup + Foundational → Foundation ready
2. Add US1 → Test VDB resolution → Deploy (MVP - fast large graphs!)
3. Add US2 → Test hybrid mode → Deploy (full quality for small graphs)
4. Add US3 → Test configuration → Deploy (flexible deployment options)
5. Add US4 → Test logging → Deploy (production observability)

### Recommended Execution

For single developer:
1. T001-T003 (Setup) - 30 min
2. T004-T010 (Foundational) - 2 hours
3. T011-T018 (US1) - 3 hours
4. T019-T025 (US2) - 2 hours
5. T026-T033 (US3) - 2 hours
6. T034-T039 (US4) - 1 hour
7. T040-T045 (Polish) - 1 hour

---

## Summary

| Phase | Tasks | Parallel | Description |
|-------|-------|----------|-------------|
| Setup | 3 | 3 | Constants |
| Foundational | 7 | 5 | get_node_count() + config |
| US1 (P1) | 8 | 4 | VDB-assisted resolution |
| US2 (P2) | 7 | 3 | Hybrid mode |
| US3 (P3) | 8 | 4 | Configuration |
| US4 (P4) | 6 | 2 | Observability |
| Polish | 6 | 2 | Integration & docs |
| **Total** | **45** | **23** | |

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- US1 is the MVP - delivers core performance improvement
- US2-US4 add refinements but US1 alone is valuable
- Verify tests fail before implementing
- Commit after each task or logical group
