# Tasks: Entity Resolution (Linking & Conflict Detection)

**Input**: Design documents from `/specs/007-entity-resolution/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, quickstart.md

**Tests**: Included - unit tests for both modules as per plan.md verification requirements.

**Organization**: Tasks grouped by user story for independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)
- Exact file paths included in descriptions

## Path Conventions

- **Project structure**: `lightrag/` for source, `tests/` for tests (at repository root)
- Based on existing LightRAG-MT project structure

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Add rapidfuzz dependency and prepare configuration structure

- [ ] T001 Add rapidfuzz>=3.0.0 dependency to pyproject.toml in dependencies list
- [ ] T002 [P] Add entity resolution configuration constants to lightrag/constants.py
- [ ] T003 [P] Add conflict detection configuration constants to lightrag/constants.py
- [ ] T004 [P] Add new environment variables to env.example (ENABLE_ENTITY_RESOLUTION, ENTITY_SIMILARITY_THRESHOLD, ENABLE_CONFLICT_DETECTION, CONFLICT_CONFIDENCE_THRESHOLD)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core classes that all user stories depend on

**⚠️ CRITICAL**: No user story work can begin until EntityResolver and ConflictDetector classes exist

- [ ] T005 Create lightrag/entity_resolution.py with EntityResolver class skeleton (init, resolve, consolidate_entities methods)
- [ ] T006 [P] Create lightrag/conflict_detection.py with ConflictInfo dataclass and ConflictDetector class skeleton
- [ ] T007 Add configuration loading for entity resolution settings in lightrag/lightrag.py (read from env/global_config)
- [ ] T008 [P] Add configuration loading for conflict detection settings in lightrag/lightrag.py

**Checkpoint**: Foundation ready - user story implementation can now begin

---

## Phase 3: User Story 1 - Entity Deduplication During Ingestion (Priority: P1) 🎯 MVP

**Goal**: Automatically merge entities with similar names during document ingestion

**Independent Test**: Ingest document with "Apple Inc", "Apple Inc.", "Apple" variations → verify single entity with all descriptions merged

### Tests for User Story 1

- [ ] T009 [P] [US1] Create tests/test_entity_resolution.py with test class structure
- [ ] T010 [P] [US1] Add test_fuzzy_matching_accuracy() - verify token set ratio works correctly
- [ ] T011 [P] [US1] Add test_threshold_behavior() - verify 0.85 default threshold
- [ ] T012 [P] [US1] Add test_case_insensitive_matching() - "APPLE" matches "Apple"
- [ ] T013 [P] [US1] Add test_short_name_exclusion() - names ≤2 chars never matched (FR-018)
- [ ] T014 [P] [US1] Add test_canonical_name_selection() - longest name wins (FR-003)
- [ ] T015 [P] [US1] Add test_entity_type_constraint() - same-type entities only (FR-006)
- [ ] T016 [P] [US1] Add test_consolidate_entities_batch() - batch processing works

### Implementation for User Story 1

- [ ] T017 [US1] Implement EntityResolver.__init__() with threshold and min_length params in lightrag/entity_resolution.py
- [ ] T018 [US1] Implement EntityResolver._normalize_name() for case-insensitive comparison in lightrag/entity_resolution.py
- [ ] T019 [US1] Implement EntityResolver._compute_similarity() using rapidfuzz.fuzz.token_set_ratio in lightrag/entity_resolution.py
- [ ] T020 [US1] Implement EntityResolver._select_canonical_name() - longest name, first if tie in lightrag/entity_resolution.py
- [ ] T021 [US1] Implement EntityResolver.resolve() method for single entity lookup in lightrag/entity_resolution.py
- [ ] T022 [US1] Implement EntityResolver.consolidate_entities() for batch entity merging in lightrag/entity_resolution.py
- [ ] T023 [US1] Add entity resolution logging (INFO level) for merge operations in lightrag/entity_resolution.py
- [ ] T024 [US1] Inject EntityResolver call in lightrag/operate.py after entity collection (~line 2460)
- [ ] T025 [US1] Add enable_entity_resolution config check before calling resolver in lightrag/operate.py
- [ ] T026 [US1] Run tests/test_entity_resolution.py and verify all tests pass

**Checkpoint**: Entity deduplication fully functional - can ingest documents with name variations

---

## Phase 4: User Story 2 - Conflict Detection in Entity Descriptions (Priority: P2)

**Goal**: Detect and log contradictory information in entity descriptions during summarization

**Independent Test**: Ingest "Tesla founded in 2003" then "Tesla founded in 2004" → verify conflict logged and uncertainty in summary

### Tests for User Story 2

- [ ] T027 [P] [US2] Create tests/test_conflict_detection.py with test class structure
- [ ] T028 [P] [US2] Add test_detect_temporal_conflict() - different years detected
- [ ] T029 [P] [US2] Add test_detect_attribution_conflict() - different founders detected
- [ ] T030 [P] [US2] Add test_detect_numerical_conflict() - different values detected
- [ ] T031 [P] [US2] Add test_no_conflict_on_extension() - "Musk" vs "Musk and others" is not conflict
- [ ] T032 [P] [US2] Add test_confidence_scoring() - conflicts have valid confidence scores
- [ ] T033 [P] [US2] Add test_n_way_conflict() - 3+ sources with different values (FR-019)

### Implementation for User Story 2

- [ ] T034 [US2] Implement ConflictInfo dataclass with all fields in lightrag/conflict_detection.py
- [ ] T035 [US2] Implement ConflictInfo.to_log_message() method in lightrag/conflict_detection.py
- [ ] T036 [US2] Implement ConflictInfo.to_prompt_context() method in lightrag/conflict_detection.py
- [ ] T037 [US2] Implement ConflictDetector.__init__() with confidence threshold in lightrag/conflict_detection.py
- [ ] T038 [US2] Add DATE_PATTERNS regex list to ConflictDetector in lightrag/conflict_detection.py
- [ ] T039 [US2] Add ATTRIBUTION_PATTERNS regex list to ConflictDetector in lightrag/conflict_detection.py
- [ ] T040 [US2] Add NUMBER_PATTERNS regex list to ConflictDetector in lightrag/conflict_detection.py
- [ ] T041 [US2] Implement ConflictDetector._extract_values() for pattern matching in lightrag/conflict_detection.py
- [ ] T042 [US2] Implement ConflictDetector._compare_values() for conflict detection logic in lightrag/conflict_detection.py
- [ ] T043 [US2] Implement ConflictDetector.detect_conflicts() main method in lightrag/conflict_detection.py
- [ ] T044 [US2] Add conflict logging (WARNING level) in lightrag/conflict_detection.py
- [ ] T045 [US2] Add PROMPTS["summarize_with_conflicts"] template to lightrag/prompt.py
- [ ] T046 [US2] Inject ConflictDetector call in lightrag/operate.py before _handle_entity_relation_summary (~line 1720)
- [ ] T047 [US2] Modify _handle_entity_relation_summary to use conflict-aware prompt when conflicts detected in lightrag/operate.py
- [ ] T048 [US2] Add enable_conflict_detection config check before calling detector in lightrag/operate.py
- [ ] T049 [US2] Run tests/test_conflict_detection.py and verify all tests pass

**Checkpoint**: Conflict detection fully functional - contradictions logged and surfaced in summaries

---

## Phase 5: User Story 3 - Configurable Resolution Behavior (Priority: P3)

**Goal**: Allow operators to tune entity matching and conflict detection thresholds

**Independent Test**: Set ENTITY_SIMILARITY_THRESHOLD=0.95 → verify stricter matching; set =0.75 → verify more merging

### Implementation for User Story 3

- [ ] T050 [US3] Verify ENABLE_ENTITY_RESOLUTION env var toggles feature on/off in lightrag/operate.py
- [ ] T051 [US3] Verify ENTITY_SIMILARITY_THRESHOLD env var changes matching sensitivity
- [ ] T052 [US3] Verify ENTITY_MIN_NAME_LENGTH env var excludes short names
- [ ] T053 [US3] Verify ENABLE_CONFLICT_DETECTION env var toggles feature on/off
- [ ] T054 [US3] Verify CONFLICT_CONFIDENCE_THRESHOLD env var changes logging threshold
- [ ] T055 [US3] Add test_configuration_changes() to tests/test_entity_resolution.py for threshold variations
- [ ] T056 [US3] Add test_disable_entity_resolution() to tests/test_entity_resolution.py
- [ ] T057 [US3] Add test_disable_conflict_detection() to tests/test_conflict_detection.py

**Checkpoint**: All configuration options working - operators can tune behavior

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Integration tests, documentation, cleanup

- [ ] T058 [P] Create tests/test_entity_integration.py for end-to-end ingestion test
- [ ] T059 [P] Add integration test: ingest doc with name variations, verify single entity in graph
- [ ] T060 [P] Add integration test: ingest conflicting docs, verify warning logged and uncertainty in description
- [ ] T061 [P] Update CLAUDE.md with new feature documentation if needed
- [ ] T062 Run full test suite: pytest tests/ -v and verify all tests pass
- [ ] T063 Verify quickstart.md scenarios work as documented
- [ ] T064 Code cleanup: remove any debug statements, ensure consistent logging
- [ ] T065 Final review: verify FR-001 through FR-019 requirements are met

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup - BLOCKS all user stories
- **User Story 1 (Phase 3)**: Depends on Foundational - can start after Phase 2
- **User Story 2 (Phase 4)**: Depends on Foundational - can start after Phase 2 (parallel with US1)
- **User Story 3 (Phase 5)**: Depends on US1 and US2 completion (tests config of both features)
- **Polish (Phase 6)**: Depends on all user stories complete

### User Story Dependencies

- **User Story 1 (P1)**: Independent - only needs EntityResolver foundation
- **User Story 2 (P2)**: Independent - only needs ConflictDetector foundation (can run parallel with US1)
- **User Story 3 (P3)**: Depends on US1 + US2 (tests configuration of both modules)

### Within Each User Story

- Tests written FIRST, verify they FAIL
- Core class methods before integration
- Integration with operate.py after core logic complete
- All tests passing before moving to next story

### Parallel Opportunities

**Setup Phase (all can run in parallel)**:
- T002, T003, T004 (different files)

**Foundational Phase**:
- T005 + T006 can run in parallel (different files)
- T007 + T008 can run in parallel (different config sections)

**User Story 1 Tests (all parallel)**:
- T009-T016 can all run in parallel (same test file, independent tests)

**User Story 2 Tests (all parallel)**:
- T027-T033 can all run in parallel

**After Foundational, US1 and US2 can run in parallel by different developers**

---

## Parallel Example: User Story 1

```bash
# Launch all tests for US1 together:
Task: T009 "Create tests/test_entity_resolution.py"
Task: T010 "Add test_fuzzy_matching_accuracy()"
Task: T011 "Add test_threshold_behavior()"
Task: T012 "Add test_case_insensitive_matching()"
Task: T013 "Add test_short_name_exclusion()"
Task: T014 "Add test_canonical_name_selection()"
Task: T015 "Add test_entity_type_constraint()"
Task: T016 "Add test_consolidate_entities_batch()"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001-T004)
2. Complete Phase 2: Foundational - EntityResolver skeleton (T005, T007)
3. Complete Phase 3: User Story 1 (T009-T026)
4. **STOP and VALIDATE**: Test entity deduplication independently
5. Deploy if entity linking is the priority need

### Incremental Delivery

1. Setup + Foundational → Foundation ready
2. User Story 1 → Entity deduplication works → Deploy (MVP!)
3. User Story 2 → Conflict detection works → Deploy
4. User Story 3 → Configuration tunable → Deploy
5. Polish → Production-ready

### Parallel Team Strategy

With 2 developers after Foundational phase:
- Developer A: User Story 1 (entity resolution)
- Developer B: User Story 2 (conflict detection)
- Both complete → Developer A does User Story 3 (tests both)

---

## Task Summary

| Phase | Tasks | Parallelizable |
|-------|-------|----------------|
| Setup | 4 | 3 |
| Foundational | 4 | 2 |
| User Story 1 | 18 | 8 (tests) |
| User Story 2 | 23 | 7 (tests) |
| User Story 3 | 8 | 0 |
| Polish | 8 | 4 |
| **Total** | **65** | **24** |

---

## Notes

- [P] tasks = different files, no dependencies on incomplete tasks
- [Story] label maps task to specific user story for traceability
- Each user story is independently completable and testable
- Tests MUST fail before implementation begins
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
