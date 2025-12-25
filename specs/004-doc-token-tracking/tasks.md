# Tasks: Document Processing Token Tracking

**Input**: Design documents from `/specs/004-doc-token-tracking/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, contracts/track-status-api.yaml, quickstart.md
**Branch**: `004-doc-token-tracking`

**Tests**: Included as required by Constitution Principle IV (Multi-Workspace Test Coverage)

**Organization**: Tasks grouped by user story for independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Repository root**: `lightrag/` for source, `tests/` for tests
- All modifications to existing files; no new modules required

---

## Phase 1: Setup (Verification)

**Purpose**: Verify existing infrastructure is ready

- [ ] T001 Review TokenTracker class in lightrag/utils.py (lines 2530-2628)
- [ ] T002 [P] Review DocProcessingStatus.metadata field in lightrag/base.py (line 701)
- [ ] T003 [P] Review lightrag_doc_status table metadata column in lightrag/kg/postgres_impl.py (lines 4848-4863)
- [ ] T004 [P] Review track_status endpoint in lightrag/api/routers/document_routes.py (lines 2881-2953)

---

## Phase 2: Foundational (Pipeline Infrastructure)

**Purpose**: Create token tracking infrastructure in document processing pipeline

**⚠️ CRITICAL**: These components are used by all user stories

### TokenTracker Integration

- [ ] T005 Add token_tracker to pipeline_status dict in lightrag/lightrag.py (apipeline_process_enqueue_documents)
- [ ] T006 Add processing_start_time capture in lightrag/lightrag.py when document processing begins
- [ ] T007 Add processing_end_time capture in lightrag/lightrag.py when document processing completes

### Pipeline Global Config

- [ ] T008 Pass token_tracker through global_config to extract_entities in lightrag/lightrag.py
- [ ] T009 Pass token_tracker through global_config to embedding calls in lightrag/lightrag.py

**Checkpoint**: Token tracking infrastructure ready - user story implementation can begin

---

## Phase 3: User Story 1 - Retrieve Token Usage After Processing (Priority: P1) 🎯 MVP

**Goal**: Return token_usage in track_status response for processed documents

**Independent Test**: Upload a document, wait for processing, call track_status and verify token_usage contains all fields

### Tests for User Story 1

- [ ] T010 [P] [US1] Add test_track_status_includes_token_usage() in tests/test_doc_token_tracking.py
- [ ] T011 [P] [US1] Add test_token_usage_structure() in tests/test_doc_token_tracking.py

### Implementation for User Story 1

- [ ] T012 [US1] Build token_usage dict from TokenTracker in lightrag/lightrag.py after processing completes
- [ ] T013 [US1] Store token_usage in doc_status.metadata in lightrag/lightrag.py when updating status to "processed"
- [ ] T014 [US1] Verify track_status endpoint returns metadata with token_usage in lightrag/api/routers/document_routes.py

**Checkpoint**: track_status returns token_usage for processed documents - core billing capability works ✅

---

## Phase 4: User Story 2 - Accumulate Tokens Across Pipeline (Priority: P1)

**Goal**: Capture and accumulate all tokens from embedding and LLM calls

**Independent Test**: Process a multi-chunk document and verify token counts reflect ALL operations

### Tests for User Story 2

- [ ] T015 [P] [US2] Add test_embedding_tokens_accumulated() in tests/test_doc_token_tracking.py
- [ ] T016 [P] [US2] Add test_llm_tokens_accumulated() in tests/test_doc_token_tracking.py
- [ ] T017 [P] [US2] Add test_total_chunks_counted() in tests/test_doc_token_tracking.py

### Implementation for User Story 2

- [ ] T018 [US2] Capture embedding tokens via add_embedding_usage() calls in lightrag/lightrag.py
- [ ] T019 [US2] Capture LLM tokens from extract_entities() in lightrag/operate.py
- [ ] T020 [US2] Pass token_tracker to use_llm_func_with_cache() calls in lightrag/operate.py
- [ ] T021 [US2] Add LLM token capture in extract_entities glean loop in lightrag/operate.py
- [ ] T022 [US2] Count total_chunks during chunking phase in lightrag/lightrag.py
- [ ] T023 [US2] Include model names (embedding_model, llm_model) in token_usage dict in lightrag/lightrag.py

**Checkpoint**: All token types accumulated across full pipeline ✅

---

## Phase 5: User Story 3 - Persist Token Usage (Priority: P2)

**Goal**: Ensure token_usage is durably stored and retrievable after service restart

**Independent Test**: Process document, restart service, verify token_usage still available

### Tests for User Story 3

- [ ] T024 [P] [US3] Add test_token_usage_persistence() in tests/test_doc_token_tracking.py
- [ ] T025 [P] [US3] Add test_partial_usage_on_failure() in tests/test_doc_token_tracking.py

### Implementation for User Story 3

- [ ] T026 [US3] Ensure metadata update happens atomically with status update in lightrag/lightrag.py
- [ ] T027 [US3] Handle partial token_usage storage on processing failure in lightrag/lightrag.py
- [ ] T028 [US3] Add processing timestamps (start_time, end_time) to metadata in lightrag/lightrag.py

**Checkpoint**: Token usage durably stored in database ✅

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final verification, edge cases, and cleanup

- [ ] T029 [P] Handle missing usage data from local models (zero tokens, null model) in lightrag/lightrag.py
- [ ] T030 [P] Handle empty documents (zero chunks) in lightrag/lightrag.py
- [ ] T031 [P] Add test for workspace isolation of token_usage in tests/test_doc_token_tracking.py
- [ ] T032 Run all tests to verify no regressions in tests/
- [ ] T033 Execute manual verification steps from quickstart.md (MANUAL)
- [ ] T034 Code review: verify API backward compatibility (Constitution Principle I)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies - start immediately
- **Phase 2 (Foundational)**: Depends on Phase 1 - creates pipeline infrastructure
- **Phase 3 (US1)**: Depends on Phase 2 - token_usage in track_status response
- **Phase 4 (US2)**: Depends on Phase 2 - can run parallel to US1
- **Phase 5 (US3)**: Depends on Phase 2 - can run parallel to US1/US2
- **Phase 6 (Polish)**: Depends on all user stories

### User Story Dependencies

```
Phase 2 (Foundational) ──┬──> Phase 3 (US1: track_status response) ──┐
                         │                                           │
                         ├──> Phase 4 (US2: Token accumulation) ─────┼──> Phase 6 (Polish)
                         │                                           │
                         └──> Phase 5 (US3: Persistence) ────────────┘
```

- **US1 (track_status)**: Foundation → P1 - core value
- **US2 (Accumulation)**: Foundation → P1 - parallel to US1
- **US3 (Persistence)**: Foundation → P2 - parallel to US1/US2

### Within Each User Story

- Tests SHOULD be written first (TDD when practical)
- Infrastructure before capture
- Capture before storage
- Storage before response
- Story complete before checkpoint

### Parallel Opportunities

**Phase 1 (Setup)**:
```
T002 [P] DocProcessingStatus review
T003 [P] Database schema review
T004 [P] track_status endpoint review
```

**Phase 3 + Phase 4 + Phase 5 (Can run in parallel after Phase 2)**:
```
US1: T010, T011 (tests) → T012-T014 (impl)
US2: T015-T017 (tests) → T018-T023 (impl)
US3: T024, T025 (tests) → T026-T028 (impl)
```

**Phase 6 (Polish)**:
```
T029 [P], T030 [P], T031 [P]
```

---

## Parallel Example: User Stories 1 and 2

```bash
# US1 and US2 can run in parallel after Phase 2:

# Developer A - US1 Token Response:
Task: "Add test_track_status_includes_token_usage() in tests/test_doc_token_tracking.py"
Task: "Build token_usage dict from TokenTracker in lightrag/lightrag.py"

# Developer B - US2 Token Accumulation:
Task: "Add test_embedding_tokens_accumulated() in tests/test_doc_token_tracking.py"
Task: "Capture LLM tokens from extract_entities() in lightrag/operate.py"
```

---

## Implementation Strategy

### MVP First (US1 Only)

1. Complete Phase 1: Setup (verification)
2. Complete Phase 2: Foundational (pipeline infrastructure)
3. Complete Phase 3: US1 track_status response
4. **STOP and VALIDATE**: Verify token_usage in track_status
5. This alone enables basic billing

### Full Feature (US1 + US2 + US3)

1. Complete Setup + Foundational
2. Complete US1, US2, US3 in parallel
3. Run full test suite
4. Deploy

### Incremental Delivery

1. US1 alone → track_status has token_usage (billing ready)
2. US1 + US2 → Accurate accumulated totals (granular billing)
3. US1 + US2 + US3 → Durable persistence (audit ready)

---

## Files to Modify

| File | Change | Description |
|------|--------|-------------|
| `lightrag/lightrag.py` | MODIFY | Create TokenTracker, pass through pipeline, save to metadata |
| `lightrag/operate.py` | MODIFY | Accept TokenTracker, capture LLM tokens from extract_entities |
| `lightrag/api/routers/document_routes.py` | VERIFY | track_status returns metadata (already does) |
| `tests/test_doc_token_tracking.py` | CREATE | Unit and integration tests |

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story
- US1 and US2 are both P1 but can be developed in parallel
- US3 enhances US1/US2 but is independent
- All tests should FAIL before implementation (TDD)
- Commit after each task or logical group
- Stop at any checkpoint to validate independently

---

## Task Summary

| Phase | Story | Task Count | Parallel |
|-------|-------|------------|----------|
| Setup | - | 4 | 3 |
| Foundational | - | 5 | 0 |
| US1 (track_status) | P1 | 5 | 2 |
| US2 (Accumulation) | P1 | 9 | 3 |
| US3 (Persistence) | P2 | 5 | 2 |
| Polish | - | 6 | 3 |
| **Total** | | **34** | **13** |

**MVP Scope**: Phases 1-3 (14 tasks) - track_status with token_usage
**Full Scope**: All phases (34 tasks) - complete token tracking with persistence
