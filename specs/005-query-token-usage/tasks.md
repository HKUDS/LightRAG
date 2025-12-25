# Tasks: Query Token Usage for Billing

**Input**: Design documents from `/specs/005-query-token-usage/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: Included as this is a billing-critical feature requiring validation.

**Organization**: Tasks grouped by user story for independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)

## Path Conventions

- **Source**: `lightrag/api/` (existing API structure)
- **Tests**: `tests/` (existing test structure)

---

## Phase 1: Setup

**Purpose**: Review existing infrastructure and prepare for changes

- [x] T001 Review existing TokenTracker in lightrag/utils.py
- [x] T002 Review existing UsageInfo model in lightrag/api/models/usage.py
- [x] T003 Review existing QueryResponse model in lightrag/api/routers/query_routes.py

---

## Phase 2: Foundational (Model Layer)

**Purpose**: Create the QueryTokenUsage model that all user stories depend on

**⚠️ CRITICAL**: User story work cannot begin until model is complete

- [x] T004 Add QueryTokenUsage model in lightrag/api/models/usage.py
- [x] T005 Add from_token_tracker classmethod to QueryTokenUsage in lightrag/api/models/usage.py
- [x] T006 Export QueryTokenUsage from lightrag/api/models/__init__.py

**Checkpoint**: Foundation ready - user story implementation can begin ✅

---

## Phase 3: User Story 1 - Standard Query Token Tracking (Priority: P1) 🎯 MVP

**Goal**: Return token_usage in POST /query response with all 5 fields populated

**Independent Test**: Make a standard query and verify response includes token_usage with llm_model, llm_input_tokens, llm_output_tokens, embedding_model, embedding_tokens

### Tests for User Story 1

- [ ] T007 [P] [US1] Add test for token_usage in standard query response in tests/test_query_token_usage.py
- [ ] T008 [P] [US1] Add test for model names matching API model IDs in tests/test_query_token_usage.py

### Implementation for User Story 1

- [x] T009 [US1] Import QueryTokenUsage in lightrag/api/routers/query_routes.py
- [x] T010 [US1] Add token_usage field to QueryResponse model in lightrag/api/routers/query_routes.py
- [x] T011 [US1] Build QueryTokenUsage from token_tracker in query_text function in lightrag/api/routers/query_routes.py
- [x] T012 [US1] Include token_usage in QueryResponse return in lightrag/api/routers/query_routes.py

**Checkpoint**: Standard queries return token_usage - MVP complete ✅

---

## Phase 4: User Story 2 - Context-Only Query Token Tracking (Priority: P1)

**Goal**: Return token_usage with LLM tokens = 0 when only_need_context=true

**Independent Test**: Make a query with only_need_context=true and verify llm_input_tokens=0, llm_output_tokens=0, llm_model=null

### Tests for User Story 2

- [ ] T013 [P] [US2] Add test for zero LLM tokens when only_need_context=true in tests/test_query_token_usage.py
- [ ] T014 [P] [US2] Add test for embedding tokens populated when only_need_context=true in tests/test_query_token_usage.py

### Implementation for User Story 2

- [x] T015 [US2] Verify TokenTracker returns zero LLM tokens when no LLM call made in lightrag/api/routers/query_routes.py
- [x] T016 [US2] Verify context-only path still populates embedding fields in lightrag/api/routers/query_routes.py

**Checkpoint**: Context-only queries correctly report zero LLM tokens ✅

---

## Phase 5: User Story 3 - Streaming Query Token Tracking (Priority: P2)

**Goal**: Include token_usage in final SSE event for streaming queries

**Independent Test**: Make a streaming query and verify final chunk includes token_usage with all fields

### Tests for User Story 3

- [ ] T017 [P] [US3] Add test for token_usage in streaming final chunk in tests/test_query_token_usage.py

### Implementation for User Story 3

- [x] T018 [US3] Add token_usage field to StreamChunkResponse model in lightrag/api/routers/query_routes.py
- [x] T019 [US3] Build QueryTokenUsage in streaming generator in lightrag/api/routers/query_routes.py
- [x] T020 [US3] Include token_usage in final streaming chunk in lightrag/api/routers/query_routes.py

**Checkpoint**: Streaming queries include token_usage in final event ✅

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Validation and cleanup

- [ ] T021 [P] Verify backward compatibility - usage field still present in tests/test_query_token_usage.py
- [x] T022 [P] Run existing test suite to verify no regressions (101 passed, 36 skipped)
- [ ] T023 Run quickstart.md validation scenarios manually
- [ ] T024 Update API documentation if needed

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - start immediately
- **Foundational (Phase 2)**: Depends on Setup - BLOCKS all user stories
- **User Stories (Phase 3-5)**: All depend on Foundational completion
- **Polish (Phase 6)**: Depends on all user stories

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational - No dependencies on other stories
- **User Story 2 (P1)**: Can start after Foundational - Shares model with US1 but independently testable
- **User Story 3 (P2)**: Can start after Foundational - Independent streaming path

### Within Each User Story

- Tests written first (verify they reference correct assertions)
- Implementation follows test specifications
- Story complete when tests pass

### Parallel Opportunities

- Setup tasks T001-T003 can run in parallel (reading only)
- Tests for each user story can run in parallel
- User stories 1, 2, 3 can run in parallel after Foundational phase
- Polish tasks T021-T022 can run in parallel

---

## Parallel Example: Foundational + User Story Tests

```bash
# After Setup, launch all foundational tasks:
Task: "Add QueryTokenUsage model in lightrag/api/models/usage.py"

# After Foundational, launch all US1 tests in parallel:
Task: "Add test for token_usage in standard query response"
Task: "Add test for model names matching API model IDs"

# Launch all user story implementations in parallel (if team capacity):
Task: "[US1] Build QueryTokenUsage from token_tracker"
Task: "[US2] Verify context-only path"
Task: "[US3] Add token_usage to StreamChunkResponse"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (T001-T003)
2. Complete Phase 2: Foundational (T004-T006)
3. Complete Phase 3: User Story 1 (T007-T012)
4. **STOP and VALIDATE**: Test standard query returns token_usage
5. Deploy if ready - Cleo can start billing standard queries

### Incremental Delivery

1. Setup + Foundational → QueryTokenUsage model ready
2. Add User Story 1 → Standard queries billable (MVP!)
3. Add User Story 2 → Context-only queries billable
4. Add User Story 3 → Streaming queries billable
5. Each story adds billing capability without breaking others

### Suggested MVP Scope

**MVP = Phases 1-3 (14 tasks)**
- Setup: 3 tasks
- Foundational: 3 tasks
- User Story 1: 6 tasks + 2 tests

This delivers:
- token_usage field in /query response
- All 5 required fields (llm_model, llm_input_tokens, llm_output_tokens, embedding_model, embedding_tokens)
- Standard query billing capability for Cleo

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story
- Existing TokenTracker infrastructure handles all token counting
- This feature adds new response field format, not new tracking logic
- Backward compatible: existing `usage` field preserved
