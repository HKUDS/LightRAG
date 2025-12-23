# Tasks: API Usage Metering and Cost Tracking

**Input**: Design documents from `/specs/003-api-usage-metering/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, contracts/usage-api.yaml, quickstart.md
**Branch**: `003-api-usage-metering`

**Tests**: Included as required by Constitution Principle IV (Multi-Workspace Test Coverage)

**Organization**: Tasks grouped by user story for independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3, US4)
- Include exact file paths in descriptions

## Path Conventions

- **Repository root**: `lightrag/` for source, `tests/` for tests
- Changes to 4 existing files + 4 new files

---

## Phase 1: Setup (Verification)

**Purpose**: Verify current state and understand existing infrastructure

- [x] T001 Review TokenTracker class in lightrag/utils.py (lines 2528-2582)
- [x] T002 [P] Review LLM token capture in lightrag/llm/openai.py (lines 424-434, 576-584)
- [x] T003 [P] Review embedding token capture in lightrag/llm/openai.py (lines 770-775)
- [x] T004 [P] Review QueryResponse model in lightrag/api/routers/query_routes.py (lines 158-165)

---

## Phase 2: Foundational (Response Models & Token Tracking)

**Purpose**: Create shared models and extend token tracking infrastructure

**⚠️ CRITICAL**: These components are used by all user stories

### Pydantic Models

- [x] T005 Create UsageInfo, LLMUsageInfo, EmbeddingUsageInfo models in lightrag/api/models/usage.py
- [x] T006 [P] Create UsageAggregateResponse model in lightrag/api/models/usage.py

### TokenTracker Extension

- [x] T007 Extend TokenTracker to track model names in lightrag/utils.py
- [x] T008 Add separate embedding tracking methods to TokenTracker in lightrag/utils.py
- [x] T009 Add get_llm_usage() and get_embedding_usage() methods to TokenTracker in lightrag/utils.py

### Configuration

- [x] T010 Add pricing environment variables to env.example (LIGHTRAG_LLM_PRICE_PER_1K_INPUT, LIGHTRAG_LLM_PRICE_PER_1K_OUTPUT, LIGHTRAG_EMBEDDING_PRICE_PER_1K)

**Checkpoint**: Models and extended TokenTracker ready - user story implementation can begin

---

## Phase 3: User Story 1 - Per-Request Usage Visibility (Priority: P1) 🎯 MVP

**Goal**: Include usage data in query endpoint responses

**Independent Test**: Make a query request and verify response contains "usage" field with LLM token counts

### Tests for User Story 1

- [x] T011 [P] [US1] Add test_query_response_includes_usage() in tests/test_usage_metering.py
- [x] T012 [P] [US1] Add test_usage_backward_compatible() in tests/test_usage_metering.py

### Implementation for User Story 1

- [x] T013 [US1] Add optional usage field to QueryResponse model in lightrag/api/routers/query_routes.py
- [x] T014 [US1] Capture model name from LLM calls in lightrag/llm/openai.py (streaming path)
- [x] T015 [US1] Capture model name from LLM calls in lightrag/llm/openai.py (non-streaming path)
- [x] T016 [US1] Build UsageInfo from TokenTracker in query endpoint in lightrag/api/routers/query_routes.py
- [x] T017 [US1] Return usage field in query response in lightrag/api/routers/query_routes.py

**Checkpoint**: Query endpoint returns usage data - core billing capability works ✅

---

## Phase 4: User Story 2 - Embedding Usage Tracking (Priority: P1)

**Goal**: Track embedding operations separately from LLM operations

**Independent Test**: Upload a document and verify response contains separate embedding usage metrics

### Tests for User Story 2

- [x] T018 [P] [US2] Add test_document_upload_includes_embedding_usage() in tests/test_usage_metering.py
- [x] T019 [P] [US2] Add test_embedding_tracked_separately_from_llm() in tests/test_usage_metering.py

### Implementation for User Story 2

- [x] T020 [US2] Capture embedding model name in lightrag/llm/openai.py (embedding endpoint)
- [x] T021 [US2] Track embedding tokens separately in TokenTracker in lightrag/utils.py
- [ ] T022 [US2] Add usage field to document upload response model in lightrag/api/routers/document_routes.py (DEFERRED)
- [ ] T023 [US2] Build UsageInfo with embedding data in document routes in lightrag/api/routers/document_routes.py (DEFERRED)
- [ ] T024 [US2] Return usage in POST /documents/text response in lightrag/api/routers/document_routes.py (DEFERRED)

**Checkpoint**: Embedding tracking infrastructure ready - query endpoints include embedding usage ✅

---

## Phase 5: User Story 3 - Estimated Cost Calculation (Priority: P2)

**Goal**: Calculate and include estimated cost in responses when pricing is configured

**Independent Test**: Configure pricing env vars, make a request, verify estimated_cost_usd in response

### Tests for User Story 3

- [x] T025 [P] [US3] Add test_cost_estimation_with_pricing_config() in tests/test_usage_metering.py
- [x] T026 [P] [US3] Add test_cost_estimation_omitted_without_config() in tests/test_usage_metering.py

### Implementation for User Story 3

- [x] T027 [US3] Create calculate_estimated_cost() function in lightrag/api/models/usage.py
- [x] T028 [US3] Load pricing from env vars in calculate_estimated_cost() in lightrag/api/models/usage.py
- [x] T029 [US3] Integrate cost calculation into UsageInfo builder in lightrag/api/routers/query_routes.py
- [ ] T030 [US3] Integrate cost calculation into document routes in lightrag/api/routers/document_routes.py (DEFERRED)

**Checkpoint**: Cost estimation available when pricing configured - query endpoints complete ✅

---

## Phase 6: User Story 4 - Per-Workspace Usage Aggregation (Priority: P3) 🔄 DEFERRED

**Goal**: Persist usage and provide aggregation endpoint

**Status**: DEFERRED - Requires significant database persistence work. Core usage tracking (US1-US3) complete and enables per-request billing.

**Independent Test**: Make requests, query GET /usage, verify totals match

### Tests for User Story 4

- [ ] T031 [P] [US4] Add test_usage_persistence() in tests/test_usage_metering.py (DEFERRED)
- [ ] T032 [P] [US4] Add test_usage_aggregation_endpoint() in tests/test_usage_metering.py (DEFERRED)
- [x] T033 [P] [US4] Add test_workspace_usage_isolation() in tests/test_usage_metering.py

### Implementation for User Story 4

- [ ] T034 [US4] Create UsageStorage class with PostgreSQL implementation in lightrag/kg/usage_storage.py (DEFERRED)
- [ ] T035 [US4] Implement save_usage_record() method in lightrag/kg/usage_storage.py (DEFERRED)
- [ ] T036 [US4] Implement get_usage_aggregate() method with date range in lightrag/kg/usage_storage.py (DEFERRED)
- [ ] T037 [US4] Create usage_routes.py with GET /usage endpoint in lightrag/api/routers/usage_routes.py (DEFERRED)
- [ ] T038 [US4] Register usage router in main API app in lightrag/api/lightrag_server.py (DEFERRED)
- [ ] T039 [US4] Persist usage record after each request in query routes in lightrag/api/routers/query_routes.py (DEFERRED)
- [ ] T040 [US4] Persist usage record after each request in document routes in lightrag/api/routers/document_routes.py (DEFERRED)

**Checkpoint**: DEFERRED - Core per-request usage tracking complete, aggregation to be added as follow-up

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Final verification, edge cases, and cleanup

- [ ] T041 [P] Handle partial usage when requests fail (FR-009) in lightrag/api/routers/query_routes.py (DEFERRED)
- [x] T042 [P] Handle missing usage data from local models gracefully in lightrag/api/models/usage.py
- [ ] T043 [P] Add logging for usage tracking operations in lightrag/kg/usage_storage.py (DEFERRED - with US4)
- [x] T044 Run all tests to verify no regressions in tests/
- [ ] T045 Execute manual verification steps from quickstart.md (MANUAL)
- [x] T046 Code review: verify API backward compatibility (Constitution Principle I)
- [x] T047 [P] Multi-workspace usage isolation test in tests/test_usage_metering.py

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies - start immediately
- **Phase 2 (Foundational)**: Depends on Phase 1 - creates shared infrastructure
- **Phase 3 (US1)**: Depends on Phase 2 - LLM usage in query responses
- **Phase 4 (US2)**: Depends on Phase 2 - can run parallel to US1
- **Phase 5 (US3)**: Depends on Phase 2 - can run parallel to US1/US2
- **Phase 6 (US4)**: Depends on Phase 3 AND Phase 4 (needs usage data to persist)
- **Phase 7 (Polish)**: Depends on all user stories

### User Story Dependencies

```
Phase 2 (Foundational) ──┬──> Phase 3 (US1: Query Usage) ──────┬──> Phase 6 (US4: Aggregation)
                         │                                     │
                         ├──> Phase 4 (US2: Embedding Usage) ──┤
                         │                                     │
                         └──> Phase 5 (US3: Cost Estimation) ──┘
```

- **US1 (Query Usage)**: Foundation → P1 - core value
- **US2 (Embedding Usage)**: Foundation → P1 - parallel to US1
- **US3 (Cost Estimation)**: Foundation → P2 - parallel to US1/US2
- **US4 (Aggregation)**: Requires US1 + US2 data flowing → P3

### Within Each User Story

- Tests SHOULD be written first (TDD when practical)
- Models before services
- Services before endpoints
- Story complete before checkpoint

### Parallel Opportunities

**Phase 1 (Setup)**:
```
T002 [P] LLM token capture review
T003 [P] Embedding token capture review
T004 [P] QueryResponse model review
```

**Phase 2 (Foundational)**:
```
T005 + T006 [P] (model creation)
```

**Phase 3 + Phase 4 + Phase 5 (Can run in parallel after Phase 2)**:
```
US1: T011, T012 (tests) → T013-T017 (impl)
US2: T018, T019 (tests) → T020-T024 (impl)
US3: T025, T026 (tests) → T027-T030 (impl)
```

**Phase 6 (User Story 4)**:
```
T031 [P], T032 [P], T033 [P] (all tests)
```

**Phase 7 (Polish)**:
```
T041 [P], T042 [P], T043 [P], T047 [P]
```

---

## Parallel Example: User Stories 1 and 2

```bash
# US1 and US2 can run in parallel after Phase 2:

# Developer A - US1 Query Usage:
Task: "Add test_query_response_includes_usage() in tests/test_usage_metering.py"
Task: "Add optional usage field to QueryResponse in lightrag/api/routers/query_routes.py"

# Developer B - US2 Embedding Usage:
Task: "Add test_document_upload_includes_embedding_usage() in tests/test_usage_metering.py"
Task: "Track embedding tokens separately in TokenTracker in lightrag/utils.py"
```

---

## Implementation Strategy

### MVP First (US1 Only)

1. Complete Phase 1: Setup (verification)
2. Complete Phase 2: Foundational (models, TokenTracker extension)
3. Complete Phase 3: US1 Query Usage
4. **STOP and VALIDATE**: Query endpoint returns usage data
5. This alone enables basic billing

### Full Feature (US1 + US2 + US3 + US4)

1. Complete Setup + Foundational
2. Complete US1, US2, US3 in parallel
3. Complete US4 (persistence + aggregation)
4. Run full test suite
5. Deploy

### Incremental Delivery

1. US1 alone → Query usage visible (billing ready)
2. US1 + US2 → Embedding usage separate (granular billing)
3. US1 + US2 + US3 → Cost estimates included (convenience)
4. All stories → Full persistence and aggregation (reporting)

---

## Files to Create/Modify

| File | Status | Description |
|------|--------|-------------|
| `lightrag/api/models/usage.py` | NEW | Pydantic models for usage data |
| `lightrag/api/config/pricing.py` | NEW | Pricing configuration loader |
| `lightrag/kg/usage_storage.py` | NEW | Usage persistence (PostgreSQL) |
| `lightrag/api/routers/usage_routes.py` | NEW | GET /usage endpoint |
| `tests/test_usage_metering.py` | NEW | Unit and integration tests |
| `lightrag/utils.py` | MODIFY | Extend TokenTracker |
| `lightrag/llm/openai.py` | MODIFY | Capture model names |
| `lightrag/api/routers/query_routes.py` | MODIFY | Add usage to response |
| `lightrag/api/routers/document_routes.py` | MODIFY | Add usage to response |
| `lightrag/api/lightrag_server.py` | MODIFY | Register usage router |
| `env.example` | MODIFY | Add pricing env vars |

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story
- US1 and US2 are both P1 but can be developed in parallel
- US3 enhances US1/US2 but is independent
- US4 requires US1/US2 data to exist
- All tests should FAIL before implementation (TDD)
- Commit after each task or logical group
- Stop at any checkpoint to validate independently

---

## Task Summary

| Phase | Story | Task Count | Parallel |
|-------|-------|------------|----------|
| Setup | - | 4 | 3 |
| Foundational | - | 6 | 2 |
| US1 (Query Usage) | P1 | 7 | 2 |
| US2 (Embedding Usage) | P1 | 7 | 2 |
| US3 (Cost Estimation) | P2 | 6 | 2 |
| US4 (Aggregation) | P3 | 10 | 3 |
| Polish | - | 7 | 4 |
| **Total** | | **47** | **18** |

**MVP Scope**: Phases 1-3 (17 tasks) - query usage visibility
**Full Scope**: All phases (47 tasks) - complete usage tracking with persistence
