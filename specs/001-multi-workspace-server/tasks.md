# Tasks: Multi-Workspace Server Support

**Input**: Design documents from `/specs/001-multi-workspace-server/`
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, contracts/ ✅

**Tests**: Required per SC-007 ("All multi-workspace functionality is covered by automated tests demonstrating isolation")

**Organization**: Tasks are grouped by user story to enable independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3, US4, US5)
- Include exact file paths in descriptions

## Path Conventions

Based on plan.md structure:
- **Source**: `lightrag/api/` for API server code
- **Tests**: `tests/` at repository root
- **Config**: `lightrag/api/config.py`

---

## Phase 1: Setup

**Purpose**: Create new module and configuration infrastructure

- [x] T001 Create workspace_manager.py module skeleton in lightrag/api/workspace_manager.py
- [x] T002 [P] Add multi-workspace configuration options to lightrag/api/config.py
- [x] T003 [P] Create test file skeleton in tests/test_multi_workspace_server.py

---

## Phase 2: Foundational (Core Infrastructure)

**Purpose**: WorkspacePool and workspace resolution - MUST complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T004 Implement WorkspaceConfig dataclass in lightrag/api/workspace_manager.py
- [x] T005 Implement workspace identifier validation (regex, length) in lightrag/api/workspace_manager.py
- [x] T006 Implement WorkspacePool class with asyncio.Lock in lightrag/api/workspace_manager.py
- [x] T007 Implement get_lightrag_for_workspace() async helper in lightrag/api/workspace_manager.py
- [x] T008 Implement LRU tracking in WorkspacePool in lightrag/api/workspace_manager.py
- [x] T009 Implement workspace eviction logic in WorkspacePool in lightrag/api/workspace_manager.py
- [x] T010 Implement get_workspace_from_request() header extraction in lightrag/api/workspace_manager.py
- [x] T011 Implement get_rag FastAPI dependency in lightrag/api/workspace_manager.py
- [x] T012 Add workspace logging (non-sensitive) in lightrag/api/workspace_manager.py
- [x] T013 [P] Add unit tests for workspace validation in tests/test_multi_workspace_server.py
- [x] T014 [P] Add unit tests for WorkspacePool in tests/test_multi_workspace_server.py

**Checkpoint**: Foundation ready - WorkspacePool and dependency available for route integration

---

## Phase 3: User Story 1+2 - Tenant Isolation & Header Routing (Priority: P1) 🎯 MVP

**Goal**: Enable workspace isolation via HTTP headers - the core multi-tenant capability

**Independent Test**: Ingest document in Tenant A, query from Tenant B, verify isolation

> Note: US1 (isolation) and US2 (routing) are combined because routing is required to test isolation

### Tests for User Story 1+2

- [ ] T015 [P] [US1] Add isolation test: ingest in workspace A, query from workspace B returns nothing in tests/test_multi_workspace_server.py
- [ ] T016 [P] [US1] Add isolation test: query from workspace A returns own documents in tests/test_multi_workspace_server.py
- [ ] T017 [P] [US2] Add routing test: LIGHTRAG-WORKSPACE header routes correctly in tests/test_multi_workspace_server.py
- [ ] T018 [P] [US2] Add routing test: X-Workspace-ID fallback works in tests/test_multi_workspace_server.py
- [ ] T019 [P] [US2] Add routing test: LIGHTRAG-WORKSPACE takes precedence over X-Workspace-ID in tests/test_multi_workspace_server.py

### Implementation for User Story 1+2

- [x] T020 [US1] Refactor create_document_routes() to accept workspace dependency in lightrag/api/routers/document_routes.py
- [x] T021 [US1] Update document upload endpoints to use workspace-resolved RAG in lightrag/api/routers/document_routes.py
- [x] T022 [US1] Update document scan endpoints to use workspace-resolved RAG in lightrag/api/routers/document_routes.py
- [x] T023 [US2] Refactor create_query_routes() to accept workspace dependency in lightrag/api/routers/query_routes.py
- [x] T024 [US2] Update query endpoints to use workspace-resolved RAG in lightrag/api/routers/query_routes.py
- [x] T025 [US2] Update streaming query endpoint to use workspace-resolved RAG in lightrag/api/routers/query_routes.py
- [x] T026 [P] [US1] Refactor create_graph_routes() to use workspace dependency in lightrag/api/routers/graph_routes.py
- [x] T027 [P] [US2] Refactor OllamaAPI class to use workspace dependency in lightrag/api/routers/ollama_api.py
- [x] T028 [US1] Integrate workspace pool initialization in create_app() in lightrag/api/lightrag_server.py
- [x] T029 [US2] Wire workspace dependency into router registration in lightrag/api/lightrag_server.py
- [x] T030 [US1] Add workspace identifier to request logging in lightrag/api/lightrag_server.py

**Checkpoint**: Multi-workspace routing and isolation functional - MVP complete ✅

---

## Phase 4: User Story 3 - Backward Compatible Single-Workspace Mode (Priority: P2)

**Goal**: Existing deployments continue working without any configuration changes

**Independent Test**: Deploy new version with existing config, verify all functionality unchanged

### Tests for User Story 3

- [ ] T031 [P] [US3] Add backward compat test: no header uses WORKSPACE env var in tests/test_multi_workspace_server.py
- [ ] T032 [P] [US3] Add backward compat test: existing routes unchanged in tests/test_multi_workspace_server.py
- [ ] T033 [P] [US3] Add backward compat test: response formats unchanged in tests/test_multi_workspace_server.py

### Implementation for User Story 3

- [x] T034 [US3] Implement WORKSPACE env var fallback for default workspace in lightrag/api/config.py
- [x] T035 [US3] Implement LIGHTRAG_DEFAULT_WORKSPACE with WORKSPACE fallback in lightrag/api/workspace_manager.py
- [x] T036 [US3] Ensure default workspace is used when no header present in lightrag/api/workspace_manager.py
- [x] T037 [US3] Verify auth dependency runs before workspace resolution in lightrag/api/workspace_manager.py

**Checkpoint**: Existing single-workspace deployments work unchanged

---

## Phase 5: User Story 4 - Configurable Missing Header Behavior (Priority: P2)

**Goal**: Allow strict multi-tenant mode that rejects requests without workspace headers

**Independent Test**: Set LIGHTRAG_ALLOW_DEFAULT_WORKSPACE=false, send request without header, verify 400 error

### Tests for User Story 4

- [ ] T038 [P] [US4] Add strict mode test: missing header returns 400 when default disabled in tests/test_multi_workspace_server.py
- [ ] T039 [P] [US4] Add strict mode test: error message clearly indicates missing header in tests/test_multi_workspace_server.py
- [ ] T040 [P] [US4] Add permissive mode test: missing header uses default when enabled in tests/test_multi_workspace_server.py

### Implementation for User Story 4

- [x] T041 [US4] Add LIGHTRAG_ALLOW_DEFAULT_WORKSPACE config option in lightrag/api/config.py
- [x] T042 [US4] Implement missing header rejection when default disabled in lightrag/api/workspace_manager.py
- [x] T043 [US4] Return clear 400 error with message for missing workspace header in lightrag/api/workspace_manager.py
- [x] T044 [US4] Add invalid workspace identifier 400 error handling in lightrag/api/workspace_manager.py

**Checkpoint**: Strict multi-tenant mode available for security-conscious deployments

---

## Phase 6: User Story 5 - Workspace Instance Management (Priority: P3)

**Goal**: Efficient memory management with configurable pool size and LRU eviction

**Independent Test**: Configure max pool size, create more workspaces than limit, verify LRU eviction

### Tests for User Story 5

- [x] T045 [P] [US5] Add pool test: new workspace initializes on first request in tests/test_multi_workspace_server.py
- [x] T046 [P] [US5] Add pool test: LRU eviction when pool full in tests/test_multi_workspace_server.py
- [x] T047 [P] [US5] Add pool test: concurrent requests for same new workspace share initialization in tests/test_multi_workspace_server.py
- [x] T048 [P] [US5] Add pool test: LIGHTRAG_MAX_WORKSPACES_IN_POOL config respected in tests/test_multi_workspace_server.py

### Implementation for User Story 5

- [x] T049 [US5] Add LIGHTRAG_MAX_WORKSPACES_IN_POOL config option in lightrag/api/config.py
- [x] T050 [US5] Implement finalize_storages() call on eviction in lightrag/api/workspace_manager.py
- [x] T051 [US5] Add pool finalize_all() for graceful shutdown in lightrag/api/workspace_manager.py
- [x] T052 [US5] Wire pool finalize_all() into lifespan shutdown in lightrag/api/lightrag_server.py
- [x] T053 [US5] Add workspace initialization timing logs in lightrag/api/workspace_manager.py

**Checkpoint**: Memory management and pool eviction functional for large-scale deployments

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Documentation, cleanup, and validation

- [x] T054 [P] Update lightrag/api/README.md with multi-workspace section
- [x] T055 [P] Add env.example entries for new configuration options
- [x] T056 [P] Add type hints and docstrings to workspace_manager.py in lightrag/api/workspace_manager.py
- [x] T057 Run all tests and verify isolation in tests/
- [ ] T058 Run quickstart.md validation scenarios manually
- [x] T059 Update conftest.py with multi-workspace test fixtures in tests/conftest.py

---

## Dependencies & Execution Order

### Phase Dependencies

```
Phase 1: Setup
    │
    ▼
Phase 2: Foundational ◄─── BLOCKS ALL USER STORIES
    │
    ├─────────────────────────────────────────┐
    ▼                                         ▼
Phase 3: US1+2 (P1)                    Can start in parallel
    │                                  after Phase 2
    ▼
Phase 4: US3 (P2) ◄─── Depends on US1+2 for route integration
    │
    ▼
Phase 5: US4 (P2) ◄─── Depends on US3 for default workspace logic
    │
    ▼
Phase 6: US5 (P3) ◄─── Can start after Phase 2, but sequential for pool logic
    │
    ▼
Phase 7: Polish ◄─── After all user stories
```

### User Story Dependencies

| Story | Can Start After | Dependencies |
|-------|-----------------|--------------|
| US1+2 | Phase 2 (Foundational) | WorkspacePool, get_rag dependency |
| US3 | US1+2 | Routes must use workspace dependency |
| US4 | US3 | Default workspace logic must exist |
| US5 | Phase 2 | Pool must exist, can parallel with US1-4 |

### Within Each User Story

1. Tests written FIRST (marked [P] for parallel)
2. Verify tests FAIL before implementation
3. Implementation tasks in dependency order
4. Story complete when checkpoint passes

### Parallel Opportunities

**Phase 1 (Setup)**:
- T002 and T003 can run in parallel

**Phase 2 (Foundational)**:
- T013 and T014 (tests) can run in parallel after T004-T012

**Phase 3 (US1+2)**:
- T015-T019 (all tests) can run in parallel
- T026 and T027 (graph and ollama routes) can run in parallel

**Phase 4-6**:
- All test tasks within each phase can run in parallel

**Phase 7 (Polish)**:
- T054, T055, T056 can run in parallel

---

## Parallel Example: User Story 1+2 Tests

```bash
# Launch all US1+2 tests together:
Task: T015 - isolation test: ingest in workspace A, query from workspace B
Task: T016 - isolation test: query from workspace A returns own documents
Task: T017 - routing test: LIGHTRAG-WORKSPACE header routes correctly
Task: T018 - routing test: X-Workspace-ID fallback works
Task: T019 - routing test: LIGHTRAG-WORKSPACE takes precedence
```

---

## Implementation Strategy

### MVP First (User Stories 1+2 Only)

1. Complete Phase 1: Setup (T001-T003)
2. Complete Phase 2: Foundational (T004-T014)
3. Complete Phase 3: US1+2 (T015-T030)
4. **STOP and VALIDATE**: Test multi-workspace isolation independently
5. Deploy/demo if ready - this is the core value

### Incremental Delivery

1. Setup + Foundational → Infrastructure ready
2. Add US1+2 → Test isolation → Deploy (MVP!)
3. Add US3 → Test backward compat → Deploy
4. Add US4 → Test strict mode → Deploy
5. Add US5 → Test pool management → Deploy
6. Polish → Full release

### Recommended Order (Single Developer)

```
T001 → T002 → T003 (Setup)
T004 → T005 → T006 → T007 → T008 → T009 → T010 → T011 → T012 (Foundational)
T013 + T014 (parallel tests)
T015-T019 (parallel US1+2 tests - write first, expect failures)
T020 → T021 → T022 → T023 → T024 → T025 (routes)
T026 + T027 (parallel graph/ollama)
T028 → T029 → T030 (server integration)
[US1+2 MVP checkpoint - validate isolation]
T031-T033 (US3 tests) → T034-T037 (US3 impl)
T038-T040 (US4 tests) → T041-T044 (US4 impl)
T045-T048 (US5 tests) → T049-T053 (US5 impl)
T054-T059 (Polish)
```

---

## Task Summary

| Phase | Tasks | Parallel Tasks |
|-------|-------|----------------|
| Phase 1: Setup | 3 | 2 |
| Phase 2: Foundational | 11 | 2 |
| Phase 3: US1+2 (P1) | 16 | 7 |
| Phase 4: US3 (P2) | 7 | 3 |
| Phase 5: US4 (P2) | 7 | 3 |
| Phase 6: US5 (P3) | 9 | 4 |
| Phase 7: Polish | 6 | 3 |
| **Total** | **59** | **24** |

---

## Notes

- [P] tasks = different files, no dependencies on incomplete tasks
- [USx] label maps task to specific user story
- Each user story independently completable and testable
- Verify tests fail before implementing
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Constitution compliance verified at each phase boundary
