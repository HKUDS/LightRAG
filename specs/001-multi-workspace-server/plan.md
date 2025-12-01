# Implementation Plan: Multi-Workspace Server Support

**Branch**: `001-multi-workspace-server` | **Date**: 2025-12-01 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/001-multi-workspace-server/spec.md`

## Summary

Implement server-level multi-workspace support for LightRAG Server by introducing:
1. A process-local pool of LightRAG instances keyed by workspace identifier
2. HTTP header-based workspace routing (`LIGHTRAG-WORKSPACE`, fallback `X-Workspace-ID`)
3. A FastAPI dependency that resolves the appropriate LightRAG instance per request
4. Configuration options for default workspace behavior and pool size limits

This builds on the existing workspace isolation in the LightRAG core (storage namespacing, pipeline status isolation) without re-implementing isolation at the storage level.

## Technical Context

**Language/Version**: Python 3.10+
**Primary Dependencies**: FastAPI, Pydantic, asyncio, uvicorn
**Storage**: Delegates to existing backends (JsonKV, NanoVectorDB, NetworkX, Postgres, Neo4j, etc.)
**Testing**: pytest 8.4+, pytest-asyncio 1.2+ with `asyncio_mode = "auto"`
**Target Platform**: Linux server (also Windows/macOS for development)
**Project Type**: Single project - Python package with API server
**Performance Goals**: <10ms workspace routing overhead, <5s first-request initialization per workspace
**Constraints**: Full backward compatibility with existing single-workspace deployments
**Scale/Scope**: Support 50+ concurrent workspace instances (configurable)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Requirement | Design Compliance |
|-----------|-------------|-------------------|
| **I. API Backward Compatibility** | No breaking changes to public API | ✅ No route/payload changes; existing behavior preserved when no workspace header |
| **II. Workspace/Tenant Isolation** | Data must never cross workspace boundaries | ✅ Leverages existing core isolation; each workspace gets separate LightRAG instance |
| **III. Explicit Configuration** | Config must be documented and validated | ✅ New env vars documented; startup validation for invalid configs |
| **IV. Multi-Workspace Test Coverage** | Tests for all new isolation logic | ✅ Test plan includes isolation, backward compat, config validation tests |

**Constitution Status**: ✅ All gates pass

## Project Structure

### Documentation (this feature)

```text
specs/001-multi-workspace-server/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output (no new API contracts needed)
└── tasks.md             # Phase 2 output (/speckit.tasks command)
```

### Source Code (repository root)

```text
lightrag/
├── api/
│   ├── lightrag_server.py    # MODIFY: Integrate workspace pool and dependency
│   ├── config.py             # MODIFY: Add multi-workspace config options
│   ├── workspace_manager.py  # NEW: Instance pool and workspace resolution
│   ├── routers/
│   │   ├── document_routes.py  # MODIFY: Use workspace dependency
│   │   ├── query_routes.py     # MODIFY: Use workspace dependency
│   │   ├── graph_routes.py     # MODIFY: Use workspace dependency
│   │   └── ollama_api.py       # MODIFY: Use workspace dependency
│   └── utils_api.py          # MODIFY: Add workspace-aware auth dependency
└── ...

tests/
├── conftest.py                         # MODIFY: Add multi-workspace fixtures
├── test_workspace_isolation.py         # EXISTS: Core workspace isolation tests
└── test_multi_workspace_server.py      # NEW: Server-level multi-workspace tests
```

**Structure Decision**: Extends existing single-project structure. New `workspace_manager.py` module encapsulates all multi-workspace logic to minimize changes to existing files.

## Complexity Tracking

> No Constitution Check violations requiring justification.

| Decision | Rationale |
|----------|-----------|
| Single new module (`workspace_manager.py`) | Centralizes multi-workspace logic; minimizes changes to existing code |
| LRU eviction for pool | Simple, well-understood algorithm; matches access patterns |
| Closure-to-dependency migration | Required for per-request workspace resolution; additive change |
