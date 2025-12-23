# Implementation Plan: API Usage Metering and Cost Tracking

**Branch**: `003-api-usage-metering` | **Date**: 2025-12-23 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/003-api-usage-metering/spec.md`

## Summary

Add comprehensive API usage metering to track token consumption for LLM and embedding operations. The system will capture usage data from OpenAI/Azure SDK responses, persist per-request metrics, and expose them in API responses. This enables SaaS operators to accurately bill end-users based on actual API consumption.

**Technical Approach**: Extend the existing `TokenTracker` class to include workspace context, add a new `UsageMetricsStorage` component for persistence, and extend API response models to include usage data.

## Technical Context

**Language/Version**: Python 3.10+ (existing codebase)
**Primary Dependencies**: FastAPI, Pydantic, asyncio, uvicorn, OpenAI SDK
**Storage**: PostgreSQL (via existing `postgres_impl.py` patterns), workspace-namespaced
**Testing**: pytest with async fixtures
**Target Platform**: Linux server (Render deployment)
**Project Type**: Single project (API server extension)
**Performance Goals**: <50ms overhead per request for usage tracking
**Constraints**: Must maintain backward compatibility (existing clients don't break)
**Scale/Scope**: Multi-tenant, workspace-isolated usage data

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. API Backward Compatibility | PASS | Usage field is additive; existing responses unchanged |
| II. Workspace and Tenant Isolation | PASS | Usage data scoped to workspace via namespace pattern |
| III. Explicit Server Configuration | PASS | Optional pricing config via env vars |
| IV. Multi-Workspace Test Coverage | REQUIRED | Must add tests for workspace-scoped usage queries |

**Security Requirements**:
- Usage data access is workspace-scoped (FR-012)
- No cross-workspace usage queries allowed
- Usage endpoint protected by existing auth

**Performance Standards**:
- Usage tracking adds <50ms latency (SC-004)
- No degradation to existing query performance

## Project Structure

### Documentation (this feature)

```text
specs/003-api-usage-metering/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
│   └── usage-api.yaml   # OpenAPI spec for usage endpoints
└── tasks.md             # Phase 2 output
```

### Source Code (repository root)

```text
lightrag/
├── api/
│   ├── routers/
│   │   ├── query_routes.py      # Extend responses with usage
│   │   ├── document_routes.py   # Extend responses with usage
│   │   └── usage_routes.py      # NEW: GET /usage endpoint
│   └── usage_metering.py        # NEW: Usage tracking module
├── llm/
│   └── openai.py                # Already has TokenTracker integration
├── kg/
│   └── usage_storage.py         # NEW: Usage storage implementation
└── utils.py                     # Extend TokenTracker

tests/
├── test_usage_metering.py       # NEW: Unit tests
└── test_usage_api.py            # NEW: API integration tests
```

**Structure Decision**: Extend existing single-project structure with new modules for usage metering. Follows existing patterns from `kg/` storage implementations.

## Key Integration Points

| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| TokenTracker | `lightrag/utils.py` | 2528-2582 | Existing class to extend |
| LLM Token Capture | `lightrag/llm/openai.py` | 424-434, 576-584 | Already captures usage |
| Embedding Capture | `lightrag/llm/openai.py` | 770-775 | Already captures usage |
| Query Response | `lightrag/api/routers/query_routes.py` | 158-165 | Extend with usage field |
| Workspace Manager | `lightrag/api/workspace_manager.py` | 32-236 | Provides workspace context |
| Namespace Pattern | `lightrag/kg/shared_storage.py` | 99-112 | `workspace:namespace` format |

## Design Decisions

### D1: Usage Response Structure

```json
{
  "response": "...",
  "usage": {
    "llm": {
      "prompt_tokens": 1250,
      "completion_tokens": 340,
      "total_tokens": 1590,
      "calls": 2,
      "model": "gpt-4o-mini"
    },
    "embedding": {
      "tokens": 512,
      "calls": 3,
      "model": "text-embedding-3-small"
    },
    "estimated_cost_usd": 0.0023
  }
}
```

### D2: Storage Schema

Usage records stored with namespace: `{workspace}:usage_metrics`

Fields: request_id, timestamp, operation_type, llm_model, prompt_tokens, completion_tokens, embedding_model, embedding_tokens, cost_estimate

### D3: Aggregation Endpoint

`GET /usage?start_date=2025-01-01&end_date=2025-01-31`

Returns workspace-scoped aggregates (sum of tokens, call counts, costs).

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Performance degradation | Low | Medium | Async storage writes, benchmark testing |
| Storage bloat | Medium | Low | TTL-based cleanup, aggregation for old data |
| Missing usage data | Low | High | Graceful degradation, log on failure |
| Backward compatibility break | Low | High | Optional usage field, version testing |

## Test Strategy

1. **Unit tests**: Mock TokenTracker, verify usage accumulation
2. **Integration tests**: Full query flow with usage in response
3. **Multi-workspace tests**: Verify workspace isolation for usage data
4. **Backward compatibility tests**: Existing clients work without changes
5. **Performance tests**: Verify <50ms overhead

## Complexity Tracking

> No Constitution violations requiring justification.

| Aspect | Complexity | Justification |
|--------|------------|---------------|
| New storage table | Low | Follows existing KV storage pattern |
| Response model extension | Low | Additive field, backward compatible |
| Aggregation endpoint | Medium | Requires date range queries |
