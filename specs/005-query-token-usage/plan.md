# Implementation Plan: Query Token Usage for Billing

**Branch**: `005-query-token-usage` | **Date**: 2025-12-25 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/005-query-token-usage/spec.md`

## Summary

Add a flat `token_usage` field to query responses for Cleo billing integration. The infrastructure already exists from feature 003-api-usage-metering - this feature adds a new response field format while maintaining backward compatibility.

## Technical Context

**Language/Version**: Python 3.10+
**Primary Dependencies**: FastAPI, Pydantic, asyncio
**Storage**: N/A (in-memory token tracking)
**Testing**: pytest
**Target Platform**: Linux server
**Project Type**: Single (LightRAG server)
**Performance Goals**: < 1ms overhead for token_usage field generation
**Constraints**: Backward compatible with existing `usage` field
**Scale/Scope**: All query endpoints (/query, /query/stream)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. API Backward Compatibility | ✅ PASS | Existing `usage` field preserved; `token_usage` is additive |
| II. Workspace/Tenant Isolation | ✅ N/A | Token tracking is per-request, no cross-workspace data |
| III. Explicit Server Config | ✅ N/A | No new configuration required |
| IV. Multi-Workspace Test Coverage | ✅ PASS | Tests will verify token_usage in multi-workspace context |

**Gate Result**: PASS - No violations

## Project Structure

### Documentation (this feature)

```text
specs/005-query-token-usage/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
│   └── query-api.yaml   # OpenAPI contract
└── tasks.md             # Phase 2 output (created by /speckit.tasks)
```

### Source Code (repository root)

```text
lightrag/
├── api/
│   ├── models/
│   │   └── usage.py          # Add QueryTokenUsage model
│   └── routers/
│       └── query_routes.py   # Add token_usage to responses

tests/
└── test_query_token_usage.py # New test file
```

**Structure Decision**: Minimal changes to existing structure. New model added to existing usage.py, response models extended in query_routes.py.

## Implementation Phases

### Phase 1: Add QueryTokenUsage Model

**File**: `lightrag/api/models/usage.py`

Add new Pydantic model:
```python
class QueryTokenUsage(BaseModel):
    """Flat token usage structure for Cleo billing."""
    llm_model: Optional[str] = None
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    embedding_model: Optional[str] = None
    embedding_tokens: int = 0

    @classmethod
    def from_token_tracker(cls, token_tracker) -> "QueryTokenUsage":
        llm = token_tracker.get_llm_usage()
        emb = token_tracker.get_embedding_usage()
        return cls(
            llm_model=llm.get("model"),
            llm_input_tokens=llm.get("prompt_tokens", 0),
            llm_output_tokens=llm.get("completion_tokens", 0),
            embedding_model=emb.get("model"),
            embedding_tokens=emb.get("total_tokens", 0),
        )
```

### Phase 2: Extend QueryResponse Model

**File**: `lightrag/api/routers/query_routes.py`

Add `token_usage` field to QueryResponse:
```python
class QueryResponse(BaseModel):
    response: str
    references: Optional[List[ReferenceItem]] = None
    usage: Optional[UsageInfo] = None  # Existing
    token_usage: Optional[QueryTokenUsage] = None  # NEW
```

### Phase 3: Build token_usage in query_text

**File**: `lightrag/api/routers/query_routes.py` (around line 493)

Build token_usage alongside usage_info:
```python
token_usage = QueryTokenUsage.from_token_tracker(token_tracker)
return QueryResponse(
    response=response_content,
    references=references,
    usage=usage_info,
    token_usage=token_usage,  # NEW
)
```

### Phase 4: Add token_usage to StreamChunkResponse

**File**: `lightrag/api/routers/query_routes.py`

1. Add field to StreamChunkResponse model
2. Include in final streaming chunk (around line 766)

### Phase 5: Tests

**File**: `tests/test_query_token_usage.py`

Test cases:
1. Standard query returns token_usage with all fields
2. Context-only query returns 0 LLM tokens
3. Streaming query includes token_usage in final chunk
4. Backward compatibility: usage field still present

## Complexity Tracking

> No Constitution violations - table not needed

## Dependencies

- Feature 003-api-usage-metering: TokenTracker class (already merged)
- Existing QueryResponse model
- Existing StreamChunkResponse model
