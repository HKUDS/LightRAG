# Research: Query Token Usage for Billing

**Feature**: 005-query-token-usage
**Date**: 2025-12-25

## Executive Summary

The token tracking infrastructure already exists from feature 003-api-usage-metering. The current API already returns token usage in a nested `usage` structure. This feature requires adding a flat `token_usage` field to match Cleo's expected format while maintaining backward compatibility.

## 1. Existing Infrastructure Analysis

### Current Response Structure

**Location**: `lightrag/api/routers/query_routes.py`

The API already returns token usage:

```json
{
  "response": "Generated answer...",
  "references": [...],
  "usage": {
    "llm": {
      "prompt_tokens": 2450,
      "completion_tokens": 380,
      "total_tokens": 2830,
      "calls": 1,
      "model": "gpt-4o-mini"
    },
    "embedding": {
      "tokens": 45,
      "calls": 1,
      "model": "text-embedding-3-small"
    },
    "estimated_cost_usd": 0.0023
  }
}
```

### Cleo's Required Format

```json
{
  "response": "Generated answer...",
  "sources": [...],
  "token_usage": {
    "llm_model": "gpt-4o-mini",
    "llm_input_tokens": 2450,
    "llm_output_tokens": 380,
    "embedding_model": "text-embedding-3-small",
    "embedding_tokens": 45
  }
}
```

### Field Mapping

| Cleo Field | Current Field | Notes |
|------------|---------------|-------|
| `token_usage.llm_model` | `usage.llm.model` | Same data |
| `token_usage.llm_input_tokens` | `usage.llm.prompt_tokens` | Renamed |
| `token_usage.llm_output_tokens` | `usage.llm.completion_tokens` | Renamed |
| `token_usage.embedding_model` | `usage.embedding.model` | Same data |
| `token_usage.embedding_tokens` | `usage.embedding.tokens` | Same data |

## 2. Decision: Add Flat token_usage Field

**Decision**: Add a new `token_usage` field with flat structure alongside existing `usage` field.

**Rationale**:
- Maintains backward compatibility with existing clients using `usage`
- Provides Cleo-specific format without breaking other integrations
- Minimal code changes - just add a new field builder
- Both formats derive from the same TokenTracker data

**Alternatives Considered**:
- Replace `usage` with `token_usage`: Rejected - breaks backward compatibility
- Have Cleo use existing `usage` format: Rejected - requires Cleo code changes and nested parsing

## 3. Implementation Locations

### Models to Create

**Location**: `lightrag/api/models/usage.py`

Add new model:
```python
class QueryTokenUsage(BaseModel):
    """Flat token usage format for Cleo billing integration."""
    llm_model: Optional[str] = None
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    embedding_model: Optional[str] = None
    embedding_tokens: int = 0
```

### Routes to Modify

**Location**: `lightrag/api/routers/query_routes.py`

1. Add `token_usage` field to `QueryResponse` model (line 159)
2. Build `QueryTokenUsage` from TokenTracker in `query_text` (line 493)
3. Build `QueryTokenUsage` from TokenTracker in streaming final chunk (line 766)

### Context-Only Handling

When `only_need_context=true`:
- `llm_input_tokens = 0`
- `llm_output_tokens = 0`
- `llm_model = None` (no LLM call made)
- `embedding_tokens` and `embedding_model` populated normally

**Current behavior verified**: TokenTracker already tracks zero LLM tokens when no LLM call is made.

## 4. Streaming Response Analysis

**Location**: `lightrag/api/routers/query_routes.py` lines 718-767

Current streaming implementation:
- Uses NDJSON format
- Final chunk includes `usage` field
- TokenTracker is passed through async generator

**Required change**: Add `token_usage` to `StreamChunkResponse` model (line 184)

## 5. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Breaking existing clients | Low | High | Keep existing `usage` field unchanged |
| Token count mismatch | Low | Medium | Reuse existing TokenTracker logic |
| Streaming token tracking incomplete | Low | Low | Verify final chunk includes all tokens |

## Conclusions

1. **Minimal changes required** - infrastructure exists from 003-api-usage-metering
2. **Add new model** - QueryTokenUsage with flat structure
3. **Extend existing responses** - add token_usage alongside usage
4. **No backend changes** - TokenTracker already captures all needed data
5. **Backward compatible** - existing usage field remains unchanged
