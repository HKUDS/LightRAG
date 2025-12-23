# Research: API Usage Metering and Cost Tracking

**Feature**: 003-api-usage-metering
**Date**: 2025-12-23
**Status**: Complete

## Executive Summary

Research confirms that LightRAG already has token tracking infrastructure via `TokenTracker` class. The OpenAI SDK returns usage data in responses which is already being captured. Implementation requires extending existing patterns rather than building from scratch.

---

## Research Area 1: Token Tracking Infrastructure

### Question
Does LightRAG already track token usage, and if so, how?

### Investigation

Found existing `TokenTracker` class in `lightrag/utils.py` (lines 2528-2582):

```python
class TokenTracker:
    def __init__(self):
        self.reset()

    def add_usage(self, token_counts):
        self.prompt_tokens += token_counts.get("prompt_tokens", 0)
        self.completion_tokens += token_counts.get("completion_tokens", 0)
        self.total_tokens += token_counts.get("total_tokens", 0)
        self.call_count += 1

    def get_usage(self):
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "call_count": self.call_count,
        }
```

### Decision
Extend existing `TokenTracker` to include workspace context and model names.

### Rationale
- Infrastructure already exists and is tested
- Already integrated with LLM and embedding calls
- Minimal changes required

### Alternatives Considered

| Alternative | Why Rejected |
|------------|--------------|
| New tracking class | Duplicates existing functionality |
| Middleware-only approach | Would miss internal LLM calls |
| External metering service | Over-engineered for current needs |

---

## Research Area 2: LLM Usage Data Availability

### Question
What usage data is available from LLM providers?

### Investigation

**OpenAI SDK Response Structure** (from `lightrag/llm/openai.py`):

Streaming (lines 424-434):
```python
if token_tracker and final_chunk_usage:
    token_counts = {
        "prompt_tokens": getattr(final_chunk_usage, "prompt_tokens", 0),
        "completion_tokens": getattr(final_chunk_usage, "completion_tokens", 0),
        "total_tokens": getattr(final_chunk_usage, "total_tokens", 0),
    }
    token_tracker.add_usage(token_counts)
```

Non-streaming (lines 576-584):
```python
if token_tracker and hasattr(response, "usage"):
    token_counts = {
        "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
        "completion_tokens": getattr(response.usage, "completion_tokens", 0),
        "total_tokens": getattr(response.usage, "total_tokens", 0),
    }
    token_tracker.add_usage(token_counts)
```

**Available Data**:
- `prompt_tokens`: Input tokens consumed
- `completion_tokens`: Output tokens generated
- `total_tokens`: Sum of prompt + completion
- Model name available from request params

### Decision
Use existing token capture points. Add model name tracking.

### Rationale
- Data already captured, just needs to be exposed
- No additional API calls required
- Accurate to provider billing

---

## Research Area 3: Embedding Usage Data

### Question
How is embedding usage tracked?

### Investigation

**Embedding API Response** (`lightrag/llm/openai.py`, lines 770-775):
```python
if token_tracker and hasattr(response, "usage"):
    token_counts = {
        "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
        "total_tokens": getattr(response.usage, "total_tokens", 0),
    }
    token_tracker.add_usage(token_counts)
```

**Key Difference from LLM**:
- No `completion_tokens` (embeddings are input-only)
- Only `prompt_tokens` and `total_tokens` available

### Decision
Track embedding separately from LLM usage with distinct fields.

### Rationale
- Different pricing tiers for embedding vs LLM
- Different metrics (no completion tokens for embedding)
- Operators need granular breakdown

---

## Research Area 4: API Response Structure

### Question
How are responses currently structured and how to extend them?

### Investigation

**Current QueryResponse** (`lightrag/api/routers/query_routes.py`, lines 158-165):
```python
class QueryResponse(BaseModel):
    response: str = Field(description="The generated response")
    references: Optional[List[ReferenceItem]] = Field(
        default=None,
        description="Reference list"
    )
```

**Extension Pattern**:
Add optional `usage` field with default `None` for backward compatibility.

### Decision
Add optional `usage: Optional[UsageInfo]` field to response models.

### Rationale
- Backward compatible (field is optional)
- Follows existing pattern with `references`
- Pydantic handles serialization automatically

---

## Research Area 5: Storage Pattern for Usage Data

### Question
How should usage data be persisted for historical queries?

### Investigation

**Existing Namespace Pattern** (`lightrag/kg/shared_storage.py`, lines 99-112):
```python
def get_final_namespace(namespace: str, workspace: str | None = None):
    if workspace is None:
        workspace = _default_workspace
    final_namespace = f"{workspace}:{namespace}" if workspace else f"{namespace}"
    return final_namespace
```

**Storage Options Available**:
1. PostgreSQL (`lightrag/kg/postgres_impl.py`) - Full SQL support
2. MongoDB (`lightrag/kg/mongo_impl.py`) - Document storage
3. Redis (`lightrag/kg/redis_impl.py`) - Fast KV access

### Decision
Use PostgreSQL with workspace-namespaced table for usage storage.

### Rationale
- SQL queries for aggregation (SUM, GROUP BY, date ranges)
- Already deployed infrastructure
- Consistent with existing storage patterns
- Workspace isolation via namespace prefix

### Alternatives Considered

| Alternative | Why Rejected |
|------------|--------------|
| Redis only | No SQL aggregation, harder to query history |
| New dedicated DB | Over-engineered, adds ops complexity |
| File-based JSON | Poor query performance at scale |

---

## Research Area 6: Pricing Configuration

### Question
How should pricing be configured for cost estimation?

### Investigation

**Existing Config Pattern**: Environment variables in `.env` / `env.example`

**Pricing Complexity**:
- OpenAI prices vary by model
- Azure pricing may differ
- Prices change over time

### Decision
Use environment variables for pricing config:
```
LIGHTRAG_LLM_PRICE_PER_1K_INPUT=0.0015
LIGHTRAG_LLM_PRICE_PER_1K_OUTPUT=0.002
LIGHTRAG_EMBEDDING_PRICE_PER_1K=0.0001
```

### Rationale
- Simple configuration
- Easy to update without code changes
- Optional (cost estimation disabled if not set)
- Follows existing env var pattern

---

## Summary of Decisions

| Area | Decision | Confidence |
|------|----------|------------|
| Token tracking | Extend existing TokenTracker | High |
| LLM usage | Use existing capture points | High |
| Embedding usage | Separate tracking from LLM | High |
| Response structure | Add optional usage field | High |
| Storage | PostgreSQL with namespace | High |
| Pricing config | Environment variables | High |

## Open Questions

None - all clarifications resolved through code analysis.

## Files to Modify

| File | Changes |
|------|---------|
| `lightrag/utils.py` | Extend TokenTracker with model names |
| `lightrag/api/routers/query_routes.py` | Add usage to QueryResponse |
| `lightrag/api/routers/document_routes.py` | Add usage to upload responses |
| NEW: `lightrag/api/routers/usage_routes.py` | GET /usage endpoint |
| NEW: `lightrag/kg/usage_storage.py` | Usage persistence |
| NEW: `tests/test_usage_metering.py` | Unit tests |
