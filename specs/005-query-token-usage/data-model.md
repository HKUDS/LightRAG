# Data Model: Query Token Usage for Billing

**Feature**: 005-query-token-usage
**Date**: 2025-12-25

## Entities

### QueryTokenUsage

Flat token usage structure for Cleo billing integration.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| llm_model | string | no | Model ID used for response generation (null if no LLM call) |
| llm_input_tokens | integer | yes | Total input tokens (system prompt + context + query), 0 if no LLM call |
| llm_output_tokens | integer | yes | Tokens in generated response, 0 if no LLM call |
| embedding_model | string | no | Model ID used for query embedding (null if local model) |
| embedding_tokens | integer | yes | Tokens used to embed the query |

**Validation Rules**:
- All token counts must be >= 0
- Model names are optional (null for local models or no call)
- When `only_need_context=true`, LLM fields should be 0/null

**Example**:
```json
{
  "llm_model": "gpt-4o-mini",
  "llm_input_tokens": 2450,
  "llm_output_tokens": 380,
  "embedding_model": "text-embedding-3-small",
  "embedding_tokens": 45
}
```

### Extended QueryResponse

Extended response model with token_usage field.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| response | string | yes | Generated response text |
| references | array | no | Reference list (when include_references=true) |
| usage | UsageInfo | no | Existing nested usage structure (backward compat) |
| token_usage | QueryTokenUsage | no | Flat token usage for Cleo billing |

### Extended StreamChunkResponse

Extended streaming response with token_usage in final chunk.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| references | array | no | Reference list (first chunk only) |
| response | string | no | Response content chunk |
| error | string | no | Error message if processing fails |
| usage | object | no | Existing usage info (final chunk only) |
| token_usage | QueryTokenUsage | no | Flat token usage (final chunk only) |

## State Transitions

```
Query Request
    │
    ├── Embedding Call (always)
    │   └── embedding_tokens += tokens_used
    │       embedding_model = model_id
    │
    ├── only_need_context=true?
    │   └── YES: llm_input_tokens=0, llm_output_tokens=0, llm_model=null
    │
    └── only_need_context=false (default)
        └── LLM Call
            ├── llm_input_tokens = prompt_tokens
            ├── llm_output_tokens = completion_tokens
            └── llm_model = model_id
```

## Relationships

```
QueryResponse (1) ─────────── (0..1) QueryTokenUsage
                                      │
                                      ├── llm_model
                                      ├── llm_input_tokens
                                      ├── llm_output_tokens
                                      ├── embedding_model
                                      └── embedding_tokens

TokenTracker (internal) ──builds──> QueryTokenUsage
```

## Derivation from TokenTracker

QueryTokenUsage is built from the existing TokenTracker:

```python
token_usage = QueryTokenUsage(
    llm_model=token_tracker.get_llm_usage().get("model"),
    llm_input_tokens=token_tracker.get_llm_usage().get("prompt_tokens", 0),
    llm_output_tokens=token_tracker.get_llm_usage().get("completion_tokens", 0),
    embedding_model=token_tracker.get_embedding_usage().get("model"),
    embedding_tokens=token_tracker.get_embedding_usage().get("total_tokens", 0),
)
```
