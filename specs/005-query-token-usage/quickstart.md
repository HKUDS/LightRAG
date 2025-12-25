# Quickstart: Query Token Usage for Billing

**Feature**: 005-query-token-usage
**Branch**: `005-query-token-usage`

## Overview

This feature adds a `token_usage` field to query responses for Cleo billing integration.

## Usage

### Standard Query

```bash
curl -X POST 'http://localhost:9621/query' \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "What are the main findings?",
    "mode": "hybrid"
  }'
```

**Response**:
```json
{
  "response": "Based on the knowledge graph, the main findings are...",
  "references": [
    {"reference_id": "chunk-abc123", "file_path": "report.pdf"}
  ],
  "token_usage": {
    "llm_model": "gpt-4o-mini",
    "llm_input_tokens": 2450,
    "llm_output_tokens": 380,
    "embedding_model": "text-embedding-3-small",
    "embedding_tokens": 45
  }
}
```

### Context-Only Query

```bash
curl -X POST 'http://localhost:9621/query' \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "What are the main findings?",
    "only_need_context": true
  }'
```

**Response** (no LLM tokens):
```json
{
  "response": "[Retrieved context chunks]",
  "references": [...],
  "token_usage": {
    "llm_model": null,
    "llm_input_tokens": 0,
    "llm_output_tokens": 0,
    "embedding_model": "text-embedding-3-small",
    "embedding_tokens": 45
  }
}
```

### Streaming Query

```bash
curl -X POST 'http://localhost:9621/query/stream' \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "What are the main findings?"
  }'
```

**Final chunk includes token_usage**:
```json
{"response": "...", "token_usage": {"llm_model": "gpt-4o-mini", ...}}
```

## Billing Integration

Use the `token_usage` fields with Cleo's credit_pricing table:

```python
# Example cost calculation
cost = (
    (token_usage['llm_input_tokens'] / 1_000_000) * 0.15 +  # gpt-4o-mini input
    (token_usage['llm_output_tokens'] / 1_000_000) * 0.60 + # gpt-4o-mini output
    (token_usage['embedding_tokens'] / 1_000_000) * 0.02   # text-embedding-3-small
)
```

## Field Reference

| Field | Type | Description |
|-------|------|-------------|
| `llm_model` | string | Model ID (e.g., `gpt-4o-mini`) |
| `llm_input_tokens` | int | Input tokens to LLM |
| `llm_output_tokens` | int | Output tokens from LLM |
| `embedding_model` | string | Embedding model ID |
| `embedding_tokens` | int | Query embedding tokens |

## Backward Compatibility

The existing `usage` field with nested structure remains available for existing clients.
