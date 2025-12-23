# Quickstart: API Usage Metering

**Feature**: 003-api-usage-metering
**Date**: 2025-12-23

## Prerequisites

- Python 3.10+
- LightRAG-MT server running
- API key configured

## Quick Verification

### Step 1: Make a Query and Check Usage

```bash
curl -X POST "http://localhost:9621/query" \
  -H "X-API-Key: your-api-key" \
  -H "LIGHTRAG-WORKSPACE: test-workspace" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the capital of France?"}'
```

**Expected Response** (with usage feature enabled):
```json
{
  "response": "The capital of France is Paris...",
  "references": [...],
  "usage": {
    "llm": {
      "prompt_tokens": 1250,
      "completion_tokens": 85,
      "total_tokens": 1335,
      "calls": 1,
      "model": "gpt-4o-mini"
    },
    "embedding": {
      "tokens": 12,
      "calls": 1,
      "model": "text-embedding-3-small"
    },
    "estimated_cost_usd": 0.0015
  }
}
```

### Step 2: Query Historical Usage

```bash
curl "http://localhost:9621/usage?start_date=2025-01-01&end_date=2025-01-31" \
  -H "X-API-Key: your-api-key" \
  -H "LIGHTRAG-WORKSPACE: test-workspace"
```

**Expected Response**:
```json
{
  "workspace": "test-workspace",
  "start_date": "2025-01-01",
  "end_date": "2025-01-31",
  "llm": {
    "total_prompt_tokens": 125000,
    "total_completion_tokens": 8500,
    "total_calls": 100
  },
  "embedding": {
    "total_tokens": 1200,
    "total_calls": 100
  },
  "total_estimated_cost_usd": 0.25,
  "request_count": 100
}
```

## Configuration

### Optional: Enable Cost Estimation

Add to your `.env` file:

```bash
# Pricing per 1000 tokens (adjust for your model)
LIGHTRAG_LLM_PRICE_PER_1K_INPUT=0.0015
LIGHTRAG_LLM_PRICE_PER_1K_OUTPUT=0.002
LIGHTRAG_EMBEDDING_PRICE_PER_1K=0.0001
```

If not configured, `estimated_cost_usd` will be omitted from responses.

## Test Scenarios

### Scenario 1: Query with Usage Tracking

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Send POST /query | Response includes `usage` object |
| 2 | Check `usage.llm.prompt_tokens` | Non-zero integer |
| 3 | Check `usage.llm.model` | Model name present |

### Scenario 2: Document Upload with Usage

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Upload document via POST /documents/upload_file | Response includes usage |
| 2 | Check `usage.embedding.tokens` | Non-zero (embeddings generated) |
| 3 | Check `usage.llm.calls` | May be >0 if entity extraction ran |

### Scenario 3: Historical Aggregation

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Make 5 queries to workspace | Each returns usage |
| 2 | Query GET /usage for today | `request_count` = 5 |
| 3 | Verify token totals | Sum matches individual requests |

### Scenario 4: Workspace Isolation

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Make queries to workspace A | Usage recorded for A |
| 2 | Make queries to workspace B | Usage recorded for B |
| 3 | Query GET /usage for A | Only A's usage returned |
| 4 | Query GET /usage for B | Only B's usage returned |

### Scenario 5: Backward Compatibility

| Step | Action | Expected Result |
|------|--------|-----------------|
| 1 | Use existing client (no usage handling) | Client works unchanged |
| 2 | Parse response as before | `response` field still present |
| 3 | Ignore `usage` field | No errors |

## Troubleshooting

### Usage field is null/missing

1. Check that the LLM provider returns usage data (OpenAI does by default)
2. Verify the request completed successfully (not cached)
3. Check server logs for usage tracking errors

### estimated_cost_usd is missing

1. Verify pricing env vars are set
2. Check values are valid floats (e.g., `0.0015`, not `0,0015`)
3. Restart server after changing env vars

### GET /usage returns empty results

1. Verify date range includes requests
2. Check workspace header matches where queries were made
3. Ensure usage persistence is working (check database)

## Success Criteria Checklist

- [ ] Query responses include `usage` field
- [ ] Token counts match provider dashboard
- [ ] LLM and embedding usage tracked separately
- [ ] Model names included in usage
- [ ] GET /usage returns aggregated data
- [ ] Workspace isolation verified
- [ ] Existing clients work unchanged
- [ ] Cost estimation works when configured
