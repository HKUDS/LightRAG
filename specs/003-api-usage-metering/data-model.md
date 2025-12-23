# Data Model: API Usage Metering

**Feature**: 003-api-usage-metering
**Date**: 2025-12-23

## Entities

### UsageRecord

Per-request usage metrics persisted for historical queries.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| id | string | Yes | Unique record ID (UUID) |
| workspace | string | Yes | Workspace identifier |
| request_id | string | Yes | Request correlation ID |
| timestamp | datetime | Yes | When the request was processed |
| operation_type | enum | Yes | "query", "upload", "insert_text" |
| llm_model | string | No | LLM model used (e.g., "gpt-4o-mini") |
| llm_prompt_tokens | integer | No | LLM input tokens |
| llm_completion_tokens | integer | No | LLM output tokens |
| llm_total_tokens | integer | No | Total LLM tokens |
| llm_calls | integer | No | Number of LLM API calls |
| embedding_model | string | No | Embedding model used |
| embedding_tokens | integer | No | Embedding tokens consumed |
| embedding_calls | integer | No | Number of embedding API calls |
| estimated_cost_usd | decimal | No | Calculated cost (if pricing configured) |

**Indexes**:
- Primary: `id`
- Query: `(workspace, timestamp)` for date range queries
- Aggregation: `(workspace, operation_type, timestamp)`

**Retention**: Follows workspace data retention policy (configurable)

---

### UsageInfo (Response Model)

Embedded in API responses to provide per-request usage data.

```python
class LLMUsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    calls: int
    model: Optional[str] = None

class EmbeddingUsageInfo(BaseModel):
    tokens: int
    calls: int
    model: Optional[str] = None

class UsageInfo(BaseModel):
    llm: Optional[LLMUsageInfo] = None
    embedding: Optional[EmbeddingUsageInfo] = None
    estimated_cost_usd: Optional[float] = None
```

---

### WorkspaceUsageAggregate (Query Response)

Aggregated usage for billing/reporting queries.

| Field | Type | Description |
|-------|------|-------------|
| workspace | string | Workspace identifier |
| start_date | date | Aggregation period start |
| end_date | date | Aggregation period end |
| total_llm_prompt_tokens | integer | Sum of LLM input tokens |
| total_llm_completion_tokens | integer | Sum of LLM output tokens |
| total_llm_calls | integer | Total LLM API calls |
| total_embedding_tokens | integer | Sum of embedding tokens |
| total_embedding_calls | integer | Total embedding API calls |
| total_estimated_cost_usd | decimal | Sum of estimated costs |
| request_count | integer | Number of requests in period |

---

### PricingConfig (Configuration)

Optional pricing configuration for cost estimation.

| Environment Variable | Type | Default | Description |
|---------------------|------|---------|-------------|
| LIGHTRAG_LLM_PRICE_PER_1K_INPUT | float | None | USD per 1000 input tokens |
| LIGHTRAG_LLM_PRICE_PER_1K_OUTPUT | float | None | USD per 1000 output tokens |
| LIGHTRAG_EMBEDDING_PRICE_PER_1K | float | None | USD per 1000 embedding tokens |

When not configured, `estimated_cost_usd` is omitted from responses.

---

## Relationships

```
Workspace (1) ────────── (*) UsageRecord
    │                         │
    │                         ├── llm_model
    │                         ├── embedding_model
    │                         └── operation_type
    │
    └── PricingConfig (optional, global)
```

## State Transitions

UsageRecord is immutable after creation. No state transitions.

## Validation Rules

1. **workspace**: Must be a valid workspace ID (sanitized, no path traversal)
2. **request_id**: Must be unique per workspace
3. **timestamp**: Must be UTC, not in future
4. **token counts**: Must be non-negative integers
5. **cost estimate**: Must be non-negative decimal, max 2 decimal places

## Data Volume Estimates

| Metric | Estimate | Notes |
|--------|----------|-------|
| Records per query | 1 | One record per API request |
| Record size | ~500 bytes | JSON serialized |
| Daily volume (per workspace) | 100-10,000 | Depends on usage |
| Retention | 30-90 days | Configurable |
| Storage per workspace/month | 1-50 MB | Depends on activity |

## Storage Implementation

**Namespace**: `{workspace}:usage_metrics`

**PostgreSQL Schema**:
```sql
CREATE TABLE usage_metrics (
    id UUID PRIMARY KEY,
    workspace VARCHAR(255) NOT NULL,
    request_id VARCHAR(255) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    operation_type VARCHAR(50) NOT NULL,
    llm_model VARCHAR(100),
    llm_prompt_tokens INTEGER,
    llm_completion_tokens INTEGER,
    llm_total_tokens INTEGER,
    llm_calls INTEGER,
    embedding_model VARCHAR(100),
    embedding_tokens INTEGER,
    embedding_calls INTEGER,
    estimated_cost_usd DECIMAL(10, 6),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_usage_workspace_timestamp
    ON usage_metrics (workspace, timestamp);
CREATE INDEX idx_usage_workspace_operation
    ON usage_metrics (workspace, operation_type, timestamp);
```

## Backward Compatibility

- All new response fields are optional
- Existing clients receive `usage: null` or field omitted
- No changes to existing request formats
