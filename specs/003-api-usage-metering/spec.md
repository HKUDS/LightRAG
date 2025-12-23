# Feature Specification: API Usage Metering and Cost Tracking

**Feature Branch**: `003-api-usage-metering`
**Created**: 2025-12-23
**Status**: Draft
**Input**: User description: "Add API usage metering and cost tracking for external API calls. Expose token usage (input/output) and call counts for LLM and embedding operations in API responses."

## Clarifications

### Session 2025-12-23

- Q: Should per-request usage be persisted or only returned in response? → A: Persist per-request (store each usage record for history and aggregation)
- Q: Who can access usage data in a multi-tenant context? → A: Workspace-scoped (each workspace can only access its own usage data)

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Per-Request Usage Visibility (Priority: P1)

As a SaaS operator, when I make a query or upload a document via the API, I want to receive detailed usage metrics in the response so I can accurately bill my end-users based on actual consumption.

**Why this priority**: This is the core value proposition. Without per-request usage data in responses, operators cannot implement accurate billing for their users.

**Independent Test**: Can be fully tested by making a single query request and verifying the response contains a "usage" field with token counts and model information.

**Acceptance Scenarios**:

1. **Given** a query request to POST /query, **When** the request completes successfully, **Then** the response must include a "usage" object with LLM token counts (input, output, total) and the model name used.

2. **Given** a document upload to POST /documents/upload_file, **When** the processing completes, **Then** the response must include usage data for all embedding and LLM calls made during processing.

3. **Given** a text insertion to POST /documents/text, **When** the processing completes, **Then** the response must include cumulative usage data for all external API calls.

---

### User Story 2 - Embedding Usage Tracking (Priority: P1)

As a SaaS operator, when documents are processed, I want to see the token usage for embedding operations separately from LLM operations, so I can apply different pricing tiers for each service type.

**Why this priority**: Embedding and LLM calls have very different costs. Operators need granular data to set fair pricing.

**Independent Test**: Can be tested by uploading a document and verifying the response contains separate embedding usage metrics.

**Acceptance Scenarios**:

1. **Given** a document upload request, **When** embeddings are generated for chunks, **Then** the usage response must show embedding token count and number of embedding calls separately from LLM usage.

2. **Given** a query that triggers embedding generation, **When** the query completes, **Then** embedding usage must be reported distinctly from LLM completion usage.

---

### User Story 3 - Estimated Cost Calculation (Priority: P2)

As a SaaS operator, I want to receive an estimated cost in USD for each request based on configurable pricing, so I can provide real-time cost feedback to my users.

**Why this priority**: Enhances user experience but not strictly required for billing. Operators can calculate costs client-side if needed.

**Independent Test**: Can be tested by configuring pricing and verifying the response includes an estimated_cost_usd field.

**Acceptance Scenarios**:

1. **Given** pricing configuration is set for the models in use, **When** a request completes, **Then** the usage response must include an estimated_cost_usd field calculated from token counts and configured rates.

2. **Given** no pricing configuration is set, **When** a request completes, **Then** the usage response must omit the estimated_cost_usd field (not return 0 or null).

---

### User Story 4 - Per-Workspace Usage Aggregation (Priority: P3)

As a platform administrator, I want to query aggregated usage statistics per workspace over a time period, so I can generate billing reports and monitor consumption patterns.

**Why this priority**: Useful for reporting but operators can aggregate per-request data client-side initially.

**Independent Test**: Can be tested by making multiple requests to a workspace, then querying the usage endpoint and verifying totals match.

**Acceptance Scenarios**:

1. **Given** multiple requests have been made to a workspace, **When** I query GET /usage with workspace and date range parameters, **Then** I receive aggregated totals for LLM tokens, embedding tokens, and call counts.

2. **Given** a workspace with no activity in the requested period, **When** I query usage, **Then** I receive zero values (not an error).

---

### Edge Cases

- What happens when external API calls fail mid-request?
  - Partial usage should still be reported for completed calls before the failure.

- What happens when using local/self-hosted models that don't report tokens?
  - Usage fields should indicate "unavailable" or be omitted for that operation type.

- What happens when a request uses multiple different models?
  - Usage should be broken down by model, not aggregated into misleading totals.

- What happens during streaming responses?
  - Final usage should be included in the last chunk or as a separate final message.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST include a "usage" object in API responses for /query, /documents/upload_file, and /documents/text endpoints.

- **FR-002**: System MUST track and report prompt_tokens (input) and completion_tokens (output) for each LLM call.

- **FR-003**: System MUST track and report total_tokens for each embedding call.

- **FR-004**: System MUST report the model name/identifier used for each operation type (LLM, embedding).

- **FR-005**: System MUST report the count of external API calls made per operation type.

- **FR-006**: System MUST aggregate usage across all external calls made during a single request.

- **FR-007**: System MUST support optional cost estimation based on configurable per-token pricing.

- **FR-008**: System MUST maintain backward compatibility - existing clients not expecting usage data should not break.

- **FR-009**: System MUST report partial usage when requests fail after some external calls have completed.

- **FR-010**: System SHOULD provide a dedicated endpoint for querying historical usage aggregates per workspace.

- **FR-011**: System MUST persist each request's usage metrics to enable historical queries and aggregation.

- **FR-012**: System MUST enforce workspace-scoped access control - each workspace can only query its own usage data.

### Key Entities

- **UsageMetrics**: Per-request usage data including LLM tokens (input/output), embedding tokens, call counts, models used, and optional estimated cost.

- **WorkspaceUsage**: Aggregated usage statistics for a workspace over a time period, supporting billing and analytics use cases.

- **PricingConfiguration**: Optional configuration mapping model names to per-token costs for cost estimation.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of successful query and document processing responses include usage metrics.

- **SC-002**: Usage token counts match the actual values returned by external API providers (validated against provider dashboards).

- **SC-003**: Operators can calculate accurate bills within 1% variance compared to provider invoices.

- **SC-004**: Usage data adds less than 50ms overhead to request processing time.

- **SC-005**: Existing API clients continue to function without modification (backward compatibility).

- **SC-006**: Usage endpoint can return aggregated data for up to 30 days of history within 2 seconds.

## Assumptions

- External LLM/embedding providers (OpenAI, Azure, etc.) return usage data in their responses.
- Usage data is returned synchronously with the request (not via webhooks).
- Token counts are integers (not fractional).
- Cost estimation uses USD as the standard currency.
- Historical usage data retention follows workspace data retention policies.
- Streaming responses will include usage in the final chunk.

## Out of Scope

- Real-time usage dashboards or visualizations.
- Automatic billing integration with payment providers.
- Usage quotas or rate limiting based on consumption.
- Multi-currency cost estimation.
- Usage prediction or forecasting.
