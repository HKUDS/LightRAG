# Feature Specification: Query Token Usage for Billing

**Feature Branch**: `005-query-token-usage`
**Created**: 2025-12-25
**Status**: Draft
**Input**: User description: "Add token_usage to Query Response for Cleo billing. Return token usage information in /query endpoint response for accurate user billing based on credit_pricing table rates."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Standard Query Token Tracking (Priority: P1)

As Cleo's billing system, I need to receive token usage information with every query response so that I can accurately calculate and charge users for their RAG query costs.

**Why this priority**: This is the core billing requirement. Without token usage data, Cleo cannot charge users for query costs, making the RAG service unbillable.

**Independent Test**: Can be fully tested by making a standard query and verifying the response includes all token_usage fields with accurate values.

**Acceptance Scenarios**:

1. **Given** a user makes a standard query, **When** the query is processed with LLM response generation, **Then** the response includes `token_usage` with `llm_model`, `llm_input_tokens`, `llm_output_tokens`, `embedding_model`, and `embedding_tokens`.
2. **Given** a user makes a query, **When** the response is returned, **Then** `llm_input_tokens` reflects the total input (system prompt + context chunks + user query).
3. **Given** a user makes a query, **When** the response is returned, **Then** `llm_output_tokens` reflects the generated response tokens.
4. **Given** a user makes a query, **When** the response is returned, **Then** model names match the exact model IDs used (e.g., `gpt-4o-mini` not display names).

---

### User Story 2 - Context-Only Query Token Tracking (Priority: P1)

As Cleo's billing system, I need accurate token tracking when users request context only (no LLM generation) so that I only charge for embedding costs, not LLM costs.

**Why this priority**: Equal priority to US1 because incorrect billing for context-only queries would overcharge users and cause billing disputes.

**Independent Test**: Can be fully tested by making a query with `only_need_context=true` and verifying LLM tokens are zero while embedding tokens are populated.

**Acceptance Scenarios**:

1. **Given** a user makes a query with `only_need_context=true`, **When** the response is returned, **Then** `llm_input_tokens` equals 0 and `llm_output_tokens` equals 0.
2. **Given** a user makes a query with `only_need_context=true`, **When** the response is returned, **Then** `embedding_tokens` reflects the query embedding cost.
3. **Given** a user makes a query with `only_need_context=true`, **When** the response is returned, **Then** `embedding_model` is populated with the model used.

---

### User Story 3 - Streaming Query Token Tracking (Priority: P2)

As Cleo's billing system, I need token usage information for streaming queries so that streaming responses can be billed accurately.

**Why this priority**: Streaming is a secondary use case but still requires accurate billing. Can be delivered after standard query tracking.

**Independent Test**: Can be fully tested by making a streaming query and verifying the final SSE event includes token_usage data.

**Acceptance Scenarios**:

1. **Given** a user makes a streaming query, **When** the stream completes, **Then** the final SSE event includes `token_usage` with all required fields.
2. **Given** a user makes a streaming query, **When** the stream is interrupted, **Then** partial token usage up to the interruption point is available.

---

### Edge Cases

- What happens when the LLM call fails after embedding? System should still report embedding_tokens consumed.
- What happens when no context chunks are retrieved? LLM tokens should still be tracked (system prompt + query only).
- What happens when using a local/self-hosted model? Model names should still be populated; token counts may be estimated.
- How does system handle cached responses? Token usage should reflect actual API calls made (zero if fully cached).

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST include a `token_usage` object in the response for `POST /query` endpoint.
- **FR-002**: System MUST include `llm_model` field containing the exact model ID used for response generation (e.g., `gpt-4o-mini`).
- **FR-003**: System MUST include `llm_input_tokens` field with total input tokens (system prompt + context + query).
- **FR-004**: System MUST include `llm_output_tokens` field with tokens in the generated response.
- **FR-005**: System MUST include `embedding_model` field containing the exact model ID used for query embedding.
- **FR-006**: System MUST include `embedding_tokens` field with tokens used to embed the query.
- **FR-007**: System MUST set `llm_input_tokens` and `llm_output_tokens` to 0 when `only_need_context=true`.
- **FR-008**: System MUST use token counts from the provider's API response metadata (not estimated counts).
- **FR-009**: System MUST include `token_usage` in the final SSE event for `POST /query/stream` endpoint.
- **FR-010**: System MUST preserve backward compatibility - existing response fields (`response`, `sources`) remain unchanged.

### Key Entities

- **TokenUsage**: Represents token consumption for a single query. Contains `llm_model`, `llm_input_tokens`, `llm_output_tokens`, `embedding_model`, `embedding_tokens`.
- **QueryResponse**: Extended to include the new `token_usage` field alongside existing `response` and `sources` fields.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of query responses include complete `token_usage` data with all 5 required fields.
- **SC-002**: Token counts match provider API response metadata with 100% accuracy.
- **SC-003**: Context-only queries (`only_need_context=true`) correctly report 0 LLM tokens.
- **SC-004**: Streaming queries include token_usage in final event for 100% of completed streams.
- **SC-005**: Cleo can calculate query costs using token_usage data and credit_pricing rates without additional API calls.
- **SC-006**: Existing API clients continue to function without modification (backward compatibility).

## Assumptions

- Token counts are available from LLM/embedding provider API responses (OpenAI, Azure OpenAI).
- The existing TokenTracker infrastructure from feature 003-api-usage-metering can be leveraged.
- Model names use the exact API model ID format, not display names.
- For local models without token counting, the system will estimate tokens or return null (to be confirmed during implementation).

## Dependencies

- Feature 003-api-usage-metering: TokenTracker class and usage tracking infrastructure.
- LLM provider APIs must return token usage in response metadata.

## Out of Scope

- Aggregated token usage reporting across multiple queries.
- Historical token usage storage or analytics.
- Cost calculation within the API (Cleo handles this with credit_pricing table).
- Token usage for document insertion (covered by feature 004-doc-token-tracking).
