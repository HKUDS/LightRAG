# Feature Specification: Document Processing Token Tracking

**Feature Branch**: `004-doc-token-tracking`
**Created**: 2025-12-25
**Status**: Draft
**Input**: User description: "Add document processing token tracking for billing. Track and store token usage during document insertion: embedding_tokens, llm_input_tokens, llm_output_tokens, total_chunks. Store in metadata JSONB field when document status becomes processed. Include model names. Return token_usage in track_status response."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Retrieve Token Usage After Document Processing (Priority: P1)

As a SaaS operator (Cleo), I need to retrieve detailed token consumption data after a document has been processed so I can accurately bill end-users based on their actual resource usage.

**Why this priority**: Core billing capability - without accurate usage data, operators cannot charge users appropriately for document processing.

**Independent Test**: Upload a document via API, wait for processing to complete, call the track_status endpoint, and verify the response contains complete token_usage data with all token counts and model names.

**Acceptance Scenarios**:

1. **Given** a document has been uploaded and fully processed, **When** the operator queries the track_status endpoint, **Then** the response includes token_usage with embedding_tokens, llm_input_tokens, llm_output_tokens, total_chunks, embedding_model, and llm_model.

2. **Given** a document is still being processed, **When** the operator queries the track_status endpoint, **Then** the response shows the current status without token_usage (token_usage is null or absent until processing completes).

3. **Given** a document processing failed, **When** the operator queries the track_status endpoint, **Then** the response shows the error status and may include partial token_usage if any tokens were consumed before failure.

---

### User Story 2 - Accumulate Tokens Across Pipeline Steps (Priority: P1)

As a platform operator, I need the system to accumulate all token usage across the entire document processing pipeline (multiple embedding calls and multiple LLM calls for entity/relation extraction) so that billing reflects the complete cost of processing.

**Why this priority**: A single document may require multiple LLM calls and embedding operations. Billing must capture the total, not just partial usage.

**Independent Test**: Process a document that generates multiple chunks and requires multiple LLM calls, then verify the returned token counts reflect the sum of ALL operations in the pipeline.

**Acceptance Scenarios**:

1. **Given** a document that produces 5 chunks, **When** processing completes, **Then** embedding_tokens reflects the total tokens across all 5 embedding operations.

2. **Given** a document that requires 3 LLM calls for entity extraction, **When** processing completes, **Then** llm_input_tokens and llm_output_tokens reflect the sum of all 3 calls.

3. **Given** a document with mixed content, **When** processing completes, **Then** total_chunks accurately reflects the number of text chunks created.

---

### User Story 3 - Persist Token Usage in Document Status (Priority: P2)

As a platform operator, I need token usage data to be persisted with the document status so I can retrieve billing data at any time after processing, even days or weeks later.

**Why this priority**: Billing reconciliation may happen asynchronously. Data must be durable and retrievable for audit purposes.

**Independent Test**: Process a document, verify token_usage is stored, restart the service, and confirm token_usage is still retrievable from the track_status endpoint.

**Acceptance Scenarios**:

1. **Given** a document has been processed, **When** the system stores the document status, **Then** the metadata includes processing_start_time, processing_end_time, and complete token_usage object.

2. **Given** token_usage has been persisted, **When** the operator queries track_status days later, **Then** the complete token_usage data is returned unchanged.

---

### Edge Cases

- What happens when the embedding model returns no usage data (e.g., local model)? System stores zero for embedding_tokens and null for embedding_model.
- What happens when LLM calls fail mid-processing? System stores partial token counts reflecting consumption up to the failure point.
- What happens when a document produces zero chunks (empty document)? System stores total_chunks: 0 and zero for all token counts.
- What happens when processing is retried after failure? System resets and re-accumulates tokens for the successful processing attempt.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST track embedding tokens consumed during document processing and store the total as embedding_tokens.
- **FR-002**: System MUST track LLM input tokens (prompt tokens) consumed during entity/relation extraction and store the total as llm_input_tokens.
- **FR-003**: System MUST track LLM output tokens (completion tokens) generated during entity/relation extraction and store the total as llm_output_tokens.
- **FR-004**: System MUST count the number of text chunks created during document processing and store as total_chunks.
- **FR-005**: System MUST capture and store the embedding model name used (embedding_model).
- **FR-006**: System MUST capture and store the LLM model name used (llm_model).
- **FR-007**: System MUST accumulate token counts across ALL embedding and LLM calls within a single document processing pipeline.
- **FR-008**: System MUST store token_usage data in the existing metadata field when document status transitions to "processed".
- **FR-009**: System MUST include token_usage in the GET /documents/track_status/{track_id} response when the document status is "processed".
- **FR-010**: System MUST NOT modify existing database table schema - all data stored in the existing metadata column.
- **FR-011**: System MUST store processing_start_time and processing_end_time timestamps in the metadata.
- **FR-012**: System MUST handle missing usage data gracefully by storing zero for token counts and null for model names when data is unavailable.

### Key Entities

- **Token Usage**: Represents the resource consumption for a document processing operation. Contains embedding_tokens, llm_input_tokens, llm_output_tokens, total_chunks, embedding_model, llm_model.
- **Document Status Metadata**: Extended metadata object stored with document status. Contains processing_start_time, processing_end_time, and token_usage.
- **Processing Pipeline**: The sequence of operations (chunking, embedding, entity extraction) that consumes tokens and must be tracked holistically.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of successfully processed documents have complete token_usage data available via the track_status endpoint.
- **SC-002**: Token counts accurately reflect the sum of all operations - verified by comparing API-reported usage with actual provider billing.
- **SC-003**: Operators can retrieve token_usage data for any processed document at any time after processing.
- **SC-004**: No changes required to database schema - all data fits within existing metadata structure.
- **SC-005**: Existing document processing functionality remains unchanged (backward compatible).

## Assumptions

- The embedding and LLM providers return usage data in their API responses (prompt_tokens, completion_tokens, total_tokens).
- The existing metadata column has sufficient capacity for the additional token_usage object.
- Processing timestamps use Unix epoch seconds format.
- Token counts are integers (whole numbers).
- Model names are strings as returned by the API providers (e.g., "text-embedding-3-small", "openai/gpt-4o-mini").

## Out of Scope

- Historical aggregation or reporting dashboards for token usage.
- Real-time streaming of token counts during processing.
- Cost calculation (Cleo handles pricing based on token counts).
- Usage quotas or rate limiting based on tokens.
- Token usage for query operations (covered by feature 003-api-usage-metering).
