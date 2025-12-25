# Implementation Plan: Document Processing Token Tracking

**Branch**: `004-doc-token-tracking` | **Date**: 2025-12-25 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/004-doc-token-tracking/spec.md`

## Summary

Add token usage tracking to the document processing pipeline, storing embedding and LLM token counts in the document status metadata when processing completes. Return token_usage data via the existing track_status API endpoint to enable billing for document indexing operations.

## Technical Context

**Language/Version**: Python 3.10+ (existing codebase)
**Primary Dependencies**: FastAPI, Pydantic, asyncio, uvicorn (from 001-multi-workspace-server)
**Storage**: PostgreSQL with JSONB metadata column (existing lightrag_doc_status table)
**Testing**: pytest (existing test infrastructure)
**Target Platform**: Linux server (Docker/cloud deployment)
**Project Type**: Single Python package with API server
**Performance Goals**: No measurable impact on document processing latency (<1% overhead)
**Constraints**: Do NOT modify database schema - use existing metadata JSONB column
**Scale/Scope**: Thousands of documents per workspace, multiple concurrent uploads

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. API Backward Compatibility | ✅ PASS | Adding optional token_usage field to track_status response - existing clients unaffected |
| II. Workspace and Tenant Isolation | ✅ PASS | Token usage stored per-document in workspace-scoped doc_status table |
| III. Explicit Server Configuration | ✅ PASS | No new configuration required - uses existing storage |
| IV. Multi-Workspace Test Coverage | ✅ PASS | Will add tests for workspace-isolated token tracking |

**All gates pass. Proceeding with implementation.**

## Project Structure

### Documentation (this feature)

```text
specs/004-doc-token-tracking/
├── plan.md              # This file
├── research.md          # Existing infrastructure analysis
├── data-model.md        # Token usage data structures
├── quickstart.md        # Integration guide for Cleo
├── contracts/           # API response schema updates
│   └── track-status-api.yaml
└── tasks.md             # Implementation tasks (Phase 2)
```

### Source Code (repository root)

```text
lightrag/
├── utils.py                    # TokenTracker already exists (lines 2530-2628)
├── base.py                     # DocProcessingStatus with metadata field
├── lightrag.py                 # Document processing pipeline
├── operate.py                  # Entity extraction with LLM calls
├── kg/
│   └── postgres_impl.py        # lightrag_doc_status table schema
└── api/
    └── routers/
        └── document_routes.py  # track_status endpoint

tests/
└── test_doc_token_tracking.py  # New test file
```

**Structure Decision**: Extend existing files. No new modules required - TokenTracker already exists and can be passed through the document processing pipeline.

## Complexity Tracking

> No violations. Feature uses existing infrastructure.

| Component | Status | Notes |
|-----------|--------|-------|
| TokenTracker | EXISTS | Already has add_usage() and add_embedding_usage() methods |
| metadata JSONB | EXISTS | lightrag_doc_status.metadata column ready for token_usage |
| track_status API | EXISTS | Returns metadata - just need to populate it |

## Key Integration Points

### 1. Token Capture Points

| Operation | Location | Method |
|-----------|----------|--------|
| Embedding tokens | `lightrag/llm/openai.py` | `add_embedding_usage()` via token_tracker kwarg |
| LLM prompt tokens | `lightrag/operate.py:extract_entities()` | LLM call returns usage in response |
| LLM completion tokens | `lightrag/operate.py:extract_entities()` | Same as above |
| Chunk count | `lightrag/lightrag.py:apipeline_process_enqueue_documents()` | Count chunks created |

### 2. Data Flow

```
Document Upload → Chunking → Embedding → Entity Extraction → Status Update
                     ↓            ↓              ↓                ↓
              count chunks  track tokens   track tokens    save metadata
```

### 3. Metadata Schema

```json
{
  "processing_start_time": 1766671645,
  "processing_end_time": 1766671752,
  "token_usage": {
    "embedding_tokens": 93,
    "llm_input_tokens": 7850,
    "llm_output_tokens": 462,
    "total_chunks": 1,
    "embedding_model": "text-embedding-3-small",
    "llm_model": "openai/gpt-4o-mini"
  }
}
```

## Implementation Approach

### Phase 1: Pass TokenTracker Through Pipeline

1. Create TokenTracker at document enqueue time
2. Store in pipeline_status dict (already exists for progress tracking)
3. Pass to embedding and LLM calls via kwargs

### Phase 2: Capture Tokens

1. Embedding: Already captured via `add_embedding_usage()` in openai.py
2. LLM: Capture from extract_entities() responses
3. Chunks: Count during chunking phase

### Phase 3: Persist to Metadata

1. Build token_usage dict from TokenTracker
2. Add processing timestamps
3. Store in DocProcessingStatus.metadata
4. Persisted automatically when doc status updated to "processed"

### Phase 4: Return in API

1. track_status endpoint already returns metadata
2. token_usage included automatically
3. No API changes needed

## Files to Modify

| File | Change | Description |
|------|--------|-------------|
| `lightrag/lightrag.py` | MODIFY | Create TokenTracker, pass through pipeline, save to metadata |
| `lightrag/operate.py` | MODIFY | Accept TokenTracker, capture LLM tokens from extract_entities |
| `lightrag/api/routers/document_routes.py` | MODIFY | Ensure token_usage in track_status response |
| `tests/test_doc_token_tracking.py` | CREATE | Unit and integration tests |

## Dependencies on Existing Code

- TokenTracker class (utils.py) - ready to use
- DocProcessingStatus.metadata (base.py) - ready to use
- lightrag_doc_status.metadata column (postgres_impl.py) - ready to use
- track_status endpoint (document_routes.py) - returns metadata already
