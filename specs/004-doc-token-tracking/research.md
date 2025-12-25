# Research: Document Processing Token Tracking

**Feature**: 004-doc-token-tracking
**Date**: 2025-12-25

## Executive Summary

All required infrastructure already exists in the codebase. The TokenTracker class is ready to use, the metadata JSONB column is available, and the track_status endpoint already returns metadata. Implementation requires only passing TokenTracker through the pipeline and persisting token counts.

## 1. Existing TokenTracker Infrastructure

**Location**: `lightrag/utils.py` (lines 2530-2628)

**Decision**: Use existing TokenTracker class without modification.

**Rationale**: The class already has all required methods:
- `add_usage(token_counts, model)` - for LLM tokens
- `add_embedding_usage(token_counts, model)` - for embedding tokens
- `get_llm_usage()` - returns dict with prompt_tokens, completion_tokens, total_tokens, call_count, model
- `get_embedding_usage()` - returns dict with total_tokens, call_count, model

**Alternatives Considered**:
- Create new DocumentTokenTracker class - rejected, duplicates existing functionality
- Modify TokenTracker for document-specific needs - rejected, current API is sufficient

## 2. Document Status Storage

**Location**: `lightrag/kg/postgres_impl.py` (lines 4848-4863)

**Decision**: Use existing `metadata` JSONB column in `lightrag_doc_status` table.

**Table Schema**:
```sql
CREATE TABLE lightrag_doc_status (
    workspace VARCHAR(1024) NOT NULL,
    id VARCHAR(255) NOT NULL,
    content_summary VARCHAR(255),
    content_length INT,
    chunks_count INT,
    status VARCHAR(64) DEFAULT 'pending',
    file_path TEXT,
    chunks_list JSONB DEFAULT '[]',
    track_id VARCHAR(255),
    metadata JSONB DEFAULT '{}',  -- Token usage stored here
    error_msg TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (workspace, id)
);
```

**Rationale**:
- No schema migration required
- JSONB supports flexible token_usage structure
- Already indexed for workspace-based queries

**Alternatives Considered**:
- Add dedicated token columns - rejected, requires schema migration
- Create separate token_usage table - rejected, over-engineering for simple use case

## 3. Document Processing Pipeline

**Location**: `lightrag/lightrag.py`

**Key Methods**:
- `ainsert()` (line 1142) - Entry point, generates track_id
- `apipeline_enqueue_documents()` - Enqueues documents for processing
- `apipeline_process_enqueue_documents()` (line 1597) - Main processing loop
- `_process_extract_entities()` (line 2160) - Calls entity extraction

**Decision**: Create TokenTracker in pipeline and pass via global_config dict.

**Integration Points**:
1. Create TokenTracker at start of `apipeline_process_enqueue_documents()`
2. Record processing_start_time
3. Pass token_tracker to:
   - Embedding function (via kwargs, already supported)
   - extract_entities() (via global_config)
4. Record processing_end_time
5. Build token_usage dict and store in metadata before status update

**Rationale**:
- global_config pattern already used for passing configuration
- Minimal changes to existing code
- TokenTracker instance scoped to single document processing

## 4. LLM Token Capture

**Location**: `lightrag/operate.py`

**Key Function**: `extract_entities()` (line 2768)

**Current LLM Call Pattern**:
```python
result = await use_llm_func_with_cache(
    use_llm_func=use_model_func,
    input_text=chunk_key,
    ...
)
```

**Decision**: Capture token counts from LLM response usage field.

**Implementation**:
1. OpenAI-compatible responses include `response.usage.prompt_tokens` and `response.usage.completion_tokens`
2. The LLM wrapper in `lightrag/llm/openai.py` already extracts this data
3. Pass token_tracker via global_config["token_tracker"]
4. Call `token_tracker.add_usage()` after each LLM call

**Rationale**: Consistent with query-side token tracking (003-api-usage-metering)

## 5. Embedding Token Capture

**Location**: `lightrag/llm/openai.py`

**Current Implementation**:
- Embedding function accepts `token_tracker` kwarg
- Calls `token_tracker.add_embedding_usage(token_counts, model=model)`
- Already implemented in 003-api-usage-metering feature

**Decision**: Ensure token_tracker is passed to embedding calls during document processing.

**Implementation**:
- EmbeddingFunc wrapper passes kwargs through to underlying function
- Just need to include token_tracker in kwargs when calling embeddings

## 6. Track Status API

**Location**: `lightrag/api/routers/document_routes.py` (lines 2881-2953)

**Current Response**:
```python
{
    "track_id": "...",
    "documents": [
        {
            "id": "...",
            "status": "processed",
            "chunks_count": 5,
            "metadata": {},  # token_usage will appear here
            "error_msg": null
        }
    ],
    "status_summary": {...}
}
```

**Decision**: No API changes needed.

**Rationale**:
- Endpoint already returns `metadata` field
- token_usage stored in metadata is automatically included
- Backward compatible - existing clients ignore new fields

## 7. Chunk Count Tracking

**Location**: `lightrag/lightrag.py`

**Current Implementation**:
- `chunks_count` already stored in doc_status during processing
- Calculated from chunking results

**Decision**: Copy chunks_count to token_usage.total_chunks.

**Rationale**:
- Consistent with Cleo's billing model
- Already available, no additional computation needed

## Conclusions

1. **No new dependencies required**
2. **No database schema changes required**
3. **No API changes required**
4. **Reuse existing TokenTracker class**
5. **Minimal code changes** - pass TokenTracker through pipeline and persist to metadata

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Token tracking adds latency | Low | Low | TokenTracker is in-memory, O(1) operations |
| Missing tokens from failed calls | Medium | Low | Store partial token_usage on failure |
| Concurrent processing interference | Low | Medium | TokenTracker is per-document, no sharing |
