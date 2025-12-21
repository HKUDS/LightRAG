# Data Model: Fix Graph Not Updating After Document Deletion

**Feature**: 002-fix-graph-deletion-sync
**Date**: 2025-12-21

## Overview

This document describes the data entities involved in the document deletion flow and their relationships. No new entities are introduced; this is a bug fix affecting existing data flow.

## Entities Affected

### Document
Primary content unit being deleted.

| Field | Type | Description |
|-------|------|-------------|
| doc_id | string | Unique identifier (hash-based) |
| file_path | string | Original file path |
| status | enum | PENDING, PROCESSING, PROCESSED, FAILED |
| chunks_list | list[string] | Associated chunk IDs |
| created_at | datetime | Creation timestamp |
| updated_at | datetime | Last update timestamp |

**Storage**: `doc_status` KV storage (PGDocStatusStorage)

### Chunk
Text segment derived from a document.

| Field | Type | Description |
|-------|------|-------------|
| chunk_id | string | Unique identifier (hash-based) |
| content | string | Text content |
| tokens | int | Token count |
| llm_cache_list | list[string] | Associated LLM cache entry IDs |
| order | int | Position in document |

**Storage**: `text_chunks` KV storage (PGKVStorage)

### Entity (Graph Node)
Knowledge graph node extracted from chunks.

| Field | Type | Description |
|-------|------|-------------|
| entity_id | string | Normalized entity name |
| entity_type | string | Entity category |
| description | string | LLM-generated description |
| source_id | string | Pipe-separated chunk IDs |

**Storage**: `chunk_entity_relation_graph` (NetworkXStorage)

### Relationship (Graph Edge)
Knowledge graph edge connecting entities.

| Field | Type | Description |
|-------|------|-------------|
| source | string | Source entity ID |
| target | string | Target entity ID |
| description | string | Relationship description |
| keywords | string | Relationship keywords |
| weight | float | Relationship strength |
| source_id | string | Pipe-separated chunk IDs |

**Storage**: `chunk_entity_relation_graph` (NetworkXStorage)

### LLM Cache Entry
Cached LLM extraction result.

| Field | Type | Description |
|-------|------|-------------|
| cache_id | string | Unique identifier |
| cache_type | string | "extract" for entity extraction |
| chunk_id | string | Associated chunk ID |
| return | string | LLM response text |
| create_time | int | Unix timestamp |

**Storage**: `llm_response_cache` KV storage (PGKVStorage)

## Data Flow During Deletion

### Current Flow (Buggy)

```
1. Get document status → doc_id
2. Get chunks_list → [chunk_ids]
3. Collect llm_cache_list from each chunk → [cache_ids]
4. Analyze affected entities/relationships
5. Delete chunks from VDB and KV storage
6. Delete relationships from graph (remove_edges)
7. Delete entities from graph (remove_nodes)
8. Persist graph (index_done_callback)          ← Graph may reload here
9. Rebuild affected entities from cache         ← Reads STALE cache
10. Delete from full_entities/full_relations
11. Delete document and status
12. Delete LLM cache                            ← TOO LATE
```

### Fixed Flow

```
1. Get document status → doc_id
2. Get chunks_list → [chunk_ids]
3. Collect llm_cache_list from each chunk → [cache_ids]
4. Analyze affected entities/relationships
5. Delete chunks from VDB and KV storage
6. Delete relationships from graph (remove_edges)
7. Delete entities from graph (remove_nodes)
8. [NEW] Delete LLM cache entries              ← BEFORE rebuild
9. Persist graph (with reload prevention)       ← No stale data reload
10. Rebuild affected entities from cache        ← Cache entries already gone
11. Delete from full_entities/full_relations
12. Delete document and status
```

## State Transitions

### Document Deletion State

```
ACTIVE ──────────────────────────────────────────────> DELETED
       │                                                 ↑
       ├─> Chunks deleted                               │
       ├─> Cache invalidated (MOVED EARLIER)            │
       ├─> Graph entities removed                       │
       ├─> Graph persisted                              │
       └─> Document record deleted ─────────────────────┘
```

### Graph Storage State

```
IDLE ─────> IN_DELETION ─────> PERSISTING ─────> IDLE
     │           │                  │
     │           │                  └─> storage_updated = True
     │           │                      (for other processes)
     │           │
     │           └─> Reload BLOCKED during this state
     │
     └─> Normal operations, reload allowed if storage_updated
```

## New Flag: Deletion In Progress

Added to `NetworkXStorage` class:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| _deletion_in_progress | bool | False | Prevents graph reload during deletion |

**Lifecycle**:
1. Set to `True` at start of `remove_nodes()` or `remove_edges()` batch
2. Cleared to `False` after `index_done_callback()` completes
3. When `True`, `_get_graph()` skips reload even if `storage_updated.value` is True

## Referential Integrity

### Cascade Delete Order

```
Document
  └─> Chunks (delete from text_chunks, chunks_vdb)
        └─> LLM Cache Entries (delete from llm_response_cache) [MOVED UP]
  └─> Entity References (full_entities)
        └─> Graph Entities (remove_nodes if no other sources)
  └─> Relationship References (full_relations)
        └─> Graph Relationships (remove_edges if no other sources)
  └─> Document Status (doc_status)
```

### Source ID Management

Entities and relationships track source chunks via `source_id` field:
- Format: `chunk_id_1<|>chunk_id_2<|>chunk_id_3`
- Deletion removes chunk_id from this list
- When list becomes empty, entity/relationship is deleted from graph
- When list has remaining IDs, entity/relationship is rebuilt from cache
