# Data Model: Hybrid Cross-Document Entity Resolution

**Phase**: 1 - Design
**Date**: 2025-01-21

## Entities

### 1. CrossDocResolutionMode (Enum)

Defines the algorithm used for cross-document entity resolution.

| Value | Description |
|-------|-------------|
| `full` | Full fuzzy matching against all existing entities (O(n × m)) |
| `vdb` | VDB-assisted matching with top-K candidates (O(n × log m)) |
| `hybrid` | Auto-switch: full below threshold, vdb above threshold |
| `disabled` | Skip cross-document resolution entirely |

**Default**: `hybrid`

---

### 2. CrossDocResolutionConfig

Configuration parameters for resolution behavior.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mode` | CrossDocResolutionMode | `hybrid` | Resolution algorithm to use |
| `threshold_entities` | int | 5000 | Entity count at which hybrid switches to VDB mode |
| `vdb_top_k` | int | 10 | Number of VDB candidates to retrieve |
| `similarity_threshold` | float | 0.85 | Minimum similarity score for matching |

**Source**: Environment variables with fallback to constants.py defaults

---

### 3. CrossDocResolutionMetrics

Metrics captured during resolution for observability and billing.

| Field | Type | Description |
|-------|------|-------------|
| `mode_used` | str | Actual mode used ("full", "vdb") |
| `entities_checked` | int | Number of new entities processed |
| `duplicates_found` | int | Number of entities merged with existing |
| `time_ms` | float | Processing time in milliseconds |
| `comparisons_count` | int | Number of similarity comparisons made |

**Lifecycle**: Created per indexing operation, logged at completion

---

### 4. EntityVDBMetadata (Extension)

Additional metadata for entities in the VDB to support type-filtered queries.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `entity_name` | str | Yes | Canonical entity name |
| `entity_type` | str | Yes | Entity type (PERSON, ORG, LOCATION, etc.) |
| `workspace` | str | Yes | Workspace ID for isolation |

**Note**: Verify current VDB metadata; add `entity_type` if missing.

---

## Relationships

```
CrossDocResolutionConfig
    │
    ▼ (used by)
┌─────────────────────────────────────┐
│  _resolve_cross_document_entities   │
│  (operate.py)                       │
└─────────────────────────────────────┘
    │                           │
    ▼ (mode=full)               ▼ (mode=vdb)
┌────────────────┐      ┌────────────────────┐
│ Full Matching  │      │ VDB-Assisted       │
│ O(n × m)       │      │ O(n × log m)       │
└────────────────┘      └────────────────────┘
    │                           │
    ▼                           ▼
┌─────────────────────────────────────┐
│  BaseGraphStorage.get_node_count()  │
│  (determines mode in hybrid)        │
└─────────────────────────────────────┘
    │
    ▼ (outputs)
CrossDocResolutionMetrics
```

---

## State Transitions

### Hybrid Mode Switching

```
                          ┌─────────────────┐
                          │  Start Indexing │
                          └────────┬────────┘
                                   │
                          ┌────────▼────────┐
                          │ Get node count  │
                          └────────┬────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
          count < threshold             count >= threshold
                    │                             │
           ┌────────▼────────┐          ┌────────▼────────┐
           │   FULL MODE     │          │   VDB MODE      │
           │  Max precision  │          │  Fast lookup    │
           └────────┬────────┘          └────────┬────────┘
                    │                             │
                    └──────────────┬──────────────┘
                                   │
                          ┌────────▼────────┐
                          │  Log Metrics    │
                          └────────┬────────┘
                                   │
                          ┌────────▼────────┐
                          │  Continue       │
                          │  Indexing       │
                          └─────────────────┘
```

---

## Validation Rules

### Configuration Validation

| Field | Rule | Error |
|-------|------|-------|
| `mode` | Must be one of: full, vdb, hybrid, disabled | Invalid resolution mode |
| `threshold_entities` | Must be > 0 | Threshold must be positive |
| `vdb_top_k` | Must be >= 1 and <= 100 | Top-K must be between 1 and 100 |
| `similarity_threshold` | Must be > 0 and <= 1.0 | Similarity threshold must be (0, 1] |

### Runtime Validation

| Condition | Behavior |
|-----------|----------|
| VDB unavailable in vdb/hybrid mode | Fall back to full mode with warning |
| get_node_count() fails | Fall back to full mode with warning |
| Entity embedding fails | Skip entity (keep as-is), log warning |

---

## Storage Impact

### New Storage Requirements

| Storage | Change | Impact |
|---------|--------|--------|
| `BaseGraphStorage` | Add `get_node_count()` method | Minimal - O(1) query |
| Entity VDB metadata | May need `entity_type` field | Backward compatible |

### No Schema Changes Required

- No new tables/collections
- No migration scripts needed
- Existing data structures unchanged
