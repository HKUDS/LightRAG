# Quickstart: Entity Resolution

**Feature**: 007-entity-resolution
**Date**: 2026-01-16

## Overview

Entity Resolution automatically deduplicates similar entity names and detects conflicts in entity descriptions during document ingestion.

## Configuration

Add the following to your `.env` file or environment:

```bash
# Entity Resolution (Deduplication)
ENABLE_ENTITY_RESOLUTION=true           # Enable/disable entity deduplication
ENTITY_SIMILARITY_THRESHOLD=0.85        # Fuzzy match threshold (0.0-1.0)
ENTITY_MIN_NAME_LENGTH=3                # Min chars for fuzzy matching

# Conflict Detection
ENABLE_CONFLICT_DETECTION=true          # Enable/disable conflict detection
CONFLICT_CONFIDENCE_THRESHOLD=0.7       # Min confidence for logging conflicts
```

## Default Behavior

With default settings:

1. **Entity Deduplication** is ENABLED
   - "Apple Inc", "Apple Inc.", "Apple" → merged into "Apple Inc"
   - Only entities of the same type are merged
   - Names with 2 or fewer characters are not matched

2. **Conflict Detection** is ENABLED
   - Conflicting dates, numbers, and attributions are logged
   - Conflicts appear in entity summaries with uncertainty language

## Usage Examples

### Basic Usage (Defaults)

No configuration needed. Entity resolution works automatically during ingestion:

```python
from lightrag import LightRAG

rag = LightRAG(working_dir="./my_workspace")

# Ingest documents - entity resolution happens automatically
await rag.ainsert("Apple Inc was founded in 1976. Apple makes iPhones.")

# Query - returns consolidated entity information
result = await rag.aquery("Tell me about Apple")
```

### Strict Matching (Fewer False Positives)

```bash
ENTITY_SIMILARITY_THRESHOLD=0.95
```

Only very similar names will be merged (e.g., "Apple Inc" and "Apple Inc." but not "Apple").

### Lenient Matching (More Aggressive Deduplication)

```bash
ENTITY_SIMILARITY_THRESHOLD=0.75
```

More variations will be merged. Use with caution - may cause false positives.

### Disable Entity Resolution

```bash
ENABLE_ENTITY_RESOLUTION=false
```

Entities stored with exact names as extracted. Useful for:
- Debugging extraction issues
- Preserving original entity names
- Performance testing

### Disable Conflict Detection

```bash
ENABLE_CONFLICT_DETECTION=false
```

Conflicts not detected or logged. Entity summaries created without uncertainty language.

## Monitoring

### Resolution Logs

Entity resolutions are logged at INFO level:

```
INFO: Entity resolution: {'Apple Inc.', 'Apple', 'APPLE INC'} → 'Apple Inc' (type: ORGANIZATION)
```

### Conflict Logs

Conflicts are logged at WARNING level:

```
WARNING: Conflict[temporal] in 'Tesla': '2003' vs '2004' (confidence: 0.95)
```

### Log Filtering

To see only entity resolution activity:

```bash
# In your logging config
LIGHTRAG_LOG_LEVEL=INFO
```

```python
import logging
logging.getLogger("lightrag.entity_resolution").setLevel(logging.DEBUG)
logging.getLogger("lightrag.conflict_detection").setLevel(logging.DEBUG)
```

## Verification

### Test Entity Resolution

1. Create a test document:
   ```text
   Apple Inc is a technology company.
   Apple Inc. was founded by Steve Jobs.
   Apple makes iPhones and Macs.
   APPLE INC has a market cap over $2 trillion.
   ```

2. Ingest the document

3. Check logs for:
   ```
   INFO: Entity resolution: {'Apple Inc.', 'Apple', 'APPLE INC'} → 'Apple Inc'
   ```

4. Query the knowledge graph - should return ONE entity with all information merged

### Test Conflict Detection

1. Create test documents:
   ```text
   Document 1: Tesla was founded in 2003.
   Document 2: Tesla was founded in 2004.
   ```

2. Ingest both documents

3. Check logs for:
   ```
   WARNING: Conflict[temporal] in 'Tesla': '2003' vs '2004' (confidence: 0.95)
   ```

4. Query Tesla entity - description should mention both dates with uncertainty

## Troubleshooting

### Too Many False Positives (Wrong Merges)

**Symptom**: Different entities being merged incorrectly

**Solutions**:
1. Increase threshold: `ENTITY_SIMILARITY_THRESHOLD=0.90`
2. Check entity types - ensure extraction correctly identifies types
3. Review logs to identify problematic patterns

### Too Many Duplicates (Not Enough Merging)

**Symptom**: Same entity appears multiple times with slight name variations

**Solutions**:
1. Decrease threshold: `ENTITY_SIMILARITY_THRESHOLD=0.80`
2. Check if names are too short (≤2 chars excluded)
3. Verify entity types match

### Conflict Detection Missing Obvious Conflicts

**Symptom**: Known contradictions not being logged

**Solutions**:
1. Decrease confidence threshold: `CONFLICT_CONFIDENCE_THRESHOLD=0.5`
2. Check if patterns match your data format
3. Ensure conflict detection is enabled

### Performance Degradation

**Symptom**: Ingestion slower than expected

**Solutions**:
1. Check entity count per document - large counts impact resolution time
2. Consider disabling resolution for bulk imports: `ENABLE_ENTITY_RESOLUTION=false`
3. Profile with: `LIGHTRAG_LOG_LEVEL=DEBUG`

## Technical Details

### Matching Algorithm

Uses **Token Set Ratio** from `rapidfuzz` library:
- Handles word order variations: "Apple Inc" ≈ "Inc Apple"
- Handles partial matches: "Apple" ≈ "Apple Inc"
- Case-insensitive
- Punctuation-tolerant

### Canonical Name Selection

When multiple names match:
1. **Longest name** is selected as canonical
2. If equal length, **first encountered** is used
3. Other names become aliases (logged)

### Conflict Types Detected

| Type | Pattern Examples |
|------|------------------|
| temporal | Years (1976, 2003), dates (01/15/2020) |
| attribution | "founded by X", "created by Y" |
| numerical | "$100M", "50%", "1000 employees" |

### Performance Characteristics

- Entity resolution: O(n²) worst case, optimized with early filtering
- Typical overhead: <5% of ingestion time
- Memory: O(n) for entity cache during ingestion
