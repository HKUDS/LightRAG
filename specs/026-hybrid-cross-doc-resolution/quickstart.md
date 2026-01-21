# Quickstart: Hybrid Cross-Document Entity Resolution

**Phase**: 1 - Design
**Date**: 2025-01-21

## Overview

This feature optimizes cross-document entity resolution for large knowledge graphs by implementing a hybrid approach that automatically switches between:

- **Full mode**: Maximum precision fuzzy matching (best for small graphs)
- **VDB mode**: Fast vector-based candidate selection (best for large graphs)

## Configuration

### Environment Variables

Add to your `.env` file:

```bash
# Resolution mode: "full" | "vdb" | "hybrid" | "disabled"
CROSS_DOC_RESOLUTION_MODE=hybrid

# Entity count threshold for hybrid mode switching
CROSS_DOC_THRESHOLD_ENTITIES=5000

# Number of VDB candidates in VDB mode
CROSS_DOC_VDB_TOP_K=10
```

### Mode Descriptions

| Mode | Use Case | Performance | Precision |
|------|----------|-------------|-----------|
| `hybrid` | **Production (recommended)** | Auto-adaptive | Best of both |
| `full` | Small graphs, max quality | Slow for large graphs | Maximum |
| `vdb` | Large graphs, consistent speed | Fast | High (90%+) |
| `disabled` | Testing, debugging | Instant | None |

## Usage

### Default Behavior (No Configuration)

By default, hybrid mode is enabled:

```python
from lightrag import LightRAG

# Hybrid mode auto-enabled
rag = LightRAG(workspace="my_workspace")

# Indexing uses:
# - Full matching if <5000 entities
# - VDB matching if >=5000 entities
await rag.ainsert(documents)
```

### Force Specific Mode

```python
import os

# Force full matching (max quality)
os.environ["CROSS_DOC_RESOLUTION_MODE"] = "full"

# Force VDB matching (max speed)
os.environ["CROSS_DOC_RESOLUTION_MODE"] = "vdb"

# Disable resolution (testing)
os.environ["CROSS_DOC_RESOLUTION_MODE"] = "disabled"
```

### Adjust Threshold

```python
import os

# Lower threshold for earlier VDB switching
os.environ["CROSS_DOC_THRESHOLD_ENTITIES"] = "2000"

# Higher threshold for more full matching
os.environ["CROSS_DOC_THRESHOLD_ENTITIES"] = "10000"
```

## Monitoring

### Log Output

Resolution metrics are logged at INFO level:

```
PERF cross_doc_resolution mode=vdb entities=47 duplicates=3 time_ms=234.5
```

### Interpreting Metrics

| Metric | Description |
|--------|-------------|
| `mode` | Actual mode used (full/vdb) |
| `entities` | New entities processed |
| `duplicates` | Entities merged with existing |
| `time_ms` | Processing time |

## Troubleshooting

### Resolution Taking Too Long

**Symptom**: Cross-doc resolution >5 seconds per document

**Solution**: Check entity count and adjust threshold:
```bash
# If graph has >5000 entities but using full mode
CROSS_DOC_RESOLUTION_MODE=vdb
# Or lower hybrid threshold
CROSS_DOC_THRESHOLD_ENTITIES=3000
```

### Duplicate Entities Appearing

**Symptom**: Similar entities not being merged

**Solution**: Check similarity threshold or force full mode:
```bash
# Lower similarity threshold (more aggressive matching)
ENTITY_SIMILARITY_THRESHOLD=0.80

# Force full matching for max precision
CROSS_DOC_RESOLUTION_MODE=full
```

### VDB Mode Not Activating

**Symptom**: Logs show "mode=full" even with large graph

**Solution**: Verify entity count and threshold:
```python
# Check actual entity count
count = await rag.chunk_entity_relation_graph.get_node_count()
print(f"Entity count: {count}")

# Verify threshold
print(f"Threshold: {os.getenv('CROSS_DOC_THRESHOLD_ENTITIES', 5000)}")
```

## Performance Expectations

| Graph Size | Mode | Time per Document |
|------------|------|-------------------|
| <5K entities | full (hybrid) | ~2-3 seconds |
| 5K-10K entities | vdb (hybrid) | ~0.5-1 second |
| 10K-50K entities | vdb | ~0.5-1 second |
| >50K entities | vdb | ~1-2 seconds |

## Next Steps

- See [design.md](design.md) for technical implementation details
- See [spec.md](spec.md) for full requirements
- See [research.md](research.md) for design decisions
