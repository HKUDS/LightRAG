# Hybrid Cross-Document Entity Resolution

## Problem Statement

Current cross-document entity resolution uses brute-force fuzzy matching with O(n × m) complexity:
- n = new entities from current document
- m = existing entities in the graph

**Performance impact observed:**
- 11,000 entities in graph → ~7.6 seconds per document for cross-doc resolution
- 300 documents took ~10 hours to index
- Cross-doc resolution becomes the bottleneck as the graph grows

## Current Implementation

Location: `lightrag/operate.py` - `_resolve_cross_document_entities()` (line ~2471)

```python
# Current algorithm (simplified)
existing_nodes = await knowledge_graph_inst.get_all_nodes()  # Get ALL entities

for entity_name in all_nodes:                               # For each new entity
    for existing_name in existing_by_type[entity_type]:     # Compare with ALL existing
        score = compute_entity_similarity(entity_name, existing_name)  # O(m)
```

**Complexity:** O(n × m) where m grows unboundedly

## Proposed Solution: Hybrid Approach

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Cross-Doc Resolution                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   entities_count < THRESHOLD      entities_count >= THRESHOLD│
│          │                              │                    │
│          ▼                              ▼                    │
│   ┌─────────────┐                ┌─────────────┐            │
│   │    FULL     │                │     VDB     │            │
│   │   Fuzzy     │                │  Assisted   │            │
│   │  Matching   │                │  Matching   │            │
│   └─────────────┘                └─────────────┘            │
│          │                              │                    │
│   O(n × m) comparisons           O(n × k) + VDB lookup      │
│   High precision                 Good precision              │
│   ~500ms/entity                  ~10ms/entity                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Configuration Options

```python
# New configuration parameters
CROSS_DOC_RESOLUTION_MODE = "hybrid"  # "full" | "vdb" | "hybrid" | "disabled"
CROSS_DOC_THRESHOLD_ENTITIES = 5000   # Switch to VDB mode after this many entities
CROSS_DOC_VDB_TOP_K = 10              # Number of VDB candidates to check
```

### Mode Comparison

| Mode     | Precision    | Performance      | Use Case                    |
|----------|-------------|------------------|----------------------------|
| full     | ⭐⭐⭐⭐⭐    | 🐢 Slow          | Small graphs, max quality  |
| vdb      | ⭐⭐⭐⭐      | 🚀 Fast          | Large graphs               |
| hybrid   | ⭐⭐⭐⭐⭐→⭐⭐⭐⭐ | 🚀 Auto-adaptive | Production                 |
| disabled | N/A          | ⚡ Instant       | Debug/tests                |

## VDB-Assisted Algorithm

### How It Works

1. **For each new entity**, get its embedding (already computed during indexing)
2. **Query VDB** for top-K similar entities using approximate nearest neighbor (ANN)
   - VDB uses HNSW or IVF algorithms → O(log m) instead of O(m)
3. **Fuzzy match only against the K candidates** returned by VDB
4. **Resolve to best match** if score exceeds threshold

### Implementation Sketch

```python
async def _resolve_cross_document_entities_vdb(
    all_nodes: dict,
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    global_config: dict,
) -> tuple[dict, dict]:
    """Cross-doc resolution using VDB for candidate selection."""

    embedding_func = global_config["embedding_func"]
    similarity_threshold = global_config.get("entity_similarity_threshold", 0.85)
    top_k = global_config.get("cross_doc_vdb_top_k", 10)

    resolved_nodes = defaultdict(list)
    resolution_map = {}

    for entity_name, entities in all_nodes.items():
        entity_type = entities[0].get("entity_type", "UNKNOWN").upper()

        # 1. Get embedding for entity name
        entity_embedding = await embedding_func([entity_name])

        # 2. Query VDB for top-K similar entities (O(log m))
        candidates = await entity_vdb.query(
            query_embedding=entity_embedding[0],
            top_k=top_k,
            filter={"entity_type": entity_type},  # Same type only
        )

        # 3. Fuzzy match only against candidates (O(k))
        best_match = None
        best_score = 0.0

        for candidate in candidates:
            candidate_name = candidate["entity_name"]
            score = compute_entity_similarity(entity_name, candidate_name)

            if score >= similarity_threshold and score > best_score:
                best_match = candidate_name
                best_score = score

        # 4. Resolve or keep original
        if best_match and best_match != entity_name:
            resolution_map[entity_name] = (best_match, best_score)
            resolved_nodes[best_match].extend(entities)
        else:
            resolved_nodes[entity_name].extend(entities)

    return dict(resolved_nodes), resolution_map
```

### Hybrid Mode Logic

```python
async def _resolve_cross_document_entities_hybrid(
    all_nodes: dict,
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    global_config: dict,
) -> tuple[dict, dict]:
    """Hybrid resolution: full mode for small graphs, VDB for large."""

    threshold = global_config.get("cross_doc_threshold_entities", 5000)

    # Count existing entities
    existing_count = await knowledge_graph_inst.get_node_count()

    if existing_count < threshold:
        # Small graph: use full fuzzy matching for maximum quality
        return await _resolve_cross_document_entities_full(
            all_nodes, knowledge_graph_inst, global_config
        )
    else:
        # Large graph: use VDB-assisted for performance
        return await _resolve_cross_document_entities_vdb(
            all_nodes, knowledge_graph_inst, entity_vdb, global_config
        )
```

## Advanced Options

### Option A: Fixed Threshold (Simple)

```python
CROSS_DOC_THRESHOLD_ENTITIES = 5000
```

### Option B: Per-Workspace Threshold (Flexible)

```python
async def get_threshold(workspace_id: str) -> int:
    # Small clients: max quality
    # Large clients: performance
    return workspace_config.get("cross_doc_threshold", 5000)
```

### Option C: Gradual Transition

```python
# Avoid abrupt quality change at threshold
if count < 3000:
    mode = "full"
elif count < 7000:
    # Mix: high-confidence matches via VDB, others via full
    mode = "gradual"
else:
    mode = "vdb"
```

## Metrics to Track

```python
@dataclass
class CrossDocMetrics:
    mode_used: str              # "full" | "vdb" | "hybrid"
    entities_checked: int
    duplicates_found: int
    time_ms: float
    comparisons_count: int
    comparisons_per_sec: float
```

## Impact on Billing

The VDB mode also reduces LLM summarization calls (fewer merged entities = fewer summaries):

| Mode              | LLM Summarize Calls | Estimated Tokens |
|-------------------|---------------------|------------------|
| full (5K entities) | ~2500               | ~500K            |
| vdb (5K entities)  | ~500                | ~100K            |
| disabled          | 0                   | 0                |

## Implementation Checklist

- [ ] Add configuration parameters to `LightRAG` class
- [ ] Implement `_resolve_cross_document_entities_vdb()` function
- [ ] Implement `_resolve_cross_document_entities_hybrid()` wrapper
- [ ] Add `get_node_count()` method to graph storage backends
- [ ] Add metrics logging for cross-doc resolution
- [ ] Update documentation
- [ ] Add tests for all modes
- [ ] Performance benchmarks

## Files to Modify

1. **lightrag/lightrag.py** - Add configuration parameters
2. **lightrag/operate.py** - Implement VDB and hybrid resolution
3. **lightrag/kg/*.py** - Add `get_node_count()` method
4. **tests/test_entity_resolution.py** - Add tests

## References

- Current implementation: `lightrag/operate.py:2471` (`_resolve_cross_document_entities`)
- Entity VDB: `lightrag/lightrag.py` - `entity_vdb` storage
- Similarity computation: `lightrag/entity_resolution.py` - `compute_entity_similarity()`
