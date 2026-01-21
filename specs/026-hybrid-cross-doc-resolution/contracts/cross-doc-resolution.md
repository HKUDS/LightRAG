# Contract: Cross-Document Resolution Functions

**Phase**: 1 - Design
**Date**: 2025-01-21

## Function Signatures

### _resolve_cross_document_entities_vdb()

VDB-assisted resolution using semantic similarity for candidate selection.

```python
async def _resolve_cross_document_entities_vdb(
    all_nodes: dict[str, list[dict]],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    global_config: dict,
) -> tuple[dict[str, list[dict]], dict[str, tuple[str, float]]]:
    """Cross-doc resolution using VDB for candidate selection.

    Args:
        all_nodes: Dict mapping entity names to list of entity data dicts.
        knowledge_graph_inst: Graph storage instance for the workspace.
        entity_vdb: Vector storage for entity embeddings.
        global_config: Global configuration dict containing:
            - embedding_func: Function to generate embeddings
            - entity_similarity_threshold: Minimum score for matching (default: 0.85)
            - cross_doc_vdb_top_k: Number of VDB candidates (default: 10)

    Returns:
        Tuple of:
        - resolved_nodes: Dict mapping resolved names to merged entity lists
        - resolution_map: Dict mapping original names to (resolved_name, score)

    Raises:
        No exceptions - falls back gracefully on errors.
    """
```

### _resolve_cross_document_entities_hybrid()

Hybrid resolution that automatically selects mode based on graph size.

```python
async def _resolve_cross_document_entities_hybrid(
    all_nodes: dict[str, list[dict]],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    global_config: dict,
) -> tuple[dict[str, list[dict]], dict[str, tuple[str, float]]]:
    """Hybrid resolution: full mode for small graphs, VDB for large.

    Args:
        all_nodes: Dict mapping entity names to list of entity data dicts.
        knowledge_graph_inst: Graph storage instance.
        entity_vdb: Vector storage for entity embeddings.
        global_config: Global configuration dict containing:
            - cross_doc_threshold_entities: Switch threshold (default: 5000)
            - (plus all args from _resolve_cross_document_entities_vdb)

    Returns:
        Same as _resolve_cross_document_entities_vdb.

    Behavior:
        - If node_count < threshold: uses full fuzzy matching
        - If node_count >= threshold: uses VDB-assisted matching
    """
```

## Configuration Contract

### Environment Variables

| Variable | Type | Default | Validation |
|----------|------|---------|------------|
| `CROSS_DOC_RESOLUTION_MODE` | str | "hybrid" | Must be: full, vdb, hybrid, disabled |
| `CROSS_DOC_THRESHOLD_ENTITIES` | int | 5000 | Must be > 0 |
| `CROSS_DOC_VDB_TOP_K` | int | 10 | Must be 1-100 |

### Constants (lightrag/constants.py)

```python
DEFAULT_CROSS_DOC_RESOLUTION_MODE = "hybrid"
DEFAULT_CROSS_DOC_THRESHOLD_ENTITIES = 5000
DEFAULT_CROSS_DOC_VDB_TOP_K = 10
```

## Logging Contract

### Performance Log Format

```
PERF cross_doc_resolution mode={mode} entities={count} duplicates={count} time_ms={ms}
```

Example:
```
PERF cross_doc_resolution mode=vdb entities=47 duplicates=3 time_ms=234.5
```

### Log Levels

| Event | Level | When |
|-------|-------|------|
| Mode switch in hybrid | INFO | Threshold crossed |
| Resolution complete | INFO | Always (PERF log) |
| VDB query failure | WARNING | Fallback to full |
| Embedding failure | WARNING | Entity skipped |

## Test Contract

```python
@pytest.mark.asyncio
async def test_vdb_resolution_basic():
    """VDB mode resolves similar entities."""
    all_nodes = {
        "Apple Inc": [{"entity_type": "ORG"}],
        "Apple Incorporated": [{"entity_type": "ORG"}],  # Should merge
    }
    # Mock VDB to return "Apple Inc" as candidate for "Apple Incorporated"
    resolved, map = await _resolve_cross_document_entities_vdb(...)
    assert "Apple Inc" in resolved
    assert "Apple Incorporated" not in resolved
    assert map["Apple Incorporated"][0] == "Apple Inc"

@pytest.mark.asyncio
async def test_hybrid_uses_full_below_threshold():
    """Hybrid mode uses full matching for small graphs."""
    # Mock get_node_count() to return 1000 (below 5000 threshold)
    resolved, _ = await _resolve_cross_document_entities_hybrid(...)
    # Verify full matching was used (check log or mock)

@pytest.mark.asyncio
async def test_hybrid_uses_vdb_above_threshold():
    """Hybrid mode uses VDB matching for large graphs."""
    # Mock get_node_count() to return 10000 (above 5000 threshold)
    resolved, _ = await _resolve_cross_document_entities_hybrid(...)
    # Verify VDB matching was used (check log or mock)

@pytest.mark.asyncio
async def test_disabled_mode_skips_resolution():
    """Disabled mode returns entities unchanged."""
    # Set mode to "disabled"
    resolved, map = await _resolve_cross_document_entities(...)
    assert len(map) == 0  # No resolutions
```
