# Contract: BaseGraphStorage Extension

**Phase**: 1 - Design
**Date**: 2025-01-21

## New Abstract Method

### get_node_count()

Returns the total number of nodes (entities) in the graph for the current workspace.

```python
@abstractmethod
async def get_node_count(self) -> int:
    """Get the total count of nodes in the graph.

    Returns:
        int: Number of nodes in the graph for this workspace.

    Notes:
        - MUST be O(1) complexity
        - MUST be workspace-scoped (count only nodes in current workspace)
        - MUST return 0 for empty graphs (not raise exception)
    """
    pass
```

## Implementation Requirements

### PostgreSQL (`postgres_impl.py`)

```python
async def get_node_count(self) -> int:
    query = "SELECT count(*) FROM entity_nodes WHERE workspace = $1"
    result = await self.pool.fetchval(query, self.workspace)
    return result or 0
```

**Performance**: O(1) with workspace index on entity_nodes table.

### MongoDB (`mongo_impl.py`)

```python
async def get_node_count(self) -> int:
    return await self.collection.count_documents({})
```

**Performance**: O(1) - MongoDB maintains collection statistics.

### NetworkX (`networkx_impl.py`)

```python
async def get_node_count(self) -> int:
    return len(self._graph.nodes())
```

**Performance**: O(1) - NetworkX Graph tracks node count.

### Neo4j (`neo4j_impl.py`)

```python
async def get_node_count(self) -> int:
    query = "MATCH (n) RETURN count(n) as count"
    result = await self.session.run(query)
    record = await result.single()
    return record["count"] if record else 0
```

**Performance**: O(1) - Neo4j maintains count store.

## Test Contract

```python
@pytest.mark.asyncio
async def test_get_node_count_empty_graph():
    """Empty graph returns 0."""
    storage = create_test_storage()
    assert await storage.get_node_count() == 0

@pytest.mark.asyncio
async def test_get_node_count_with_nodes():
    """Count reflects actual nodes."""
    storage = create_test_storage()
    await storage.upsert_node("entity1", {"type": "PERSON"})
    await storage.upsert_node("entity2", {"type": "ORG"})
    assert await storage.get_node_count() == 2

@pytest.mark.asyncio
async def test_get_node_count_workspace_isolation():
    """Count is workspace-scoped."""
    storage_a = create_test_storage(workspace="workspace_a")
    storage_b = create_test_storage(workspace="workspace_b")

    await storage_a.upsert_node("entity1", {"type": "PERSON"})
    await storage_b.upsert_node("entity2", {"type": "ORG"})
    await storage_b.upsert_node("entity3", {"type": "ORG"})

    assert await storage_a.get_node_count() == 1
    assert await storage_b.get_node_count() == 2
```
