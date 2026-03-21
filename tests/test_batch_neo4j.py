"""Tests for Neo4JStorage batch upsert methods."""
import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def neo4j_storage():
    from lightrag.kg.neo4j_impl import Neo4JStorage

    storage = Neo4JStorage.__new__(Neo4JStorage)
    storage._DATABASE = "neo4j"
    storage.workspace = "test"

    # Mock the driver and session
    mock_tx = AsyncMock()
    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    async def _exec_write(fn):
        return await fn(mock_tx)

    mock_session.execute_write = AsyncMock(side_effect=_exec_write)

    mock_driver = MagicMock()
    mock_driver.session = MagicMock(return_value=mock_session)

    storage._driver = mock_driver
    storage._get_workspace_label = MagicMock(return_value="test_workspace")

    return storage, mock_tx


@pytest.mark.asyncio
async def test_batch_upsert_nodes_unwind(neo4j_storage):
    storage, mock_tx = neo4j_storage
    mock_tx.run = AsyncMock()

    nodes = [
        ("alice", {"entity_id": "alice", "entity_type": "PERSON", "description": "Alice"}),
        ("bob", {"entity_id": "bob", "entity_type": "PERSON", "description": "Bob"}),
        ("acme", {"entity_id": "acme", "entity_type": "ORG", "description": "Acme Corp"}),
    ]
    await storage.batch_upsert_nodes(nodes)

    # Should group by entity_type: 2 PERSON nodes in one call, 1 ORG in another
    assert mock_tx.run.call_count == 2
    for c in mock_tx.run.call_args_list:
        query = c[0][0]
        assert "UNWIND" in query
        assert "MERGE" in query


@pytest.mark.asyncio
async def test_batch_upsert_nodes_empty(neo4j_storage):
    storage, mock_tx = neo4j_storage
    mock_tx.run = AsyncMock()
    await storage.batch_upsert_nodes([])
    mock_tx.run.assert_not_called()


@pytest.mark.asyncio
async def test_batch_upsert_edges_unwind(neo4j_storage):
    storage, mock_tx = neo4j_storage
    mock_result = AsyncMock()
    mock_result.consume = AsyncMock()
    mock_tx.run = AsyncMock(return_value=mock_result)

    edges = [
        ("alice", "bob", {"weight": "1.0", "description": "knows"}),
        ("alice", "acme", {"weight": "0.5", "description": "works_at"}),
    ]
    await storage.batch_upsert_edges(edges)

    assert mock_tx.run.call_count == 1
    query = mock_tx.run.call_args[0][0]
    assert "UNWIND" in query
    assert "MERGE" in query
    assert "DIRECTED" in query


@pytest.mark.asyncio
async def test_batch_upsert_edges_empty(neo4j_storage):
    storage, mock_tx = neo4j_storage
    mock_tx.run = AsyncMock()
    await storage.batch_upsert_edges([])
    mock_tx.run.assert_not_called()


@pytest.mark.asyncio
async def test_batch_upsert_nodes_large_batch_chunked(neo4j_storage):
    """Batches larger than _BATCH_CHUNK_SIZE should be split into sub-batches."""
    storage, mock_tx = neo4j_storage
    mock_tx.run = AsyncMock()

    nodes = [
        (f"node_{i}", {"entity_id": f"node_{i}", "entity_type": "ITEM", "description": f"Item {i}"})
        for i in range(1200)
    ]
    await storage.batch_upsert_nodes(nodes)

    # 1200 / 500 = 3 sub-batches
    assert mock_tx.run.call_count == 3


@pytest.mark.asyncio
async def test_has_nodes_batch(neo4j_storage):
    storage, mock_tx = neo4j_storage

    # Mock the session for read access
    mock_records = [{"entity_id": "alice"}, {"entity_id": "charlie"}]

    class MockResult:
        def __aiter__(self):
            return self

        def __init__(self, records):
            self._records = iter(records)

        async def __anext__(self):
            try:
                return next(self._records)
            except StopIteration:
                raise StopAsyncIteration

        async def consume(self):
            pass

    mock_result = MockResult(mock_records)

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.run = AsyncMock(return_value=mock_result)

    storage._driver.session = MagicMock(return_value=mock_session)

    result = await storage.has_nodes_batch(["alice", "bob", "charlie"])
    assert result == {"alice", "charlie"}
    # Verify UNWIND query was used
    query = mock_session.run.call_args[0][0]
    assert "UNWIND" in query


@pytest.mark.asyncio
async def test_has_nodes_batch_empty(neo4j_storage):
    storage, mock_tx = neo4j_storage
    result = await storage.has_nodes_batch([])
    assert result == set()
