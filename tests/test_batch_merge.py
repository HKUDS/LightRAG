"""Tests that merge phase uses batch upsert instead of individual calls."""

import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def global_config():
    return {
        "source_ids_limit_method": "FIFO",
        "max_source_ids_per_entity": 100,
        "max_source_ids_per_relation": 100,
        "max_file_paths": 10,
        "file_path_more_placeholder": "more",
        "use_llm_func": AsyncMock(),
        "entity_summary_to_max_tokens": 500,
        "summary_language": "English",
    }


@pytest.fixture
def mock_kg():
    kg = AsyncMock()
    kg.get_node = AsyncMock(return_value=None)
    kg.upsert_node = AsyncMock()
    kg.has_edge = AsyncMock(return_value=False)
    kg.get_edge = AsyncMock(return_value=None)
    kg.upsert_edge = AsyncMock()
    return kg


@pytest.fixture
def mock_entity_chunks():
    chunks = AsyncMock()
    chunks.get_by_id = AsyncMock(return_value=None)
    return chunks


@pytest.mark.asyncio
async def test_merge_nodes_returns_data_without_graph_call(mock_kg, mock_entity_chunks, global_config):
    from lightrag.operate import _merge_nodes_then_upsert

    nodes_data = [
        {
            "entity_type": "PERSON",
            "description": "A test entity",
            "source_id": "chunk-1",
            "file_path": "test.txt",
        }
    ]

    result = await _merge_nodes_then_upsert(
        entity_name="TEST_ENTITY",
        nodes_data=nodes_data,
        knowledge_graph_inst=mock_kg,
        entity_vdb=None,
        global_config=global_config,
        pipeline_status=None,
        pipeline_status_lock=None,
        llm_response_cache=None,
        entity_chunks_storage=mock_entity_chunks,
    )

    assert result is not None
    assert result["entity_name"] == "TEST_ENTITY"
    assert result["entity_type"] == "PERSON"
    mock_kg.upsert_node.assert_not_called()


@pytest.mark.asyncio
async def test_merge_edges_returns_data_without_graph_call(mock_kg, global_config):
    from lightrag.operate import _merge_edges_then_upsert

    mock_kg.get_node = AsyncMock(
        return_value={
            "entity_id": "EXISTS",
            "entity_type": "THING",
            "source_id": "chunk-1",
            "description": "existing",
            "file_path": "test.txt",
        }
    )

    edges_data = [
        {
            "description": "A relates to B",
            "keywords": "test",
            "weight": 1.0,
            "source_id": "chunk-1",
            "file_path": "test.txt",
        }
    ]

    result = await _merge_edges_then_upsert(
        src_id="ENTITY_A",
        tgt_id="ENTITY_B",
        edges_data=edges_data,
        knowledge_graph_inst=mock_kg,
        relationships_vdb=None,
        entity_vdb=None,
        global_config=global_config,
        pipeline_status=None,
        pipeline_status_lock=None,
        llm_response_cache=None,
        added_entities=None,
        relation_chunks_storage=None,
        entity_chunks_storage=None,
    )

    assert result is not None
    assert result["src_id"] == "ENTITY_A"
    assert result["tgt_id"] == "ENTITY_B"
    mock_kg.upsert_edge.assert_not_called()


@pytest.mark.asyncio
async def test_skip_graph_upsert_flag_on_early_return():
    from lightrag.operate import _merge_nodes_then_upsert

    mock_kg = AsyncMock()
    mock_kg.get_node = AsyncMock(
        return_value={
            "entity_id": "TEST",
            "entity_type": "PERSON",
            "description": "existing",
            "source_id": "c1<SEP>c2<SEP>c3",
            "file_path": "test.txt",
        }
    )

    mock_entity_chunks = AsyncMock()
    mock_entity_chunks.get_by_id = AsyncMock(
        return_value={"chunk_ids": ["c1", "c2", "c3"], "count": 3}
    )

    global_config = {
        "source_ids_limit_method": "KEEP",
        "max_source_ids_per_entity": 3,
        "max_file_paths": 10,
        "file_path_more_placeholder": "more",
        "use_llm_func": AsyncMock(),
        "entity_summary_to_max_tokens": 500,
        "summary_language": "English",
    }

    nodes_data = [
        {
            "entity_type": "PERSON",
            "description": "new desc",
            "source_id": "c-new",
            "file_path": "new.txt",
        }
    ]

    result = await _merge_nodes_then_upsert(
        entity_name="TEST",
        nodes_data=nodes_data,
        knowledge_graph_inst=mock_kg,
        entity_vdb=None,
        global_config=global_config,
        pipeline_status=None,
        pipeline_status_lock=None,
        llm_response_cache=None,
        entity_chunks_storage=mock_entity_chunks,
    )

    assert result is not None
    assert result.get("_skip_graph_upsert") is True
