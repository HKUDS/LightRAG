# test_skip_unchanged_summary.py

import pytest
from unittest.mock import AsyncMock, patch
from lightrag.utils import GRAPH_FIELD_SEP


def _make_global_config(**overrides):
    cfg = {
        "source_ids_limit_method": "FIFO",
        "max_source_ids_per_entity": 100,
        "max_file_paths": 50,
        "file_path_more_placeholder": "more",
    }
    cfg.update(overrides)
    return cfg


@pytest.fixture
def mock_kg_with_existing_person():
    kg = AsyncMock()
    kg.get_node = AsyncMock(return_value={
        "entity_type": "PERSON",
        "source_id": "doc1",
        "file_path": "file1.pdf",
        "description": f"Alice is a person{GRAPH_FIELD_SEP}Alice works at Acme",
    })
    kg.upsert_node = AsyncMock()
    return kg


@pytest.mark.asyncio
async def test_skip_summary_when_no_new_descriptions(mock_kg_with_existing_person):
    from lightrag.operate import _merge_nodes_then_upsert

    nodes_data = [
        {
            "entity_type": "PERSON",
            "source_id": "doc1",
            "file_path": "file1.pdf",
            "description": "Alice is a person",
        },
        {
            "entity_type": "PERSON",
            "source_id": "doc1",
            "file_path": "file1.pdf",
            "description": "Alice works at Acme",
        },
    ]

    with patch("lightrag.operate._handle_entity_relation_summary") as mock_summary:
        await _merge_nodes_then_upsert(
            entity_name="Alice",
            nodes_data=nodes_data,
            knowledge_graph_inst=mock_kg_with_existing_person,
            entity_vdb=None,
            global_config=_make_global_config(),
        )
        mock_summary.assert_not_called()

    mock_kg_with_existing_person.upsert_node.assert_called_once()


@pytest.mark.asyncio
async def test_calls_summary_when_new_descriptions(mock_kg_with_existing_person):
    from lightrag.operate import _merge_nodes_then_upsert

    nodes_data = [
        {
            "entity_type": "PERSON",
            "source_id": "doc2",
            "file_path": "file2.pdf",
            "description": "Alice is the CEO of Acme",
        },
    ]

    with patch("lightrag.operate._handle_entity_relation_summary") as mock_summary:
        mock_summary.return_value = ("Alice is a person and CEO of Acme", True)
        await _merge_nodes_then_upsert(
            entity_name="Alice",
            nodes_data=nodes_data,
            knowledge_graph_inst=mock_kg_with_existing_person,
            entity_vdb=None,
            global_config=_make_global_config(),
        )
        mock_summary.assert_called_once()


@pytest.mark.asyncio
async def test_calls_summary_for_new_entity():
    from lightrag.operate import _merge_nodes_then_upsert

    mock_kg = AsyncMock()
    mock_kg.get_node = AsyncMock(return_value=None)
    mock_kg.upsert_node = AsyncMock()

    nodes_data = [
        {
            "entity_type": "ORG",
            "source_id": "doc1",
            "file_path": "file1.pdf",
            "description": "Acme Corp makes widgets",
        },
    ]

    with patch("lightrag.operate._handle_entity_relation_summary") as mock_summary:
        mock_summary.return_value = ("Acme Corp makes widgets", True)
        await _merge_nodes_then_upsert(
            entity_name="Acme",
            nodes_data=nodes_data,
            knowledge_graph_inst=mock_kg,
            entity_vdb=None,
            global_config=_make_global_config(),
        )
        mock_summary.assert_called_once()


@pytest.mark.asyncio
async def test_skip_preserves_existing_description(mock_kg_with_existing_person):
    from lightrag.operate import _merge_nodes_then_upsert

    existing_desc = f"Alice is a person{GRAPH_FIELD_SEP}Alice works at Acme"

    nodes_data = [
        {
            "entity_type": "PERSON",
            "source_id": "doc1",
            "file_path": "file1.pdf",
            "description": "Alice is a person",
        },
    ]

    with patch("lightrag.operate._handle_entity_relation_summary"):
        await _merge_nodes_then_upsert(
            entity_name="Alice",
            nodes_data=nodes_data,
            knowledge_graph_inst=mock_kg_with_existing_person,
            entity_vdb=None,
            global_config=_make_global_config(),
        )

    call_kwargs = mock_kg_with_existing_person.upsert_node.call_args
    upserted_data = call_kwargs.kwargs.get("node_data", call_kwargs[1].get("node_data"))
    assert upserted_data["description"] == existing_desc
