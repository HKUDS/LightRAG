from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from lightrag.base import QueryParam
from lightrag.operate import (
    _find_related_text_unit_from_entities,
    _find_related_text_unit_from_relations,
)


pytestmark = pytest.mark.offline


def _text_chunks_db():
    return SimpleNamespace(
        global_config={
            "kg_chunk_pick_method": "VECTOR",
            "related_chunk_number": 1,
        },
        embedding_func=Mock(),
        get_by_ids=AsyncMock(return_value=[{"content": "chunk"}]),
    )


async def _vector_picker(**kwargs):
    if kwargs["num_of_chunks"] <= 0:
        return []
    return ["chunk-1"]


@pytest.mark.asyncio
async def test_single_entity_uses_vector_picker_with_one_chunk(monkeypatch):
    vector_picker = AsyncMock(side_effect=_vector_picker)
    weighted_picker = Mock(return_value=["weighted-chunk"])
    monkeypatch.setattr("lightrag.operate.pick_by_vector_similarity", vector_picker)
    monkeypatch.setattr("lightrag.operate.pick_by_weighted_polling", weighted_picker)

    result = await _find_related_text_unit_from_entities(
        [{"entity_name": "entity", "source_id": "chunk-1"}],
        QueryParam(),
        _text_chunks_db(),
        SimpleNamespace(),
        query="query",
        chunks_vdb=SimpleNamespace(),
        query_embedding=[1.0],
    )

    assert vector_picker.await_args.kwargs["num_of_chunks"] == 1
    weighted_picker.assert_not_called()
    assert [chunk["chunk_id"] for chunk in result] == ["chunk-1"]


@pytest.mark.asyncio
async def test_single_relation_uses_vector_picker_with_one_chunk(monkeypatch):
    vector_picker = AsyncMock(side_effect=_vector_picker)
    weighted_picker = Mock(return_value=["weighted-chunk"])
    monkeypatch.setattr("lightrag.operate.pick_by_vector_similarity", vector_picker)
    monkeypatch.setattr("lightrag.operate.pick_by_weighted_polling", weighted_picker)

    result = await _find_related_text_unit_from_relations(
        [{"src_id": "source", "tgt_id": "target", "source_id": "chunk-1"}],
        QueryParam(),
        _text_chunks_db(),
        entity_chunks=[],
        query="query",
        chunks_vdb=SimpleNamespace(),
        query_embedding=[1.0],
    )

    assert vector_picker.await_args.kwargs["num_of_chunks"] == 1
    weighted_picker.assert_not_called()
    assert [chunk["chunk_id"] for chunk in result] == ["chunk-1"]


@pytest.mark.asyncio
async def test_zero_related_chunk_limit_remains_disabled(monkeypatch):
    vector_picker = AsyncMock(side_effect=_vector_picker)
    weighted_picker = Mock(return_value=[])
    monkeypatch.setattr("lightrag.operate.pick_by_vector_similarity", vector_picker)
    monkeypatch.setattr("lightrag.operate.pick_by_weighted_polling", weighted_picker)
    text_chunks_db = _text_chunks_db()
    text_chunks_db.global_config["related_chunk_number"] = 0

    result = await _find_related_text_unit_from_entities(
        [{"entity_name": "entity", "source_id": "chunk-1"}],
        QueryParam(),
        text_chunks_db,
        SimpleNamespace(),
        query="query",
        chunks_vdb=SimpleNamespace(),
        query_embedding=[1.0],
    )

    assert vector_picker.await_args.kwargs["num_of_chunks"] == 0
    assert result == []
