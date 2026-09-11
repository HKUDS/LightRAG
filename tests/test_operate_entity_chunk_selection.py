"""Budget parity between the entity and relation chunk-selection paths.

Both paths deduplicate each group's chunk list, both can empty a group that
way, and both size their chunk budget from the surviving group count. The
entity path used to keep the emptied groups, so phantom groups inflated the
WEIGHT and VECTOR budgets and over-delivered beyond ``related_chunk_number``.
"""

from unittest.mock import AsyncMock, patch

import pytest

from lightrag.base import QueryParam
from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.operate import (
    _find_related_text_unit_from_entities,
    _find_related_text_unit_from_relations,
)


pytestmark = pytest.mark.offline

CHUNK_IDS = [f"chunk-{index}" for index in range(20)]
SOURCE_ID = GRAPH_FIELD_SEP.join(CHUNK_IDS)
RELATED_CHUNK_NUMBER = 2
DUPLICATE_GROUPS = 5


def _text_chunks_db(pick_method: str) -> AsyncMock:
    text_chunks_db = AsyncMock()
    text_chunks_db.global_config = {
        "kg_chunk_pick_method": pick_method,
        "related_chunk_number": RELATED_CHUNK_NUMBER,
    }
    text_chunks_db.get_by_ids.side_effect = lambda chunk_ids: [
        {"content": f"content-{chunk_id}"} for chunk_id in chunk_ids
    ]
    return text_chunks_db


def _duplicate_entities() -> list[dict]:
    """Entities that all carry the same chunks, so one group survives dedup."""
    return [
        {"entity_name": f"entity-{index}", "source_id": SOURCE_ID}
        for index in range(DUPLICATE_GROUPS)
    ]


def _duplicate_relations() -> list[dict]:
    """The relation-path twin of :func:`_duplicate_entities`."""
    return [
        {"src_tgt": (f"src-{index}", f"tgt-{index}"), "source_id": SOURCE_ID}
        for index in range(DUPLICATE_GROUPS)
    ]


async def _select_from_entities(text_chunks_db: AsyncMock, **kwargs) -> list[dict]:
    return await _find_related_text_unit_from_entities(
        _duplicate_entities(),
        QueryParam(),
        text_chunks_db,
        AsyncMock(),
        **kwargs,
    )


async def _select_from_relations(text_chunks_db: AsyncMock, **kwargs) -> list[dict]:
    return await _find_related_text_unit_from_relations(
        _duplicate_relations(),
        QueryParam(),
        text_chunks_db,
        entity_chunks=[],
        **kwargs,
    )


def _vector_call_args(picker: AsyncMock) -> tuple[int, int]:
    """Return the (budget, group count) a path handed to the vector picker."""
    return (
        picker.call_args.kwargs["num_of_chunks"],
        len(picker.call_args.kwargs["entity_info"]),
    )


async def test_weighted_polling_sees_only_surviving_entity_groups():
    with patch(
        "lightrag.operate.pick_by_weighted_polling",
        return_value=CHUNK_IDS[:RELATED_CHUNK_NUMBER],
    ) as weighted_picker:
        result = await _select_from_entities(_text_chunks_db("WEIGHT"))

    groups, max_related_chunks = weighted_picker.call_args.args[:2]
    assert len(groups) == 1
    assert groups[0]["sorted_chunks"] == CHUNK_IDS
    assert max_related_chunks == RELATED_CHUNK_NUMBER
    assert [chunk["chunk_id"] for chunk in result] == CHUNK_IDS[:RELATED_CHUNK_NUMBER]


async def test_entity_path_honours_the_configured_weight_budget():
    """The user-visible symptom: phantom groups used to deliver 8, not 2."""
    result = await _select_from_entities(_text_chunks_db("WEIGHT"))

    assert [chunk["chunk_id"] for chunk in result] == CHUNK_IDS[:RELATED_CHUNK_NUMBER]


async def test_vector_budget_counts_only_surviving_entity_groups():
    text_chunks_db = _text_chunks_db("VECTOR")
    text_chunks_db.embedding_func = AsyncMock()

    with patch(
        "lightrag.operate.pick_by_vector_similarity",
        new=AsyncMock(return_value=[]),
    ) as vector_picker:
        await _select_from_entities(
            text_chunks_db, query="question", chunks_vdb=AsyncMock()
        )

    # int(2 * 1 / 2) over the single surviving group, not int(2 * 5 / 2).
    assert _vector_call_args(vector_picker) == (1, 1)


async def test_both_paths_deliver_the_same_weight_budget():
    """The asymmetry is what makes this easy to reintroduce, so pin it."""
    entity_result = await _select_from_entities(_text_chunks_db("WEIGHT"))
    relation_result = await _select_from_relations(_text_chunks_db("WEIGHT"))

    assert [chunk["chunk_id"] for chunk in entity_result] == [
        chunk["chunk_id"] for chunk in relation_result
    ]


async def test_both_paths_hand_the_same_budget_to_the_vector_picker():
    budgets = []
    for select in (_select_from_entities, _select_from_relations):
        text_chunks_db = _text_chunks_db("VECTOR")
        text_chunks_db.embedding_func = AsyncMock()
        with patch(
            "lightrag.operate.pick_by_vector_similarity",
            new=AsyncMock(return_value=[]),
        ) as vector_picker:
            await select(text_chunks_db, query="question", chunks_vdb=AsyncMock())
        budgets.append(_vector_call_args(vector_picker))

    assert budgets[0] == budgets[1]
