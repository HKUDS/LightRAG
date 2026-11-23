"""Budget parity between the entity and relation chunk-selection paths.

Both paths deduplicate each group's chunk list, both can empty a group that
way, and both size their chunk budget from the surviving group count. The
entity path used to keep the emptied groups, so phantom groups inflated the
WEIGHT and VECTOR budgets and over-delivered beyond ``related_chunk_number``.
Both paths also share the VECTOR -> WEIGHT fallback when VECTOR cannot run.
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


def _text_chunks_db(
    pick_method: str, related_chunk_number: int = RELATED_CHUNK_NUMBER
) -> AsyncMock:
    text_chunks_db = AsyncMock()
    text_chunks_db.global_config = {
        "kg_chunk_pick_method": pick_method,
        "related_chunk_number": related_chunk_number,
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


@pytest.mark.parametrize(
    "select",
    [_select_from_entities, _select_from_relations],
    ids=["entities", "relations"],
)
async def test_weight_path_honours_the_configured_budget(select):
    """The user-visible symptom: phantom entity groups used to deliver 8, not 2.

    Both paths are pinned because the asymmetry is what makes this easy to
    reintroduce.
    """
    result = await select(_text_chunks_db("WEIGHT"))

    assert [chunk["chunk_id"] for chunk in result] == CHUNK_IDS[:RELATED_CHUNK_NUMBER]


@pytest.mark.parametrize(
    "select",
    [_select_from_entities, _select_from_relations],
    ids=["entities", "relations"],
)
async def test_vector_budget_counts_only_surviving_groups(select):
    """``related_chunk_number=1`` pins both the dedup and the quota floor.

    With the phantom groups still counted the picker would get
    ``int(1 * 5 / 2) == 2`` chunks over 5 groups. With the groups dropped but
    without the floor added with ``_vector_chunk_quota``, ``int(1 * 1 / 2)``
    truncates to 0; ``pick_by_vector_similarity`` reads that as "no chunks" and
    the call site silently downgrades to WEIGHT. Only ``(1, 1)`` has both.
    """
    text_chunks_db = _text_chunks_db("VECTOR", related_chunk_number=1)
    text_chunks_db.embedding_func = AsyncMock()

    with patch(
        "lightrag.operate.pick_by_vector_similarity",
        new=AsyncMock(return_value=[]),
    ) as vector_picker:
        await select(text_chunks_db, query="question", chunks_vdb=AsyncMock())

    assert _vector_call_args(vector_picker) == (1, 1)


@pytest.mark.parametrize(
    ("select", "query", "chunks_vdb"),
    [
        (_select_from_entities, "", AsyncMock()),
        (_select_from_relations, "question", None),
    ],
    ids=["entities_without_query", "relations_without_chunks_vdb"],
)
async def test_vector_without_prerequisites_falls_back_to_weight(
    select, query, chunks_vdb
):
    """VECTOR without a query or ``chunks_vdb`` used to skip both branches and return []."""
    text_chunks_db = _text_chunks_db("VECTOR")
    text_chunks_db.embedding_func = AsyncMock()

    with patch(
        "lightrag.operate.pick_by_vector_similarity", new=AsyncMock()
    ) as vector_picker:
        result = await select(text_chunks_db, query=query, chunks_vdb=chunks_vdb)

    vector_picker.assert_not_awaited()
    assert [chunk["chunk_id"] for chunk in result] == CHUNK_IDS[:RELATED_CHUNK_NUMBER]
