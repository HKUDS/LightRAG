from unittest.mock import AsyncMock, patch

import pytest

from lightrag.base import QueryParam
from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.operate import _find_related_text_unit_from_entities


pytestmark = pytest.mark.offline


@pytest.mark.asyncio
async def test_entity_chunk_selection_drops_groups_emptied_by_deduplication():
    chunk_ids = [f"chunk-{index}" for index in range(20)]
    source_id = GRAPH_FIELD_SEP.join(chunk_ids)
    node_datas = [
        {"entity_name": f"entity-{index}", "source_id": source_id} for index in range(5)
    ]

    text_chunks_db = AsyncMock()
    text_chunks_db.global_config = {
        "kg_chunk_pick_method": "WEIGHT",
        "related_chunk_number": 2,
    }
    text_chunks_db.get_by_ids.return_value = [
        {"content": f"content-{chunk_id}"} for chunk_id in chunk_ids[:2]
    ]

    with patch(
        "lightrag.operate.pick_by_weighted_polling",
        return_value=chunk_ids[:2],
    ) as weighted_picker:
        result = await _find_related_text_unit_from_entities(
            node_datas,
            QueryParam(),
            text_chunks_db,
            AsyncMock(),
        )

    groups, max_related_chunks = weighted_picker.call_args.args[:2]
    assert len(groups) == 1
    assert groups[0]["sorted_chunks"] == chunk_ids
    assert max_related_chunks == 2
    assert [chunk["chunk_id"] for chunk in result] == chunk_ids[:2]
