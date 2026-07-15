"""Regression coverage for fail-closed cached-knowledge rebuilds."""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lightrag.operate import rebuild_knowledge_from_chunks

pytestmark = pytest.mark.offline


@pytest.mark.asyncio
async def test_rebuild_raises_when_no_cached_extraction_results_exist():
    """Deletion finalization must not proceed if required cache evidence is absent."""
    with patch(
        "lightrag.operate._get_cached_extraction_results",
        new_callable=AsyncMock,
        return_value={},
    ):
        with pytest.raises(RuntimeError, match="No cached extraction results found"):
            await rebuild_knowledge_from_chunks(
                entities_to_rebuild={"ENTITY": ["chunk-a"]},
                relationships_to_rebuild={},
                knowledge_graph_inst=MagicMock(),
                entities_vdb=MagicMock(),
                relationships_vdb=MagicMock(),
                text_chunks_storage=AsyncMock(),
                llm_response_cache=AsyncMock(),
                global_config={},
            )


@pytest.mark.asyncio
async def test_rebuild_raises_when_a_worker_failure_is_captured():
    """Worker exceptions caught for status reporting still fail the caller closed."""

    @asynccontextmanager
    async def keyed_lock(*_args, **_kwargs):
        yield

    with (
        patch(
            "lightrag.operate._get_cached_extraction_results",
            new_callable=AsyncMock,
            return_value={"chunk-a": [("cached result", 1)]},
        ),
        patch(
            "lightrag.operate._rebuild_from_extraction_result",
            new_callable=AsyncMock,
            return_value=({}, {}),
        ),
        patch("lightrag.operate.get_storage_keyed_lock", keyed_lock),
        patch(
            "lightrag.operate._rebuild_single_entity",
            new_callable=AsyncMock,
            side_effect=RuntimeError("storage write failed"),
        ),
    ):
        with pytest.raises(RuntimeError, match="Failed: 1 entities"):
            await rebuild_knowledge_from_chunks(
                entities_to_rebuild={"ENTITY": ["chunk-a"]},
                relationships_to_rebuild={},
                knowledge_graph_inst=MagicMock(),
                entities_vdb=MagicMock(),
                relationships_vdb=MagicMock(),
                text_chunks_storage=AsyncMock(),
                llm_response_cache=AsyncMock(),
                global_config={},
            )
