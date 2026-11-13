"""Which store a KG search queries, and with which keywords, per mode.

Pins the empty-keyword fallbacks of ``_perform_kg_search``: a mode whose own
keyword list came back empty searches ITS OWN store with the other list, and
never falls through to the store it does not own.
"""

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from lightrag.base import QueryParam
from lightrag.operate import _perform_kg_search

pytestmark = pytest.mark.offline

LL = "Tesla, SpaceX"
HL = "launch costs, tariffs"


def _make_vdb():
    vdb = AsyncMock()
    vdb.query = AsyncMock(return_value=[])
    vdb.cosine_better_than_threshold = 0.2
    return vdb


async def _search(mode: str, ll_keywords: str, hl_keywords: str):
    async def _embed(texts, **kwargs):
        return np.array(
            [np.full(8, i + 1, dtype=np.float32) for i in range(len(texts))]
        )

    embed_func = AsyncMock(side_effect=_embed)
    text_chunks_db = MagicMock()
    text_chunks_db.embedding_func = embed_func
    text_chunks_db.global_config = {"kg_chunk_pick_method": "VECTOR"}
    entities_vdb = _make_vdb()
    relationships_vdb = _make_vdb()

    await _perform_kg_search(
        query="test query",
        ll_keywords=ll_keywords,
        hl_keywords=hl_keywords,
        knowledge_graph_inst=AsyncMock(),
        entities_vdb=entities_vdb,
        relationships_vdb=relationships_vdb,
        text_chunks_db=text_chunks_db,
        query_param=QueryParam(mode=mode, top_k=5),
    )
    return entities_vdb, relationships_vdb, embed_func


def _queried_with(vdb) -> str | None:
    """The keyword string a store was searched with; None when it was skipped."""
    assert vdb.query.call_count <= 1
    return vdb.query.call_args.args[0] if vdb.query.call_count else None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode, ll_keywords, hl_keywords, entity_query, relation_query",
    [
        # Both lists present: every mode searches only the stores it owns.
        pytest.param("local", LL, HL, LL, None, id="local"),
        pytest.param("global", LL, HL, None, HL, id="global"),
        pytest.param("hybrid", LL, HL, LL, HL, id="hybrid"),
        pytest.param("mix", LL, HL, LL, HL, id="mix"),
        # Low-level list empty: high-level keywords stand in on the entity store.
        pytest.param("local", "", HL, HL, None, id="local-ll-empty"),
        pytest.param("hybrid", "", HL, HL, HL, id="hybrid-ll-empty"),
        pytest.param("mix", "", HL, HL, HL, id="mix-ll-empty"),
        pytest.param("global", "", HL, None, HL, id="global-ll-empty"),
        # High-level list empty: only `global` falls back, on its own store.
        pytest.param("global", LL, "", None, LL, id="global-hl-empty"),
        pytest.param("local", LL, "", LL, None, id="local-hl-empty"),
        pytest.param("hybrid", LL, "", LL, None, id="hybrid-hl-empty"),
        # Nothing to search with.
        pytest.param("local", "", "", None, None, id="local-both-empty"),
        pytest.param("global", "", "", None, None, id="global-both-empty"),
    ],
)
async def test_kg_search_queries_each_store_with_the_resolved_keywords(
    mode, ll_keywords, hl_keywords, entity_query, relation_query
):
    entities_vdb, relationships_vdb, _ = await _search(mode, ll_keywords, hl_keywords)

    assert _queried_with(entities_vdb) == entity_query
    assert _queried_with(relationships_vdb) == relation_query


@pytest.mark.asyncio
async def test_fallback_shared_by_both_stores_is_embedded_once():
    entities_vdb, relationships_vdb, embed_func = await _search("hybrid", "", HL)

    assert embed_func.call_count == 1
    assert embed_func.call_args.args[0] == ["test query", HL]
    entity_embedding = entities_vdb.query.call_args.kwargs["query_embedding"]
    relation_embedding = relationships_vdb.query.call_args.kwargs["query_embedding"]
    assert entity_embedding is not None
    assert np.array_equal(entity_embedding, relation_embedding)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode, ll_keywords, hl_keywords, expected",
    [
        pytest.param("local", "", HL, "querying entities", id="entity-fallback"),
        pytest.param(
            "global", LL, "", "querying relationships", id="relation-fallback"
        ),
    ],
)
async def test_fallback_is_logged(
    mode, ll_keywords, hl_keywords, expected, lightrag_log_records
):
    await _search(mode, ll_keywords, hl_keywords)

    warnings = [
        record.getMessage()
        for record in lightrag_log_records
        if record.levelname == "WARNING"
    ]
    assert sum(expected in message for message in warnings) == 1
