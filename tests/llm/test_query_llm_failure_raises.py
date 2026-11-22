"""A failed answering LLM call must raise, never come back as an empty answer.

``aquery_llm`` used to catch every exception and return
``{"status": "failure", "llm_response": {"content": None}}``. ``aquery`` turned
that into ``None``, and the REST routes into HTTP 200 with "No relevant context
found for the query." -- a timeout looked like an empty knowledge base (#4475).

Retrieval is real here except for the search transport, so the failure lands
exactly where the issue reported it: after context was built, in the call to
the answering LLM.
"""

from functools import partial
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from lightrag import LightRAG
from lightrag.base import QueryParam
from lightrag.utils import Tokenizer

pytestmark = [pytest.mark.offline, pytest.mark.asyncio]

_TIMEOUT_MESSAGE = "query LLM func: Worker execution timeout after 2s"


class _CharacterTokenizer:
    def encode(self, content):
        return [ord(char) for char in content]

    def decode(self, tokens):
        return "".join(chr(token) for token in tokens)


@pytest.fixture
def failing_rag(monkeypatch):
    chunk = {
        "id": "chunk-1",
        "chunk_id": "chunk-1",
        "content": "Retrieved evidence.",
        "file_path": "source.md",
    }
    search = {
        "final_entities": [],
        "final_relations": [],
        "vector_chunks": [chunk],
        "chunk_tracking": {"chunk-1": {"source": "C", "frequency": 1, "order": 1}},
        "query_embedding": None,
    }
    monkeypatch.setattr(
        "lightrag.operate._perform_kg_search", AsyncMock(return_value=search)
    )
    model = AsyncMock(side_effect=TimeoutError(_TIMEOUT_MESSAGE))
    config = {
        "tokenizer": Tokenizer("character", _CharacterTokenizer()),
        "role_llm_funcs": {"query": model},
        "addon_params": {"language": "en"},
    }
    cache = SimpleNamespace(
        global_config={"enable_llm_cache": False},
        get_by_id=AsyncMock(return_value=None),
        upsert=AsyncMock(),
    )
    rag = SimpleNamespace(
        _build_global_config=lambda: config,
        _query_done=AsyncMock(),
        chunk_entity_relation_graph=None,
        entities_vdb=None,
        relationships_vdb=None,
        text_chunks=SimpleNamespace(global_config=config),
        chunks_vdb=SimpleNamespace(
            cosine_better_than_threshold=0.0, query=AsyncMock(return_value=[chunk])
        ),
        llm_response_cache=cache,
    )
    rag.aquery_llm = partial(LightRAG.aquery_llm, rag)
    return rag, model


def _param(mode: str, stream: bool) -> QueryParam:
    return QueryParam(mode=mode, ll_keywords=["evidence"], stream=stream)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("mode", ["naive", "mix", "bypass"])
async def test_aquery_llm_raises_when_the_answering_llm_fails(
    failing_rag, mode, stream
):
    rag, model = failing_rag

    with pytest.raises(TimeoutError, match="Worker execution timeout"):
        await rag.aquery_llm("Find evidence", param=_param(mode, stream))

    # The fault fired in the answering call, after retrieval succeeded.
    model.assert_awaited_once()


# ``aquery`` awaits ``aquery_llm`` before it branches on streaming, so the
# exception leaves the same way in every mode; the mode x stream behaviour is
# pinned on ``aquery_llm`` by test_aquery_llm_raises_when_the_answering_llm_fails.
@pytest.mark.parametrize("stream", [False])
@pytest.mark.parametrize("mode", ["mix"])
async def test_aquery_raises_instead_of_returning_none(failing_rag, mode, stream):
    rag, model = failing_rag

    with pytest.raises(TimeoutError, match="Worker execution timeout"):
        await LightRAG.aquery(rag, "Find evidence", param=_param(mode, stream))

    model.assert_awaited_once()
