"""aquery_data and aquery_llm must answer "no keywords" with the failure shape.

When keyword extraction yields neither high- nor low-level keywords for a
query of 50+ characters, ``kg_query`` returns the canned ``fail_response``
without ``raw_data``. ``aquery_data`` documents "Both high-level and low-level
keywords are empty" as a failure case answered with
``{"status": "failure", "message": ..., "data": {}}``; ``/query/data`` builds
``QueryDataResponse(**result)`` from that dict, so an empty dict turns the
no-result case into an HTTP 500 (first described in #3727). ``aquery_llm``
returns the same failure shape, with the canned text as its ``llm_response``.
"""

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from lightrag import LightRAG
from lightrag.base import QueryParam
from lightrag.prompt import PROMPTS

# query_routes parses the server's command line on import.
_original_argv = sys.argv[:]
sys.argv = [sys.argv[0]]
_query_routes = importlib.import_module("lightrag.api.routers.query_routes")
sys.argv = _original_argv

pytestmark = [pytest.mark.offline, pytest.mark.asyncio]

# >= 50 characters, so kg_query does not fall back to using it as a keyword.
_VAGUE_QUERY = "hello there, I was just wondering how you are doing today"


# What the keyword model answers. The first is what rule 6 of
# PROMPTS["keywords_extraction"] asks for on a vague query; the second is a
# reply that is not JSON at all, which the parser also reads as no keywords.
_KEYWORD_REPLIES = [
    '{"high_level_keywords": [], "low_level_keywords": []}',
    "Sorry, I can't extract keywords from that.",
]


def _rag(monkeypatch, keyword_reply):
    search = AsyncMock()
    monkeypatch.setattr("lightrag.operate._perform_kg_search", search)
    config = {
        "role_llm_funcs": {
            "query": AsyncMock(),
            "keyword": AsyncMock(return_value=keyword_reply),
        },
        "addon_params": {"language": "English"},
        "tokenizer": None,
    }
    namespace = SimpleNamespace(
        _build_global_config=lambda: config,
        _query_done=AsyncMock(),
        chunk_entity_relation_graph=None,
        entities_vdb=None,
        relationships_vdb=None,
        text_chunks=None,
        chunks_vdb=None,
        llm_response_cache=None,
    )
    return namespace, search


@pytest.mark.parametrize("keyword_reply", _KEYWORD_REPLIES)
@pytest.mark.parametrize("mode", ["local", "global", "hybrid", "mix"])
async def test_aquery_data_reports_failure_when_no_keywords_are_extracted(
    monkeypatch, mode, keyword_reply
):
    assert len(_VAGUE_QUERY) >= 50
    rag, search = _rag(monkeypatch, keyword_reply)

    result = await LightRAG.aquery_data(rag, _VAGUE_QUERY, param=QueryParam(mode=mode))

    search.assert_not_awaited()
    assert result.get("status") == "failure", result
    assert result.get("data") == {}
    # The /query/data route builds QueryDataResponse(**result) from this dict;
    # a missing required field is a ValidationError there, i.e. an HTTP 500.
    _query_routes.QueryDataResponse(**result)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("keyword_reply", _KEYWORD_REPLIES)
@pytest.mark.parametrize("mode", ["local", "global", "hybrid", "mix"])
async def test_aquery_llm_reports_failure_when_no_keywords_are_extracted(
    monkeypatch, mode, keyword_reply, stream
):
    rag, search = _rag(monkeypatch, keyword_reply)

    result = await LightRAG.aquery_llm(
        rag, _VAGUE_QUERY, param=QueryParam(mode=mode, stream=stream)
    )

    search.assert_not_awaited()
    assert result.get("status") == "failure", result
    assert result.get("data") == {}
    assert result["metadata"] == {"failure_reason": "no_results", "mode": mode}
    # /query and /query/stream serve llm_response; it stays the canned text.
    assert result["llm_response"] == {
        "content": PROMPTS["fail_response"],
        "response_iterator": None,
        "is_streaming": False,
        "llm_generated": False,
    }
