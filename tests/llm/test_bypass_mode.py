"""``LightRAG.aquery_llm(mode="bypass")``: how the direct LLM answer is queued
and how its result is classified as streaming or not."""

import pytest

from lightrag import LightRAG
from lightrag.base import QueryParam
from lightrag.constants import DEFAULT_QUERY_PRIORITY, DEFAULT_SUMMARY_PRIORITY
from lightrag.utils import TruncatedResponse

pytestmark = pytest.mark.offline


class _FakeRAG:
    """Minimal stand-in exposing only what the bypass branch touches."""

    def __init__(self, llm_func):
        self._llm = llm_func

    def _build_global_config(self):
        return {"role_llm_funcs": {"query": self._llm}}


@pytest.mark.asyncio
async def test_bypass_non_streaming_answer_at_query_priority():
    """A TruncatedResponse stays non-streaming, the call is queued at query
    priority, and a prompt below the RAG minimum still reaches the LLM.

    TruncatedResponse is a str subclass: an exact ``type(response) is str``
    check would push it into the streaming branch, handing callers a bare str
    as ``response_iterator``. A plain str takes the same branch, so this input
    covers it too.

    The bypass answer is interactive like the kg/naive answers and the Ollama
    ``/api/chat`` bypass; summary priority would queue it behind every pending
    RAG answer on the query role queue.

    ``"中"`` is below the RAG minimum, which bypass skips because it does no
    retrieval; the empty-prompt rejection is pinned in
    ``tests/test_lightrag_query_validation.py``.
    """
    seen: list[dict] = []

    async def llm(prompt, **kwargs):
        seen.append(kwargs)
        return TruncatedResponse("partial answer")

    result = await LightRAG.aquery_llm(
        _FakeRAG(llm), "中", param=QueryParam(mode="bypass", stream=False)
    )

    llm_response = result["llm_response"]
    assert llm_response["is_streaming"] is False
    assert llm_response["content"] == "partial answer"
    assert llm_response["response_iterator"] is None
    assert len(seen) == 1
    # Unequal constants, so the assertion tells query from summary priority.
    assert DEFAULT_QUERY_PRIORITY != DEFAULT_SUMMARY_PRIORITY
    assert seen[0]["_priority"] == DEFAULT_QUERY_PRIORITY


@pytest.mark.asyncio
async def test_bypass_iterator_stays_streaming():
    async def _chunks():
        yield "part"

    iterator = _chunks()

    async def llm(prompt, **_kwargs):
        return iterator

    result = await LightRAG.aquery_llm(
        _FakeRAG(llm), "question", param=QueryParam(mode="bypass", stream=True)
    )

    llm_response = result["llm_response"]
    assert llm_response["is_streaming"] is True
    assert llm_response["response_iterator"] is iterator
