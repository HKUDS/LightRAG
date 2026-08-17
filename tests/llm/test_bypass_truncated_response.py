import pytest

from lightrag import LightRAG
from lightrag.base import QueryParam
from lightrag.utils import TruncatedResponse


class _FakeRAG:
    """Minimal stand-in exposing only what the bypass branch touches."""

    def __init__(self, llm_func):
        self._llm = llm_func

    def _build_global_config(self):
        return {"role_llm_funcs": {"query": self._llm}}


def _make_llm(response):
    async def llm(prompt, **_kwargs):
        return response

    return llm


@pytest.mark.offline
@pytest.mark.asyncio
async def test_bypass_truncated_response_is_not_misclassified_as_streaming():
    """A TruncatedResponse (str subclass) must stay on the non-streaming branch.

    The bypass branch distinguishes a plain string from a streaming iterator;
    an exact ``type(response) is str`` check would push the truncated string
    into the streaming branch, handing callers a bare str as
    ``response_iterator``.
    """
    rag = _FakeRAG(_make_llm(TruncatedResponse("partial answer")))

    result = await LightRAG.aquery_llm(
        rag, "question", param=QueryParam(mode="bypass", stream=False)
    )

    llm_response = result["llm_response"]
    assert llm_response["is_streaming"] is False
    assert llm_response["content"] == "partial answer"
    assert llm_response["response_iterator"] is None


@pytest.mark.offline
@pytest.mark.asyncio
async def test_bypass_plain_string_stays_non_streaming():
    rag = _FakeRAG(_make_llm("complete answer"))

    result = await LightRAG.aquery_llm(
        rag, "question", param=QueryParam(mode="bypass", stream=False)
    )

    llm_response = result["llm_response"]
    assert llm_response["is_streaming"] is False
    assert llm_response["content"] == "complete answer"


@pytest.mark.offline
@pytest.mark.asyncio
async def test_bypass_iterator_stays_streaming():
    async def _chunks():
        yield "part"

    iterator = _chunks()
    rag = _FakeRAG(_make_llm(iterator))

    result = await LightRAG.aquery_llm(
        rag, "question", param=QueryParam(mode="bypass", stream=True)
    )

    llm_response = result["llm_response"]
    assert llm_response["is_streaming"] is True
    assert llm_response["response_iterator"] is iterator
