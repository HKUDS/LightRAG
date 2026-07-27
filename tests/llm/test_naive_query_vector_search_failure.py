import pytest

from lightrag.base import QueryParam
from lightrag.operate import _get_vector_context, naive_query


class _FakeTokenizer:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


class _RaisingChunksVDB:
    cosine_better_than_threshold = 0.0

    async def query(self, *_args, **_kwargs):
        raise TimeoutError("embedding worker timed out")


def _query_global_config() -> dict:
    async def query_model(*_args, **_kwargs):
        return "unused"

    return {
        "tokenizer": _FakeTokenizer(),
        "role_llm_funcs": {"query": query_model},
        "min_rerank_score": 0.0,
        "max_total_tokens": 4096,
    }


@pytest.mark.offline
@pytest.mark.asyncio
async def test_get_vector_context_suppresses_by_default():
    """Legacy callers (e.g. mix mode) keep the existing graceful-degradation behavior."""
    param = QueryParam(mode="naive", enable_rerank=False)
    chunks = await _get_vector_context("q", _RaisingChunksVDB(), param, None)
    assert chunks == []


@pytest.mark.offline
@pytest.mark.asyncio
async def test_get_vector_context_raises_when_not_suppressed():
    param = QueryParam(mode="naive", enable_rerank=False)
    with pytest.raises(TimeoutError):
        await _get_vector_context(
            "q", _RaisingChunksVDB(), param, None, suppress_errors=False
        )


@pytest.mark.offline
@pytest.mark.asyncio
async def test_naive_query_propagates_vector_search_failure():
    """A vector-search backend failure must not be reported as 'no relevant chunks'.

    Before the fix, _get_vector_context swallowed every exception into an empty
    list, so naive_query returned None exactly as it would for a genuine
    zero-match query -- the caller (and eventually the /query API) could not
    tell an embedding-worker timeout apart from "nothing matched".
    """
    param = QueryParam(mode="naive", enable_rerank=False)

    with pytest.raises(TimeoutError):
        await naive_query(
            "same query",
            _RaisingChunksVDB(),
            param,
            _query_global_config(),
            hashing_kv=None,
        )
