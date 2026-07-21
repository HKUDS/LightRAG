import pytest

from lightrag.base import QueryParam
from lightrag.operate import naive_query
from lightrag.utils import TruncatedResponse


class _FakeTokenizer:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


class _FakeKVStorage:
    def __init__(self):
        self.global_config = {"enable_llm_cache": True}
        self._store = {}

    async def get_by_id(self, key):
        return self._store.get(key)

    async def upsert(self, entries):
        self._store.update(entries)


class _FakeChunksVDB:
    cosine_better_than_threshold = 0.0

    async def query(self, *_args, **_kwargs):
        return [
            {
                "id": "chunk-1",
                "content": "Truncated response cache guard test chunk.",
                "file_path": "test.md",
            }
        ]


def _query_global_config(llm_func) -> dict:
    return {
        "tokenizer": _FakeTokenizer(),
        "role_llm_funcs": {"query": llm_func},
        "min_rerank_score": 0.0,
        "max_total_tokens": 4096,
    }


@pytest.mark.offline
@pytest.mark.asyncio
async def test_naive_query_truncated_answer_not_cached():
    """A token-limit-truncated answer is returned but never enters the query cache.

    The next identical query must reach the LLM again instead of replaying the
    partial answer; a subsequent complete answer is then cached as usual.
    """
    cache = _FakeKVStorage()
    chunks_vdb = _FakeChunksVDB()
    responses = [
        TruncatedResponse("partial ans"),
        "complete answer",
    ]
    calls = 0

    async def query_model(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return responses[calls - 1]

    param = QueryParam(mode="naive", enable_rerank=False)

    first = await naive_query(
        "same query",
        chunks_vdb,
        param,
        _query_global_config(query_model),
        hashing_kv=cache,
    )
    assert first.content == "partial ans"
    assert cache._store == {}

    second = await naive_query(
        "same query",
        chunks_vdb,
        param,
        _query_global_config(query_model),
        hashing_kv=cache,
    )
    assert second.content == "complete answer"
    assert calls == 2
    assert len(cache._store) == 1
