import pytest

from lightrag.base import QueryParam
from lightrag.operate import naive_query


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
                "content": "LightRAG cache identity test chunk.",
                "file_path": "test.md",
            }
        ]


def _query_global_config(model: str, llm_func) -> dict:
    return {
        "tokenizer": _FakeTokenizer(),
        "role_llm_funcs": {"query": llm_func},
        "llm_cache_identities": {
            "query": {
                "role": "query",
                "binding": "openai",
                "model": model,
                "host": "https://api.example.com/v1",
            }
        },
        "min_rerank_score": 0.0,
        "max_total_tokens": 4096,
    }


@pytest.mark.offline
@pytest.mark.asyncio
async def test_naive_query_partitions_query_cache_by_llm_identity():
    cache = _FakeKVStorage()
    chunks_vdb = _FakeChunksVDB()
    calls = 0

    async def query_model(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return f"answer-{calls}"

    param = QueryParam(mode="naive", enable_rerank=False)

    first = await naive_query(
        "same query",
        chunks_vdb,
        param,
        _query_global_config("model-a", query_model),
        hashing_kv=cache,
    )
    second = await naive_query(
        "same query",
        chunks_vdb,
        param,
        _query_global_config("model-b", query_model),
        hashing_kv=cache,
    )

    assert first.content == "answer-1"
    assert second.content == "answer-2"
    assert calls == 2
    assert len(cache._store) == 2
