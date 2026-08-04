from unittest.mock import AsyncMock

import pytest

from lightrag.utils import TruncatedResponse, use_llm_func_with_cache


class _FakeKVStorage:
    def __init__(self):
        self.global_config = {"enable_llm_cache_for_entity_extract": True}
        self._store = {}

    async def get_by_id(self, key):
        return self._store.get(key)

    async def upsert(self, entries):
        self._store.update(entries)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_use_llm_func_with_cache_partitions_cache_by_response_format():
    cache = _FakeKVStorage()
    llm_func = AsyncMock(side_effect=["plain-text", '{"answer":"json"}'])

    plain_result, _ = await use_llm_func_with_cache(
        "same prompt",
        llm_func,
        llm_response_cache=cache,
    )
    json_result, _ = await use_llm_func_with_cache(
        "same prompt",
        llm_func,
        llm_response_cache=cache,
        response_format={"type": "json_object"},
    )

    assert plain_result == "plain-text"
    assert json_result == '{"answer":"json"}'
    assert llm_func.await_count == 2
    assert len(cache._store) == 2


@pytest.mark.offline
@pytest.mark.asyncio
async def test_use_llm_func_with_cache_partitions_cache_by_llm_identity():
    cache = _FakeKVStorage()
    llm_func = AsyncMock(side_effect=["model-a", "model-b"])

    first_result, _ = await use_llm_func_with_cache(
        "same prompt",
        llm_func,
        llm_response_cache=cache,
        llm_cache_identity={
            "role": "query",
            "binding": "openai",
            "model": "model-a",
            "host": "https://api.example.com/v1",
        },
    )
    second_result, _ = await use_llm_func_with_cache(
        "same prompt",
        llm_func,
        llm_response_cache=cache,
        llm_cache_identity={
            "role": "query",
            "binding": "openai",
            "model": "model-b",
            "host": "https://api.example.com/v1",
        },
    )

    assert first_result == "model-a"
    assert second_result == "model-b"
    assert llm_func.await_count == 2
    assert len(cache._store) == 2


@pytest.mark.offline
@pytest.mark.asyncio
async def test_use_llm_func_with_cache_skips_caching_truncated_response():
    """A token-limit-truncated response is returned but never persisted.

    Caching a partial extraction payload would replay the incomplete data on
    every later run, even once a larger token budget would have produced the
    complete output. The content is still returned for best-effort salvage.
    """
    cache = _FakeKVStorage()
    truncated = TruncatedResponse('{"entities":[{"name":"Ali')
    llm_func = AsyncMock(return_value=truncated)

    result, _ = await use_llm_func_with_cache(
        "extract prompt",
        llm_func,
        llm_response_cache=cache,
        response_format={"type": "json_object"},
    )

    # Content is returned to the caller for tolerant parsing/salvage...
    assert result == '{"entities":[{"name":"Ali'
    # ...but nothing was written to the cache.
    assert cache._store == {}
    llm_func.assert_awaited_once()


@pytest.mark.offline
@pytest.mark.asyncio
async def test_use_llm_func_with_cache_truncated_response_is_not_reused():
    """A skipped truncated write means the next call re-invokes the LLM.

    First call truncates (not cached); a retry with more budget must reach the
    LLM again and then cache the complete result.
    """
    cache = _FakeKVStorage()
    llm_func = AsyncMock(
        side_effect=[
            TruncatedResponse('{"entities":[{"name":"Ali'),
            '{"entities":[{"name":"Alice"}]}',
        ]
    )

    first, _ = await use_llm_func_with_cache(
        "same prompt",
        llm_func,
        llm_response_cache=cache,
    )
    second, _ = await use_llm_func_with_cache(
        "same prompt",
        llm_func,
        llm_response_cache=cache,
    )

    assert first == '{"entities":[{"name":"Ali'
    assert second == '{"entities":[{"name":"Alice"}]}'
    # Both calls hit the LLM (the truncated first result was not cached);
    # only the complete second result is now persisted.
    assert llm_func.await_count == 2
    assert len(cache._store) == 1


@pytest.mark.offline
@pytest.mark.asyncio
async def test_use_llm_func_with_cache_rejects_json_schema_response_format():
    llm_func = AsyncMock()

    with pytest.raises(ValueError, match="json_schema"):
        await use_llm_func_with_cache(
            "same prompt",
            llm_func,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "answer_payload",
                    "schema": {"type": "object"},
                },
            },
        )

    llm_func.assert_not_awaited()
