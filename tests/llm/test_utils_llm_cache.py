from unittest.mock import AsyncMock

import pytest

from lightrag.exceptions import EmptyTruncatedResponseError
from lightrag.utils import (
    TruncatedResponse,
    is_truncated_response,
    use_llm_func_with_cache,
)


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
    assert is_truncated_response(result)
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
async def test_truncation_marker_survives_when_cache_is_disabled():
    """Callers must observe truncation even without an extraction cache."""
    llm_func = AsyncMock(
        return_value=TruncatedResponse("<think>reasoning</think>Partial result")
    )

    result, _ = await use_llm_func_with_cache(
        "extract prompt",
        llm_func,
        llm_response_cache=None,
    )

    assert result == "Partial result"
    assert is_truncated_response(result)


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


@pytest.mark.offline
@pytest.mark.asyncio
@pytest.mark.parametrize("cache_enabled", [False, True])
async def test_truncated_response_emptied_by_think_removal_is_rejected(cache_enabled):
    """The one empty+length shape no binding can see.

    A thinking model that exhausts its budget inside the reasoning trace
    returns ``<think>...</think>`` with no answer after it. That payload is
    NON-empty, so every binding's own empty-content check passes it through;
    it only becomes visibly empty after think-tag removal. Returning it let
    extraction index an empty graph and still report PROCESSED.
    """
    llm_func = AsyncMock(
        return_value=TruncatedResponse("<think>let me carefully consider</think>")
    )

    with pytest.raises(EmptyTruncatedResponseError) as excinfo:
        await use_llm_func_with_cache(
            "extract prompt",
            llm_func,
            llm_response_cache=_FakeKVStorage() if cache_enabled else None,
            chunk_id="chunk-001",
        )

    message = str(excinfo.value)
    assert "Received empty extract content after think-tag removal" in message
    assert "chunk_id=chunk-001" in message
    # Everything the model produced was reasoning, by construction.
    assert "reasoning_content_len=40" in message
    assert "budget consumed by reasoning" in message
    assert "output token limit" in message


@pytest.mark.offline
@pytest.mark.asyncio
async def test_an_untruncated_empty_response_is_still_returned():
    """Scope: only the token-limit case escalates. A model that legitimately
    answers with nothing (or with reasoning only, having finished normally)
    keeps its previous behavior."""
    llm_func = AsyncMock(return_value="<think>done thinking</think>")

    result, _ = await use_llm_func_with_cache(
        "extract prompt", llm_func, llm_response_cache=None
    )

    assert result == ""


@pytest.mark.offline
@pytest.mark.asyncio
async def test_a_truncated_response_with_content_after_the_think_block_survives():
    """The rejection must not swallow the salvage path."""
    llm_func = AsyncMock(
        return_value=TruncatedResponse('<think>reasoning</think>{"entities":[{"name')
    )

    result, _ = await use_llm_func_with_cache(
        "extract prompt", llm_func, llm_response_cache=None
    )

    assert result == '{"entities":[{"name'
    assert is_truncated_response(result)


class _FakeChunkKV:
    """text_chunks stand-in for the reference-before-row path (#3833).

    ``writes`` is shared with the cache double so a test can assert the ORDER
    of the two storage writes rather than merely that both happened.
    """

    def __init__(self, rows: dict | None = None, writes: list | None = None):
        self.data = dict(rows or {})
        self.writes = writes if writes is not None else []
        self.fail_on_upsert = False
        self.fail_on_read = False

    async def get_by_id(self, key):
        if self.fail_on_read:
            raise RuntimeError("text_chunks read is down")
        return self.data.get(key)

    async def upsert(self, rows: dict):
        if self.fail_on_upsert:
            raise RuntimeError("text_chunks write is down")
        self.data.update(rows)
        self.writes.append("chunk")

    async def index_done_callback(self):
        return None


class _RecordingCache(_FakeKVStorage):
    """Cache double that records its writes into a shared order log."""

    def __init__(self, writes: list):
        super().__init__()
        self.writes = writes

    async def upsert(self, entries):
        self._store.update(entries)
        self.writes.append("cache")


def _attached(chunks: _FakeChunkKV, chunk_id: str = "chunk-1") -> list[str]:
    return list((chunks.data.get(chunk_id) or {}).get("llm_cache_list") or [])


@pytest.mark.offline
@pytest.mark.asyncio
async def test_extract_cache_row_is_written_only_after_its_reference():
    """The reference must be durable before the row exists (#3833).

    A row written first is unreachable for good if anything cuts the gap
    short; a reference written first only ever dangles, which every reader
    tolerates.
    """
    writes: list[str] = []
    chunks = _FakeChunkKV({"chunk-1": {"content": "c"}}, writes=writes)
    cache = _RecordingCache(writes)
    llm_func = AsyncMock(return_value="extracted")

    result, _ = await use_llm_func_with_cache(
        "extract prompt",
        llm_func,
        llm_response_cache=cache,
        chunk_id="chunk-1",
        text_chunks_storage=chunks,
    )

    assert result == "extracted"
    assert writes == ["chunk", "cache"], writes
    assert _attached(chunks) == list(cache._store)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_cache_write_is_skipped_when_the_reference_cannot_be_recorded():
    """No reference, no row: an unrecordable key must not leave a row behind."""
    chunks = _FakeChunkKV({"chunk-1": {"content": "c"}})
    chunks.fail_on_upsert = True
    cache = _FakeKVStorage()
    llm_func = AsyncMock(return_value="extracted")

    result, _ = await use_llm_func_with_cache(
        "extract prompt",
        llm_func,
        llm_response_cache=cache,
        chunk_id="chunk-1",
        text_chunks_storage=chunks,
    )

    # The caller still gets its answer; only the caching is given up.
    assert result == "extracted"
    assert cache._store == {}


@pytest.mark.offline
@pytest.mark.asyncio
async def test_a_failing_chunk_read_also_skips_the_cache_write():
    chunks = _FakeChunkKV({"chunk-1": {"content": "c"}})
    chunks.fail_on_read = True
    cache = _FakeKVStorage()

    result, _ = await use_llm_func_with_cache(
        "extract prompt",
        AsyncMock(return_value="extracted"),
        llm_response_cache=cache,
        chunk_id="chunk-1",
        text_chunks_storage=chunks,
    )

    assert result == "extracted"
    assert cache._store == {}


@pytest.mark.offline
@pytest.mark.asyncio
async def test_a_missing_chunk_row_skips_the_cache_write():
    """Nothing could carry the reference, so the row must not be written."""
    chunks = _FakeChunkKV()  # the chunk row is absent
    cache = _FakeKVStorage()

    result, _ = await use_llm_func_with_cache(
        "extract prompt",
        AsyncMock(return_value="extracted"),
        llm_response_cache=cache,
        chunk_id="chunk-1",
        text_chunks_storage=chunks,
    )

    assert result == "extracted"
    assert cache._store == {}


@pytest.mark.offline
@pytest.mark.asyncio
async def test_an_empty_response_attaches_no_reference():
    """``save_to_cache`` no-ops on falsy content, so attaching would dangle.

    Not merely cosmetic: a misconfigured model answering empty for every chunk
    would otherwise fill every chunk row with references to rows that were
    never written.
    """
    chunks = _FakeChunkKV({"chunk-1": {"content": "c"}})
    cache = _FakeKVStorage()

    result, _ = await use_llm_func_with_cache(
        "extract prompt",
        AsyncMock(return_value=""),
        llm_response_cache=cache,
        chunk_id="chunk-1",
        text_chunks_storage=chunks,
    )

    assert result == ""
    assert cache._store == {}
    assert _attached(chunks) == []


@pytest.mark.offline
@pytest.mark.asyncio
async def test_a_truncated_response_attaches_no_reference():
    """Truncated output is deliberately not cached, so nothing may be attached."""
    chunks = _FakeChunkKV({"chunk-1": {"content": "c"}})
    cache = _FakeKVStorage()

    result, _ = await use_llm_func_with_cache(
        "extract prompt",
        AsyncMock(return_value=TruncatedResponse("partial")),
        llm_response_cache=cache,
        chunk_id="chunk-1",
        text_chunks_storage=chunks,
    )

    assert result == "partial"
    assert cache._store == {}
    assert _attached(chunks) == []


@pytest.mark.offline
@pytest.mark.asyncio
async def test_a_cache_hit_reattaches_a_wiped_reference():
    """Re-ingest self-heal: stage 1 rewrites the chunk row and empties the list.

    The rows it named still exist, so the hit branch has to put the reference
    back -- otherwise re-ingesting a cached document orphans all of its rows.
    """
    writes: list[str] = []
    chunks = _FakeChunkKV({"chunk-1": {"content": "c", "llm_cache_list": []}}, writes)
    cache = _RecordingCache(writes)
    llm_func = AsyncMock(return_value="extracted")

    # First pass populates both stores.
    await use_llm_func_with_cache(
        "extract prompt",
        llm_func,
        llm_response_cache=cache,
        chunk_id="chunk-1",
        text_chunks_storage=chunks,
    )
    cache_keys = list(cache._store)
    # Stage 1 of a re-ingest replaces the row's business value wholesale.
    chunks.data["chunk-1"] = {"content": "c", "llm_cache_list": []}

    await use_llm_func_with_cache(
        "extract prompt",
        llm_func,
        llm_response_cache=cache,
        chunk_id="chunk-1",
        text_chunks_storage=chunks,
    )

    assert llm_func.await_count == 1, "second call must be served from cache"
    assert _attached(chunks) == cache_keys


@pytest.mark.offline
@pytest.mark.asyncio
async def test_callers_without_a_chunk_storage_keep_the_legacy_order():
    """The parse stage and the summary path have no owning chunk to attach to.

    They must keep collecting keys in a list, unchanged.
    """
    cache = _FakeKVStorage()
    collector: list[str] = []

    result, _ = await use_llm_func_with_cache(
        "smartheading prompt",
        AsyncMock(return_value="judged"),
        llm_response_cache=cache,
        cache_type="smartheading",
        chunk_id="chunk-1",  # named, but no storage to record it on
        cache_keys_collector=collector,
    )

    assert result == "judged"
    assert list(cache._store) == collector
