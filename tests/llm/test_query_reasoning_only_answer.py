"""Regression: a reasoning-only LLM response is never returned or cached as the answer.

The query paths call the LLM with ``enable_cot=True``. With a reasoning model
whose provider returns ``reasoning_content`` but an empty ``content``, the
OpenAI binding used to return ``<think>{reasoning}</think>``: non-empty, so it
skipped the binding's empty-content check and its retry, was returned as the
answer and written to the query cache (replayed on every identical query).
With ``enable_cot=False`` the same response was retried and then raised.

Now the binding treats that response as empty whatever ``enable_cot`` says,
the query layer rejects any reasoning-only answer that still arrives (inline
``<think>`` in ``content``, custom functions), and a reasoning-only answer
already in the cache is treated as a miss.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from tenacity import wait_none

from lightrag import LightRAG
from lightrag.base import QueryContextResult, QueryParam
from lightrag.exceptions import EmptyTruncatedResponseError, ReasoningOnlyResponseError
from lightrag.llm.openai import InvalidResponseError, openai_complete_if_cache
from lightrag.operate import kg_query, naive_query
from lightrag.utils import Tokenizer, TruncatedResponse

pytestmark = pytest.mark.offline

QUERY = "How do I reset the device?"
REASONING = "The manual says to hold the reset button; I should explain that."
ANSWER = "Hold the reset button for ten seconds."


class _FakeTokenizerImpl:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


class _RecordingKV:
    def __init__(self, cached_answer=None):
        self.global_config = {"enable_llm_cache": True}
        self.writes = []
        self._cached_answer = cached_answer

    async def get_by_id(self, _key):
        if self._cached_answer is None:
            return None
        return {"return": self._cached_answer, "create_time": 0}

    async def upsert(self, entries):
        self.writes.append(entries)


class _ChunksVDB:
    cosine_better_than_threshold = 0.0

    async def query(self, *_args, **_kwargs):
        return [
            {
                "id": "chunk-1",
                "content": "Hold the reset button for ten seconds.",
                "file_path": "device/reset.md",
            }
        ]


def _completion(content, reasoning_content, finish_reason="stop"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(
                    content=content, parsed=None, reasoning_content=reasoning_content
                ),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30),
    )


@pytest.fixture(autouse=True)
def _no_retry_wait(monkeypatch):
    """Keep the binding's retry loop, drop its real exponential backoff."""
    monkeypatch.setattr(openai_complete_if_cache.retry, "wait", wait_none())


def _openai_query_llm(*completions):
    """The query role backed by the real OpenAI binding over a mocked client.

    Successive requests (retries included) receive ``completions`` in order;
    the last one repeats. ``llm.create`` records every request.
    """
    responses = list(completions)

    async def _create(*_args, **_kwargs):
        return responses.pop(0) if len(responses) > 1 else responses[0]

    create = AsyncMock(side_effect=_create)
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        close=AsyncMock(),
    )

    async def _llm(prompt, system_prompt=None, history_messages=None, **kwargs):
        with patch(
            "lightrag.llm.openai.create_openai_async_client", return_value=client
        ):
            return await openai_complete_if_cache(
                "reasoning-model",
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages or [],
                api_key="test-key",
                **kwargs,
            )

    _llm.create = create
    return _llm


async def _inline_reasoning_only(*_args, **_kwargs):
    """A provider that puts the trace inline in ``content``: no binding sees it."""
    return f"<think>{REASONING}</think>"


def _config(query_llm) -> dict:
    return {
        "tokenizer": Tokenizer("fake", _FakeTokenizerImpl()),
        "role_llm_funcs": {"keyword": query_llm, "query": query_llm},
        "addon_params": {"language": "en"},
        "min_rerank_score": 0.0,
        "max_total_tokens": 4096,
    }


@pytest.fixture
def stub_query_context(monkeypatch):
    async def _fake_build_query_context(*_args, **_kwargs):
        return QueryContextResult(
            context="KG CONTEXT",
            raw_data={"metadata": {"processing_info": {"final_chunks_count": 1}}},
        )

    monkeypatch.setattr(
        "lightrag.operate._build_query_context", _fake_build_query_context
    )


def _kg_param(**kwargs) -> QueryParam:
    return QueryParam(
        mode="mix",
        enable_rerank=False,
        stream=False,
        hl_keywords=["reset"],
        ll_keywords=["device"],
        **kwargs,
    )


async def _kg_query(llm, cache):
    return await kg_query(
        QUERY, None, None, None, None, _kg_param(), _config(llm), hashing_kv=cache
    )


@pytest.mark.parametrize("enable_cot", [True, False])
async def test_binding_treats_reasoning_only_as_empty_and_retries(enable_cot):
    """Root cause: enable_cot no longer turns an empty answer into a
    <think> block; the response is retried as a sampling artifact either way."""
    llm = _openai_query_llm(_completion("", REASONING))

    with pytest.raises(InvalidResponseError) as caught:
        await llm(QUERY, enable_cot=enable_cot)

    assert llm.create.await_count == 3
    assert "reasoning-only output" in str(caught.value)
    assert REASONING not in str(caught.value)


async def test_binding_truncated_reasoning_only_fails_fast():
    """Token-limit exhaustion is deterministic: one request, no retry."""
    llm = _openai_query_llm(_completion("", REASONING, finish_reason="length"))

    with pytest.raises(EmptyTruncatedResponseError):
        await llm(QUERY, enable_cot=True)

    assert llm.create.await_count == 1


async def test_kg_query_retry_recovers_the_answer(stub_query_context):
    """A reasoning-only first attempt is retried; the answer that follows is
    returned and cached, the trace is not."""
    cache = _RecordingKV()
    llm = _openai_query_llm(_completion("", REASONING), _completion(ANSWER, REASONING))

    result = await _kg_query(llm, cache)

    assert result.content == ANSWER
    assert llm.create.await_count == 2
    assert len(cache.writes) == 1
    assert REASONING not in str(cache.writes)


async def test_provider_without_reasoning_content_is_unaffected(stub_query_context):
    """Control: the official OpenAI Chat Completions API returns no
    ``reasoning_content``, so enable_cot adds nothing and the answer is
    returned and cached as before."""
    completion = _completion(ANSWER, None)
    delattr(completion.choices[0].message, "reasoning_content")
    cache = _RecordingKV()

    result = await _kg_query(_openai_query_llm(completion), cache)

    assert result.content == ANSWER
    assert len(cache.writes) == 1


async def test_kg_query_reasoning_only_answer_is_never_cached(stub_query_context):
    cache = _RecordingKV()
    llm = _openai_query_llm(_completion("", REASONING))

    with pytest.raises(InvalidResponseError):
        await _kg_query(llm, cache)

    assert cache.writes == []


async def test_kg_query_rejects_inline_reasoning_only_answer(stub_query_context):
    cache = _RecordingKV()

    with pytest.raises(ReasoningOnlyResponseError):
        await _kg_query(_inline_reasoning_only, cache)

    assert cache.writes == []


async def test_kg_query_truncated_reasoning_only_answer_is_a_truncation_error(
    stub_query_context,
):
    cache = _RecordingKV()
    llm = _openai_query_llm(_completion("", REASONING, finish_reason="length"))

    with pytest.raises(EmptyTruncatedResponseError):
        await _kg_query(llm, cache)

    assert cache.writes == []


async def test_naive_query_rejects_reasoning_only_answer_and_does_not_cache_it():
    cache = _RecordingKV()

    with pytest.raises(ReasoningOnlyResponseError):
        await naive_query(
            QUERY,
            _ChunksVDB(),
            QueryParam(mode="naive", enable_rerank=False, stream=False),
            _config(_inline_reasoning_only),
            hashing_kv=cache,
        )

    assert cache.writes == []


async def test_answer_with_reasoning_trace_is_still_returned_and_cached(
    stub_query_context,
):
    """Control: the intended enable_cot output (trace, then answer) is unchanged."""
    cache = _RecordingKV()
    response = f"<think>{REASONING}</think>{ANSWER}"

    async def _llm(*_args, **_kwargs):
        return response

    result = await _kg_query(_llm, cache)

    assert result.content == response
    assert len(cache.writes) == 1


async def test_truncated_partial_answer_is_still_returned(stub_query_context):
    """Control: a truncated answer with text keeps its best-effort salvage."""
    cache = _RecordingKV()

    async def _llm(*_args, **_kwargs):
        return TruncatedResponse("Hold the reset button for")

    result = await _kg_query(_llm, cache)

    assert result.content == "Hold the reset button for"
    assert cache.writes == []


class _BypassRAG:
    """Minimal stand-in exposing only what the bypass branch touches."""

    def __init__(self, llm_func):
        self._llm = llm_func

    def _build_global_config(self):
        return {"role_llm_funcs": {"query": self._llm}}


async def test_bypass_rejects_reasoning_only_answer():
    rag = _BypassRAG(_inline_reasoning_only)

    with pytest.raises(ReasoningOnlyResponseError):
        await LightRAG.aquery_llm(
            rag, QUERY, param=QueryParam(mode="bypass", stream=False)
        )


async def test_bypass_answer_with_reasoning_trace_is_unchanged():
    response = f"<think>{REASONING}</think>{ANSWER}"

    async def _llm(*_args, **_kwargs):
        return response

    result = await LightRAG.aquery_llm(
        _BypassRAG(_llm), QUERY, param=QueryParam(mode="bypass", stream=False)
    )

    assert result["llm_response"]["content"] == response


async def _naive_query(llm, cache):
    return await naive_query(
        QUERY,
        _ChunksVDB(),
        QueryParam(mode="naive", enable_rerank=False, stream=False),
        _config(llm),
        hashing_kv=cache,
    )


@pytest.mark.parametrize("run_query", [_kg_query, _naive_query], ids=["kg", "naive"])
async def test_cached_reasoning_only_answer_is_regenerated(
    stub_query_context, run_query
):
    """An entry written before the guard existed is a miss, not a replay: the
    answer is regenerated and overwrites it."""
    cache = _RecordingKV(cached_answer=f"<think>{REASONING}</think>")
    calls = []

    async def _llm(*_args, **_kwargs):
        calls.append(1)
        return ANSWER

    result = await run_query(_llm, cache)

    assert result.content == ANSWER
    assert len(calls) == 1
    assert len(cache.writes) == 1
    assert ANSWER in str(cache.writes)


@pytest.mark.parametrize("run_query", [_kg_query, _naive_query], ids=["kg", "naive"])
async def test_cached_answer_with_reasoning_trace_is_still_served(
    stub_query_context, run_query
):
    """Control: a cached trace-plus-answer is a hit; the LLM is not called."""
    cached = f"<think>{REASONING}</think>{ANSWER}"
    cache = _RecordingKV(cached_answer=cached)
    calls = []

    async def _llm(*_args, **_kwargs):
        calls.append(1)
        return "should not be generated"

    result = await run_query(_llm, cache)

    assert result.content == cached
    assert calls == []
    assert cache.writes == []
