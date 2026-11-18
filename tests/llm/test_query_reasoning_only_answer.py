"""Regression: a reasoning-only LLM response is never returned or cached as the answer.

The query paths call the LLM with ``enable_cot=True``. With a reasoning model
whose provider returns ``reasoning_content`` but an empty ``content``, the
OpenAI binding returns ``<think>{reasoning}</think>``: non-empty, so it passed
the binding's empty-content check, was returned as the answer and written to
the query cache (replayed on every identical query). With ``enable_cot=False``
the same response raises.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

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
    def __init__(self):
        self.global_config = {"enable_llm_cache": True}
        self.writes = []

    async def get_by_id(self, _key):
        return None

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


def _openai_query_llm(completion):
    """The query role backed by the real OpenAI binding over a mocked client."""

    async def _llm(prompt, system_prompt=None, history_messages=None, **kwargs):
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=AsyncMock(return_value=completion))
            ),
            close=AsyncMock(),
        )
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

    return _llm


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


async def test_binding_root_cause_enable_cot_turns_empty_into_think_block():
    """Root cause, pinned: the binding is unchanged and behaves as designed."""
    llm = _openai_query_llm(_completion("", REASONING))

    assert await llm(QUERY, enable_cot=True) == f"<think>{REASONING}</think>"
    # Same response without enable_cot: the binding rejects it (after retries).
    with pytest.raises(InvalidResponseError):
        await llm(QUERY, enable_cot=False)


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


async def test_kg_query_rejects_reasoning_only_answer_and_does_not_cache_it(
    stub_query_context,
):
    cache = _RecordingKV()

    with pytest.raises(ReasoningOnlyResponseError):
        await _kg_query(_openai_query_llm(_completion("", REASONING)), cache)

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

    async def _inline_reasoning_only(*_args, **_kwargs):
        # A provider that puts the trace inline in `content`.
        return f"<think>{REASONING}</think>"

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
    rag = _BypassRAG(_openai_query_llm(_completion("", REASONING)))

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
