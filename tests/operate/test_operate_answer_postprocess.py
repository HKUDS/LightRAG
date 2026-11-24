"""A non-streaming query answer must reach the caller as the LLM produced it.

``kg_query`` and ``naive_query`` both carried a legacy clean-up block, gated
on ``len(response) > len(sys_prompt)``, that ran ``.replace("user", "")``,
``.replace("model", "")`` and ``.replace(query, "")`` over the answer, and
``naive_query`` additionally sliced off its first ``len(sys_prompt)``
characters. It was meant for bindings that echoed the prompt back, but the
only such binding (``lightrag/llm/hf.py``) already drops the prompt tokens
before decoding, so by the time the answer reaches ``operate.py`` there is no
echo left to remove: the block can only damage a legitimate answer. It fires
whenever the answer is longer than the rendered system prompt -- a small
retrieved context plus a detailed answer is enough -- and only on the
non-streaming path, so ``/query`` and ``/query/stream`` returned different
text for the same model output.
"""

import pytest

from lightrag.base import QueryContextResult, QueryParam
from lightrag.operate import kg_query, naive_query
from lightrag.utils import Tokenizer


class _CharTokenizer:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


class _ChunksVDB:
    cosine_better_than_threshold = 0.0

    async def query(self, *_args, **_kwargs):
        return [
            {
                "id": "chunk-1",
                "content": "The user manual describes the model settings.",
                "file_path": "manual.md",
            }
        ]


class _TextChunks:
    def __init__(self, global_config):
        self.global_config = global_config


# Longer than any system prompt these tests render (~3.3k chars), and it
# contains "user" / "model" as parts of ordinary words. The query text is kept
# out of it so the assertions do not depend on `.replace(query, "")`.
LONG_ANSWER = (
    "The user manual explains how each model is configured. "
    + "Every setting is described step by step for the operator. " * 80
    + "Finally, users can reset the model to its defaults."
)


QUERY = "How is it configured?"


class _AnswerCacheKV:
    """Answer-cache storage that hits on every key with ``answer``."""

    def __init__(self, answer: str):
        self.global_config = {"enable_llm_cache": True}
        self._answer = answer

    async def get_by_id(self, _key):
        return {"return": self._answer, "create_time": 0}


def _config(answer: str, seen: dict) -> dict:
    async def query_model(*_args, system_prompt=None, **_kwargs):
        seen["sys_prompt"] = system_prompt or ""
        return answer

    return {
        "tokenizer": Tokenizer("fake", _CharTokenizer()),
        "role_llm_funcs": {"query": query_model},
        "addon_params": {"language": "en"},
        "min_rerank_score": 0.0,
        "max_total_tokens": 30000,
        "related_chunk_number": 1,
        "kg_chunk_pick_method": "WEIGHT",
    }


async def _run_query(kind: str, monkeypatch, config: dict, hashing_kv=None):
    if kind == "naive":
        return await naive_query(
            QUERY,
            _ChunksVDB(),
            QueryParam(mode="naive", enable_rerank=False, stream=False),
            config,
            hashing_kv=hashing_kv,
        )

    async def fake_keywords(*_args, **_kwargs):
        return ["configuration"], ["model settings"]

    async def fake_context(*_args, **_kwargs):
        return QueryContextResult(context="small context", raw_data={})

    monkeypatch.setattr("lightrag.operate.get_keywords_from_query", fake_keywords)
    monkeypatch.setattr("lightrag.operate._build_query_context", fake_context)
    return await kg_query(
        QUERY,
        None,
        None,
        None,
        _TextChunks(config),
        QueryParam(mode="mix", enable_rerank=False, stream=False),
        config,
        hashing_kv=hashing_kv,
    )


def _assert_old_block_would_fire(seen: dict) -> None:
    # The removed block only ran when the answer outgrew the rendered system
    # prompt. If prompt templates ever grow past LONG_ANSWER, this test would
    # pass even with the block restored, so pin the precondition explicitly.
    assert seen["sys_prompt"], "stub LLM did not receive a system prompt"
    assert len(LONG_ANSWER) > len(seen["sys_prompt"]), (
        "LONG_ANSWER no longer outgrows the system prompt; lengthen it so "
        "this test still guards the removed clean-up block"
    )


@pytest.mark.offline
@pytest.mark.parametrize("kind", ["naive", "kg"])
async def test_query_returns_long_answer_unmodified(kind, monkeypatch):
    seen: dict = {}
    result = await _run_query(kind, monkeypatch, _config(LONG_ANSWER, seen))

    _assert_old_block_would_fire(seen)
    assert result.content == LONG_ANSWER


@pytest.mark.offline
@pytest.mark.parametrize("kind", ["naive", "kg"])
async def test_query_cache_hit_returns_cached_answer_unmodified(kind, monkeypatch):
    # The cache stores the raw answer and the clean-up block used to run on
    # every hit as well. Render the prompt once without a cache to prove the
    # precondition, then serve the same request from the cache.
    seen: dict = {}
    await _run_query(kind, monkeypatch, _config(LONG_ANSWER, seen))
    _assert_old_block_would_fire(seen)

    async def must_not_call(*_args, **_kwargs):
        raise AssertionError("cache hit must not call the LLM")

    config = _config(LONG_ANSWER, {})
    config["role_llm_funcs"] = {"query": must_not_call}
    result = await _run_query(
        kind, monkeypatch, config, hashing_kv=_AnswerCacheKV(LONG_ANSWER)
    )

    assert result.content == LONG_ANSWER
