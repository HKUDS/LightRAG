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


def _config(answer: str) -> dict:
    async def query_model(*_args, **_kwargs):
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


@pytest.mark.offline
@pytest.mark.asyncio
async def test_naive_query_returns_long_answer_unmodified():
    result = await naive_query(
        "How is it configured?",
        _ChunksVDB(),
        QueryParam(mode="naive", enable_rerank=False, stream=False),
        _config(LONG_ANSWER),
        hashing_kv=None,
    )

    assert result.content == LONG_ANSWER


@pytest.mark.offline
@pytest.mark.asyncio
async def test_kg_query_returns_long_answer_unmodified(monkeypatch):
    async def fake_keywords(*_args, **_kwargs):
        return ["configuration"], ["model settings"]

    async def fake_context(*_args, **_kwargs):
        return QueryContextResult(context="small context", raw_data={})

    monkeypatch.setattr("lightrag.operate.get_keywords_from_query", fake_keywords)
    monkeypatch.setattr("lightrag.operate._build_query_context", fake_context)

    config = _config(LONG_ANSWER)
    result = await kg_query(
        "How is it configured?",
        None,
        None,
        None,
        _TextChunks(config),
        QueryParam(mode="mix", enable_rerank=False, stream=False),
        config,
        hashing_kv=None,
    )

    assert result.content == LONG_ANSWER
