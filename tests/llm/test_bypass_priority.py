"""``aquery_llm(mode="bypass")`` queues its answer at query priority.

The bypass answer is an interactive query-role call like the kg/naive
answers (operate.py) and the Ollama ``/api/chat`` bypass, so it must carry
``DEFAULT_QUERY_PRIORITY``; summary priority would queue it behind every
pending RAG answer on the same role queue.
"""

import pytest

from lightrag import LightRAG
from lightrag.base import QueryParam
from lightrag.constants import DEFAULT_QUERY_PRIORITY, DEFAULT_SUMMARY_PRIORITY

pytestmark = pytest.mark.offline


class _FakeRAG:
    """Minimal stand-in exposing only what the bypass branch touches."""

    def __init__(self, llm_func):
        self._llm = llm_func

    def _build_global_config(self):
        return {"role_llm_funcs": {"query": self._llm}}


@pytest.mark.asyncio
async def test_bypass_answer_uses_query_priority():
    seen: list[dict] = []

    async def llm(prompt, **kwargs):
        seen.append(kwargs)
        return "answer"

    result = await LightRAG.aquery_llm(
        _FakeRAG(llm), "question", param=QueryParam(mode="bypass", stream=False)
    )

    assert result["llm_response"]["content"] == "answer"
    assert len(seen) == 1
    # The two constants differ, so the assertion tells the right one apart.
    assert DEFAULT_QUERY_PRIORITY != DEFAULT_SUMMARY_PRIORITY
    assert seen[0]["_priority"] == DEFAULT_QUERY_PRIORITY
