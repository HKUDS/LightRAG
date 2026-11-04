"""Non-positive batch/concurrency settings are rejected at construction.

``embedding_batch_num`` is the stride of the vector-store flush loop, and
``embedding_func_max_async`` / ``llm_model_max_async`` /
``rerank_model_max_async`` (plus per-role ``max_async`` overrides) are the
worker counts of the ``priority_limit_async_func_call`` queues. All of
them used to accept 0 or negative values silently: the batch size only
failed at the first document insert with a ``range()``/concatenate error
that never named the setting, and a non-positive concurrency limit
dispatched zero workers, so the wrapped calls never executed and the
pipeline hung without an error. Construction now fails fast, matching the
neighbouring ``__post_init__`` checks (``MAX_PENDING_DOCUMENTS`` and
friends).
"""

from __future__ import annotations

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.utils import EmbeddingFunc, Tokenizer

pytestmark = pytest.mark.offline


class _SimpleTokenizerImpl:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


async def _dummy_embedding(texts: list[str]) -> np.ndarray:
    return np.ones((len(texts), 8), dtype=float)


async def _dummy_llm(*args, **kwargs) -> str:
    return "ok"


async def _dummy_rerank(query: str, documents: list[str], **kwargs):
    return list(range(len(documents)))


@pytest.fixture(autouse=True)
def _shared():
    initialize_share_data()
    yield
    finalize_share_data()


def _make_rag(tmp_path, **kwargs) -> LightRAG:
    return LightRAG(
        working_dir=str(tmp_path / "wd"),
        workspace="nonpositive-config",
        llm_model_func=_dummy_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8, max_token_size=8192, func=_dummy_embedding
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
        **kwargs,
    )


@pytest.mark.parametrize("bad", [0, -1])
def test_embedding_batch_num_nonpositive_is_rejected(tmp_path, bad):
    with pytest.raises(ValueError, match="EMBEDDING_BATCH_NUM"):
        _make_rag(tmp_path, embedding_batch_num=bad)


@pytest.mark.parametrize("bad", [0, -2])
def test_embedding_func_max_async_nonpositive_is_rejected(tmp_path, bad):
    with pytest.raises(ValueError, match="EMBEDDING_FUNC_MAX_ASYNC"):
        _make_rag(tmp_path, embedding_func_max_async=bad)


def test_llm_model_max_async_nonpositive_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="MAX_ASYNC_LLM"):
        _make_rag(tmp_path, llm_model_max_async=0)


def test_rerank_model_max_async_nonpositive_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="MAX_ASYNC_RERANK"):
        _make_rag(
            tmp_path,
            rerank_model_func=_dummy_rerank,
            rerank_model_max_async=0,
        )


def test_role_max_async_override_nonpositive_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="max_async for LLM role 'query'"):
        _make_rag(tmp_path, role_llm_configs={"query": {"max_async": 0}})


def test_role_max_async_env_nonpositive_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("QUERY_MAX_ASYNC_LLM", "0")
    with pytest.raises(ValueError, match="max_async for LLM role 'query'"):
        _make_rag(tmp_path)


def test_role_max_async_runtime_update_nonpositive_is_rejected(tmp_path):
    rag = _make_rag(tmp_path)
    with pytest.raises(ValueError, match="max_async for LLM role 'query'"):
        rag.update_llm_role_config("query", max_async=0)
    # The failed update rolls back: the previous limit stays in effect.
    assert rag._get_effective_role_llm_max_async("query") == 4


def test_positive_values_are_accepted(tmp_path):
    rag = _make_rag(
        tmp_path,
        embedding_batch_num=1,
        embedding_func_max_async=1,
        llm_model_max_async=1,
        rerank_model_func=_dummy_rerank,
        rerank_model_max_async=1,
        role_llm_configs={"query": {"max_async": 1}},
    )
    assert rag.embedding_batch_num == 1
    assert rag.embedding_func_max_async == 1
    assert rag.llm_model_max_async == 1
    assert rag.rerank_model_max_async == 1
    assert rag._get_effective_role_llm_max_async("query") == 1
