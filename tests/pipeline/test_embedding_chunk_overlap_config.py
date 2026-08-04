"""``embedding_chunk_overlap_token_size`` contract (design: "Tokenizer safe
splitting and truncation refactor").

Independent from the chunker's own ``chunk_overlap_token_size`` — this field
only governs the overlap used by the embedding hard fallback
(``enforce_chunk_token_limit_before_embedding``) when a chunk is still over
the embedding model's context limit after chunking. Some chunker strategies
(V) deliberately zero out ``chunk_overlap_token_size`` for unrelated reasons,
so the two fields must never share a value or a fallback.
"""

from __future__ import annotations

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.constants import DEFAULT_EMBEDDING_CHUNK_OVERLAP_TOKEN_SIZE
from lightrag.utils import EmbeddingFunc, Tokenizer

pytestmark = pytest.mark.offline


class _SimpleTokenizerImpl:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


async def _dummy_embedding(texts: list[str]) -> np.ndarray:
    return np.ones((len(texts), 8), dtype=float)


async def _dummy_llm(*_args, **_kwargs) -> str:
    return "mock"


def _make_rag(tmp_path, **overrides):
    kwargs = dict(
        working_dir=str(tmp_path / "embedding-overlap-cfg"),
        workspace="embedding-overlap-cfg",
        llm_model_func=_dummy_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8, max_token_size=8192, func=_dummy_embedding
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
    )
    kwargs.update(overrides)
    return LightRAG(**kwargs)


def test_default_matches_chunk_overlap_token_size_magnitude(tmp_path):
    rag = _make_rag(tmp_path)
    assert (
        rag.embedding_chunk_overlap_token_size
        == DEFAULT_EMBEDDING_CHUNK_OVERLAP_TOKEN_SIZE
    )
    assert DEFAULT_EMBEDDING_CHUNK_OVERLAP_TOKEN_SIZE == 100


def test_zero_disables_the_fallback_overlap(tmp_path):
    rag = _make_rag(tmp_path, embedding_chunk_overlap_token_size=0)
    assert rag.embedding_chunk_overlap_token_size == 0


def test_negative_value_is_rejected_at_construction(tmp_path):
    with pytest.raises(ValueError, match="EMBEDDING_CHUNK_OVERLAP_TOKEN_SIZE"):
        _make_rag(tmp_path, embedding_chunk_overlap_token_size=-1)


def test_does_not_read_or_fall_back_to_chunk_overlap_token_size(tmp_path):
    """The two fields are independent: setting one must not move the other."""
    rag = _make_rag(
        tmp_path,
        chunk_overlap_token_size=0,  # e.g. V-strategy's own deliberate zeroing
        embedding_chunk_overlap_token_size=42,
    )
    assert rag.chunk_overlap_token_size == 0
    assert rag.embedding_chunk_overlap_token_size == 42


def test_global_config_carries_the_field_for_pipeline_use(tmp_path):
    rag = _make_rag(tmp_path, embedding_chunk_overlap_token_size=7)
    assert rag._build_global_config()["embedding_chunk_overlap_token_size"] == 7
