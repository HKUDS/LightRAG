"""Unit tests for ``chunking_by_semantic_vector`` (process_options=V)."""

import asyncio
import logging

import numpy as np
import pytest

pytest.importorskip("langchain_experimental")

from lightrag.chunker import chunking_by_semantic_vector  # noqa: E402
from lightrag.utils import EmbeddingFunc, Tokenizer, TokenizerInterface  # noqa: E402


class _CharTokenizer(TokenizerInterface):
    """1 char ≈ 1 token."""

    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)


def _tok() -> Tokenizer:
    return Tokenizer("char-tokenizer", _CharTokenizer())


def _make_deterministic_embedding(dim: int = 8) -> EmbeddingFunc:
    """A toy async embedding func that hashes each input text into a
    stable unit vector — enough to drive SemanticChunker without needing
    a real model."""

    async def _embed(texts, **kwargs):
        rng = np.random.default_rng(seed=0)
        # Use a simple hash → seeded rng to get reproducible vectors per text.
        rows = []
        for text in texts:
            seed = abs(hash(text)) % (2**32)
            rng = np.random.default_rng(seed=seed)
            vec = rng.normal(size=dim).astype(np.float32)
            vec /= np.linalg.norm(vec) or 1.0
            rows.append(vec)
        return np.vstack(rows)

    return EmbeddingFunc(embedding_dim=dim, max_token_size=4096, func=_embed)


@pytest.mark.offline
def test_v_chunker_runs_with_stub_embedding():
    """Async chunker should split a multi-sentence body into ≥1 chunk
    when given a working embedding func."""
    body = (
        "Quantum mechanics describes nature at small scales. "
        "It contradicts classical intuition. "
        "Bread is baked from flour. "
        "Sourdough requires a long fermentation. "
    )

    async def _run():
        chunks = await chunking_by_semantic_vector(
            _tok(),
            body,
            chunk_token_size=200,
            embedding_func=_make_deterministic_embedding(),
        )
        return chunks

    chunks = asyncio.run(_run())

    assert len(chunks) >= 1
    # Each chunk dict has the canonical schema.
    assert all({"tokens", "content", "chunk_order_index"} <= set(c) for c in chunks)
    # chunk_order_index is contiguous starting at 0.
    assert [c["chunk_order_index"] for c in chunks] == list(range(len(chunks)))
    # No empty content rows.
    assert all(c["content"].strip() for c in chunks)


class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.mark.offline
def test_v_chunker_falls_back_to_recursive_when_no_embedding():
    """When ``embedding_func`` is None, V must log a warning and route
    to chunking_by_recursive_character (R) — V's only differentiator
    is embeddings, so without them R is the closest neighbour."""
    body = "Para A.\n\nPara B for fallback test.\n\nPara C."

    lightrag_logger = logging.getLogger("lightrag")
    handler = _ListHandler()
    handler.setLevel(logging.WARNING)
    lightrag_logger.addHandler(handler)
    try:

        async def _run():
            return await chunking_by_semantic_vector(
                _tok(),
                body,
                chunk_token_size=20,
                embedding_func=None,
            )

        chunks = asyncio.run(_run())
    finally:
        lightrag_logger.removeHandler(handler)

    assert len(chunks) >= 1
    assert any(
        "embedding_func is None" in rec.getMessage()
        for rec in handler.records
        if rec.levelno == logging.WARNING
    )


@pytest.mark.offline
def test_v_chunker_empty_input_returns_empty_list():
    async def _run():
        return await chunking_by_semantic_vector(_tok(), "")

    assert asyncio.run(_run()) == []
