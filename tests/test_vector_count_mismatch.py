"""
Tests for vector count mismatch handling in EmbeddingFunc.

Verifies that vector count mismatches raise ValueError by default,
and that the opt-in allow_extra_vectors flag enables safe slicing
for providers known to return extra padding vectors.
"""

import numpy as np
import pytest

from lightrag.utils import EmbeddingFunc


@pytest.mark.offline
@pytest.mark.asyncio
async def test_overflow_raises_by_default():
    """Overflow raises ValueError when allow_extra_vectors is False (default)."""

    async def mock_embed(texts, **kwargs):
        dim = kwargs.get("embedding_dim", 128)
        return np.random.rand(len(texts) * 2, dim).astype(np.float32)

    func = EmbeddingFunc(embedding_dim=128, func=mock_embed, model_name="test-model")
    with pytest.raises(ValueError, match="allow_extra_vectors"):
        await func(["hello", "world"])


@pytest.mark.offline
@pytest.mark.asyncio
async def test_overflow_slices_with_opt_in():
    """Overflow slices to expected count when allow_extra_vectors=True."""

    async def mock_embed(texts, **kwargs):
        dim = kwargs.get("embedding_dim", 128)
        return np.random.rand(len(texts) * 2, dim).astype(np.float32)

    func = EmbeddingFunc(
        embedding_dim=128,
        func=mock_embed,
        model_name="test-model",
        allow_extra_vectors=True,
    )
    result = await func(["hello", "world"])
    assert result.shape == (2, 128)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_underflow_raises_even_with_opt_in():
    """Underflow always raises ValueError, even with allow_extra_vectors=True."""

    async def mock_embed(texts, **kwargs):
        dim = kwargs.get("embedding_dim", 128)
        return np.random.rand(1, dim).astype(np.float32)

    func = EmbeddingFunc(
        embedding_dim=128,
        func=mock_embed,
        model_name="test-model",
        allow_extra_vectors=True,
    )
    with pytest.raises(ValueError, match="Vector count mismatch"):
        await func(["hello", "world", "foo"])


@pytest.mark.offline
@pytest.mark.asyncio
async def test_vector_count_match_passes():
    """When vector count matches, result is returned unchanged."""

    async def mock_embed(texts, **kwargs):
        dim = kwargs.get("embedding_dim", 128)
        return np.random.rand(len(texts), dim).astype(np.float32)

    func = EmbeddingFunc(embedding_dim=128, func=mock_embed, model_name="test-model")
    result = await func(["hello", "world"])
    assert result.shape == (2, 128)


@pytest.mark.offline
@pytest.mark.asyncio
async def test_error_message_includes_provider_name():
    """Error message includes provider name, batch size, and counts."""

    async def mock_embed(texts, **kwargs):
        dim = kwargs.get("embedding_dim", 128)
        return np.random.rand(len(texts) * 3, dim).astype(np.float32)

    func = EmbeddingFunc(embedding_dim=128, func=mock_embed, model_name="azure-openai")
    with pytest.raises(ValueError, match="azure-openai") as exc_info:
        await func(["a", "b"])
    assert "expected 2" in str(exc_info.value)
    assert "got 6" in str(exc_info.value)
