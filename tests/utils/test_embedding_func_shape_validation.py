"""EmbeddingFunc.__call__: result shape validation must identify WHICH shape
mismatch happened, not just that total elements did not add up.

Issue #2549: multimodal PDF ingestion crashed when an embedding provider
returned twice as many vectors as expected. A prior fix (rejected, see PR
#3537) tried to detect the 2x case from total element count and silently
slice the "extra" rows off. That is unsafe: a (N, 2D) result (wrong output
dimension) divides out to the same "actual vectors" number as a genuine
(2N, D) result (wrong vector count), so a total-element check reports the
wrong diagnosis for one of the two cases, and slicing rows off a (2N, D)
result has no general contract for which N of the 2N rows are the real ones.

The fix instead checks result.ndim / result.shape[0] / result.shape[1]
directly, so each shape mismatch is identified and rejected on its own
terms -- never reshaped, sliced, or silently accepted.
"""

from __future__ import annotations

import numpy as np
import pytest

from lightrag.utils import EmbeddingFunc

pytestmark = pytest.mark.offline


def _make(func, embedding_dim=4):
    return EmbeddingFunc(embedding_dim=embedding_dim, func=func)


@pytest.mark.asyncio
async def test_correct_shape_passes_through_unchanged():
    async def embed(texts):
        return np.zeros((len(texts), 4), dtype=np.float32)

    result = await _make(embed)(["a", "b", "c"])
    assert result.shape == (3, 4)


@pytest.mark.asyncio
async def test_single_item_batch_with_2d_result_passes():
    # Real providers wrap a single embedding in a (1, D) array, not a bare
    # 1D (D,) vector -- confirm the strict ndim==2 check accepts that shape.
    async def embed(texts):
        return np.zeros((len(texts), 4), dtype=np.float32)

    result = await _make(embed)(["only one"])
    assert result.shape == (1, 4)


@pytest.mark.asyncio
async def test_doubled_vector_count_is_rejected_not_sliced():
    """(2N, D): a genuine vector-count doubling. Must raise -- never
    silently truncated to the first N rows, since there is no contract
    that the first N rows are the real ones."""

    async def embed(texts):
        return np.zeros((len(texts) * 2, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="Vector count mismatch"):
        await _make(embed)(["a", "b", "c"])


@pytest.mark.asyncio
async def test_doubled_dimension_is_rejected_as_dimension_not_count():
    """(N, 2D): a genuine dimension mismatch. Must be reported as a
    dimension mismatch, not misdiagnosed as a doubled vector count -- a
    total-element-count check cannot tell these two cases apart since both
    divide out to the same "actual vectors" total."""

    async def embed(texts):
        return np.zeros((len(texts), 8), dtype=np.float32)

    with pytest.raises(ValueError, match="Embedding dimension mismatch"):
        await _make(embed)(["a", "b", "c"])


@pytest.mark.asyncio
async def test_1d_result_is_rejected():
    async def embed(texts):
        return np.zeros(4, dtype=np.float32)

    with pytest.raises(ValueError, match="unexpected shape"):
        await _make(embed)(["only one"])


@pytest.mark.asyncio
async def test_3d_result_is_rejected():
    async def embed(texts):
        return np.zeros((len(texts), 2, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="unexpected shape"):
        await _make(embed)(["a", "b"])
