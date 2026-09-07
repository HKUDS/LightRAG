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
terms -- never reshaped, sliced, or silently accepted. The raised message
stays short; the likely cause and the fix go to logger.error next to it.

The one shape that is NOT a mismatch is the empty batch: zero inputs carry
no vectors and no dimension to check, so a provider that short-circuits with
``if not texts: return np.array([])`` keeps working.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from lightrag.utils import EmbeddingFunc

pytestmark = pytest.mark.offline


@pytest.fixture
def _propagate_lightrag_logs():
    """The ``lightrag`` logger sets propagate=False; caplog needs it on."""
    lg = logging.getLogger("lightrag")
    old = lg.propagate
    lg.propagate = True
    try:
        yield
    finally:
        lg.propagate = old


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
async def test_too_few_vectors_is_rejected():
    """(N//2, D): underflow is as fatal as overflow. Backends index the
    result positionally, so a short result would silently pair vectors with
    the wrong texts (or IndexError) rather than fail here."""

    async def embed(texts):
        return np.zeros((len(texts) // 2, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="Vector count mismatch"):
        await _make(embed)(["a", "b", "c", "d"])


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
async def test_dimension_is_checked_when_texts_are_not_passed_positionally():
    """The vector count check needs the input list to compare against, so it
    is skipped when the texts arrive by keyword. The dimension check does not
    depend on the input count and must still run -- under the old
    total-element logic this path accepted (N, 2D) outright."""

    async def embed(texts=None):
        return np.zeros((len(texts), 8), dtype=np.float32)

    with pytest.raises(ValueError, match="Embedding dimension mismatch"):
        await _make(embed)(texts=["a", "b", "c"])


@pytest.mark.asyncio
async def test_count_is_checked_when_texts_arrive_by_keyword():
    """The input batch is resolved from the wrapped function's first
    parameter name when the caller does not pass it positionally, so the
    vector count is verifiable on that path too. Before this, a keyword call
    skipped the count check entirely."""

    async def embed(texts):
        return np.zeros((len(texts) * 2, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="Vector count mismatch"):
        await _make(embed)(texts=["a", "b", "c"])


@pytest.mark.asyncio
async def test_keyword_batch_resolution_is_not_hardcoded_to_texts():
    """Resolution reads the wrapped function's actual first parameter, so a
    custom embedding function is free to name it something else."""

    async def embed(sentences):
        return np.zeros((len(sentences) * 2, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="Vector count mismatch"):
        await _make(embed)(sentences=["a", "b", "c"])


@pytest.mark.asyncio
async def test_unresolvable_keyword_batch_skips_the_count_check():
    """A *args-only signature cannot map a keyword to the batch. That makes
    the count unverifiable, which must degrade to skipping that one check --
    not to an error and not to a guess. The dimension check still runs."""

    async def embed(**kwargs):
        return np.zeros((5, 4), dtype=np.float32)

    result = await _make(embed)(texts=["a", "b", "c"])
    assert result.shape == (5, 4)


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


@pytest.mark.asyncio
async def test_empty_batch_with_bare_empty_array_is_accepted():
    """Zero inputs means zero vectors and no dimension to validate. A
    provider short-circuiting with ``np.array([])`` returns a 1D (0,) array;
    rejecting it on ndim would break a common custom-embedding idiom for a
    request that asked for nothing."""

    async def embed(texts):
        if not texts:
            return np.array([])
        return np.zeros((len(texts), 4), dtype=np.float32)

    result = await _make(embed)([])
    assert result.size == 0


@pytest.mark.asyncio
async def test_empty_batch_by_keyword_is_accepted():
    """Same allowance on the keyword path. Keying the empty-batch exemption
    to positional arguments only would reject `np.array([])` here while
    accepting it two lines above -- and the element-count check this PR
    replaces accepted both."""

    async def embed(texts):
        if not texts:
            return np.array([])
        return np.zeros((len(texts), 4), dtype=np.float32)

    result = await _make(embed)(texts=[])
    assert result.size == 0


@pytest.mark.asyncio
async def test_empty_batch_with_empty_2d_array_is_accepted():
    async def embed(texts):
        return np.zeros((0, 4), dtype=np.float32)

    result = await _make(embed)([])
    assert result.shape == (0, 4)


@pytest.mark.asyncio
async def test_empty_input_with_non_empty_result_is_rejected():
    """The empty-batch allowance is keyed on an empty result too: vectors
    returned for zero inputs are a real mismatch, not a no-op."""

    async def embed(texts):
        return np.zeros((1, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="Vector count mismatch"):
        await _make(embed)([])


@pytest.mark.asyncio
async def test_unexpected_ndim_logs_the_required_shape(
    caplog, _propagate_lightrag_logs
):
    """The ndim rejection is the one an author of a custom embedding function
    hits first, so the log has to state the shape actually required rather
    than only the one that arrived."""

    async def embed(texts):
        return np.zeros(4, dtype=np.float32)

    with caplog.at_level(logging.ERROR, logger="lightrag"):
        with pytest.raises(ValueError):
            await _make(embed)(["only one"])

    assert "(len(texts), 4)" in caplog.text
    assert "(1, 4)" in caplog.text


@pytest.mark.asyncio
async def test_count_mismatch_logs_token_limit_remedy(caplog, _propagate_lightrag_logs):
    """A provider splitting an over-long input internally is the documented
    primary cause of this mismatch (issue #2549), and declaring the model's
    token limit is the fix. Keep that remedy in the operator-facing log --
    the short exception message alone does not point anywhere."""

    async def embed(texts):
        return np.zeros((len(texts) * 2, 4), dtype=np.float32)

    with caplog.at_level(logging.ERROR, logger="lightrag"):
        with pytest.raises(ValueError):
            await _make(embed)(["a", "b"])

    assert "EMBEDDING_TOKEN_LIMIT" in caplog.text
    assert "max_token_size" in caplog.text


@pytest.mark.asyncio
async def test_dimension_mismatch_logs_declared_dimension_remedy(
    caplog, _propagate_lightrag_logs
):
    """The actionable fix for (N, 2D) is reconciling the declared dimension
    with the model actually being called -- and clearing the data directory
    afterwards, since existing vectors live in the old space."""

    async def embed(texts):
        return np.zeros((len(texts), 8), dtype=np.float32)

    with caplog.at_level(logging.ERROR, logger="lightrag"):
        with pytest.raises(ValueError):
            await _make(embed)(["a", "b"])

    assert "EMBEDDING_DIM" in caplog.text
    assert "returned 8-dimensional" in caplog.text
    assert "declares 4" in caplog.text
