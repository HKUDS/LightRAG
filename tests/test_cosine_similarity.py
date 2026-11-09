"""Regression tests for :func:`lightrag.utils.cosine_similarity`."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest

from lightrag.utils import cosine_similarity, pick_by_vector_similarity

pytestmark = pytest.mark.offline


def test_zero_vectors_return_zero_not_nan():
    score = cosine_similarity(np.zeros(3), np.zeros(3))
    assert score == 0.0
    assert not np.isnan(score)


def test_zero_against_nonzero_returns_zero():
    score = cosine_similarity(np.zeros(2), np.array([1.0, 0.0]))
    assert score == 0.0
    assert not np.isnan(score)


def test_identical_unit_vectors_return_one():
    v = np.array([1.0, 0.0, 0.0])
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_orthogonal_unit_vectors_return_zero():
    assert cosine_similarity(
        np.array([1.0, 0.0]), np.array([0.0, 1.0])
    ) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "dtype,magnitude",
    [
        (np.float16, 300.0),
        (np.int8, 100),
        (np.float32, 1e-30),
        (np.float32, 1e30),
        (np.float64, 1e-200),
        (np.float64, 1e200),
    ],
)
@pytest.mark.parametrize("direction", [1, -1])
def test_finite_vectors_keep_their_direction_across_scales(dtype, magnitude, direction):
    vector = np.array([magnitude, magnitude], dtype=dtype)
    other = vector * direction
    original = vector.copy()
    with np.errstate(all="ignore"):
        score = cosine_similarity(vector, other)
    assert score == pytest.approx(float(direction))
    np.testing.assert_array_equal(vector, original)


def test_different_finite_scales_still_compare_as_parallel():
    small = np.array([1e-200, 2e-200])
    large = np.array([1e200, 2e200])
    with np.errstate(all="ignore"):
        score = cosine_similarity(small, large)
    assert score == pytest.approx(1.0)


def test_empty_vectors_keep_zero_similarity():
    assert cosine_similarity([], []) == 0.0


@pytest.mark.asyncio
async def test_vector_chunk_selection_ranks_parallel_above_opposite_float16():
    query_vector = np.array([300.0, 300.0], dtype=np.float16)
    vectors = SimpleNamespace(
        get_vectors_by_ids=AsyncMock(
            return_value={"opposite": -query_vector, "parallel": query_vector.copy()}
        )
    )
    with np.errstate(all="ignore"):
        selected = await pick_by_vector_similarity(
            query="query",
            text_chunks_storage=None,
            chunks_vdb=vectors,
            num_of_chunks=1,
            entity_info=[{"sorted_chunks": ["opposite", "parallel"]}],
            embedding_func=None,
            query_embedding=query_vector,
        )
    assert selected == ["parallel"]
