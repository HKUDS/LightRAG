"""Regression tests for :func:`lightrag.utils.cosine_similarity`."""

import numpy as np
import pytest

from lightrag.utils import cosine_similarity

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
