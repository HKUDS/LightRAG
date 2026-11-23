"""Regression tests for non-finite vector handling in :func:`lightrag.utils.cosine_similarity`."""

import numpy as np
import pytest

from lightrag.utils import cosine_similarity

pytestmark = pytest.mark.offline


def test_cosine_similarity_nan_vectors_return_zero():
    v1 = np.array([np.nan, 1.0])
    v2 = np.array([1.0, 1.0])
    assert cosine_similarity(v1, v2) == 0.0


def test_cosine_similarity_inf_vectors_return_zero():
    v1 = np.array([np.inf, 1.0])
    v2 = np.array([1.0, 1.0])
    assert cosine_similarity(v1, v2) == 0.0
