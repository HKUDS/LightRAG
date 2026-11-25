"""pick_by_weighted_polling must not reverse-slice on negative max."""

import pytest

from lightrag.utils import pick_by_weighted_polling

pytestmark = pytest.mark.offline


def test_negative_max_does_not_reverse_slice():
    # On buggy code, chunks[:-1] returns all-but-last instead of [].
    chunks = [{"sorted_chunks": ["c1", "c2", "c3"]}]
    assert pick_by_weighted_polling(chunks, -1) == []


def test_zero_max_returns_empty():
    chunks = [{"sorted_chunks": ["c1", "c2", "c3"]}]
    assert pick_by_weighted_polling(chunks, 0) == []


def test_positive_max_still_truncates():
    chunks = [{"sorted_chunks": ["c1", "c2", "c3"]}]
    assert pick_by_weighted_polling(chunks, 2) == ["c1", "c2"]
