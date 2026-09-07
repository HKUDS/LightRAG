"""Determinism tests for :func:`lightrag.utils.pick_by_vector_similarity`.

Candidate chunk IDs are deduplicated while preserving first-occurrence order.
That order is the tie-break for equal cosine similarities and is also what the
storage-error fallback truncates, so a hash-seed dependent order would make the
evidence handed to the LLM vary between processes for the same query.
"""

from copy import deepcopy
from itertools import permutations
from unittest.mock import AsyncMock

import numpy as np
import pytest

from lightrag.utils import pick_by_vector_similarity

pytestmark = pytest.mark.offline


class VectorStore:
    def __init__(self, vectors):
        self.vectors = vectors
        self.requested_ids = None

    async def get_vectors_by_ids(self, ids):
        self.requested_ids = list(ids)
        # The returned mapping's order must not control tie-breaking either.
        return {key: self.vectors[key] for key in reversed(ids) if key in self.vectors}


async def _select(entities, vectors, count, *, embedding=None, query_embedding=None):
    store = VectorStore(vectors)
    result = await pick_by_vector_similarity(
        query="evidence",
        text_chunks_storage=None,
        chunks_vdb=store,
        num_of_chunks=count,
        entity_info=entities,
        embedding_func=embedding,
        query_embedding=(
            np.array([1.0, 0.0]) if query_embedding is None else query_embedding
        ),
    )
    return result, store


@pytest.mark.parametrize("order", list(permutations(["chunk-a", "chunk-b", "chunk-c"])))
async def test_similarity_ties_preserve_candidate_order(order):
    vectors = {key: np.array([1.0, 0.0]) for key in order}
    result, store = await _select([{"sorted_chunks": list(order)}], vectors, 2)
    assert result == list(order[:2])
    assert store.requested_ids == list(order)


async def test_duplicate_chunks_keep_their_first_position():
    entities = [
        {"sorted_chunks": ["chunk-c", "chunk-a", "chunk-c"]},
        {"sorted_chunks": ["chunk-b", "chunk-a"]},
    ]
    before = deepcopy(entities)
    vectors = {key: np.array([1.0, 0.0]) for key in ["chunk-a", "chunk-b", "chunk-c"]}
    result, store = await _select(entities, vectors, 10)
    assert result == ["chunk-c", "chunk-a", "chunk-b"]
    assert store.requested_ids == result
    assert entities == before


async def test_similarity_still_takes_precedence_over_candidate_order():
    vectors = {
        "low": np.array([-1.0, 0.0]),
        "high": np.array([1.0, 0.0]),
        "middle": np.array([0.0, 1.0]),
    }
    result, _ = await _select(
        [{"sorted_chunks": ["low", "high", "middle"]}], vectors, 2
    )
    assert result == ["high", "middle"]


async def test_ties_only_affect_equal_scores():
    vectors = {
        "tie-b": np.array([0.0, 1.0]),
        "best": np.array([1.0, 0.0]),
        "tie-a": np.array([0.0, -1.0]),
        "worst": np.array([-1.0, 0.0]),
    }
    result, _ = await _select([{"sorted_chunks": list(vectors)}], vectors, 3)
    assert result == ["best", "tie-b", "tie-a"]


@pytest.mark.parametrize("order", list(permutations(["chunk-a", "chunk-b", "chunk-c"])))
async def test_storage_error_fallback_preserves_candidate_order(order):
    class FailingStore:
        async def get_vectors_by_ids(self, ids):
            raise RuntimeError("storage unavailable")

    result = await pick_by_vector_similarity(
        "evidence",
        None,
        FailingStore(),
        2,
        [{"sorted_chunks": list(order)}],
        None,
        query_embedding=np.array([1.0, 0.0]),
    )
    assert result == list(order[:2])


@pytest.mark.parametrize(
    "entities, count",
    [([], 2), ([{}], 2), ([{"sorted_chunks": ["a"]}], 0)],
)
async def test_empty_selection_does_not_read_vectors(entities, count):
    result, store = await _select(entities, {}, count)
    assert result == []
    assert store.requested_ids is None


async def test_embedding_computation_path_uses_same_ordering():
    embedding = AsyncMock(return_value=np.array([[1.0, 0.0]]))
    vectors = {key: np.array([1.0, 0.0]) for key in ["chunk-c", "chunk-a", "chunk-b"]}
    store = VectorStore(vectors)
    result = await pick_by_vector_similarity(
        "evidence",
        None,
        store,
        2,
        [{"sorted_chunks": list(vectors)}],
        embedding,
    )
    embedding.assert_awaited_once_with(["evidence"], context="query")
    assert result == ["chunk-c", "chunk-a"]
