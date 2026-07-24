"""Regression tests for :func:`lightrag.utils.pick_by_vector_similarity`.

A single candidate chunk missing its stored vector (common after deletions or
partial re-embeds) must not discard the ranking of every other candidate.
"""

import numpy as np
import pytest

from lightrag.utils import pick_by_vector_similarity

pytestmark = pytest.mark.offline


class _StubChunksVDB:
    """Minimal vector store exposing only get_vectors_by_ids."""

    def __init__(self, vectors):
        self._vectors = vectors

    async def get_vectors_by_ids(self, ids):
        # Mimic a real backend: only return entries that actually exist.
        return {cid: self._vectors[cid] for cid in ids if cid in self._vectors}


async def test_partial_missing_vectors_still_ranks_available_chunks():
    query_embedding = np.array([1.0, 0.0, 0.0])
    entity_info = [{"sorted_chunks": ["c0", "c1", "c2"]}]
    # c2 has no stored vector; c0 is the closest match to the query.
    vdb = _StubChunksVDB(
        {
            "c0": np.array([1.0, 0.0, 0.0]),
            "c1": np.array([0.9, 0.1, 0.0]),
        }
    )

    selected = await pick_by_vector_similarity(
        query="q",
        text_chunks_storage=None,
        chunks_vdb=vdb,
        num_of_chunks=5,
        entity_info=entity_info,
        embedding_func=None,
        query_embedding=query_embedding,
    )

    # Before the fix this returned [] because one of three chunks lacked a
    # vector, silently disabling the default VECTOR pick method for the query.
    assert selected == ["c0", "c1"]


async def test_no_vectors_at_all_returns_empty():
    entity_info = [{"sorted_chunks": ["c0", "c1"]}]
    vdb = _StubChunksVDB({})

    selected = await pick_by_vector_similarity(
        query="q",
        text_chunks_storage=None,
        chunks_vdb=vdb,
        num_of_chunks=5,
        entity_info=entity_info,
        embedding_func=None,
        query_embedding=np.array([1.0, 0.0, 0.0]),
    )

    assert selected == []
