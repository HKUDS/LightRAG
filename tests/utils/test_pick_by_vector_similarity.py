"""Regression tests for :func:`lightrag.utils.pick_by_vector_similarity`.

Missing stored vectors for candidate chunks are treated as a data
inconsistency. The function returns an empty list so callers fall back to the
WEIGHT retrieval method, while emitting a diagnostic warning that includes
counts and a sample of missing chunk IDs.
"""

import logging

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


async def test_partial_missing_vectors_falls_back_and_warns(caplog):
    query_embedding = np.array([1.0, 0.0, 0.0])
    entity_info = [{"sorted_chunks": ["c0", "c1", "c2"]}]
    # c2 has no stored vector; the function should not rank the partial set.
    vdb = _StubChunksVDB(
        {
            "c0": np.array([1.0, 0.0, 0.0]),
            "c1": np.array([0.9, 0.1, 0.0]),
        }
    )

    with caplog.at_level(logging.WARNING):
        selected = await pick_by_vector_similarity(
            query="q",
            text_chunks_storage=None,
            chunks_vdb=vdb,
            num_of_chunks=5,
            entity_info=entity_info,
            embedding_func=None,
            query_embedding=query_embedding,
        )

    assert selected == []

    warning_record = next(
        (r for r in caplog.records if "data inconsistency detected" in r.message), None
    )
    assert warning_record is not None, "Expected a data-inconsistency warning"
    assert "expected 3" in warning_record.message
    assert "retrieved 2" in warning_record.message
    assert "missing 1" in warning_record.message
    assert "'c2'" in warning_record.message


async def test_no_vectors_at_all_returns_empty(caplog):
    entity_info = [{"sorted_chunks": ["c0", "c1"]}]
    vdb = _StubChunksVDB({})

    with caplog.at_level(logging.WARNING):
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
    assert any(
        "no vectors retrieved from chunks_vdb" in r.message for r in caplog.records
    )


async def test_all_vectors_present_ranks_normally():
    query_embedding = np.array([1.0, 0.0, 0.0])
    entity_info = [{"sorted_chunks": ["c0", "c1", "c2"]}]
    vdb = _StubChunksVDB(
        {
            "c0": np.array([1.0, 0.0, 0.0]),
            "c1": np.array([0.9, 0.1, 0.0]),
            "c2": np.array([0.0, 1.0, 0.0]),
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

    assert selected == ["c0", "c1", "c2"]
