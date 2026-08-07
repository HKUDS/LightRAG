from lightrag.utils import compute_incremental_chunk_ids


def test_compute_incremental_chunk_ids_preserves_new_chunks_missing_from_existing():
    # Scenario: existing_full_chunk_ids is missing chunk-1, but chunk-1 is in old_chunk_ids
    # and remains in new_chunk_ids (re-indexing / partial storage state).
    existing = ["chunk-3"]
    old = ["chunk-1", "chunk-3"]
    new = ["chunk-1", "chunk-2", "chunk-3"]

    updated = compute_incremental_chunk_ids(existing, old, new)

    # All chunk IDs present in `new` must be preserved in the result.
    assert "chunk-1" in updated
    assert "chunk-2" in updated
    assert "chunk-3" in updated
    assert set(updated) == {"chunk-1", "chunk-2", "chunk-3"}
