from lightrag.utils import compute_incremental_chunk_ids


def test_compute_incremental_chunk_ids_standard_update():
    existing = ["chunk-1", "chunk-2", "chunk-3"]
    old = ["chunk-1", "chunk-2"]
    new = ["chunk-2", "chunk-4"]

    updated = compute_incremental_chunk_ids(existing, old, new)
    assert updated == ["chunk-2", "chunk-3", "chunk-4"]


def test_compute_incremental_chunk_ids_does_not_resurrect_stale_graph_ids():
    # Scenario: chunk-1 is in old_chunk_ids (from graph source_id) and new_chunk_ids,
    # but was previously deleted from existing tracking storage (chunk-3 only).
    # Incremental update must append chunk-2 (new addition) without resurrecting stale chunk-1.
    existing = ["chunk-3"]
    old = ["chunk-1", "chunk-3"]
    new = ["chunk-1", "chunk-2", "chunk-3"]

    updated = compute_incremental_chunk_ids(existing, old, new)

    assert "chunk-1" not in updated
    assert "chunk-2" in updated
    assert "chunk-3" in updated
    assert updated == ["chunk-3", "chunk-2"]
