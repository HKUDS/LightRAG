"""Regression tests for ``compute_incremental_chunk_ids``.

The helper applies a graph ``source_id`` delta (old -> new) onto the authoritative
entity/relation chunk-tracking row. See the function's "Authority model" docstring
section before changing any expectation here: tracking wins over the graph, so IDs
that tracking has pruned must stay pruned even when the graph still names them.
"""

from lightrag.utils import compute_incremental_chunk_ids


def test_compute_incremental_chunk_ids_standard_update():
    """Removals drop out in place; genuine additions land at the end."""
    existing = ["chunk-1", "chunk-2", "chunk-3"]
    old = ["chunk-1", "chunk-2"]
    new = ["chunk-2", "chunk-4"]

    updated = compute_incremental_chunk_ids(existing, old, new)

    assert updated == ["chunk-2", "chunk-3", "chunk-4"]


def test_compute_incremental_chunk_ids_does_not_resurrect_stale_graph_ids():
    """An ID in old AND new but missing from tracking was pruned on purpose.

    Tracking is authoritative and the graph's source_id may still name a chunk that
    a previous purge already removed. Appending every new_chunk_ids entry would write
    that stale attribution back into the authoritative store.
    """
    existing = ["chunk-3"]
    old = ["chunk-1", "chunk-3"]
    new = ["chunk-1", "chunk-2", "chunk-3"]

    updated = compute_incremental_chunk_ids(existing, old, new)

    assert "chunk-1" not in updated
    assert updated == ["chunk-3", "chunk-2"]


def test_rename_does_not_reseed_pruned_ids_when_source_id_is_unchanged():
    """The motivating production path for the rule above.

    ``_edit_entity_impl`` calls this helper on every rename — including when the
    source_id did not change at all (``is_renaming`` alone satisfies its guard), so
    old == new. The pruned ID must not reappear when the row is re-keyed.
    """
    existing = ["live"]
    old = ["stale", "live"]
    new = ["stale", "live"]

    updated = compute_incremental_chunk_ids(existing, old, new)

    assert updated == ["live"]


def test_no_duplicate_when_existing_already_holds_a_new_id():
    """``existing`` being a strict superset of ``old`` is the normal case.

    The graph source_id is a truncated view of the tracking row
    (``apply_source_ids_limit``), so an ID can be classified as an addition
    (new - old) while already sitting in tracking. It must not be appended twice.
    """
    existing = ["chunk-a", "chunk-b"]
    old = ["chunk-b"]
    new = ["chunk-a", "chunk-b"]

    updated = compute_incremental_chunk_ids(existing, old, new)

    assert updated == ["chunk-a", "chunk-b"]


def test_empty_chunk_ids_are_dropped_from_both_inputs():
    """Empty IDs never reach storage, so they never inflate the stored count."""
    existing = ["", "chunk-a"]
    old = []
    new = ["", "chunk-b"]

    updated = compute_incremental_chunk_ids(existing, old, new)

    assert updated == ["chunk-a", "chunk-b"]


def test_all_chunks_removed_yields_empty_list():
    """A delta that removes everything leaves nothing behind."""
    existing = ["chunk-1", "chunk-2"]
    old = ["chunk-1", "chunk-2"]
    new = []

    updated = compute_incremental_chunk_ids(existing, old, new)

    assert updated == []
