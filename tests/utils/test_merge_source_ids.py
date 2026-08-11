"""Regression tests for ``merge_source_ids`` normalization.

``merge_source_ids`` is the single collection point for every chunk-id list that
reaches ``entity_chunks_storage`` / ``relation_chunks_storage``, so it has to
flatten ``GRAPH_FIELD_SEP``-joined elements instead of storing them verbatim: a
joined id matches no key in ``text_chunks``, which would drop the chunks from
provenance, from ``apply_source_ids_limit``'s counting, and from retrieval
lookups.
"""

import logging
from contextlib import contextmanager

from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.utils import logger as lightrag_logger, merge_source_ids


@contextmanager
def _captured_logs(caplog, level):
    """Capture lightrag logger records.

    The lightrag logger sets ``propagate = False``, so caplog cannot see it
    unless propagation is re-enabled for the duration of the call.
    """
    original_propagate = lightrag_logger.propagate
    lightrag_logger.propagate = True
    try:
        with caplog.at_level(level, logger=lightrag_logger.name):
            yield
    finally:
        lightrag_logger.propagate = original_propagate


def test_merge_source_ids_splits_graph_field_sep():
    # Scenario: new_ids contains combined source_id strings with GRAPH_FIELD_SEP.
    existing = ["chunk-1"]
    new = [f"chunk-2{GRAPH_FIELD_SEP}chunk-3", "chunk-3"]

    merged = merge_source_ids(existing, new)

    # Every item in merged must be an individual chunk ID (no un-split GRAPH_FIELD_SEP strings).
    assert merged == ["chunk-1", "chunk-2", "chunk-3"]
    assert not any(GRAPH_FIELD_SEP in item for item in merged)


def test_merge_source_ids_splits_existing_ids_too():
    # The stored side is normalized as well: a corrupted entity_chunks_storage
    # row must not survive a merge just because it came from storage.
    existing = [f"chunk-1{GRAPH_FIELD_SEP}chunk-2"]
    new = ["chunk-2", "chunk-3"]

    assert merge_source_ids(existing, new) == ["chunk-1", "chunk-2", "chunk-3"]


def test_merge_source_ids_preserves_first_seen_order_across_both_sequences():
    existing = ["chunk-b", "chunk-a"]
    new = ["chunk-a", "chunk-c", "chunk-b"]

    assert merge_source_ids(existing, new) == ["chunk-b", "chunk-a", "chunk-c"]


def test_merge_source_ids_strips_and_drops_blank_fragments():
    # A whitespace-only element, and blank fragments inside a joined element,
    # contribute nothing; surviving fragments are trimmed.
    merged = merge_source_ids(
        ["   "],
        [f"  chunk-1  {GRAPH_FIELD_SEP}{GRAPH_FIELD_SEP}   {GRAPH_FIELD_SEP}chunk-2"],
    )

    assert merged == ["chunk-1", "chunk-2"]


def test_merge_source_ids_dedups_after_stripping():
    # Fragments differing only by surrounding whitespace collapse into one entry.
    # This also covers the file_path/description union fields, which reuse this
    # helper in the mongo/opensearch edge-dedup merges.
    assert merge_source_ids(["chunk-1"], ["  chunk-1  "]) == ["chunk-1"]


def test_merge_source_ids_handles_empty_and_none_inputs():
    assert merge_source_ids(None, None) == []
    assert merge_source_ids([], None) == []
    assert merge_source_ids(None, ["chunk-1"]) == ["chunk-1"]
    assert merge_source_ids(["chunk-1"], []) == ["chunk-1"]
    # Falsy elements are skipped rather than emitted as empty ids.
    assert merge_source_ids(["", "chunk-1"], [None, "chunk-2"]) == [
        "chunk-1",
        "chunk-2",
    ]


def test_merge_source_ids_coerces_non_string_ids_with_a_warning(caplog):
    # Coercion (not skipping) keeps provenance; the warning surfaces the caller's
    # type bug instead of letting it pass silently.
    with _captured_logs(caplog, logging.WARNING):
        merged = merge_source_ids([123], ["chunk-1"])

    assert merged == ["123", "chunk-1"]
    assert any("non-string id" in r.getMessage() for r in caplog.records)


def test_merge_source_ids_logs_when_it_has_to_split(caplog):
    # A joined element means an upstream producer skipped its own split; the
    # repair is silent in the data but not in the log.
    with _captured_logs(caplog, logging.DEBUG):
        merge_source_ids([], [f"chunk-1{GRAPH_FIELD_SEP}chunk-2"])

    assert any("GRAPH_FIELD_SEP-joined" in r.getMessage() for r in caplog.records)


def test_merge_source_ids_does_not_log_for_normal_input(caplog):
    with _captured_logs(caplog, logging.DEBUG):
        merge_source_ids(["chunk-1"], ["chunk-2"])

    assert not [r for r in caplog.records if r.name == lightrag_logger.name]
