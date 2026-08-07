from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.utils import merge_source_ids


def test_merge_source_ids_splits_graph_field_sep():
    # Scenario: new_ids contains combined source_id strings with GRAPH_FIELD_SEP.
    existing = ["chunk-1"]
    new = [f"chunk-2{GRAPH_FIELD_SEP}chunk-3", "chunk-3"]

    merged = merge_source_ids(existing, new)

    # Every item in merged must be an individual chunk ID (no un-split GRAPH_FIELD_SEP strings).
    assert merged == ["chunk-1", "chunk-2", "chunk-3"]
    assert not any(GRAPH_FIELD_SEP in item for item in merged)
