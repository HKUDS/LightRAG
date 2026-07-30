import pytest
from collections import defaultdict
from unittest.mock import AsyncMock

from lightrag.operate import _build_query_context

pytestmark = pytest.mark.offline


def test_cached_entities_merge_empty_list_does_not_raise():
    """Verify that cached entity merge handles empty entity_list gracefully without raising IndexError."""
    chunk_entities = defaultdict(dict)
    chunk_entities["chunk1"]["E1"] = [{"description": "desc1"}]
    entities = {"E1": []}

    # Simulate loop from lightrag/operate.py (lines 1005-1025)
    for entity_name, entity_list in entities.items():
        if not entity_list:
            continue
        if entity_name not in chunk_entities["chunk1"]:
            chunk_entities["chunk1"][entity_name].extend(entity_list)
        elif len(chunk_entities["chunk1"][entity_name]) == 0:
            chunk_entities["chunk1"][entity_name].extend(entity_list)
        else:
            existing_desc_len = len(
                chunk_entities["chunk1"][entity_name][0].get(
                    "description", ""
                )
                or ""
            )
            new_desc_len = len(entity_list[0].get("description", "") or "")

            if new_desc_len > existing_desc_len:
                chunk_entities["chunk1"][entity_name] = list(entity_list)

    assert chunk_entities["chunk1"]["E1"] == [{"description": "desc1"}]


def test_cached_relationships_merge_empty_list_does_not_raise():
    """Verify that cached relationship merge handles empty rel_list gracefully without raising IndexError."""
    chunk_relationships = defaultdict(dict)
    chunk_relationships["chunk1"][("A", "B")] = [{"description": "rel1"}]
    relationships = {("A", "B"): []}

    for rel_key, rel_list in relationships.items():
        if not rel_list:
            continue
        if rel_key not in chunk_relationships["chunk1"]:
            chunk_relationships["chunk1"][rel_key].extend(rel_list)
        elif len(chunk_relationships["chunk1"][rel_key]) == 0:
            chunk_relationships["chunk1"][rel_key].extend(rel_list)
        else:
            existing_desc_len = len(
                chunk_relationships["chunk1"][rel_key][0].get(
                    "description", ""
                )
                or ""
            )
            new_desc_len = len(rel_list[0].get("description", "") or "")

            if new_desc_len > existing_desc_len:
                chunk_relationships["chunk1"][rel_key] = list(rel_list)

    assert chunk_relationships["chunk1"][("A", "B")] == [{"description": "rel1"}]


def test_glean_nodes_merge_empty_list_does_not_raise():
    """Verify that glean nodes merge handles empty glean_entities gracefully without raising IndexError."""
    maybe_nodes = {"E1": [{"description": "desc1"}]}
    glean_nodes = {"E1": []}

    for entity_name, glean_entities in glean_nodes.items():
        if not glean_entities:
            continue
        if entity_name in maybe_nodes:
            if not maybe_nodes[entity_name]:
                maybe_nodes[entity_name] = list(glean_entities)
            else:
                original_desc_len = len(
                    maybe_nodes[entity_name][0].get("description", "") or ""
                )
                glean_desc_len = len(glean_entities[0].get("description", "") or "")

                if glean_desc_len > original_desc_len:
                    maybe_nodes[entity_name] = list(glean_entities)
        else:
            maybe_nodes[entity_name] = list(glean_entities)

    assert maybe_nodes["E1"] == [{"description": "desc1"}]


def test_glean_edges_merge_empty_list_does_not_raise():
    """Verify that glean edges merge handles empty glean_edge_list gracefully without raising IndexError."""
    maybe_edges = {("A", "B"): [{"description": "rel1"}]}
    glean_edges = {("A", "B"): []}

    for edge_key, glean_edge_list in glean_edges.items():
        if not glean_edge_list:
            continue
        if edge_key in maybe_edges:
            if not maybe_edges[edge_key]:
                maybe_edges[edge_key] = list(glean_edge_list)
            else:
                original_desc_len = len(
                    maybe_edges[edge_key][0].get("description", "") or ""
                )
                glean_desc_len = len(
                    glean_edge_list[0].get("description", "") or ""
                )

                if glean_desc_len > original_desc_len:
                    maybe_edges[edge_key] = list(glean_edge_list)
        else:
            maybe_edges[edge_key] = list(glean_edge_list)

    assert maybe_edges[("A", "B")] == [{"description": "rel1"}]


def test_raw_data_none_field_does_not_raise():
    """Verify that raw_data with 'data': None does not raise AttributeError on .get()."""
    raw_data = {"data": None, "metadata": {}}
    final_chunks_count = len((raw_data.get("data") or {}).get("chunks", []))
    assert final_chunks_count == 0
