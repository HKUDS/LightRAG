"""Verify that skip_rebuild=True defers the KG rebuild and returns targets."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lightrag.base import DeletionResult, DocStatus

pytestmark = pytest.mark.offline


# -------------------------------------------------------------------
# Unit: DeletionResult carries rebuild targets
# -------------------------------------------------------------------


def test_deletion_result_default_rebuild_fields():
    """New fields default to None so existing callers aren't affected."""
    r = DeletionResult(status="success", doc_id="d1", message="ok")
    assert r.entities_to_rebuild is None
    assert r.relationships_to_rebuild is None


def test_deletion_result_with_rebuild_fields():
    """Rebuild targets can be populated when skip_rebuild is used."""
    entities = {"entity_a": ["chunk_1", "chunk_2"]}
    relations = {("entity_a", "entity_b"): ["chunk_3"]}
    r = DeletionResult(
        status="success",
        doc_id="d1",
        message="ok",
        entities_to_rebuild=entities,
        relationships_to_rebuild=relations,
    )
    assert r.entities_to_rebuild == entities
    assert r.relationships_to_rebuild == relations


# -------------------------------------------------------------------
# Integration: adelete_by_doc_id with skip_rebuild
# -------------------------------------------------------------------


def _make_rag_stub():
    """Build a minimal mock that satisfies adelete_by_doc_id's storage calls."""
    rag = MagicMock()
    rag.workspace = "/tmp/test-workspace"

    rag.doc_status = AsyncMock()
    rag.doc_status.get_by_id = AsyncMock(
        return_value={
            "status": DocStatus.PROCESSED.value,
            "file_path": "/fake/path.txt",
            "chunks_list": ["chunk_a", "chunk_b"],
        }
    )

    rag.full_entities = AsyncMock()
    rag.full_entities.get_by_id = AsyncMock(
        return_value={"entity_names": ["EntityX"]}
    )
    rag.full_entities.delete = AsyncMock()

    rag.full_relations = AsyncMock()
    rag.full_relations.get_by_id = AsyncMock(
        return_value={"relation_pairs": [["EntityX", "EntityY"]]}
    )
    rag.full_relations.delete = AsyncMock()

    rag.chunk_entity_relation_graph = AsyncMock()
    rag.chunk_entity_relation_graph.get_nodes_batch = AsyncMock(
        return_value={
            "EntityX": {
                "entity_id": "EntityX",
                # chunk_a and chunk_b will be removed; chunk_c remains -> rebuild
                "source_id": "chunk_a<SEP>chunk_b<SEP>chunk_c",
            }
        }
    )
    rag.chunk_entity_relation_graph.get_edges_batch = AsyncMock(
        return_value={
            ("EntityX", "EntityY"): {
                "source": "EntityX",
                "target": "EntityY",
                "source_id": "chunk_a<SEP>chunk_d",
            }
        }
    )
    rag.chunk_entity_relation_graph.remove_edges = AsyncMock()
    rag.chunk_entity_relation_graph.remove_nodes = AsyncMock()
    rag.chunk_entity_relation_graph.get_nodes_edges_batch = AsyncMock(
        return_value={}
    )

    rag.chunks_vdb = AsyncMock()
    rag.entities_vdb = AsyncMock()
    rag.relationships_vdb = AsyncMock()

    rag.text_chunks = AsyncMock()
    rag.full_docs = AsyncMock()
    rag.llm_response_cache = None

    rag.entity_chunks = AsyncMock()
    rag.entity_chunks.get_by_id = AsyncMock(return_value=None)
    rag.entity_chunks.upsert = AsyncMock()
    rag.relation_chunks = AsyncMock()
    rag.relation_chunks.get_by_id = AsyncMock(return_value=None)
    rag.relation_chunks.upsert = AsyncMock()

    rag._insert_done = AsyncMock()

    return rag


@pytest.mark.asyncio
async def test_skip_rebuild_returns_targets():
    """When skip_rebuild=True, rebuild should be skipped and targets returned."""
    from lightrag.lightrag import LightRAG

    rag = _make_rag_stub()

    mock_pipeline_status = {"busy": False, "history_messages": []}
    mock_lock = asyncio.Lock()

    with (
        patch(
            "lightrag.lightrag.get_namespace_data",
            new_callable=AsyncMock,
            return_value=mock_pipeline_status,
        ),
        patch(
            "lightrag.lightrag.get_namespace_lock",
            return_value=mock_lock,
        ),
        patch(
            "lightrag.lightrag.rebuild_knowledge_from_chunks",
            new_callable=AsyncMock,
        ) as mock_rebuild,
    ):
        # Call the unbound method on our mock
        result = await LightRAG.adelete_by_doc_id(
            rag, "test-doc-001", skip_rebuild=True
        )

    assert result.status == "success"
    mock_rebuild.assert_not_called()

    # At least one set of rebuild targets should have data
    has_entities = bool(result.entities_to_rebuild)
    has_relations = bool(result.relationships_to_rebuild)
    assert has_entities or has_relations, "Expected deferred rebuild targets"

    # doc_status should NOT be deleted when skip_rebuild=True — the caller
    # is responsible for cleaning it up after a successful deferred rebuild.
    doc_status_delete_calls = [
        c for c in rag.doc_status.delete.call_args_list
    ]
    assert len(doc_status_delete_calls) == 0, (
        "doc_status.delete should not be called with skip_rebuild=True"
    )


@pytest.mark.asyncio
async def test_default_rebuild_still_runs():
    """Without skip_rebuild (default), rebuild_knowledge_from_chunks runs."""
    from lightrag.lightrag import LightRAG

    rag = _make_rag_stub()

    mock_pipeline_status = {"busy": False, "history_messages": []}
    mock_lock = asyncio.Lock()

    with (
        patch(
            "lightrag.lightrag.get_namespace_data",
            new_callable=AsyncMock,
            return_value=mock_pipeline_status,
        ),
        patch(
            "lightrag.lightrag.get_namespace_lock",
            return_value=mock_lock,
        ),
        patch(
            "lightrag.lightrag.rebuild_knowledge_from_chunks",
            new_callable=AsyncMock,
        ) as mock_rebuild,
        patch(
            "lightrag.lightrag.asdict",
            return_value={},
        ),
    ):
        result = await LightRAG.adelete_by_doc_id(rag, "test-doc-001")

    assert result.status == "success"
    mock_rebuild.assert_called_once()
    # Default mode should NOT populate the rebuild fields
    assert result.entities_to_rebuild is None
    assert result.relationships_to_rebuild is None
    # Default mode removes doc_status immediately
    rag.doc_status.delete.assert_called()
