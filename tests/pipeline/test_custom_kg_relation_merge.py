"""``ainsert_custom_kg`` must merge into an already-existing relation, not
silently overwrite it.

``upsert_edges_batch`` writes ``edge_data``'s keys wholesale -- no backend
merges weight/source_id with what was already stored. Building ``edge_data``
purely from the current call's data therefore replaces (and can shrink) the
evidence a prior document or ``ainsert_custom_kg`` call already established
for that relation, in violation of the relation weight contract (weight must
track the union of distinct real source IDs, not just this call's own).
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data

pytestmark = pytest.mark.offline


@pytest.fixture
def single_process_shared_data():
    """``ainsert_custom_kg`` calls ``_raise_if_recovery_required``, which reads
    the ``pipeline_status`` namespace. Mirrors the fixture in
    ``tests/pipeline/test_custom_kg_self_loop_rejection.py``.
    """
    finalize_share_data()
    initialize_share_data(1)
    yield
    finalize_share_data()


def _make_vdb_mock():
    vdb = MagicMock()
    vdb.global_config = {"workspace": ""}
    vdb.upsert = AsyncMock(return_value=None)
    vdb.delete = AsyncMock(return_value=None)
    vdb.index_done_callback = AsyncMock(return_value=None)
    return vdb


def _make_rag(*, existing_edge):
    from lightrag.lightrag import LightRAG

    rag = LightRAG.__new__(LightRAG)
    rag.workspace = ""
    rag.tokenizer = MagicMock()
    rag.tokenizer.encode = lambda _content: []
    rag.chunks_vdb = _make_vdb_mock()
    rag.text_chunks = _make_vdb_mock()
    # Bypasses asdict(self), which needs every dataclass field a __new__
    # instance never got -- unrelated to what this test pins.
    rag._build_global_config = MagicMock(return_value={})

    graph = MagicMock()
    graph.has_nodes_batch = AsyncMock(return_value={"Alice", "Bob"})
    graph.get_edges_batch = AsyncMock(
        return_value={("Alice", "Bob"): existing_edge} if existing_edge else {}
    )
    graph.upsert_nodes_batch = AsyncMock(return_value=None)
    graph.upsert_edges_batch = AsyncMock(return_value=None)
    rag.chunk_entity_relation_graph = graph

    rag.entities_vdb = _make_vdb_mock()
    rag.relationships_vdb = _make_vdb_mock()
    rag._insert_done = AsyncMock(return_value=None)
    return rag


@pytest.mark.asyncio
async def test_ainsert_custom_kg_merges_source_id_and_weight_into_existing_edge(
    single_process_shared_data,
):
    rag = _make_rag(
        existing_edge={
            "weight": 1.0,
            "description": "old description",
            "keywords": "old",
            "source_id": "chunk-old",
            "file_path": "custom_kg",
        }
    )

    await rag.ainsert_custom_kg(
        {
            "chunks": [{"content": "Alice mentored Bob.", "source_id": "note-1"}],
            "entities": [],
            "relationships": [
                {
                    "src_id": "Alice",
                    "tgt_id": "Bob",
                    "description": "Alice mentored Bob",
                    "keywords": "mentorship",
                    "weight": 1.0,
                    "source_id": "note-1",
                }
            ],
        }
    )

    args, _ = rag.chunk_entity_relation_graph.upsert_edges_batch.call_args
    edge_list = args[0]
    assert len(edge_list) == 1
    src, tgt, edge_data = edge_list[0]
    assert (src, tgt) == ("Alice", "Bob")

    # Old evidence must survive alongside the new one, not be replaced by it.
    assert "chunk-old" in edge_data["source_id"].split(GRAPH_FIELD_SEP)
    new_source = [
        s
        for s in edge_data["source_id"].split(GRAPH_FIELD_SEP)
        if s != "chunk-old"
    ]
    assert len(new_source) == 1
    # Two distinct real sources now back this relation, so weight must be >= 2.
    assert edge_data["weight"] >= 2.0


@pytest.mark.asyncio
async def test_ainsert_custom_kg_writes_new_relation_unmerged(
    single_process_shared_data,
):
    """No prior edge: behavior is unchanged, weight/source_id come from this
    call alone."""
    rag = _make_rag(existing_edge=None)

    await rag.ainsert_custom_kg(
        {
            "chunks": [{"content": "Alice mentored Bob.", "source_id": "note-1"}],
            "entities": [],
            "relationships": [
                {
                    "src_id": "Alice",
                    "tgt_id": "Bob",
                    "description": "Alice mentored Bob",
                    "keywords": "mentorship",
                    "weight": 1.0,
                    "source_id": "note-1",
                }
            ],
        }
    )

    args, _ = rag.chunk_entity_relation_graph.upsert_edges_batch.call_args
    _, _, edge_data = args[0][0]
    assert edge_data["weight"] == 1.0
    assert "chunk-old" not in edge_data["source_id"]
