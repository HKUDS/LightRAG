"""``ainsert_custom_kg`` must not put a self-loop into the graph.

Companion to ``tests/api/routes/test_graph_self_loop_rejection.py``: together
they close both manual paths that could write ``src == tgt``, which extraction
(``operate.py``) and ``amerge_entities`` already refuse to produce.

The refusal happens in the identifier-validation phase, before any chunk or
graph object is written, so a rejected batch leaves storage untouched.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data

pytestmark = pytest.mark.offline


@pytest.fixture
def single_process_shared_data():
    """``ainsert_custom_kg`` calls ``_raise_if_recovery_required``, which reads
    the ``pipeline_status`` namespace. Mirrors the fixture in
    ``tests/pipeline/test_graph_keyed_locks.py``.
    """
    finalize_share_data()
    initialize_share_data(1)
    yield
    finalize_share_data()


def _make_rag():
    from lightrag.lightrag import LightRAG

    rag = LightRAG.__new__(LightRAG)
    rag.workspace = ""
    rag.tokenizer = MagicMock()
    rag.tokenizer.encode = lambda _content: []
    rag.chunks_vdb = _make_vdb_mock()
    rag.text_chunks = _make_vdb_mock()
    rag.chunk_entity_relation_graph = MagicMock()
    rag.chunk_entity_relation_graph.upsert_node = AsyncMock(return_value=None)
    rag.chunk_entity_relation_graph.upsert_edge = AsyncMock(return_value=None)
    rag.entities_vdb = _make_vdb_mock()
    rag.relationships_vdb = _make_vdb_mock()
    rag._insert_done = AsyncMock(return_value=None)
    return rag


def _make_vdb_mock():
    vdb = MagicMock()
    vdb.global_config = {"workspace": ""}
    vdb.upsert = AsyncMock(return_value=None)
    vdb.index_done_callback = AsyncMock(return_value=None)
    return vdb


@pytest.mark.asyncio
async def test_ainsert_custom_kg_rejects_self_loop_relationship(
    single_process_shared_data,
):
    rag = _make_rag()

    with pytest.raises(ValueError, match=r"relationships\[0\] is a self-loop on 'A'"):
        await rag.ainsert_custom_kg(
            {
                "chunks": [],
                "entities": [],
                "relationships": [
                    {
                        "src_id": "A",
                        "tgt_id": "A",
                        "description": "A refers to itself",
                        "keywords": "self",
                    }
                ],
            }
        )

    # Validation precedes every write, so nothing reached storage.
    rag.chunk_entity_relation_graph.upsert_edge.assert_not_called()
    rag.chunk_entity_relation_graph.upsert_node.assert_not_called()
    rag.relationships_vdb.upsert.assert_not_called()


@pytest.mark.asyncio
async def test_ainsert_custom_kg_rejects_self_loop_after_normalization(
    single_process_shared_data,
):
    """Two spellings that canonicalize to the same name are the same self-loop.

    The comparison must therefore run on the normalized identifiers — the ones
    actually written as the graph edge — not on the caller's raw strings.
    """
    rag = _make_rag()

    with pytest.raises(ValueError, match=r"relationships\[0\] is a self-loop"):
        await rag.ainsert_custom_kg(
            {
                "chunks": [],
                "entities": [],
                "relationships": [
                    {
                        "src_id": '"A"',
                        "tgt_id": "A",
                        "description": "A refers to itself",
                        "keywords": "self",
                    }
                ],
            }
        )

    rag.chunk_entity_relation_graph.upsert_edge.assert_not_called()


@pytest.mark.asyncio
async def test_ainsert_custom_kg_reports_the_offending_index(
    single_process_shared_data,
):
    """The error names which relationship failed, like the sibling validators."""
    rag = _make_rag()

    with pytest.raises(ValueError, match=r"relationships\[1\] is a self-loop on 'B'"):
        await rag.ainsert_custom_kg(
            {
                "chunks": [],
                "entities": [],
                "relationships": [
                    {
                        "src_id": "A",
                        "tgt_id": "B",
                        "description": "A relates to B",
                        "keywords": "k",
                    },
                    {
                        "src_id": "B",
                        "tgt_id": "B",
                        "description": "B refers to itself",
                        "keywords": "self",
                    },
                ],
            }
        )

    rag.chunk_entity_relation_graph.upsert_edge.assert_not_called()
