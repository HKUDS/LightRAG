"""``ainsert_custom_kg`` must refuse a ``file_path`` the stores cannot hold.

Chunk content was always cleaned, but a chunk's ``file_path`` went to
``chunks_vdb`` / ``text_chunks`` as given: a lone surrogate there made
``NanoVectorDBStorage``'s snapshot save raise on every later commit, and an
entity or relation carrying one was refused by the graph guard only after the
chunks had been written. Every ``file_path`` in the payload is now checked with
the graph-attribute rule in the validation phase, so a rejected batch leaves
storage untouched.
"""

import re
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.utils import EmbeddingFunc, compute_mdhash_id

pytestmark = pytest.mark.offline


@pytest.fixture
def single_process_shared_data():
    """``ainsert_custom_kg`` reads the ``pipeline_status`` namespace."""
    finalize_share_data()
    initialize_share_data(1)
    yield
    finalize_share_data()


def _make_storage_mock():
    storage = MagicMock()
    storage.global_config = {"workspace": ""}
    storage.upsert = AsyncMock(return_value=None)
    storage.index_done_callback = AsyncMock(return_value=None)
    return storage


def _make_rag():
    from lightrag.lightrag import LightRAG

    rag = LightRAG.__new__(LightRAG)
    rag.workspace = ""
    rag.tokenizer = MagicMock()
    rag.tokenizer.encode = lambda _content: []
    rag.chunks_vdb = _make_storage_mock()
    rag.text_chunks = _make_storage_mock()
    rag.chunk_entity_relation_graph = MagicMock()
    rag.chunk_entity_relation_graph.upsert_nodes_batch = AsyncMock(return_value=None)
    rag.chunk_entity_relation_graph.upsert_edges_batch = AsyncMock(return_value=None)
    rag.entities_vdb = _make_storage_mock()
    rag.relationships_vdb = _make_storage_mock()
    rag._insert_done = AsyncMock(return_value=None)
    return rag


def _payload(bad_section: str, bad_path: str) -> dict:
    def path_for(section: str) -> str:
        return bad_path if section == bad_section else "doc.txt"

    return {
        "chunks": [
            {"content": "Alpha meets Beta.", "source_id": "c1", "file_path": "ok.txt"},
            {
                "content": "Beta meets Gamma.",
                "source_id": "c2",
                "file_path": path_for("chunks"),
            },
        ],
        "entities": [
            {"entity_name": "Alpha", "source_id": "c1", "file_path": "ok.txt"},
            {
                "entity_name": "Beta",
                "source_id": "c2",
                "file_path": path_for("entities"),
            },
        ],
        "relationships": [
            {
                "src_id": "Alpha",
                "tgt_id": "Gamma",
                "source_id": "c1",
                "file_path": "ok.txt",
            },
            {
                "src_id": "Beta",
                "tgt_id": "Gamma",
                "source_id": "c2",
                "file_path": path_for("relationships"),
            },
        ],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("section", ["chunks", "entities", "relationships"])
@pytest.mark.parametrize(
    ("bad_char", "code_point"),
    [
        # The character classes are swept by tests/utils/
        # test_graph_attribute_validator.py; each call site keeps one offender.
        pytest.param("\ud800", "U+D800", id="lone-surrogate"),
    ],
)
async def test_unstorable_file_path_is_rejected_before_any_write(
    single_process_shared_data, section, bad_char, code_point
):
    rag = _make_rag()

    with pytest.raises(
        ValueError,
        match=rf"Custom KG {section}\[1\]\.file_path must not contain the "
        rf"character {re.escape(code_point)}",
    ):
        await rag.ainsert_custom_kg(_payload(section, f"bad{bad_char}name.txt"))

    # Validation precedes every write, so nothing reached storage.
    rag.chunks_vdb.upsert.assert_not_called()
    rag.text_chunks.upsert.assert_not_called()
    rag.chunk_entity_relation_graph.upsert_nodes_batch.assert_not_called()
    rag.chunk_entity_relation_graph.upsert_edges_batch.assert_not_called()
    rag.entities_vdb.upsert.assert_not_called()
    rag.relationships_vdb.upsert.assert_not_called()
    rag._insert_done.assert_not_called()


def _real_rag(tmp_path):
    from lightrag import LightRAG

    async def _llm(prompt, **kwargs):
        return ""

    async def _embed(texts, **kwargs):
        return np.ones((len(texts), 8), dtype=np.float32)

    return LightRAG(
        working_dir=str(tmp_path),
        llm_model_func=_llm,
        embedding_func=EmbeddingFunc(embedding_dim=8, max_token_size=512, func=_embed),
    )


@pytest.mark.asyncio
async def test_rejected_chunk_path_leaves_default_storages_clean_and_writable(
    tmp_path,
):
    """On the default file storages the rejected chunk is not stored, and the
    namespace still commits the next valid batch."""
    finalize_share_data()
    rag = _real_rag(tmp_path)
    await rag.initialize_storages()
    try:
        bad_chunk = {"content": "Poisoned chunk.", "source_id": "c1"}
        with pytest.raises(ValueError, match=r"chunks\[0\]\.file_path"):
            await rag.ainsert_custom_kg(
                {
                    "chunks": [{**bad_chunk, "file_path": "bad\udcffname.txt"}],
                    "entities": [],
                    "relationships": [],
                }
            )
        bad_id = compute_mdhash_id("Poisoned chunk.", prefix="chunk-")
        assert await rag.text_chunks.get_by_id(bad_id) is None
        assert await rag.chunks_vdb.get_by_id(bad_id) is None

        await rag.ainsert_custom_kg(
            {
                "chunks": [
                    {
                        "content": "Clean chunk.",
                        "source_id": "c2",
                        "file_path": "notes\tv2.txt",
                    }
                ],
                "entities": [
                    {
                        "entity_name": "Alpha",
                        "description": "A clean entity.",
                        "source_id": "c2",
                        "file_path": "notes\tv2.txt",
                    }
                ],
                "relationships": [],
            }
        )
        good_id = compute_mdhash_id("Clean chunk.", prefix="chunk-")
        # Tab is a legal XML character, so the path is stored unchanged.
        assert (await rag.chunks_vdb.get_by_id(good_id))["file_path"] == (
            "notes\tv2.txt"
        )
        node = await rag.chunk_entity_relation_graph.get_node("Alpha")
        assert node["file_path"] == "notes\tv2.txt"
    finally:
        await rag.finalize_storages()
        finalize_share_data()


@pytest.mark.asyncio
async def test_one_shot_chunk_iterable_is_still_stored(tmp_path):
    """The validation pass must not consume a generator the build reads again."""
    finalize_share_data()
    rag = _real_rag(tmp_path)
    await rag.initialize_storages()
    try:
        contents = ["First chunk.", "Second chunk."]
        await rag.ainsert_custom_kg(
            {
                "chunks": (
                    {"content": c, "source_id": f"c{i}", "file_path": "a.txt"}
                    for i, c in enumerate(contents)
                ),
                "entities": [
                    {
                        "entity_name": "Alpha",
                        "description": "Linked to the second chunk.",
                        "source_id": "c1",
                    }
                ],
                "relationships": [],
            }
        )
        ids = [compute_mdhash_id(c, prefix="chunk-") for c in contents]
        stored = [await rag.text_chunks.get_by_id(i) is not None for i in ids]
        assert stored == [True, True]
        # The alias still resolves, so the entity names its real chunk.
        node = await rag.chunk_entity_relation_graph.get_node("Alpha")
        assert node["source_id"] == ids[1]
    finally:
        await rag.finalize_storages()
        finalize_share_data()
