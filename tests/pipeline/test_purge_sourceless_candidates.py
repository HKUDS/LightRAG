"""Deleting a document keeps a source-less object its stale anchor names.

``adelete_by_entity`` / ``adelete_by_relation`` leave the key in the
document's anchors. An object created by hand under that key afterwards has
no source, so it holds nothing of the document, yet the purge used to delete
it as "no remaining sources" while both endpoints survived. See *Candidates
without a source* in docs/design/PurgeRecoveryContract.md.

Every case runs the real pipeline on the file-backed default storages with a
stub LLM; the census document keeps both endpoints alive without the edge.
An ``ainsert_custom_kg`` relation without ``source_id`` takes the same branch
as ``acreate_relation`` and is not repeated here.
"""

from uuid import uuid4

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.kg.shared_storage import get_namespace_data
from lightrag.utils import EmbeddingFunc, Tokenizer

pytestmark = pytest.mark.offline


class _Characters:
    def encode(self, content):
        return [ord(c) for c in content]

    def decode(self, tokens):
        return "".join(chr(c) for c in tokens)


async def _embed(texts):
    return np.ones((len(texts), 8))


async def _llm(prompt, system_prompt=None, history_messages=None, **kwargs):
    entities = (
        "entity<|#|>Atlas<|#|>company<|#|>Atlas company\n"
        "entity<|#|>Borealis<|#|>company<|#|>Borealis company\n"
    )
    if "MEETING paragraph" in prompt:
        return (
            entities
            + "relation<|#|>Atlas<|#|>Borealis<|#|>met<|#|>Atlas met Borealis\n"
            + "<|COMPLETE|>"
        )
    if "CENSUS paragraph" in prompt:
        return entities + "<|COMPLETE|>"
    raise AssertionError(f"unexpected LLM prompt: {prompt[:200]!r}")


@pytest.fixture
async def rag(tmp_path):
    instance = LightRAG(
        working_dir=str(tmp_path),
        workspace=f"sourceless_{uuid4().hex}",
        llm_model_func=_llm,
        embedding_func=EmbeddingFunc(embedding_dim=8, func=_embed),
        tokenizer=Tokenizer("characters", _Characters()),
        entity_extract_max_gleaning=0,
        max_parallel_insert=1,
    )
    await instance.initialize_storages()
    try:
        yield instance
    finally:
        await instance.finalize_storages()


@pytest.mark.asyncio
@pytest.mark.parametrize("removed", ["relation", "entity"])
async def test_hand_made_object_under_a_stale_anchor_survives(rag, removed):
    await rag.ainsert(
        ["MEETING paragraph.", "CENSUS paragraph."],
        ids=["meeting", "census"],
        file_paths=["meeting.txt", "census.txt"],
    )
    graph = rag.chunk_entity_relation_graph
    assert await graph.has_edge("Atlas", "Borealis")

    # Remove the extracted object, then create a source-less one by hand under
    # the same key. The anchors still name it; that is the trigger.
    if removed == "relation":
        assert (await rag.adelete_by_relation("Atlas", "Borealis")).status == "success"
        await rag.acreate_relation(
            "Atlas",
            "Borealis",
            {"description": "Curated link", "keywords": "curated", "weight": 0.5},
        )
        anchors = await rag.full_relations.get_by_id("meeting")
        assert ["Atlas", "Borealis"] in [list(p) for p in anchors["relation_pairs"]]
        kept = "Atlas~Borealis"
    else:
        assert (await rag.adelete_by_entity("Atlas")).status == "success"
        await rag.acreate_entity(
            "Atlas", {"description": "Curated Atlas", "source_id": ""}
        )
        anchors = await rag.full_entities.get_by_id("meeting")
        assert "Atlas" in anchors["entity_names"]
        kept = "Atlas"

    result = await rag.adelete_by_doc_id("meeting")
    assert result.status == "success", result.message

    if removed == "relation":
        edge = await graph.get_edge("Atlas", "Borealis")
        assert edge is not None
        assert (edge["description"], edge["weight"]) == ("Curated link", 0.5)
    else:
        node = await graph.get_node("Atlas")
        assert node is not None
        assert node["description"] == "Curated Atlas"
    # The census document still supports Borealis; the purge rebuilt it.
    assert (await graph.get_node("Borealis"))["source_id"] == (
        await rag.doc_status.get_by_id("census")
    )["chunks_list"][0]
    # The kept object is named for an operator who wants it gone.
    status = await get_namespace_data("pipeline_status", workspace=rag.workspace)
    assert (
        f"[purge] meeting: kept 1 candidate(s) with no source: {kept}"
        in status["history_messages"]
    )
