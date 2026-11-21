"""Deletion rebuild must attribute a shared extraction cache row to every chunk
that references it.

The extract cache key hashes the prompt, which carries the chunk text but no
document identity, so two documents with one identical paragraph produce two
chunks that share one cache row. The row's own ``chunk_id`` names only its
first writer; the second chunk reaches it through its ``llm_cache_list``.
"""

from uuid import uuid4

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.utils import EmbeddingFunc, Tokenizer

pytestmark = pytest.mark.offline


class _Characters:
    def encode(self, content):
        return [ord(c) for c in content]

    def decode(self, tokens):
        return "".join(chr(c) for c in tokens)


async def _embed(texts):
    return np.ones((len(texts), 8))


# Paragraph marker -> the description the stub LLM extracts for entity Foo.
FACTS = {
    "SHARED": "Foo shared fact",
    "AONLY": "Foo fact from A",
    "BONLY": "Foo fact from B",
}

DOCS = {
    "doc-a": "SHARED paragraph about Foo.\n\nAONLY paragraph about Foo.",
    "doc-b": "SHARED paragraph about Foo.\n\nBONLY paragraph about Foo.",
}


async def _stub_llm(prompt, system_prompt=None, history_messages=None, **kwargs):
    for marker, fact in FACTS.items():
        if f"{marker} paragraph" in prompt:
            return (
                f"entity<|#|>Foo<|#|>concept<|#|>{fact}\n"
                f"entity<|#|>Bar<|#|>concept<|#|>Bar seen with {marker}\n"
                f"relation<|#|>Foo<|#|>Bar<|#|>linked<|#|>Foo links Bar in {marker}\n"
                "<|COMPLETE|>"
            )
    raise AssertionError(f"unexpected LLM prompt: {prompt[:200]!r}")


@pytest.fixture
async def rag(tmp_path):
    instance = LightRAG(
        working_dir=str(tmp_path),
        workspace=f"shared_{uuid4().hex}",
        llm_model_func=_stub_llm,
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


async def _chunks(rag: LightRAG, *doc_ids: str) -> dict[str, dict]:
    ids = []
    for doc_id in doc_ids:
        status = await rag.doc_status.get_by_id(doc_id)
        ids.extend(status["chunks_list"])
    rows = await rag.text_chunks.get_by_ids(ids)
    return {chunk_id: row for chunk_id, row in zip(ids, rows) if row}


async def _insert(rag: LightRAG, doc_id: str) -> None:
    await rag.ainsert(
        DOCS[doc_id],
        split_by_character="\n\n",
        split_by_character_only=True,
        ids=[doc_id],
        file_paths=[f"{doc_id}.txt"],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("deleted,survivor", [("doc-a", "doc-b"), ("doc-b", "doc-a")])
async def test_shared_row_survives_deletion_of_its_first_writer(rag, deleted, survivor):
    # doc-a is always ingested first, so its chunk is the shared row's writer.
    await _insert(rag, "doc-a")
    await _insert(rag, "doc-b")

    shared_rows = {
        key
        for chunk in (await _chunks(rag, "doc-a", "doc-b")).values()
        if "SHARED" in chunk["content"]
        for key in chunk["llm_cache_list"]
    }
    assert len(shared_rows) == 1, "both shared chunks must reference one row"

    surviving_chunks = set(await _chunks(rag, survivor))
    result = await rag.adelete_by_doc_id(deleted)
    assert result.status == "success"

    survivor_fact = FACTS["AONLY" if survivor == "doc-a" else "BONLY"]
    deleted_fact = FACTS["AONLY" if deleted == "doc-a" else "BONLY"]

    foo = await rag.chunk_entity_relation_graph.get_node("Foo")
    assert FACTS["SHARED"] in foo["description"]
    assert survivor_fact in foo["description"]
    assert deleted_fact not in foo["description"]
    assert foo["file_path"].split(GRAPH_FIELD_SEP) == [f"{survivor}.txt"]
    assert set(foo["source_id"].split(GRAPH_FIELD_SEP)) == surviving_chunks

    edge = await rag.chunk_entity_relation_graph.get_edge("Foo", "Bar")
    assert "Foo links Bar in SHARED" in edge["description"]
    assert set(edge["source_id"].split(GRAPH_FIELD_SEP)) == surviving_chunks
    assert edge["file_path"].split(GRAPH_FIELD_SEP) == [f"{survivor}.txt"]
