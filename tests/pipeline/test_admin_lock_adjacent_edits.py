"""Admin edits of adjacent entities are serialized on every backend.

A rename or merge rewrites every edge incident to the entity it moves without
locking the edge's far endpoint, then deletes the old node, which removes its
edges. Two concurrent edits of adjacent entities used to delete each other's
rewritten edge: both returned success and the relation between them was gone.
NetworkX was protected only because its admin-write gate is required; every
server-backed graph store ran the two edits concurrently.

The cases run the real pipeline on the file-backed storages with
``NetworkXStorage.requires_single_writer`` switched off, which is how a
server-backed graph store is gated, and a stub LLM.
"""

import asyncio
from uuid import uuid4

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.kg.networkx_impl import NetworkXStorage
from lightrag.utils import EmbeddingFunc, Tokenizer

pytestmark = pytest.mark.offline

_REPLIES = {
    "MEETING paragraph": (
        "entity<|#|>Alice<|#|>person<|#|>Alice person\n"
        "entity<|#|>Carol<|#|>person<|#|>Carol person\n"
        "relation<|#|>Alice<|#|>Carol<|#|>met<|#|>Alice met Carol\n"
    ),
    "BOB paragraph": "entity<|#|>Bob<|#|>person<|#|>Bob person\n",
    "DAVE paragraph": "entity<|#|>Dave<|#|>person<|#|>Dave person\n",
}


class _Characters:
    def encode(self, content):
        return [ord(c) for c in content]

    def decode(self, tokens):
        return "".join(chr(c) for c in tokens)


async def _embed(texts):
    return np.ones((len(texts), 8))


async def _llm(prompt, system_prompt=None, history_messages=None, **kwargs):
    for marker, reply in _REPLIES.items():
        if marker in prompt:
            return reply + "<|COMPLETE|>"
    raise AssertionError(f"unexpected LLM prompt: {prompt[:200]!r}")


@pytest.fixture
async def rag(tmp_path, monkeypatch):
    monkeypatch.setattr(NetworkXStorage, "requires_single_writer", False)
    instance = LightRAG(
        working_dir=str(tmp_path),
        workspace=f"adjacent_{uuid4().hex}",
        llm_model_func=_llm,
        embedding_func=EmbeddingFunc(embedding_dim=8, func=_embed),
        tokenizer=Tokenizer("characters", _Characters()),
        entity_extract_max_gleaning=0,
        max_parallel_insert=1,
    )
    await instance.initialize_storages()
    assert instance._admin_write_gate_required() is False
    try:
        await instance.ainsert(
            ["MEETING paragraph.", "BOB paragraph.", "DAVE paragraph."],
            ids=["meeting", "bob", "dave"],
        )
        yield instance
    finally:
        await instance.finalize_storages()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "edge"),
    [("rename", ("Alicia", "Caroline")), ("merge", ("Bob", "Dave"))],
)
async def test_concurrent_edits_of_adjacent_entities_keep_their_edge(rag, kind, edge):
    if kind == "rename":
        edits = [
            rag.aedit_entity("Alice", {"entity_name": "Alicia"}),
            rag.aedit_entity("Carol", {"entity_name": "Caroline"}),
        ]
    else:
        edits = [
            rag.amerge_entities(["Alice"], "Bob"),
            rag.amerge_entities(["Carol"], "Dave"),
        ]
    await asyncio.gather(*edits)

    graph = rag.chunk_entity_relation_graph
    stored = await graph.get_edge(*edge)
    assert stored is not None, await graph.get_all_edges()
    assert stored["description"] == "Alice met Carol"
