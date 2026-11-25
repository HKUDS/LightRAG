"""A rename or merge names its new objects in the documents' recovery anchors.

Deletion finds a document's graph contributions only through its
``full_entities`` / ``full_relations`` anchors. A rename or merge moves the
document's facts onto a name those anchors did not list, so deleting the
document used to skip the moved objects: they kept its chunk ids and
descriptions, against the purge contract's "never facts of a deleted one"
(``docs/design/PurgeRecoveryContract.md``).

The end-to-end case runs the real pipeline and deletion on the file-backed
storages with a stub LLM. The helper case pins the two anchor-writing rules
the end-to-end case cannot see.
"""

import re
from uuid import uuid4

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.utils import EmbeddingFunc, Tokenizer
from lightrag.utils_graph import _extend_recovery_anchors

pytestmark = [pytest.mark.offline, pytest.mark.asyncio]


class _Characters:
    def encode(self, content):
        return [ord(c) for c in content]

    def decode(self, tokens):
        return "".join(chr(c) for c in tokens)


async def _embed(texts):
    return np.ones((len(texts), 8))


_INPUT = re.compile(r"---Input Text---\n```\n(.*?)\n```", re.S)


async def _llm(prompt, system_prompt=None, history_messages=None, **kwargs):
    """Extract every capitalized word as an entity, linked in order."""
    text = _INPUT.search(prompt).group(1)
    names = list(dict.fromkeys(re.findall(r"\b[A-Z][a-z]+\b", text)))
    rows = [f"entity<|#|>{n}<|#|>person<|#|>{n} in: {text}" for n in names]
    rows += [
        f"relation<|#|>{a}<|#|>{b}<|#|>met<|#|>{a} met {b} in: {text}"
        for a, b in zip(names, names[1:])
    ]
    return "\n".join(rows + ["<|COMPLETE|>"])


DOCS = {
    "d0": "Alice met Bob.",
    "d1": "Bob met Dave.",
    "d2": "Alice met Carol.",
}


async def _make_rag(tmp_path) -> LightRAG:
    rag = LightRAG(
        working_dir=str(tmp_path),
        workspace=f"anchors_{uuid4().hex}",
        llm_model_func=_llm,
        embedding_func=EmbeddingFunc(embedding_dim=8, func=_embed),
        tokenizer=Tokenizer("characters", _Characters()),
        entity_extract_max_gleaning=0,
        max_parallel_insert=1,
    )
    await rag.initialize_storages()
    for doc_id, text in DOCS.items():
        await rag.ainsert(text, ids=[doc_id], file_paths=[f"{doc_id}.txt"])
    return rag


def _record(rag: LightRAG) -> list[str]:
    """Log anchor flushes and graph writes on one timeline."""
    timeline: list[str] = []

    def _wrap(obj, name, tag):
        original = getattr(obj, name)

        async def _logged(*args, **kwargs):
            timeline.append(tag)
            return await original(*args, **kwargs)

        setattr(obj, name, _logged)

    _wrap(rag.full_entities, "index_done_callback", "anchors")
    for name in ("upsert_node", "upsert_edge", "delete_node"):
        _wrap(rag.chunk_entity_relation_graph, name, "graph")
    return timeline


def _ids(value: str) -> set[str]:
    return set(value.split(GRAPH_FIELD_SEP))


@pytest.mark.parametrize(
    "operation, expected_edges",
    [
        # Alice~Bob came from d0 alone; Alicia~Carol is d2's, Bob~Dave d1's.
        ("rename", {("Alicia", "Carol"), ("Bob", "Dave")}),
        # Alice~Bob became Bob~Dave, which d1 also supports.
        ("merge", {("Bob", "Dave"), ("Carol", "Dave")}),
    ],
)
async def test_deleting_the_source_document_reaches_the_moved_objects(
    tmp_path, operation, expected_edges
):
    rag = await _make_rag(tmp_path)
    try:
        d0_chunks = set((await rag.doc_status.get_by_id("d0"))["chunks_list"])
        timeline = _record(rag)
        if operation == "rename":
            await rag.aedit_entity("Alice", {"entity_name": "Alicia"})
            moved = "Alicia"
        else:
            await rag.amerge_entities(["Alice"], "Dave")
            moved = "Dave"

        # The anchors are durable before the first graph write, as in
        # ingestion's Phase 0.
        assert timeline.index("graph") > timeline.index("anchors")
        assert moved in (await rag.full_entities.get_by_id("d0"))["entity_names"]
        del timeline[:]

        result = await rag.adelete_by_doc_id("d0")
        assert result.status == "success", result.message

        graph = rag.chunk_entity_relation_graph
        node = await graph.get_node(moved)
        assert _ids(node["source_id"]).isdisjoint(d0_chunks)
        edges = await graph.get_all_edges()
        assert {tuple(sorted((e["source"], e["target"]))) for e in edges} == (
            expected_edges
        )
        for edge in edges:
            assert _ids(edge["source_id"]).isdisjoint(d0_chunks), edge
        tracked = await rag.entity_chunks.get_by_id(moved)
        assert set(tracked["chunk_ids"]).isdisjoint(d0_chunks)
    finally:
        await rag.finalize_storages()


class _Graph:
    def __init__(self):
        self.nodes = {"Alice": {"source_id": "chunk-a" + GRAPH_FIELD_SEP + "chunk-b"}}
        self.edges = {("Alice", "Bob"): {"source_id": "chunk-a"}}

    async def get_node(self, name):
        return self.nodes.get(name)

    async def get_node_edges(self, name):
        return [pair for pair in self.edges if name in pair]

    async def get_edge(self, src, tgt):
        return self.edges.get((src, tgt)) or self.edges.get((tgt, src))


class _KV:
    def __init__(self, rows):
        self.rows = dict(rows)
        self.flushes = 0

    async def get_by_id(self, key):
        return self.rows.get(key)

    async def get_by_ids(self, keys):
        return [self.rows.get(key) for key in keys]

    async def upsert(self, data):
        self.rows.update(data)

    async def index_done_callback(self):
        self.flushes += 1


async def test_only_existing_anchor_rows_are_extended_and_nothing_is_removed():
    """doc-a has anchors and gains the new names next to the old ones; doc-b
    has none, and must not be handed an ``anchors`` deletion proof."""
    text_chunks = _KV(
        {"chunk-a": {"full_doc_id": "doc-a"}, "chunk-b": {"full_doc_id": "doc-b"}}
    )
    full_entities = _KV({"doc-a": {"entity_names": ["Alice", "Bob"], "count": 2}})
    full_relations = _KV({"doc-a": {"relation_pairs": [["Alice", "Bob"]], "count": 1}})

    await _extend_recovery_anchors(
        _Graph(),
        {"Alice": "Alicia"},
        text_chunks_storage=text_chunks,
        full_entities_storage=full_entities,
        full_relations_storage=full_relations,
    )

    assert full_entities.rows == {
        "doc-a": {"entity_names": ["Alice", "Alicia", "Bob"], "count": 3}
    }
    assert full_relations.rows == {
        "doc-a": {"relation_pairs": [["Alice", "Bob"], ["Alicia", "Bob"]], "count": 2}
    }
    assert (full_entities.flushes, full_relations.flushes) == (1, 1)
