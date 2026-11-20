"""``ainsert_custom_kg`` merges provenance into objects that already exist.

Regression for #4496: writing a custom KG entity or relation over one a
document already produced replaced its ``source_id`` / ``file_path`` and wrote
no chunk tracking, so deleting that document found no remaining source in the
tracking row (the purge's authority) and deleted the curated object while
reporting success.

The caller's description, type and keywords still replace the stored ones;
only provenance is merged. Every case runs the real pipeline on the
file-backed default storages with a stub LLM.
"""

from uuid import uuid4

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.utils import (
    EmbeddingFunc,
    Tokenizer,
    compute_mdhash_id,
    make_relation_chunk_key,
)

pytestmark = pytest.mark.offline

DOC_ID = "doc-v3"
DOC_CHUNK = "doc-v3-chunk-000"
CURATED_TEXT = "Curated note about Foo."
CURATED_CHUNK = compute_mdhash_id(CURATED_TEXT, prefix="chunk-")


class _Characters:
    def encode(self, content):
        return [ord(c) for c in content]

    def decode(self, tokens):
        return "".join(chr(c) for c in tokens)


async def _embed(texts):
    return np.ones((len(texts), 8))


async def _llm(prompt, system_prompt=None, history_messages=None, **kwargs):
    if "V3ONLY paragraph" not in prompt:
        raise AssertionError(f"unexpected LLM prompt: {prompt[:200]!r}")
    return (
        "entity<|#|>Foo<|#|>concept<|#|>Foo fact from v3\n"
        "entity<|#|>Bar<|#|>concept<|#|>Bar fact from v3\n"
        "relation<|#|>Foo<|#|>Bar<|#|>linked<|#|>Foo links Bar in v3\n"
        "<|COMPLETE|>"
    )


@pytest.fixture
async def rag(tmp_path):
    instance = LightRAG(
        working_dir=str(tmp_path),
        workspace=f"custom_kg_merge_{uuid4().hex}",
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


async def _insert_doc(rag: LightRAG) -> None:
    await rag.ainsert("V3ONLY paragraph.", ids=[DOC_ID], file_paths=["doc-v3.txt"])
    node = await rag.chunk_entity_relation_graph.get_node("Foo")
    assert node["source_id"] == DOC_CHUNK


def _custom_kg(*, entities=True, relationships=True, weight=1.0) -> dict:
    return {
        "chunks": [
            {
                "content": CURATED_TEXT,
                "source_id": "curated",
                "file_path": "curated.md",
            }
        ],
        "entities": [
            {
                "entity_name": "Foo",
                "entity_type": "concept",
                "description": "Curated Foo",
                "source_id": "curated",
                "file_path": "curated.md",
            }
        ]
        if entities
        else [],
        "relationships": [
            {
                "src_id": "Foo",
                "tgt_id": "Bar",
                "description": "Curated link",
                "keywords": "curated",
                "source_id": "curated",
                "file_path": "curated.md",
                "weight": weight,
            }
        ]
        if relationships
        else [],
    }


def _ids(value: str) -> list[str]:
    return value.split(GRAPH_FIELD_SEP)


async def _tracking(rag: LightRAG) -> tuple[list[str], list[str], list[str]]:
    foo, bar = await rag.entity_chunks.get_by_ids(["Foo", "Bar"])
    (rel,) = await rag.relation_chunks.get_by_ids(
        [make_relation_chunk_key("Foo", "Bar")]
    )
    return foo["chunk_ids"], bar["chunk_ids"], rel["chunk_ids"]


@pytest.mark.asyncio
async def test_issue_reproduction_curated_objects_survive_document_deletion(rag):
    await _insert_doc(rag)
    await rag.ainsert_custom_kg(_custom_kg())

    graph = rag.chunk_entity_relation_graph
    foo = await graph.get_node("Foo")
    edge = await graph.get_edge("Foo", "Bar")
    # The curated fields win; the document's provenance is kept beside them.
    assert foo["description"] == "Curated Foo"
    assert _ids(foo["source_id"]) == [DOC_CHUNK, CURATED_CHUNK]
    assert _ids(foo["file_path"]) == ["doc-v3.txt", "curated.md"]
    assert edge["description"] == "Curated link"
    assert _ids(edge["source_id"]) == [DOC_CHUNK, CURATED_CHUNK]
    # Bar is only a relationship endpoint here, yet gains the curated chunk
    # exactly as a pipeline relation merge gives it to an existing endpoint.
    assert _ids((await graph.get_node("Bar"))["source_id"]) == [
        DOC_CHUNK,
        CURATED_CHUNK,
    ]
    assert await _tracking(rag) == ([DOC_CHUNK, CURATED_CHUNK],) * 3

    result = await rag.adelete_by_doc_id(DOC_ID)
    assert result.status == "success", result.message

    foo = await graph.get_node("Foo")
    edge = await graph.get_edge("Foo", "Bar")
    assert foo is not None and edge is not None
    assert foo["description"] == "Curated Foo"
    assert foo["source_id"] == CURATED_CHUNK
    assert edge["description"] == "Curated link"
    assert edge["source_id"] == CURATED_CHUNK
    assert await graph.get_node("Bar") is not None
    assert await _tracking(rag) == ([CURATED_CHUNK],) * 3


@pytest.mark.asyncio
async def test_endpoint_entity_vector_carries_merged_source(rag):
    await _insert_doc(rag)
    await rag.ainsert_custom_kg(_custom_kg(entities=False))

    (bar_vector,) = await rag.entities_vdb.get_by_ids(
        [compute_mdhash_id("Bar", prefix="ent-")]
    )
    assert _ids(bar_vector["source_id"]) == [DOC_CHUNK, CURATED_CHUNK]
    bar = await rag.chunk_entity_relation_graph.get_node("Bar")
    # Not declared by the custom KG, so its own fields are untouched.
    assert bar["description"] == "Bar fact from v3"


@pytest.mark.asyncio
async def test_empty_tracking_row_is_not_reseeded_from_stale_source_id(rag):
    await _insert_doc(rag)
    # A present-but-empty row is authoritative: the node's source_id still
    # naming a chunk a purge already pruned must not come back.
    await rag.entity_chunks.upsert({"Foo": {"chunk_ids": [], "count": 0}})
    await rag.ainsert_custom_kg(_custom_kg(relationships=False))

    (foo_row,) = await rag.entity_chunks.get_by_ids(["Foo"])
    assert foo_row["chunk_ids"] == [CURATED_CHUNK]
    foo = await rag.chunk_entity_relation_graph.get_node("Foo")
    assert foo["source_id"] == CURATED_CHUNK


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stored_weight", "curated_weight", "expected"),
    [
        (1.0, 1.0, 2.0),  # raised to the merged evidence count
        (5.0, 1.0, 5.0),  # an accumulated weight is not lowered
        (1.0, 7.5, 7.5),  # a curated boost still applies
    ],
)
async def test_merged_relation_weight(rag, stored_weight, curated_weight, expected):
    await _insert_doc(rag)
    graph = rag.chunk_entity_relation_graph
    edge = await graph.get_edge("Foo", "Bar")
    await graph.upsert_edge("Foo", "Bar", {**edge, "weight": stored_weight})

    await rag.ainsert_custom_kg(_custom_kg(weight=curated_weight))

    assert (await graph.get_edge("Foo", "Bar"))["weight"] == expected


@pytest.mark.asyncio
async def test_new_objects_get_tracking_rows(rag):
    await rag.ainsert_custom_kg(_custom_kg())

    graph = rag.chunk_entity_relation_graph
    foo = await graph.get_node("Foo")
    assert foo["source_id"] == CURATED_CHUNK
    assert foo["file_path"] == "curated.md"
    # Bar is created as a placeholder endpoint from the relationship.
    bar = await graph.get_node("Bar")
    assert (bar["description"], bar["source_id"]) == ("UNKNOWN", CURATED_CHUNK)
    assert await _tracking(rag) == ([CURATED_CHUNK],) * 3


@pytest.mark.asyncio
async def test_unmapped_source_writes_no_tracking_row(rag):
    await rag.ainsert_custom_kg(
        {
            "chunks": [],
            "entities": [
                {
                    "entity_name": "Foo",
                    "entity_type": "concept",
                    "description": "Curated Foo",
                    "source_id": "no-such-alias",
                }
            ],
            "relationships": [],
        }
    )

    foo = await rag.chunk_entity_relation_graph.get_node("Foo")
    assert foo["source_id"] == "UNKNOWN"
    assert await rag.entity_chunks.get_by_ids(["Foo"]) == [None]


@pytest.mark.asyncio
async def test_edges_are_read_in_one_batch_call(rag, monkeypatch):
    """All pairs go through a single get_edges_batch, new ones included: the
    lookup must stay one round trip however many relationships are imported."""
    await _insert_doc(rag)
    graph = rag.chunk_entity_relation_graph
    requested: list[list[dict]] = []
    original = graph.get_edges_batch

    async def _recording(pairs):
        requested.append(list(pairs))
        return await original(pairs)

    monkeypatch.setattr(graph, "get_edges_batch", _recording)
    custom_kg = _custom_kg()
    custom_kg["relationships"].append(
        {
            "src_id": "Foo",
            "tgt_id": "Newcomer",
            "description": "Curated newcomer",
            "keywords": "curated",
            "source_id": "curated",
        }
    )
    await rag.ainsert_custom_kg(custom_kg)

    assert requested == [
        [{"src": "Bar", "tgt": "Foo"}, {"src": "Foo", "tgt": "Newcomer"}]
    ]
    assert (await graph.get_edge("Foo", "Newcomer"))["source_id"] == CURATED_CHUNK


@pytest.mark.asyncio
async def test_edge_with_a_missing_endpoint_node_is_still_merged(rag, monkeypatch):
    """Legacy MongoDB / OpenSearch data can hold an edge whose endpoint has no
    node document. Endpoint absence is not proof of edge absence: the stored
    weight and source_id must be merged, not overwritten."""
    await _insert_doc(rag)
    graph = rag.chunk_entity_relation_graph
    edge = await graph.get_edge("Foo", "Bar")
    await graph.upsert_edge("Foo", "Bar", {**edge, "weight": 10.0})
    original = graph.get_nodes_batch

    async def _bar_has_no_node(node_ids):
        nodes = await original(node_ids)
        nodes.pop("Bar", None)
        return nodes

    monkeypatch.setattr(graph, "get_nodes_batch", _bar_has_no_node)
    await rag.ainsert_custom_kg(_custom_kg(entities=False, weight=1.0))

    edge = await graph.get_edge("Foo", "Bar")
    assert edge["weight"] == 10.0
    assert _ids(edge["source_id"]) == [DOC_CHUNK, CURATED_CHUNK]


@pytest.mark.asyncio
async def test_tracking_rows_are_committed_before_the_graph_is_written(
    rag, monkeypatch
):
    """Committed under the keyed lock, before the first graph mutation: a
    per-process-buffered KV backend would otherwise show the next writer of
    the same keys on another worker the old row, and an immediate-write graph
    backend would make the node durable before the row carrying it."""
    await _insert_doc(rag)
    events: list[str] = []
    graph = rag.chunk_entity_relation_graph

    def _record(name, original):
        async def wrapper(*args, **kwargs):
            events.append(name)
            return await original(*args, **kwargs)

        return wrapper

    for storage, name in (
        (rag.entity_chunks, "entity_chunks"),
        (rag.relation_chunks, "relation_chunks"),
    ):
        monkeypatch.setattr(
            storage,
            "index_done_callback",
            _record(f"commit:{name}", storage.index_done_callback),
        )
    for method in ("upsert_nodes_batch", "upsert_edges_batch"):
        monkeypatch.setattr(graph, method, _record(method, getattr(graph, method)))

    await rag.ainsert_custom_kg(_custom_kg())

    first_graph_write = events.index("upsert_nodes_batch")
    assert {"commit:entity_chunks", "commit:relation_chunks"} <= set(
        events[:first_graph_write]
    )
