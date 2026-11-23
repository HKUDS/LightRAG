"""Shared fixtures for the Lance cold-export tests.

Seeds REAL file-backed storages (JsonKV / NanoVectorDB / NetworkX) through
their public upsert paths with a deterministic counting embedder -- no LLM,
no network. Seed vectors are small multiples of 0.25 so nano's float16
compression round-trips them exactly and equality assertions are exact.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from lightrag.kg.json_kv_impl import JsonKVStorage
from lightrag.kg.nano_vector_db_impl import NanoVectorDBStorage
from lightrag.kg.networkx_impl import NetworkXStorage
from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
from lightrag.namespace import NameSpace
from lightrag.utils import EmbeddingFunc, compute_mdhash_id

DIM = 4
MODEL_NAME = "counting-test-model"

DOCS = {
    "doc-1": {"content": "alpha document body"},
    "doc-2": {"content": "beta document body"},
}
CHUNKS = {
    "chunk-1": {"content": "alpha chunk one", "full_doc_id": "doc-1",
                "chunk_order_index": 0, "tokens": 3, "file_path": "a.txt"},
    "chunk-2": {"content": "alpha chunk two", "full_doc_id": "doc-1",
                "chunk_order_index": 1, "tokens": 3, "file_path": "a.txt"},
    "chunk-3": {"content": "beta chunk one", "full_doc_id": "doc-2",
                "chunk_order_index": 0, "tokens": 3, "file_path": "b.txt"},
}
NODES = {
    "Alice": {"entity_type": "person", "description": "engineer",
              "source_id": "chunk-1", "file_path": "a.txt"},
    "Bob": {"entity_type": "person", "description": "manager",
            "source_id": "chunk-3", "file_path": "b.txt"},
}
EDGES = [
    ("Alice", "Bob", {"keywords": "works with", "description": "colleagues",
                      "source_id": "chunk-1", "weight": 2.0, "file_path": "a.txt"}),
]


def expected_vector(text: str) -> list[float]:
    """The seed embedder's formula -- the independent oracle for read-back.

    Values are float16-exact, so what Lance holds must equal this list.
    """
    return [float(len(text) % 7) + 0.25 * i for i in range(DIM)]


def entity_content(name: str) -> str:
    """Entity vdb content, mirroring the rebuild write point."""
    return f"{name}\n{NODES[name]['description']}"


def relation_content(src: str, tgt: str, props: dict) -> str:
    """Relation vdb content, mirroring the rebuild write point."""
    return f"{props['keywords']}\t{src}\n{tgt}\n{props['description']}"


class CountingEmbedder:
    """Records every call; raises when called during the export phase.

    The raise is recorded BEFORE it fires so a broad except in the code under
    test cannot hide the call (BackendTestingGuide).
    """

    def __init__(self):
        self.seed_calls = 0
        self.export_calls = 0
        self.armed = False

    async def __call__(self, texts, **kwargs):
        if self.armed:
            self.export_calls += 1
            raise AssertionError("embedding function called during export")
        self.seed_calls += 1
        return np.array([expected_vector(t) for t in texts], dtype=np.float32)


@pytest.fixture(autouse=True)
def _shared_data():
    finalize_share_data()
    initialize_share_data()
    yield
    finalize_share_data()


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch, tmp_path):
    """Pin workspace/storage selection so a developer .env cannot leak in."""
    monkeypatch.setenv("WORKSPACE", "")
    monkeypatch.setenv("WORKING_DIR", str(tmp_path / "rag"))
    monkeypatch.setenv("EMBEDDING_DIM", str(DIM))
    monkeypatch.setenv("EMBEDDING_MODEL", MODEL_NAME)
    monkeypatch.setenv("LIGHTRAG_KV_STORAGE", "JsonKVStorage")
    monkeypatch.setenv("LIGHTRAG_VECTOR_STORAGE", "NanoVectorDBStorage")
    monkeypatch.setenv("LIGHTRAG_GRAPH_STORAGE", "NetworkXStorage")
    monkeypatch.setenv("LIGHTRAG_DOC_STATUS_STORAGE", "JsonDocStatusStorage")


async def seed_rag(tmp_path):
    """Build and flush a small seeded workspace; return (rag_stub, embedder).

    The stub carries exactly the attributes export_snapshot reads, so no full
    LightRAG instance is built where real storages suffice.
    """
    embedder = CountingEmbedder()
    ef = EmbeddingFunc(embedding_dim=DIM, func=embedder, model_name=MODEL_NAME)
    gc = {
        "working_dir": str(tmp_path / "rag"),
        "workspace": "",
        "embedding_batch_num": 8,
        "vector_db_storage_cls_kwargs": {"cosine_better_than_threshold": 0.2},
        "embedding_func": ef,
        "kv_storage": "JsonKVStorage",
        "vector_storage": "NanoVectorDBStorage",
        "graph_storage": "NetworkXStorage",
    }

    def kv(namespace):
        return JsonKVStorage(namespace=namespace, workspace="",
                             global_config=gc, embedding_func=ef)

    def vdb(namespace, meta_fields):
        return NanoVectorDBStorage(namespace=namespace, workspace="",
                                   global_config=gc, embedding_func=ef,
                                   meta_fields=meta_fields)

    full_docs = kv(NameSpace.KV_STORE_FULL_DOCS)
    text_chunks = kv(NameSpace.KV_STORE_TEXT_CHUNKS)
    graph = NetworkXStorage(namespace=NameSpace.GRAPH_STORE_CHUNK_ENTITY_RELATION,
                            workspace="", global_config=gc, embedding_func=ef)
    entities_vdb = vdb(NameSpace.VECTOR_STORE_ENTITIES,
                       {"entity_name", "source_id", "content", "file_path"})
    relationships_vdb = vdb(NameSpace.VECTOR_STORE_RELATIONSHIPS,
                            {"src_id", "tgt_id", "source_id", "content", "file_path"})
    chunks_vdb = vdb(NameSpace.VECTOR_STORE_CHUNKS,
                     {"full_doc_id", "content", "file_path"})
    storages = [full_docs, text_chunks, graph,
                entities_vdb, relationships_vdb, chunks_vdb]
    for storage in storages:
        await storage.initialize()

    await full_docs.upsert(DOCS)
    await text_chunks.upsert(CHUNKS)
    for name, props in NODES.items():
        await graph.upsert_node(name, props)
    for src, tgt, props in EDGES:
        await graph.upsert_edge(src, tgt, props)
    await entities_vdb.upsert({
        compute_mdhash_id(name, prefix="ent-"): {
            "entity_name": name, "content": entity_content(name),
            "source_id": props["source_id"], "file_path": props["file_path"],
        }
        for name, props in NODES.items()
    })
    rel_payloads = {}
    for src, tgt, props in EDGES:
        s, t = sorted((src, tgt))
        rel_payloads[compute_mdhash_id(s + t, prefix="rel-")] = {
            "src_id": s, "tgt_id": t,
            "content": relation_content(s, t, props),
            "source_id": props["source_id"], "file_path": props["file_path"],
        }
    await relationships_vdb.upsert(rel_payloads)
    await chunks_vdb.upsert({
        cid: {"content": c["content"], "full_doc_id": c["full_doc_id"],
              "file_path": c["file_path"]}
        for cid, c in CHUNKS.items()
    })
    for storage in storages:
        await storage.index_done_callback()

    rag = SimpleNamespace(
        full_docs=full_docs, text_chunks=text_chunks,
        chunk_entity_relation_graph=graph, entities_vdb=entities_vdb,
        relationships_vdb=relationships_vdb, chunks_vdb=chunks_vdb,
        embedding_func=ef,
    )
    return rag, embedder


@pytest.fixture
async def seeded(tmp_path):
    return await seed_rag(tmp_path)