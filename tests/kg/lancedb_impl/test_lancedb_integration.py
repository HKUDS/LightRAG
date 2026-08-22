"""Integration tests: all four LanceDB storages sharing one database
directory through a mini insert → query pipeline, plus CJK full-text search.
"""

import pytest

pytest.importorskip("lancedb", reason="lancedb is required for LanceDB storage tests")

from lightrag.base import DocStatus  # noqa: E402
from lightrag.constants import GRAPH_FIELD_SEP  # noqa: E402
from lightrag.kg.lancedb_impl import (  # noqa: E402
    LanceDBDocStatusStorage,
    LanceDBGraphStorage,
    LanceDBKVStorage,
    LanceDBVectorStorage,
)
from lightrag.utils import compute_mdhash_id  # noqa: E402

pytestmark = pytest.mark.offline


async def test_remote_uri_not_created_as_local_dir(monkeypatch, tmp_path):
    """Regression: a remote LANCEDB_URI (s3://...) must not be os.makedirs'd into
    a bogus local directory — only local paths get a directory created.
    """
    import lightrag.kg.lancedb_impl as impl

    monkeypatch.chdir(tmp_path)
    captured = {}

    async def fake_connect_async(uri, **kwargs):
        captured["uri"] = uri
        return object()

    monkeypatch.setattr(impl.lancedb, "connect_async", fake_connect_async)

    client = impl.LanceDBClient("s3://bucket/prefix")
    await client.connect()

    assert captured["uri"] == "s3://bucket/prefix"
    # The remote URI must not have been turned into a local directory.
    assert not (tmp_path / "s3:").exists()
    assert list(tmp_path.iterdir()) == []


async def test_mini_insert_query_pipeline(global_config, embedding_func):
    """Simulate the storage-level writes of one document ingestion batch,
    then query the data back the way the retrieval pipeline does.
    """
    workspace = "pipeline"
    full_docs = LanceDBKVStorage(
        namespace="full_docs",
        workspace=workspace,
        global_config=global_config,
        embedding_func=embedding_func,
    )
    text_chunks = LanceDBKVStorage(
        namespace="text_chunks",
        workspace=workspace,
        global_config=global_config,
        embedding_func=embedding_func,
    )
    chunks_vdb = LanceDBVectorStorage(
        namespace="chunks",
        workspace=workspace,
        global_config=global_config,
        embedding_func=embedding_func,
        meta_fields={"full_doc_id", "content", "file_path"},
    )
    entities_vdb = LanceDBVectorStorage(
        namespace="entities",
        workspace=workspace,
        global_config=global_config,
        embedding_func=embedding_func,
        meta_fields={"entity_name", "source_id", "content", "file_path"},
    )
    relationships_vdb = LanceDBVectorStorage(
        namespace="relationships",
        workspace=workspace,
        global_config=global_config,
        embedding_func=embedding_func,
        meta_fields={"src_id", "tgt_id", "source_id", "content", "file_path"},
    )
    graph = LanceDBGraphStorage(
        namespace="chunk_entity_relation",
        workspace=workspace,
        global_config=global_config,
        embedding_func=embedding_func,
    )
    doc_status = LanceDBDocStatusStorage(
        namespace="doc_status",
        workspace=workspace,
        global_config=global_config,
        embedding_func=None,
    )
    storages = [
        full_docs,
        text_chunks,
        chunks_vdb,
        entities_vdb,
        relationships_vdb,
        graph,
        doc_status,
    ]
    for storage in storages:
        await storage.initialize()
    try:
        doc_text = (
            "Zhu Yuanzhang founded the Ming dynasty. He made Nanjing the capital city."
        )
        doc_id = compute_mdhash_id(doc_text, prefix="doc-")
        chunk_id = compute_mdhash_id(f"{doc_id}:{doc_text}", prefix="chunk-")

        # 1. enqueue: doc status + full doc
        assert await doc_status.filter_keys({doc_id}) == {doc_id}
        await doc_status.upsert(
            {
                doc_id: {
                    "status": DocStatus.PENDING,
                    "content_summary": doc_text[:100],
                    "content_length": len(doc_text),
                    "created_at": "2024-01-01T00:00:00+00:00",
                    "updated_at": "2024-01-01T00:00:00+00:00",
                    "file_path": "ming.txt",
                    "content_hash": "hash-ming",
                    "metadata": {},
                }
            }
        )
        await full_docs.upsert({doc_id: {"content": doc_text, "file_path": "ming.txt"}})

        # 2. chunking: KV chunks + chunk vectors
        chunk_record = {
            "content": doc_text,
            "tokens": 20,
            "chunk_order_index": 0,
            "full_doc_id": doc_id,
            "file_path": "ming.txt",
        }
        await text_chunks.upsert({chunk_id: dict(chunk_record)})
        await chunks_vdb.upsert({chunk_id: dict(chunk_record)})

        # 3. extraction: graph nodes/edges + entity/relation vectors
        await graph.upsert_node(
            "Zhu Yuanzhang",
            {
                "entity_id": "Zhu Yuanzhang",
                "entity_type": "person",
                "description": "Founder of the Ming dynasty",
                "source_id": chunk_id,
                "file_path": "ming.txt",
            },
        )
        await graph.upsert_node(
            "Ming Dynasty",
            {
                "entity_id": "Ming Dynasty",
                "entity_type": "organization",
                "description": "Chinese dynasty founded in 1368",
                "source_id": chunk_id,
                "file_path": "ming.txt",
            },
        )
        await graph.upsert_edge(
            "Zhu Yuanzhang",
            "Ming Dynasty",
            {
                "weight": 1.0,
                "description": "Zhu Yuanzhang founded the Ming dynasty",
                "keywords": "founder",
                "source_id": chunk_id,
                "file_path": "ming.txt",
            },
        )
        await entities_vdb.upsert(
            {
                compute_mdhash_id("Zhu Yuanzhang", prefix="ent-"): {
                    "entity_name": "Zhu Yuanzhang",
                    "content": "Zhu Yuanzhang\nFounder of the Ming dynasty",
                    "source_id": chunk_id,
                    "file_path": "ming.txt",
                },
                compute_mdhash_id("Ming Dynasty", prefix="ent-"): {
                    "entity_name": "Ming Dynasty",
                    "content": "Ming Dynasty\nChinese dynasty founded in 1368",
                    "source_id": chunk_id,
                    "file_path": "ming.txt",
                },
            }
        )
        src, tgt = sorted(("Zhu Yuanzhang", "Ming Dynasty"))
        await relationships_vdb.upsert(
            {
                compute_mdhash_id(src + tgt, prefix="rel-"): {
                    "src_id": src,
                    "tgt_id": tgt,
                    "content": "founder\tZhu Yuanzhang founded the Ming dynasty",
                    "source_id": chunk_id,
                    "file_path": "ming.txt",
                }
            }
        )

        # 4. _insert_done: flush all storages
        for storage in storages:
            await storage.index_done_callback()
        await doc_status.upsert(
            {
                doc_id: {
                    "status": DocStatus.PROCESSED,
                    "content_summary": doc_text[:100],
                    "content_length": len(doc_text),
                    "created_at": "2024-01-01T00:00:00+00:00",
                    "updated_at": "2024-01-01T00:01:00+00:00",
                    "file_path": "ming.txt",
                    "content_hash": "hash-ming",
                    "metadata": {},
                    "chunks_count": 1,
                    "chunks_list": [chunk_id],
                }
            }
        )

        # 5. retrieval-style reads (the mock embedding is deterministic per
        # text, so query with the exact stored content to guarantee a match)
        chunk_hits = await chunks_vdb.query(doc_text, top_k=5)
        assert chunk_hits and chunk_hits[0]["id"] == chunk_id
        assert chunk_hits[0]["content"] == doc_text

        entity_hits = await entities_vdb.query(
            "Zhu Yuanzhang\nFounder of the Ming dynasty", top_k=5
        )
        assert entity_hits
        assert all("entity_name" in hit for hit in entity_hits)

        relation_hits = await relationships_vdb.query(
            "founder\tZhu Yuanzhang founded the Ming dynasty", top_k=5
        )
        assert relation_hits
        assert {relation_hits[0]["src_id"], relation_hits[0]["tgt_id"]} == {
            "Zhu Yuanzhang",
            "Ming Dynasty",
        }

        node = await graph.get_node(entity_hits[0]["entity_name"])
        assert node is not None
        assert node["source_id"] == chunk_id
        chunk = await text_chunks.get_by_id(chunk_id)
        assert chunk["content"] == doc_text
        processed = await doc_status.get_docs_by_status(DocStatus.PROCESSED)
        assert doc_id in processed
        assert processed[doc_id].chunks_list == [chunk_id]

        # 6. deletion flow (adelete_by_doc_id storage calls)
        await chunks_vdb.delete([chunk_id])
        await text_chunks.delete([chunk_id])
        await relationships_vdb.delete(
            [
                compute_mdhash_id(src + tgt, prefix="rel-"),
                compute_mdhash_id(tgt + src, prefix="rel-"),
            ]
        )
        await graph.remove_edges([("Zhu Yuanzhang", "Ming Dynasty")])
        await graph.remove_nodes(["Zhu Yuanzhang", "Ming Dynasty"])
        await entities_vdb.delete(
            [
                compute_mdhash_id("Zhu Yuanzhang", prefix="ent-"),
                compute_mdhash_id("Ming Dynasty", prefix="ent-"),
            ]
        )
        await doc_status.delete([doc_id])
        await full_docs.delete([doc_id])
        assert await full_docs.is_empty()
        assert await doc_status.is_empty()
        assert await graph.get_all_labels() == []
        assert await chunks_vdb.query(doc_text, top_k=5) == []
    finally:
        for storage in storages:
            await storage.finalize()


async def test_multiple_source_ids_round_trip(global_config, embedding_func):
    """GRAPH_FIELD_SEP-joined fields must round-trip unchanged."""
    graph = LanceDBGraphStorage(
        namespace="chunk_entity_relation",
        workspace="sep",
        global_config=global_config,
        embedding_func=embedding_func,
    )
    await graph.initialize()
    try:
        source_id = f"chunk-1{GRAPH_FIELD_SEP}chunk-2"
        await graph.upsert_node("Multi", {"entity_id": "Multi", "source_id": source_id})
        node = await graph.get_node("Multi")
        assert node["source_id"] == source_id
        assert node["source_id"].split(GRAPH_FIELD_SEP) == ["chunk-1", "chunk-2"]
    finally:
        await graph.finalize()


async def test_cjk_full_text_search(global_config, embedding_func):
    """Chinese text must be retrievable via the built-in FTS index."""
    chunks_vdb = LanceDBVectorStorage(
        namespace="chunks",
        workspace="cjk",
        global_config=global_config,
        embedding_func=embedding_func,
        meta_fields={"full_doc_id", "content", "file_path"},
    )
    await chunks_vdb.initialize()
    try:
        await chunks_vdb.upsert(
            {
                "chunk-zh-1": {
                    "content": "朱元璋是明朝的開國皇帝，定都南京。",
                    "full_doc_id": "doc-zh",
                    "file_path": "zhu.txt",
                },
                "chunk-zh-2": {
                    "content": "康熙皇帝是清朝在位時間最長的皇帝。",
                    "full_doc_id": "doc-zh",
                    "file_path": "zhu.txt",
                },
                "chunk-en-1": {
                    "content": "The quick brown fox jumps over the lazy dog.",
                    "full_doc_id": "doc-en",
                    "file_path": "fox.txt",
                },
            }
        )
        await chunks_vdb.index_done_callback()

        hits = await chunks_vdb.full_text_search("朱元璋", top_k=5)
        assert [hit["id"] for hit in hits] == ["chunk-zh-1"]
        assert hits[0]["score"] is not None

        # A two-character CJK query must match (bigram tokenizer).
        hits = await chunks_vdb.full_text_search("皇帝", top_k=5)
        assert {hit["id"] for hit in hits} == {"chunk-zh-1", "chunk-zh-2"}

        # Latin-script search still works with the same index.
        hits = await chunks_vdb.full_text_search("fox", top_k=5)
        assert [hit["id"] for hit in hits] == ["chunk-en-1"]

        # Rows added after index creation are searchable without reindexing.
        await chunks_vdb.upsert(
            {
                "chunk-zh-3": {
                    "content": "唐朝詩人李白寫了很多著名的詩。",
                    "full_doc_id": "doc-zh",
                    "file_path": "zhu.txt",
                }
            }
        )
        await chunks_vdb.index_done_callback()
        hits = await chunks_vdb.full_text_search("李白", top_k=5)
        assert [hit["id"] for hit in hits] == ["chunk-zh-3"]
    finally:
        await chunks_vdb.finalize()


async def test_cjk_vector_content_round_trip(global_config, embedding_func):
    """CJK content must survive storage/retrieval byte-exact."""
    kv = LanceDBKVStorage(
        namespace="text_chunks",
        workspace="cjk",
        global_config=global_config,
        embedding_func=embedding_func,
    )
    await kv.initialize()
    try:
        content = "測試中文內容：《明史》記載，朱元璋（1328年—1398年）。"
        await kv.upsert({"chunk-1": {"content": content, "tokens": 30}})
        record = await kv.get_by_id("chunk-1")
        assert record["content"] == content
    finally:
        await kv.finalize()
