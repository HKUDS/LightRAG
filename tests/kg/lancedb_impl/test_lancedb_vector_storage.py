"""Unit tests for LanceDBVectorStorage: deferred embedding, query, deletes."""

import pytest

pytest.importorskip("lancedb", reason="lancedb is required for LanceDB storage tests")

from lightrag.kg.lancedb_impl import LanceDBVectorStorage  # noqa: E402
from lightrag.utils import EmbeddingFunc, compute_mdhash_id  # noqa: E402

from .conftest import DIM, CountingEmbed  # noqa: E402

pytestmark = pytest.mark.offline

ENTITY_META = {"entity_name", "source_id", "content", "file_path"}
RELATION_META = {"src_id", "tgt_id", "source_id", "content", "file_path"}
CHUNK_META = {"full_doc_id", "content", "file_path"}


def _make_storage(
    global_config,
    embedding_func,
    namespace="entities",
    meta_fields=ENTITY_META,
    workspace="ws",
):
    return LanceDBVectorStorage(
        namespace=namespace,
        workspace=workspace,
        global_config=global_config,
        embedding_func=embedding_func,
        meta_fields=set(meta_fields),
    )


def _entity(name, content, source_id="chunk-1", extra=None):
    record = {
        "entity_name": name,
        "content": content,
        "source_id": source_id,
        "file_path": "test.txt",
    }
    if extra:
        record.update(extra)
    return record


async def test_missing_cosine_threshold_raises(global_config, embedding_func):
    config = dict(global_config)
    config["vector_db_storage_cls_kwargs"] = {}
    with pytest.raises(ValueError, match="cosine_better_than_threshold"):
        _make_storage(config, embedding_func)


async def test_upsert_defers_embedding_until_flush(
    global_config, embedding_func, counting_embed
):
    storage = _make_storage(global_config, embedding_func)
    await storage.initialize()
    try:
        await storage.upsert(
            {
                "ent-1": _entity("Alice", "Alice description"),
                "ent-2": _entity("Bob", "Bob description"),
            }
        )
        assert counting_embed.call_count == 0
        # query() only sees flushed data
        assert await storage.query("Alice description", top_k=5) == []
        await storage.index_done_callback()
        assert sorted(counting_embed.document_texts) == [
            "Alice description",
            "Bob description",
        ]
        results = await storage.query("Alice description", top_k=5)
        assert any(r["id"] == "ent-1" for r in results)
    finally:
        await storage.finalize()


async def test_flush_batches_by_embedding_batch_num(global_config, counting_embed):
    global_config = dict(global_config)
    global_config["embedding_batch_num"] = 2
    embedding_func = EmbeddingFunc(
        embedding_dim=DIM,
        max_token_size=512,
        func=counting_embed,
        supports_asymmetric=True,
    )
    storage = _make_storage(global_config, embedding_func)
    await storage.initialize()
    try:
        await storage.upsert(
            {f"ent-{i}": _entity(f"E{i}", f"content {i}") for i in range(5)}
        )
        await storage.index_done_callback()
        # 5 contents at batch size 2 -> 3 embedding calls
        assert counting_embed.call_count == 3
        assert len(counting_embed.document_texts) == 5
    finally:
        await storage.finalize()


async def test_reupsert_same_id_embeds_latest_content_only(
    global_config, embedding_func, counting_embed
):
    storage = _make_storage(global_config, embedding_func)
    await storage.initialize()
    try:
        await storage.upsert({"ent-1": _entity("Alice", "old content")})
        await storage.upsert({"ent-1": _entity("Alice", "new content")})
        await storage.index_done_callback()
        assert counting_embed.document_texts == ["new content"]
        record = await storage.get_by_id("ent-1")
        assert record["content"] == "new content"
    finally:
        await storage.finalize()


async def test_query_filters_by_similarity_threshold_and_orders(
    global_config, embedding_func
):
    storage = _make_storage(global_config, embedding_func)
    await storage.initialize()
    try:
        await storage.upsert(
            {
                "ent-1": _entity("Alice", "target text"),
                "ent-2": _entity("Bob", "other text"),
            }
        )
        await storage.index_done_callback()
        results = await storage.query("target text", top_k=10)
        assert results, "identical content must beat the 0.2 threshold"
        assert results[0]["id"] == "ent-1"
        # distance field carries cosine SIMILARITY (LightRAG convention)
        assert results[0]["distance"] == pytest.approx(1.0, abs=1e-5)
        for record in results:
            assert record["distance"] >= storage.cosine_better_than_threshold
            assert "entity_name" in record
            assert "created_at" in record
            assert "vector" not in record
    finally:
        await storage.finalize()


async def test_query_accepts_precomputed_embedding(
    global_config, embedding_func, counting_embed
):
    storage = _make_storage(global_config, embedding_func)
    await storage.initialize()
    try:
        await storage.upsert({"ent-1": _entity("Alice", "some content")})
        await storage.index_done_callback()
        embeddings = await counting_embed(["some content"])
        calls_before = counting_embed.call_count
        results = await storage.query("ignored", top_k=5, query_embedding=embeddings[0])
        assert counting_embed.call_count == calls_before  # no re-embedding
        assert results and results[0]["id"] == "ent-1"
    finally:
        await storage.finalize()


async def test_meta_fields_projection_drops_extras(global_config, embedding_func):
    storage = _make_storage(
        global_config, embedding_func, namespace="chunks", meta_fields=CHUNK_META
    )
    await storage.initialize()
    try:
        await storage.upsert(
            {
                "chunk-1": {
                    "content": "chunk text",
                    "full_doc_id": "doc-1",
                    "file_path": "a.txt",
                    "tokens": 99,
                    "chunk_order_index": 1,
                    "llm_cache_list": ["x"],
                }
            }
        )
        pending_view = await storage.get_by_id("chunk-1")
        assert "tokens" not in pending_view
        await storage.index_done_callback()
        record = await storage.get_by_id("chunk-1")
        assert record["full_doc_id"] == "doc-1"
        assert record["content"] == "chunk text"
        assert record["file_path"] == "a.txt"
        assert "tokens" not in record
        assert "llm_cache_list" not in record
    finally:
        await storage.finalize()


async def test_read_your_writes_on_pending_buffer(global_config, embedding_func):
    storage = _make_storage(global_config, embedding_func)
    await storage.initialize()
    try:
        await storage.upsert({"ent-1": _entity("Alice", "pending content")})
        record = await storage.get_by_id("ent-1")
        assert record is not None
        assert record["content"] == "pending content"
        assert record["id"] == "ent-1"
        records = await storage.get_by_ids(["ent-1", "missing"])
        assert records[0]["content"] == "pending content"
        assert records[1] is None
    finally:
        await storage.finalize()


async def test_get_vectors_by_ids_embeds_pending_lazily(
    global_config, embedding_func, counting_embed
):
    storage = _make_storage(global_config, embedding_func)
    await storage.initialize()
    try:
        await storage.upsert({"ent-1": _entity("Alice", "lazy content")})
        vectors = await storage.get_vectors_by_ids(["ent-1", "missing"])
        assert counting_embed.call_count == 1
        assert len(vectors["ent-1"]) == DIM
        assert "missing" not in vectors
        # Flush reuses the cached vector: no second embedding call.
        await storage.index_done_callback()
        assert counting_embed.call_count == 1
        flushed = await storage.get_vectors_by_ids(["ent-1"])
        assert flushed["ent-1"] == pytest.approx(vectors["ent-1"])
    finally:
        await storage.finalize()


async def test_delete_cancels_pending_and_removes_rows(global_config, embedding_func):
    storage = _make_storage(global_config, embedding_func)
    await storage.initialize()
    try:
        await storage.upsert(
            {"ent-1": _entity("Alice", "a"), "ent-2": _entity("B", "b")}
        )
        await storage.index_done_callback()
        await storage.upsert({"ent-3": _entity("C", "c")})
        await storage.delete(["ent-1", "ent-3", "missing"])
        assert await storage.get_by_id("ent-1") is None
        assert await storage.get_by_id("ent-3") is None
        assert await storage.get_by_id("ent-2") is not None
        await storage.index_done_callback()
        assert await storage.get_by_id("ent-3") is None  # pending was cancelled
    finally:
        await storage.finalize()


async def test_delete_entity_uses_ent_prefixed_hash(global_config, embedding_func):
    storage = _make_storage(global_config, embedding_func)
    await storage.initialize()
    try:
        entity_id = compute_mdhash_id("Alice", prefix="ent-")
        await storage.upsert({entity_id: _entity("Alice", "Alice content")})
        await storage.index_done_callback()
        await storage.delete_entity("Alice")
        assert await storage.get_by_id(entity_id) is None
    finally:
        await storage.finalize()


async def test_delete_entity_relation_matches_src_and_tgt(
    global_config, embedding_func
):
    storage = _make_storage(
        global_config,
        embedding_func,
        namespace="relationships",
        meta_fields=RELATION_META,
    )
    await storage.initialize()
    try:
        await storage.upsert(
            {
                "rel-1": {
                    "content": "A->B",
                    "src_id": "Alice",
                    "tgt_id": "Bob",
                    "source_id": "chunk-1",
                    "file_path": "f",
                },
                "rel-2": {
                    "content": "C->A",
                    "src_id": "Carol",
                    "tgt_id": "Alice",
                    "source_id": "chunk-1",
                    "file_path": "f",
                },
                "rel-3": {
                    "content": "C->B",
                    "src_id": "Carol",
                    "tgt_id": "Bob",
                    "source_id": "chunk-1",
                    "file_path": "f",
                },
            }
        )
        await storage.index_done_callback()
        # buffer one more pending relation touching Alice
        await storage.upsert(
            {
                "rel-4": {
                    "content": "A->D",
                    "src_id": "Alice",
                    "tgt_id": "Dave",
                    "source_id": "chunk-1",
                    "file_path": "f",
                }
            }
        )
        await storage.delete_entity_relation("Alice")
        assert await storage.get_by_id("rel-1") is None
        assert await storage.get_by_id("rel-2") is None
        assert await storage.get_by_id("rel-4") is None  # pruned from pending
        assert await storage.get_by_id("rel-3") is not None
    finally:
        await storage.finalize()


async def test_drop_pending_index_ops_discards_buffer(
    global_config, embedding_func, counting_embed
):
    storage = _make_storage(global_config, embedding_func)
    await storage.initialize()
    try:
        await storage.upsert({"ent-1": _entity("Alice", "abandoned")})
        await storage.drop_pending_index_ops()
        await storage.index_done_callback()
        assert counting_embed.call_count == 0
        assert await storage.get_by_id("ent-1") is None
    finally:
        await storage.finalize()


async def test_flush_failure_keeps_pending_and_raises(global_config):
    class FailingEmbed(CountingEmbed):
        def __init__(self):
            super().__init__()
            self.fail_times = 1

        async def __call__(self, texts, **kwargs):
            if self.fail_times > 0:
                self.fail_times -= 1
                raise RuntimeError("embedding failed")
            return await super().__call__(texts, **kwargs)

    failing = FailingEmbed()
    embedding_func = EmbeddingFunc(embedding_dim=DIM, max_token_size=512, func=failing)
    storage = _make_storage(global_config, embedding_func)
    await storage.initialize()
    try:
        await storage.upsert({"ent-1": _entity("Alice", "retry me")})
        with pytest.raises(RuntimeError, match="embedding failed"):
            await storage.index_done_callback()
        # Buffer intact: retry succeeds.
        await storage.index_done_callback()
        assert (await storage.get_by_id("ent-1"))["content"] == "retry me"
    finally:
        await storage.finalize()


async def test_finalize_flushes_pending(global_config, embedding_func, counting_embed):
    storage = _make_storage(global_config, embedding_func)
    await storage.initialize()
    await storage.upsert({"ent-1": _entity("Alice", "flushed at finalize")})
    await storage.finalize()
    assert counting_embed.call_count == 1

    reopened = _make_storage(global_config, embedding_func)
    await reopened.initialize()
    try:
        record = await reopened.get_by_id("ent-1")
        assert record is not None
        assert record["content"] == "flushed at finalize"
    finally:
        await reopened.finalize()


async def test_table_name_includes_model_suffix(global_config, counting_embed):
    embedding_func = EmbeddingFunc(
        embedding_dim=DIM,
        max_token_size=512,
        func=counting_embed,
        model_name="text-embedding-3-small",
    )
    storage = _make_storage(global_config, embedding_func)
    assert storage._table_name == f"ws_entities_text_embedding_3_small_{DIM}d"


async def test_drop_recreates_empty_table(global_config, embedding_func):
    storage = _make_storage(global_config, embedding_func)
    await storage.initialize()
    try:
        await storage.upsert({"ent-1": _entity("Alice", "x")})
        await storage.index_done_callback()
        result = await storage.drop()
        assert result == {"status": "success", "message": "data dropped"}
        assert await storage.query("x", top_k=5) == []
        # usable after drop
        await storage.upsert({"ent-2": _entity("Bob", "y")})
        await storage.index_done_callback()
        assert await storage.get_by_id("ent-2") is not None
    finally:
        await storage.finalize()
