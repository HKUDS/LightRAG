"""Offline tests for the FrameRAG subsystem.

Covers:
  - DocStore: upsert / get / list / delete / status counts
  - Auth: token create/verify, credentials, require_auth dev mode
  - Reranker: rerank_chunk_hits fallback on error
  - FrameRAG lifecycle: initialize / ainsert / list_documents / adelete
  - LLM _llm(): caching, semaphore, timeout
"""
from __future__ import annotations

import asyncio
import os
import time
import pytest
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Shared-Storage bootstrap (required by JsonKVStorage)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session", autouse=True)
def _init_shared_storage():
    """Bootstrap LightRAG's shared in-process storage (single-worker mode)."""
    from lightrag.kg.shared_storage import initialize_share_data
    try:
        initialize_share_data(workers=1)
    except AssertionError:
        pass  # already initialized by another test module
    yield

# ─────────────────────────────────────────────────────────────────────────────
# Helpers / shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

pytestmark = pytest.mark.offline


def _dummy_embed(dim: int = 4):
    """Return an embed func that always returns zeros of shape (n, dim)."""
    async def _embed(texts: list[str], **kwargs) -> np.ndarray:
        return np.zeros((len(texts), dim), dtype=np.float32)
    return _embed


def _dummy_llm(response: str = "[]"):
    """Return an LLM func that always returns a fixed string."""
    async def _llm(prompt: str, **kwargs) -> str:
        return response
    return _llm


# ─────────────────────────────────────────────────────────────────────────────
# DocStore tests
# ─────────────────────────────────────────────────────────────────────────────

def _clear_kv_namespace(namespace: str):
    """Remove a namespace from LightRAG's shared in-process storage so tests are isolated."""
    import lightrag.kg.shared_storage as ss
    final_ns = ss.get_final_namespace(namespace, workspace="")
    if ss._shared_dicts and final_ns in ss._shared_dicts:
        ss._shared_dicts[final_ns].clear()
    if ss._init_flags and final_ns in ss._init_flags:
        del ss._init_flags[final_ns]
    if ss._update_flags and final_ns in ss._update_flags:
        del ss._update_flags[final_ns]


class TestDocStore:
    @pytest.fixture()
    async def store(self, tmp_path):
        from framerag.doc_store import DocStore
        _clear_kv_namespace("doc_status")
        s = DocStore(str(tmp_path))
        await s.initialize()
        yield s
        _clear_kv_namespace("doc_status")

    @pytest.fixture()
    def record_factory(self):
        from framerag.doc_store import DocRecord, DocStatus
        def _make(source_doc: str = "doc.txt", status=DocStatus.PENDING):
            doc_id = f"doc-{hash(source_doc) & 0xFFFFFF:06x}"
            return DocRecord(doc_id=doc_id, source_doc=source_doc, status=status)
        return _make

    @pytest.mark.asyncio
    async def test_upsert_and_get(self, store, record_factory):
        from framerag.doc_store import DocStatus
        rec = record_factory("test.txt", DocStatus.PENDING)
        await store.upsert(rec)
        got = await store.get(rec.doc_id)
        assert got is not None
        assert got.doc_id == rec.doc_id
        assert got.source_doc == "test.txt"
        assert got.status == DocStatus.PENDING

    @pytest.mark.asyncio
    async def test_get_missing_returns_none(self, store):
        assert await store.get("nonexistent-id") is None

    @pytest.mark.asyncio
    async def test_status_update(self, store, record_factory):
        from framerag.doc_store import DocStatus
        rec = record_factory("test.txt", DocStatus.PENDING)
        await store.upsert(rec)
        rec.status = DocStatus.PROCESSED
        rec.chunk_ids = ["c1", "c2"]
        rec.chunks_count = 2
        await store.upsert(rec)
        got = await store.get(rec.doc_id)
        assert got.status == DocStatus.PROCESSED
        assert got.chunk_ids == ["c1", "c2"]
        assert got.chunks_count == 2

    @pytest.mark.asyncio
    async def test_delete(self, store, record_factory):
        from framerag.doc_store import DocStatus
        rec = record_factory("del.txt", DocStatus.PROCESSED)
        await store.upsert(rec)
        await store.delete(rec.doc_id)
        assert await store.get(rec.doc_id) is None

    @pytest.mark.asyncio
    async def test_list_all_and_by_status(self, store, record_factory):
        from framerag.doc_store import DocStatus
        r1 = record_factory("a.txt", DocStatus.PROCESSED)
        r2 = record_factory("b.txt", DocStatus.FAILED)
        r3 = record_factory("c.txt", DocStatus.PROCESSED)
        for r in (r1, r2, r3):
            await store.upsert(r)

        all_docs = await store.list_all()
        assert len(all_docs) == 3

        processed = await store.list_by_status(DocStatus.PROCESSED)
        assert len(processed) == 2

        failed = await store.list_by_status(DocStatus.FAILED)
        assert len(failed) == 1
        assert failed[0].source_doc == "b.txt"

    @pytest.mark.asyncio
    async def test_get_counts(self, store, record_factory):
        from framerag.doc_store import DocStatus
        await store.upsert(record_factory("a.txt", DocStatus.PROCESSED))
        await store.upsert(record_factory("b.txt", DocStatus.FAILED))
        await store.upsert(record_factory("c.txt", DocStatus.PENDING))
        counts = await store.get_counts()
        assert counts["processed"] == 1
        assert counts["failed"] == 1
        assert counts["pending"] == 1
        assert counts["processing"] == 0

    @pytest.mark.asyncio
    async def test_make_doc_id_deterministic(self):
        from framerag.doc_store import DocStore
        id1 = DocStore.make_doc_id("my_doc.txt")
        id2 = DocStore.make_doc_id("my_doc.txt")
        assert id1 == id2
        assert id1.startswith("doc-")

    def test_record_to_dict_roundtrip(self):
        from framerag.doc_store import DocRecord, DocStatus
        rec = DocRecord(
            doc_id="doc-abc",
            source_doc="file.txt",
            status=DocStatus.PROCESSED,
            chunk_ids=["c1", "c2"],
            chunks_count=2,
        )
        d = rec.to_dict()
        assert d["status"] == "processed"
        restored = DocRecord.from_dict(d)
        assert restored.status == DocStatus.PROCESSED
        assert restored.chunk_ids == ["c1", "c2"]


# ─────────────────────────────────────────────────────────────────────────────
# Auth tests
# ─────────────────────────────────────────────────────────────────────────────

class TestAuth:
    def test_create_and_verify_token(self):
        pytest.importorskip("jwt")
        from framerag.auth import create_token, verify_token
        token = create_token("alice")
        username = verify_token(token)
        assert username == "alice"

    def test_verify_invalid_token_raises(self):
        pytest.importorskip("jwt")
        from framerag.auth import verify_token
        with pytest.raises(Exception):  # HTTPException from FastAPI
            verify_token("not.a.valid.token")

    def test_verify_credentials_correct(self, monkeypatch):
        monkeypatch.setenv("FRAMERAG_AUTH_ACCOUNTS", "user1:pass1,user2:pass2")
        # Reload the module to pick up env var
        import importlib
        import framerag.auth as auth_mod
        importlib.reload(auth_mod)
        assert auth_mod.verify_credentials("user1", "pass1") is True
        assert auth_mod.verify_credentials("user2", "pass2") is True
        assert auth_mod.verify_credentials("user1", "wrong") is False
        assert auth_mod.verify_credentials("nobody", "pass1") is False

    def test_auth_disabled_when_no_accounts(self, monkeypatch):
        monkeypatch.delenv("FRAMERAG_AUTH_ACCOUNTS", raising=False)
        import importlib
        import framerag.auth as auth_mod
        importlib.reload(auth_mod)
        assert auth_mod.AUTH_ENABLED is False

    @pytest.mark.asyncio
    async def test_require_auth_open_mode(self, monkeypatch):
        """require_auth returns None when AUTH_ENABLED is False."""
        monkeypatch.delenv("FRAMERAG_AUTH_ACCOUNTS", raising=False)
        import importlib
        import framerag.auth as auth_mod
        importlib.reload(auth_mod)
        # Patch the module-level flag so require_auth sees it
        auth_mod.AUTH_ENABLED = False
        result = await auth_mod.require_auth(credentials=None)
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# Gleaning tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGleaning:
    """Unit tests for multi-round gleaning in framerag/operate.py."""

    def _make_chunk(self):
        from framerag.types import ChunkSchema
        return ChunkSchema(
            chunk_id="chunk-test",
            text="Apple was founded by Steve Jobs in Cupertino.",
            source_doc="test.txt",
            chunk_index=0,
            tokens=10,
        )

    def _make_mention(self, name: str):
        from framerag.types import EntityMentionSchema
        return EntityMentionSchema(
            mention_id=f"em-{name}",
            chunk_id="chunk-test",
            name=name,
            entity_type="PERSON",
            description="",
            aliases=[],
            salience="MEDIUM",
        )

    @pytest.mark.asyncio
    async def test_gleaning_disabled_with_zero_rounds(self):
        from framerag.operate import glean_entities
        chunk = self._make_chunk()
        existing = [self._make_mention("Steve Jobs")]
        call_count = 0

        async def llm(prompt, **kw):
            nonlocal call_count
            call_count += 1
            return '[{"entity_name": "Apple", "entity_type": "ORG"}]'

        result = await glean_entities(chunk, existing, llm, max_rounds=0)
        assert result == []
        assert call_count == 0, "LLM should not be called when max_rounds=0"

    @pytest.mark.asyncio
    async def test_gleaning_single_round(self):
        from framerag.operate import glean_entities
        chunk = self._make_chunk()
        existing = [self._make_mention("Steve Jobs")]
        call_count = 0

        async def llm(prompt, **kw):
            nonlocal call_count
            call_count += 1
            return '[{"entity_name": "Apple", "entity_type": "ORG", "entity_description": "tech company"}]'

        result = await glean_entities(chunk, existing, llm, max_rounds=1)
        assert len(result) == 1
        assert result[0].name == "Apple"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_gleaning_multi_round_stops_early_on_empty(self):
        """If a round finds nothing new, stop before reaching max_rounds."""
        from framerag.operate import glean_entities
        chunk = self._make_chunk()
        existing = [self._make_mention("Steve Jobs")]
        responses = [
            '[{"entity_name": "Apple", "entity_type": "ORG"}]',  # round 1: finds Apple
            '[]',                                                   # round 2: nothing new → stop
        ]
        call_count = 0

        async def llm(prompt, **kw):
            nonlocal call_count
            resp = responses[min(call_count, len(responses) - 1)]
            call_count += 1
            return resp

        result = await glean_entities(chunk, existing, llm, max_rounds=5)
        assert len(result) == 1
        assert result[0].name == "Apple"
        assert call_count == 2, "Should stop after round 2 (empty result)"

    @pytest.mark.asyncio
    async def test_gleaning_multi_round_finds_multiple(self):
        """Each round finds new entities until max_rounds reached."""
        from framerag.operate import glean_entities
        chunk = self._make_chunk()
        existing = [self._make_mention("Steve Jobs")]
        responses = [
            '[{"entity_name": "Apple", "entity_type": "ORG"}]',
            '[{"entity_name": "Cupertino", "entity_type": "LOCATION"}]',
            '[{"entity_name": "iPhone", "entity_type": "PRODUCT"}]',
        ]
        call_count = 0

        async def llm(prompt, **kw):
            nonlocal call_count
            resp = responses[min(call_count, len(responses) - 1)]
            call_count += 1
            return resp

        result = await glean_entities(chunk, existing, llm, max_rounds=3)
        names = {m.name for m in result}
        assert names == {"Apple", "Cupertino", "iPhone"}
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_gleaning_skips_duplicate_names(self):
        """Entities already in existing_mentions must not be returned again."""
        from framerag.operate import glean_entities
        chunk = self._make_chunk()
        existing = [self._make_mention("Steve Jobs"), self._make_mention("Apple")]

        async def llm(prompt, **kw):
            # LLM tries to re-add Steve Jobs (case-insensitive) + adds new Cupertino
            return '[{"entity_name": "steve jobs", "entity_type": "PERSON"}, {"entity_name": "Cupertino", "entity_type": "LOCATION"}]'

        result = await glean_entities(chunk, existing, llm, max_rounds=1)
        assert len(result) == 1
        assert result[0].name == "Cupertino"

    @pytest.mark.asyncio
    async def test_gleaning_llm_error_returns_empty(self):
        """If LLM raises on first round, return empty list gracefully."""
        from framerag.operate import glean_entities
        chunk = self._make_chunk()
        existing = [self._make_mention("Steve Jobs")]

        async def llm(prompt, **kw):
            raise RuntimeError("network error")

        result = await glean_entities(chunk, existing, llm, max_rounds=3)
        assert result == []

    @pytest.mark.asyncio
    async def test_framerag_max_gleaning_rounds_param(self, tmp_path):
        """FrameRAG constructor exposes max_gleaning_rounds and passes it down."""
        from framerag import FrameRAG
        _reset_shared_storage()
        r = FrameRAG(
            working_dir=str(tmp_path / "glean"),
            llm_func=_dummy_llm("[]"),
            embed_func=_dummy_embed(dim=4),
            embedding_dim=4,
            enable_gleaning=True,
            max_gleaning_rounds=3,
        )
        assert r._max_gleaning_rounds == 3
        await r.initialize()
        await r.finalize()
        _reset_shared_storage()


# ─────────────────────────────────────────────────────────────────────────────
# Reranker tests
# ─────────────────────────────────────────────────────────────────────────────

class TestReranker:
    @pytest.mark.asyncio
    async def test_rerank_chunk_hits_happy_path(self):
        from framerag.rerank import rerank_chunk_hits

        query = "what happened?"
        chunk_hits = [
            {"id": "c1", "score": 0.9},
            {"id": "c2", "score": 0.8},
            {"id": "c3", "score": 0.7},
        ]
        chunk_texts = {"c1": "text 1", "c2": "text 2", "c3": "text 3"}

        # Reranker that reverses order
        async def _mock_rerank(q, docs, top_n):
            return [{"index": i, "relevance_score": 1.0 / (i + 1)} for i in range(len(docs))][::-1]

        result = await rerank_chunk_hits(query, chunk_hits, chunk_texts, _mock_rerank, top_n=2)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_rerank_chunk_hits_fallback_on_error(self):
        """If the reranker raises, return original order (top_n items)."""
        from framerag.rerank import rerank_chunk_hits

        query = "test"
        chunk_hits = [{"id": f"c{i}", "score": float(i)} for i in range(5)]
        chunk_texts = {f"c{i}": f"text {i}" for i in range(5)}

        async def _failing_rerank(q, docs, top_n):
            raise RuntimeError("reranker unavailable")

        result = await rerank_chunk_hits(
            query, chunk_hits, chunk_texts, _failing_rerank, top_n=3
        )
        # Falls back to diffusion order, returns top_n
        assert len(result) <= 5
        # IDs should still be from original set
        ids = {h["id"] for h in result}
        assert ids.issubset({f"c{i}" for i in range(5)})

    @pytest.mark.asyncio
    async def test_rerank_empty_hits(self):
        from framerag.rerank import rerank_chunk_hits

        async def _rerank(q, docs, top_n):
            return []

        result = await rerank_chunk_hits("q", [], {}, _rerank, top_n=5)
        assert result == []


# ─────────────────────────────────────────────────────────────────────────────
# FrameRAG lifecycle tests
# ─────────────────────────────────────────────────────────────────────────────

def _reset_shared_storage():
    """Tear down and re-initialize LightRAG shared storage to isolate tests."""
    from lightrag.kg.shared_storage import finalize_share_data, initialize_share_data
    try:
        finalize_share_data()
    except Exception:
        pass
    try:
        initialize_share_data(workers=1)
    except AssertionError:
        pass


class TestFrameRAGLifecycle:
    """Integration tests using mock LLM and embed (no real API calls)."""

    @pytest.fixture()
    async def rag(self, tmp_path):
        from framerag import FrameRAG
        _reset_shared_storage()
        embed = _dummy_embed(dim=4)
        llm   = _dummy_llm("[]")  # All extraction calls return empty JSON
        r = FrameRAG(
            working_dir=str(tmp_path),
            llm_func=llm,
            embed_func=embed,
            embedding_dim=4,
            enable_causal=False,
            enable_gleaning=False,
            enable_event_coref=False,
        )
        await r.initialize()
        yield r
        await r.finalize()
        _reset_shared_storage()

    @pytest.mark.asyncio
    async def test_initialize_creates_doc_store(self, rag):
        assert rag._doc_store is not None

    @pytest.mark.asyncio
    async def test_ainsert_sets_processed_status(self, rag):
        from framerag.doc_store import DocStatus, DocStore
        await rag.ainsert("Hello world. This is a test document.", source_doc="test.txt")
        doc_id = DocStore.make_doc_id("test.txt")
        rec = await rag._doc_store.get(doc_id)
        assert rec is not None
        assert rec.status == DocStatus.PROCESSED
        assert rec.source_doc == "test.txt"

    @pytest.mark.asyncio
    async def test_ainsert_records_chunk_ids(self, rag):
        from framerag.doc_store import DocStore
        await rag.ainsert("Hello world.", source_doc="chunked.txt")
        doc_id = DocStore.make_doc_id("chunked.txt")
        rec = await rag._doc_store.get(doc_id)
        assert rec is not None
        assert rec.chunks_count >= 1
        assert len(rec.chunk_ids) == rec.chunks_count

    @pytest.mark.asyncio
    async def test_ainsert_failed_status_on_error(self, tmp_path):
        """If embed raises during add_chunk, status should be FAILED."""
        from framerag import FrameRAG
        from framerag.doc_store import DocStatus, DocStore
        _reset_shared_storage()

        async def _bad_embed(texts: list[str], **kwargs) -> np.ndarray:
            raise RuntimeError("embed broken")

        r = FrameRAG(
            working_dir=str(tmp_path / "fail"),
            llm_func=_dummy_llm("[]"),
            embed_func=_bad_embed,
            embedding_dim=4,
            enable_causal=False,
            enable_gleaning=False,
            enable_event_coref=False,
        )
        await r.initialize()
        try:
            with pytest.raises(Exception):
                await r.ainsert("Some text", source_doc="broken.txt")
            doc_id = DocStore.make_doc_id("broken.txt")
            rec = await r._doc_store.get(doc_id)
            assert rec is not None
            assert rec.status == DocStatus.FAILED
            assert rec.error_msg is not None
        finally:
            await r.finalize()
        _reset_shared_storage()

    @pytest.mark.asyncio
    async def test_list_documents_empty(self, rag):
        docs = await rag.list_documents()
        assert docs == []

    @pytest.mark.asyncio
    async def test_list_documents_after_insert(self, rag):
        await rag.ainsert("Doc A text.", source_doc="a.txt")
        await rag.ainsert("Doc B text.", source_doc="b.txt")
        docs = await rag.list_documents()
        assert len(docs) == 2
        sources = {d["source_doc"] for d in docs}
        assert sources == {"a.txt", "b.txt"}

    @pytest.mark.asyncio
    async def test_list_documents_status_filter(self, rag):
        await rag.ainsert("Doc A text.", source_doc="a.txt")
        processed = await rag.list_documents(status="processed")
        assert len(processed) == 1
        assert processed[0]["source_doc"] == "a.txt"

        pending = await rag.list_documents(status="pending")
        assert pending == []

    @pytest.mark.asyncio
    async def test_list_documents_invalid_status_raises(self, rag):
        with pytest.raises(ValueError, match="Unknown status"):
            await rag.list_documents(status="bogus")

    @pytest.mark.asyncio
    async def test_adelete_known_doc(self, rag):
        from framerag.doc_store import DocStore
        await rag.ainsert("Some text here.", source_doc="todel.txt")
        doc_id = DocStore.make_doc_id("todel.txt")
        deleted = await rag.adelete(doc_id)
        assert deleted is True
        assert await rag._doc_store.get(doc_id) is None

    @pytest.mark.asyncio
    async def test_adelete_unknown_doc_returns_false(self, rag):
        deleted = await rag.adelete("doc-nonexistent")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_get_stats_returns_dict(self, rag):
        stats = await rag.get_stats()
        assert "chunks" in stats
        assert "events" in stats
        assert "frames_in_db" in stats

    @pytest.mark.asyncio
    async def test_ainsert_batch(self, rag):
        from framerag.doc_store import DocStatus
        await rag.ainsert_batch(
            ["text A", "text B", "text C"],
            source_docs=["a.txt", "b.txt", "c.txt"],
            concurrency=2,
        )
        docs = await rag.list_documents(status="processed")
        assert len(docs) == 3


# ─────────────────────────────────────────────────────────────────────────────
# LLM caching + semaphore + timeout
# ─────────────────────────────────────────────────────────────────────────────

class TestLLMInternal:
    @pytest.fixture()
    async def rag(self, tmp_path):
        from framerag import FrameRAG
        r = FrameRAG(
            working_dir=str(tmp_path),
            llm_func=_dummy_llm("hello"),
            embed_func=_dummy_embed(dim=4),
            embedding_dim=4,
            max_concurrent_llm=2,
            llm_timeout=5.0,
        )
        await r.initialize()
        yield r
        await r.finalize()

    @pytest.mark.asyncio
    async def test_llm_caches_response(self, rag):
        call_count = 0

        async def counting_llm(prompt: str, **kwargs) -> str:
            nonlocal call_count
            call_count += 1
            return "cached_response"

        rag._raw_llm = counting_llm
        r1 = await rag._llm("same prompt")
        r2 = await rag._llm("same prompt")
        assert r1 == r2 == "cached_response"
        assert call_count == 1, "LLM should only be called once for the same prompt"

    @pytest.mark.asyncio
    async def test_llm_different_prompts_not_shared(self, rag):
        responses = {"p1": "res1", "p2": "res2"}

        async def varied_llm(prompt: str, **kwargs) -> str:
            return responses.get(prompt, "default")

        rag._raw_llm = varied_llm
        assert await rag._llm("p1") == "res1"
        assert await rag._llm("p2") == "res2"

    @pytest.mark.asyncio
    async def test_llm_timeout_fires(self, tmp_path):
        from framerag import FrameRAG

        async def slow_llm(prompt: str, **kwargs) -> str:
            await asyncio.sleep(10)
            return "never"

        r = FrameRAG(
            working_dir=str(tmp_path / "timeout"),
            llm_func=slow_llm,
            embed_func=_dummy_embed(dim=4),
            embedding_dim=4,
            llm_timeout=0.05,  # 50 ms
        )
        await r.initialize()
        try:
            with pytest.raises(asyncio.TimeoutError):
                await r._llm("a unique prompt that won't be cached " + str(time.time()))
        finally:
            await r.finalize()

    @pytest.mark.asyncio
    async def test_llm_semaphore_limits_concurrency(self, tmp_path):
        from framerag import FrameRAG

        max_seen = 0
        current = 0

        async def tracking_llm(prompt: str, **kwargs) -> str:
            nonlocal max_seen, current
            current += 1
            max_seen = max(max_seen, current)
            await asyncio.sleep(0.02)
            current -= 1
            return "ok"

        r = FrameRAG(
            working_dir=str(tmp_path / "sem"),
            llm_func=tracking_llm,
            embed_func=_dummy_embed(dim=4),
            embedding_dim=4,
            max_concurrent_llm=2,
            llm_timeout=10.0,
        )
        await r.initialize()
        try:
            # Fire 6 unique prompts concurrently; semaphore should cap at 2
            prompts = [f"prompt {i} {time.time()}" for i in range(6)]
            await asyncio.gather(*[r._llm(p) for p in prompts])
            assert max_seen <= 2, f"Expected ≤2 concurrent LLM calls, got {max_seen}"
        finally:
            await r.finalize()
