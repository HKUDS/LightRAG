import asyncio
import sys

import pytest

sys.argv = sys.argv[:1]

from lightrag.api.routers.document_routes import (  # noqa: E402
    DocStatusResponse,
    normalize_file_path,
    pipeline_index_texts,
)
from lightrag.base import DocStatus  # noqa: E402
from lightrag.constants import PROCESS_OPTION_CHUNK_FIXED  # noqa: E402
from lightrag.pipeline import _PipelineMixin  # noqa: E402


class DummyRAG:
    def __init__(self):
        self.enqueued_calls = []
        self.processed = False
        # _resolve_text_chunking reads addon_params; {} -> default chunker config.
        self.addon_params = {}

    async def apipeline_enqueue_documents(
        self,
        input,
        file_paths=None,
        track_id=None,
        process_options=None,
        chunk_options=None,
    ):
        self.enqueued_calls.append(
            {
                "input": input,
                "file_paths": file_paths,
                "track_id": track_id,
                "process_options": process_options,
                "chunk_options": chunk_options,
            }
        )

    async def apipeline_process_enqueue_documents(self):
        self.processed = True


class CaptureDocStatus:
    def __init__(self):
        self.upserts = []

    async def upsert(self, data):
        self.upserts.append(data)


class DummyPipeline(_PipelineMixin):
    def __init__(self):
        self.doc_status = CaptureDocStatus()


class CaptureKV:
    def __init__(self):
        self.upserts = []

    async def filter_keys(self, keys):
        return set(keys)

    async def upsert(self, data):
        self.upserts.append(data)

    async def get_by_id(self, key):
        return None

    async def get_by_ids(self, keys):
        return [None for _ in keys]

    async def index_done_callback(self):
        pass


class _FakeKeyedLock:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


def _patch_custom_chunk_saga(monkeypatch, rag):
    """Wire a bare LightRAG for the #3400 Phase-3 custom-chunk saga: a
    doc_status store for the journal, a flushable llm_response_cache for the
    staging barrier, and a stubbed per-document keyed lock (the real one
    needs initialized shared storage)."""
    import lightrag.lightrag as lightrag_module

    rag.doc_status = CaptureKV()
    rag.llm_response_cache = CaptureKV()
    monkeypatch.setattr(
        lightrag_module,
        "get_storage_keyed_lock",
        lambda keys, namespace="", enable_logging=False: _FakeKeyedLock(),
    )


@pytest.mark.asyncio
async def test_pipeline_index_texts_rejects_missing_file_sources():
    rag = DummyRAG()

    with pytest.raises(ValueError, match="valid file source"):
        await pipeline_index_texts(
            rag,
            texts=["alpha"],
            file_sources=[None],
            track_id="track-1",
        )

    assert rag.enqueued_calls == []
    assert rag.processed is False


@pytest.mark.asyncio
async def test_pipeline_index_texts_normalizes_file_sources_to_basename():
    rag = DummyRAG()

    await pipeline_index_texts(
        rag,
        texts=["alpha"],
        file_sources=["/tmp/source/alpha.txt"],
        track_id="track-1",
    )

    assert len(rag.enqueued_calls) == 1
    call = rag.enqueued_calls[0]
    assert call["input"] == ["alpha"]
    assert call["file_paths"] == ["alpha.txt"]
    assert call["track_id"] == "track-1"
    assert call["process_options"] == PROCESS_OPTION_CHUNK_FIXED
    # No chunking config supplied -> default F snapshot from addon_params.
    assert isinstance(call["chunk_options"], dict)
    assert "fixed_token" in call["chunk_options"]
    assert rag.processed is True


def test_doc_status_response_uses_non_null_unknown_source():
    response = DocStatusResponse(
        id="doc-1",
        content_summary="summary",
        content_length=5,
        status=DocStatus.PENDING,
        created_at="2026-03-19T00:00:00+00:00",
        updated_at="2026-03-19T00:00:00+00:00",
        file_path=normalize_file_path(None),
    )

    assert response.file_path == "unknown_source"


@pytest.mark.asyncio
async def test_error_document_enqueue_canonicalizes_file_path_before_upsert():
    rag = DummyPipeline()

    await rag.apipeline_enqueue_error_documents(
        [
            {
                "file_path": "/tmp/uploads/report.[native-Fi].pdf",
                "error_description": "bad file",
                "original_error": "parse failed",
            }
        ],
        track_id="track-1",
    )

    saved = next(iter(rag.doc_status.upserts[0].values()))
    assert saved["file_path"] == "report.pdf"


@pytest.mark.asyncio
async def test_custom_chunks_use_canonical_unknown_source_before_upsert(monkeypatch):
    from lightrag import LightRAG
    import lightrag.lightrag as lightrag_module

    rag = LightRAG.__new__(LightRAG)
    rag.full_docs = CaptureKV()
    rag.text_chunks = CaptureKV()
    rag.chunks_vdb = CaptureKV()
    rag.tokenizer = type("Tokenizer", (), {"encode": lambda self, text: [text]})()
    rag.workspace = "test-workspace"
    _patch_custom_chunk_saga(monkeypatch, rag)

    async def _process_extract_entities(
        chunks, pipeline_status=None, pipeline_status_lock=None
    ):
        return []

    async def _insert_done():
        return None

    rag._process_extract_entities = _process_extract_entities
    rag._insert_done = _insert_done

    async def fake_namespace_data(name, workspace=None):
        return {}

    def fake_namespace_lock(name, workspace=None):
        return asyncio.Lock()

    monkeypatch.setattr(lightrag_module, "get_namespace_data", fake_namespace_data)
    monkeypatch.setattr(lightrag_module, "get_namespace_lock", fake_namespace_lock)

    await rag.ainsert_custom_chunks("full text", ["chunk text"], doc_id="doc-1")

    assert rag.full_docs.upserts[0]["doc-1"]["file_path"] == "unknown_source"
    chunk = next(iter(rag.text_chunks.upserts[0].values()))
    assert chunk["file_path"] == "unknown_source"


@pytest.mark.asyncio
async def test_custom_chunks_merge_extracted_entities_into_kg(monkeypatch):
    """`ainsert_custom_chunks` must merge extracted entities into the KG.

    Regression: `_process_extract_entities` ran but its result was discarded,
    so `merge_nodes_and_edges` was never called and no knowledge graph was
    built from custom chunks (KG-dependent query modes returned nothing while
    the extraction LLM cost was still spent).
    """
    from lightrag import LightRAG
    import lightrag.lightrag as lightrag_module

    rag = LightRAG.__new__(LightRAG)
    rag.full_docs = CaptureKV()
    rag.text_chunks = CaptureKV()
    rag.chunks_vdb = CaptureKV()
    rag.tokenizer = type("Tokenizer", (), {"encode": lambda self, text: [text]})()
    rag.workspace = "test-workspace"
    # Referenced positionally by the merge call; unused because merge is stubbed.
    for attr in (
        "chunk_entity_relation_graph",
        "entities_vdb",
        "relationships_vdb",
        "full_entities",
        "full_relations",
        "entity_chunks",
        "relation_chunks",
        "llm_response_cache",
    ):
        setattr(rag, attr, object())
    _patch_custom_chunk_saga(monkeypatch, rag)

    extracted = [({"Entity": [{"entity_name": "Entity"}]}, {})]

    async def _process_extract_entities(
        chunks, pipeline_status=None, pipeline_status_lock=None
    ):
        return extracted

    async def _insert_done():
        return None

    rag._process_extract_entities = _process_extract_entities
    rag._insert_done = _insert_done
    rag._build_global_config = lambda: {}

    captured: dict = {}

    async def fake_merge(*, chunk_results, doc_id, **kwargs):
        captured["chunk_results"] = chunk_results
        captured["doc_id"] = doc_id

    async def fake_namespace_data(name, workspace=None):
        return {}

    def fake_namespace_lock(name, workspace=None):
        return asyncio.Lock()

    monkeypatch.setattr(lightrag_module, "merge_nodes_and_edges", fake_merge)
    monkeypatch.setattr(lightrag_module, "get_namespace_data", fake_namespace_data)
    monkeypatch.setattr(lightrag_module, "get_namespace_lock", fake_namespace_lock)

    await rag.ainsert_custom_chunks("full text", ["chunk text"], doc_id="doc-2")

    # The extracted entities/relationships must reach the KG merge step.
    assert captured["chunk_results"] == extracted
    assert captured["doc_id"] == "doc-2"


@pytest.mark.asyncio
async def test_process_extract_entities_surfaces_real_error_without_lock(monkeypatch):
    """With no pipeline_status/lock, a failing extraction must surface its REAL
    error, not a TypeError from ``async with None`` masking it.

    Regression (#3367 P2): the except block wrote to pipeline_status under
    ``async with pipeline_status_lock`` without a None guard, so a lock-less
    caller saw ``TypeError: NoneType ... async context manager`` instead of the
    actual extraction failure.
    """
    from lightrag import LightRAG
    import lightrag.lightrag as lightrag_module

    rag = LightRAG.__new__(LightRAG)
    rag.llm_response_cache = object()
    rag.text_chunks = object()
    rag._build_global_config = lambda: {}

    async def boom(*args, **kwargs):
        raise ValueError("real extraction failure")

    monkeypatch.setattr(lightrag_module, "extract_entities", boom)

    # Called with the default None status/lock (a lock-less direct caller).
    with pytest.raises(ValueError, match="real extraction failure"):
        await rag._process_extract_entities({"chunk-1": {"content": "x"}})


@pytest.mark.asyncio
async def test_custom_chunks_persist_before_extraction(monkeypatch):
    """Chunks must be persisted (Stage-1 barrier) BEFORE extraction runs.

    Regression (#3367 P8): extraction records per-chunk LLM cache references and
    reads chunks back via get_by_id; running it concurrently with the chunk
    upserts can observe a not-yet-persisted chunk and silently drop the cache
    reference. A slow text_chunks upsert makes the ordering observable: with the
    old concurrent gather, extraction ran before the upsert finished; with the
    barrier it runs after.
    """
    import asyncio

    from lightrag import LightRAG
    import lightrag.lightrag as lightrag_module

    events: list[str] = []

    class SlowText(CaptureKV):
        async def upsert(self, data):
            await asyncio.sleep(0.05)  # let a concurrent extraction race ahead
            events.append("text_chunks_persisted")
            await super().upsert(data)

    rag = LightRAG.__new__(LightRAG)
    rag.full_docs = CaptureKV()
    rag.text_chunks = SlowText()
    rag.chunks_vdb = CaptureKV()
    rag.tokenizer = type("Tokenizer", (), {"encode": lambda self, text: [text]})()
    rag.workspace = "test-workspace"
    for attr in (
        "chunk_entity_relation_graph",
        "entities_vdb",
        "relationships_vdb",
        "full_entities",
        "full_relations",
        "entity_chunks",
        "relation_chunks",
        "llm_response_cache",
    ):
        setattr(rag, attr, object())
    _patch_custom_chunk_saga(monkeypatch, rag)

    async def _process_extract_entities(
        chunks, pipeline_status=None, pipeline_status_lock=None
    ):
        events.append("extract")
        # The barrier guarantees the chunk upsert completed first.
        assert "text_chunks_persisted" in events, (
            "extraction ran before chunks were persisted"
        )
        return []

    async def _insert_done():
        return None

    rag._process_extract_entities = _process_extract_entities
    rag._insert_done = _insert_done
    rag._build_global_config = lambda: {}

    async def fake_namespace_data(name, workspace=None):
        return {}

    def fake_namespace_lock(name, workspace=None):
        return asyncio.Lock()

    monkeypatch.setattr(lightrag_module, "get_namespace_data", fake_namespace_data)
    monkeypatch.setattr(lightrag_module, "get_namespace_lock", fake_namespace_lock)

    await rag.ainsert_custom_chunks("full text", ["chunk text"], doc_id="doc-3")

    assert events.index("text_chunks_persisted") < events.index("extract")


def _custom_chunks_rag(monkeypatch, status):
    """A bare LightRAG wired for ainsert_custom_chunks with a shared, mutable
    ``status`` dict and a real asyncio lock, extraction/merge stubbed."""
    from lightrag import LightRAG
    import lightrag.lightrag as lightrag_module

    rag = LightRAG.__new__(LightRAG)
    rag.full_docs = CaptureKV()
    rag.text_chunks = CaptureKV()
    rag.chunks_vdb = CaptureKV()
    rag.tokenizer = type("Tokenizer", (), {"encode": lambda self, text: [text]})()
    rag.workspace = "test-workspace"
    for attr in (
        "chunk_entity_relation_graph",
        "entities_vdb",
        "relationships_vdb",
        "full_entities",
        "full_relations",
        "entity_chunks",
        "relation_chunks",
        "llm_response_cache",
    ):
        setattr(rag, attr, object())
    _patch_custom_chunk_saga(monkeypatch, rag)
    rag._build_global_config = lambda: {}

    async def _insert_done():
        return None

    rag._insert_done = _insert_done

    lock = asyncio.Lock()

    async def fake_namespace_data(name, workspace=None):
        return status

    def fake_namespace_lock(name, workspace=None):
        return lock

    async def fake_merge(**kwargs):
        return None

    monkeypatch.setattr(lightrag_module, "get_namespace_data", fake_namespace_data)
    monkeypatch.setattr(lightrag_module, "get_namespace_lock", fake_namespace_lock)
    monkeypatch.setattr(lightrag_module, "merge_nodes_and_edges", fake_merge)
    return rag


@pytest.mark.asyncio
async def test_custom_chunks_rejects_when_pipeline_busy(monkeypatch):
    """ainsert_custom_chunks writes the KG (Stage 3), so it must not run while
    the pipeline is busy — a second concurrent merger would double LLM
    concurrency and desync status/cancellation. It must reject, and neither
    extract nor merge may run."""
    status = {"busy": True, "history_messages": []}
    rag = _custom_chunks_rag(monkeypatch, status)

    called = {"extract": False, "merge": False}

    async def _process_extract_entities(chunks, ps=None, pl=None):
        called["extract"] = True
        return [({"E": [{"entity_name": "E"}]}, {})]

    import lightrag.lightrag as lightrag_module

    async def fake_merge(**kwargs):
        called["merge"] = True

    rag._process_extract_entities = _process_extract_entities
    monkeypatch.setattr(lightrag_module, "merge_nodes_and_edges", fake_merge)

    with pytest.raises(RuntimeError, match="busy"):
        await rag.ainsert_custom_chunks("full text", ["chunk text"], doc_id="doc-b")

    assert called["extract"] is False
    assert called["merge"] is False
    # The pre-existing busy holder's flag must be left untouched.
    assert status["busy"] is True


@pytest.mark.asyncio
async def test_custom_chunks_hands_off_busy_atomically(monkeypatch):
    """A request_pending that arrived while custom-chunks held busy is handed
    off to processing WITHOUT ever dropping busy to False first — otherwise a
    clear/delete/scan reservation could start in the gap and drop the accepted
    doc. The handoff must run with _holding_busy=True while busy is still True,
    and the (real) processing run releases busy atomically."""
    status = {"busy": False, "history_messages": [], "request_pending": False}
    rag = _custom_chunks_rag(monkeypatch, status)

    observed = {}

    async def _process_extract_entities(chunks, ps=None, pl=None):
        observed["busy_during_extract"] = status["busy"]
        # Simulate a concurrent enqueue arriving while we hold busy.
        status["request_pending"] = True
        return [({"E": [{"entity_name": "E"}]}, {})]

    drained = {}

    async def _drain(_holding_busy=False, token=None):
        drained["called"] = True
        drained["holding_busy"] = _holding_busy
        drained["token"] = token
        # The slot must NOT have been released before handing off (no window).
        drained["busy_at_handoff"] = status["busy"]
        # The real run consumes request_pending and releases busy (+owner) at exit.
        status["request_pending"] = False
        status.update({"busy": False, "busy_owner": None})

    rag._process_extract_entities = _process_extract_entities
    rag.apipeline_process_enqueue_documents = _drain

    await rag.ainsert_custom_chunks("full text", ["chunk text"], doc_id="doc-h")

    assert observed["busy_during_extract"] is True  # held during the work
    assert drained["called"] is True  # queued docs handed off
    assert drained["holding_busy"] is True  # atomic handoff, not a fresh acquire
    assert drained["token"] is not None  # handoff passes the owner token
    assert drained["busy_at_handoff"] is True  # busy never dropped before handoff
    assert status["request_pending"] is False  # consumed by the handoff
    assert status["busy"] is False  # released by the handoff run


@pytest.mark.asyncio
async def test_custom_chunks_handoff_runs_even_when_flush_errors(monkeypatch):
    """Regression (#3408 Codex P2): a doc enqueued while custom-chunks held busy
    (request_pending=True → decision="handoff", busy kept True) must still be
    drained when the flush (_insert_done_with_cleanup) then raises. Old code put
    the ``if decision == "handoff"`` branch AFTER the inner finally, so a flush
    error skipped it and the outer terminal release cleared busy without
    draining — stranding the queued docs behind request_pending=True with no
    active processor. The flush error must still propagate after the handoff."""
    status = {"busy": False, "history_messages": [], "request_pending": False}
    rag = _custom_chunks_rag(monkeypatch, status)

    async def _process_extract_entities(chunks, ps=None, pl=None):
        # A concurrent enqueue arrived while we held busy.
        status["request_pending"] = True
        return [({"E": [{"entity_name": "E"}]}, {})]

    async def _flush_boom():
        raise RuntimeError("flush failed")

    drained = {"called": False}

    async def _drain(_holding_busy=False, token=None):
        drained["called"] = True
        drained["holding_busy"] = _holding_busy
        # The slot must NOT have been released before handing off (no window),
        # even though the flush errored.
        drained["busy_at_handoff"] = status["busy"]
        # The real run consumes request_pending and releases busy (+owner).
        status["request_pending"] = False
        status.update({"busy": False, "busy_owner": None})

    rag._process_extract_entities = _process_extract_entities
    rag._insert_done_with_cleanup = _flush_boom
    rag.apipeline_process_enqueue_documents = _drain

    # The flush error still surfaces to the caller after the handoff runs.
    with pytest.raises(RuntimeError, match="flush failed"):
        await rag.ainsert_custom_chunks("full text", ["chunk text"], doc_id="doc-fe")

    # Handoff ran despite the flush error (old code skipped it).
    assert drained["called"] is True
    assert drained["holding_busy"] is True  # atomic handoff, not a fresh acquire
    assert drained["busy_at_handoff"] is True  # busy never dropped before handoff
    assert status["request_pending"] is False  # consumed by the handoff
    assert status["busy"] is False  # released by the handoff run


@pytest.mark.asyncio
async def test_custom_chunks_releases_busy_without_pending(monkeypatch):
    """With no request_pending, the busy slot is released directly and no
    handoff runs."""
    status = {"busy": False, "history_messages": [], "request_pending": False}
    rag = _custom_chunks_rag(monkeypatch, status)

    drained = {"called": False}

    async def _drain(_holding_busy=False):
        drained["called"] = True

    async def _process_extract_entities(chunks, ps=None, pl=None):
        return [({"E": [{"entity_name": "E"}]}, {})]

    rag._process_extract_entities = _process_extract_entities
    rag.apipeline_process_enqueue_documents = _drain

    await rag.ainsert_custom_chunks("full text", ["chunk text"], doc_id="doc-np")

    assert status["busy"] is False
    assert drained["called"] is False
    assert drained["called"] is False


@pytest.mark.asyncio
async def test_custom_chunks_busy_rejection_does_not_flush_shared_buffers(monkeypatch):
    """P1: a busy rejection must NOT run _insert_done_with_cleanup. The call
    neither acquired the slot nor wrote data, so flushing/discarding the SHARED
    pending buffers could commit or tear down the running job's in-flight ops."""
    status = {"busy": True, "history_messages": []}
    rag = _custom_chunks_rag(monkeypatch, status)

    cleanup = {"called": False}

    async def _insert_done_with_cleanup():
        cleanup["called"] = True

    async def _process_extract_entities(chunks, ps=None, pl=None):
        return []

    rag._insert_done_with_cleanup = _insert_done_with_cleanup
    rag._process_extract_entities = _process_extract_entities

    with pytest.raises(RuntimeError, match="busy"):
        await rag.ainsert_custom_chunks("full text", ["chunk text"], doc_id="doc-p1")

    assert cleanup["called"] is False


@pytest.mark.asyncio
async def test_custom_chunks_clears_stale_cancellation(monkeypatch):
    """A stale cancellation_requested (e.g. left by a previously cancelled
    custom-chunks job, since the cancel endpoint sets it whenever busy=True)
    must not abort the next insert. The busy lifecycle clears the cancellation
    fields on acquire (so extract/merge are not pre-killed) and on release (so
    it does not leak to the next job)."""
    status = {
        "busy": False,
        "history_messages": [],
        "request_pending": False,
        "cancellation_requested": True,  # stale from a prior cancelled job
        "cancellation_reason": "internal_error",
        "cancellation_detail": "stale",
    }
    rag = _custom_chunks_rag(monkeypatch, status)

    observed = {}

    async def _process_extract_entities(chunks, ps=None, pl=None):
        # Real extract/merge raise PipelineCancelledException when this is True;
        # it must have been cleared on acquire so this job runs.
        observed["cancel_during"] = status.get("cancellation_requested")
        return [({"E": [{"entity_name": "E"}]}, {})]

    rag._process_extract_entities = _process_extract_entities

    await rag.ainsert_custom_chunks("full text", ["chunk text"], doc_id="doc-cancel")

    assert observed["cancel_during"] is False  # cleared on acquire
    assert status["cancellation_requested"] is False  # cleared on release
    assert status["cancellation_reason"] is None
    assert status["cancellation_detail"] is None


@pytest.mark.asyncio
async def test_custom_chunks_overwrites_stale_deletion_job_name(monkeypatch):
    """A stale ``Deleting N Documents`` job_name (left by a finished batch delete,
    which releases busy but not job_name) must be overwritten when custom-chunks
    takes the busy slot. Otherwise a concurrent adelete_by_doc_id — which joins a
    running delete whenever busy=True and job_name starts with 'deleting' and
    contains 'document' — would proceed and race custom-chunks' KG/vector writes.
    """
    status = {
        "busy": False,
        "history_messages": [],
        "request_pending": False,
        "job_name": "Deleting 5 Documents",  # stale, from a finished batch delete
    }
    rag = _custom_chunks_rag(monkeypatch, status)

    observed = {}

    async def _process_extract_entities(chunks, ps=None, pl=None):
        # Captured while we hold busy — this is what a concurrent
        # adelete_by_doc_id would see and key its join-guard off.
        observed["job_name_during"] = status["job_name"]
        return [({"E": [{"entity_name": "E"}]}, {})]

    rag._process_extract_entities = _process_extract_entities

    await rag.ainsert_custom_chunks("full text", ["chunk text"], doc_id="doc-jn")

    jn = observed["job_name_during"].lower()
    # The exact adelete_by_doc_id join-guard predicate must be False while
    # custom-chunks holds busy, so a concurrent single delete is rejected.
    assert not (jn.startswith("deleting") and "document" in jn)


@pytest.mark.asyncio
async def test_custom_chunks_release_survives_cancel_at_lock_exit(monkeypatch):
    """If cancellation is delivered while exiting the acquire's status lock —
    its __aexit__ awaits an asyncio.shield, a real cancellation point — right
    after busy was set, the finally must still release busy. busy_acquired is
    recorded inside the lock (atomically with busy), so a cancel at the lock
    boundary can't leave busy=True with busy_acquired=False and wedge the
    workspace permanently busy.
    """
    from lightrag import LightRAG
    import lightrag.lightrag as lightrag_module

    status = {"busy": False, "history_messages": [], "request_pending": False}

    class _CancelOnFirstExitLock:
        """Delivers CancelledError at the FIRST __aexit__ (the acquire block's
        exit, after busy was set); later exits (the finally's release) behave
        normally, mirroring how asyncio.shield in the real lock re-raises a
        cancel at the await boundary."""

        def __init__(self):
            self._exits = 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            self._exits += 1
            if self._exits == 1:
                raise asyncio.CancelledError()
            return False

    lock = _CancelOnFirstExitLock()

    rag = LightRAG.__new__(LightRAG)
    rag.full_docs = CaptureKV()
    rag.text_chunks = CaptureKV()
    rag.chunks_vdb = CaptureKV()
    rag.tokenizer = type("Tokenizer", (), {"encode": lambda self, text: [text]})()
    rag.workspace = "test-workspace"

    async def _insert_done_with_cleanup():
        return None

    rag._insert_done_with_cleanup = _insert_done_with_cleanup

    async def fake_namespace_data(name, workspace=None):
        return status

    def fake_namespace_lock(name, workspace=None):
        return lock

    monkeypatch.setattr(lightrag_module, "get_namespace_data", fake_namespace_data)
    monkeypatch.setattr(lightrag_module, "get_namespace_lock", fake_namespace_lock)

    with pytest.raises(asyncio.CancelledError):
        await rag.ainsert_custom_chunks("full text", ["chunk text"], doc_id="doc-x")

    # Despite the cancel at the acquire-lock exit, busy must be released.
    assert status["busy"] is False
