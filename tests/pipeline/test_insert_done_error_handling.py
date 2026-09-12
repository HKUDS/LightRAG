"""Offline unit tests for the storage-flush error handling around
``index_done_callback`` (PR #3187).

These lock the ``LightRAG._insert_done`` / ``_discard_pending_index_ops`` /
``_insert_done_with_cleanup`` contract that the file pipeline relies on to
fail fast on a shared-buffer flush error instead of cascading every
subsequent document into FAILED.

The tests inject lightweight spy storages via ``_index_storages`` (and, for
the enqueue-owned / LLM-cache special cases, by binding the spy onto the real
``rag.full_docs`` / ``rag.doc_status`` / ``rag.llm_response_cache`` attributes
so the ``is`` identity checks fire). No storage driver is imported and no real
backend flush is exercised — this is pure lightrag.py logic.
"""

from __future__ import annotations

import asyncio
import logging
from uuid import uuid4

import numpy as np
import pytest

from lightrag import LightRAG
from lightrag.exceptions import IndexFlushError
from lightrag.utils import EmbeddingFunc, Tokenizer

pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def _propagate_lightrag_logs():
    """The ``lightrag`` logger sets ``propagate=False``, so caplog's root
    handler would miss its records. Re-enable propagation for these tests so
    ``caplog`` can capture the best-effort error logs we assert on."""
    lg = logging.getLogger("lightrag")
    old = lg.propagate
    lg.propagate = True
    try:
        yield
    finally:
        lg.propagate = old


class _SimpleTokenizerImpl:
    def encode(self, content: str) -> list[int]:
        return [ord(ch) for ch in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


async def _mock_embedding(texts: list[str]) -> np.ndarray:
    return np.ones((len(texts), 8), dtype=float)


async def _noop_llm(*args, **kwargs) -> str:  # pragma: no cover - never invoked
    return ""


async def _make_rag(tmp_path) -> LightRAG:
    rag = LightRAG(
        working_dir=str(tmp_path / "wd"),
        workspace=f"insdone-{uuid4().hex[:8]}",
        llm_model_func=_noop_llm,
        embedding_func=EmbeddingFunc(
            embedding_dim=8, max_token_size=1024, func=_mock_embedding
        ),
        tokenizer=Tokenizer("mock-tokenizer", _SimpleTokenizerImpl()),
    )
    await rag.initialize_storages()
    return rag


_UNSET = object()


class _SpyStorage:
    """Minimal stand-in for a StorageNameSpace with configurable flush/drop.

    ``flush_error`` / ``drop_error`` are exception *instances* raised by the
    respective coroutine. ``recorder`` is a shared list appended with
    ``(label, "flush"|"drop")`` so call ordering across storages is assertable.
    """

    def __init__(
        self,
        label: str,
        *,
        namespace: str = "ns",
        final_namespace=_UNSET,
        flush_error: BaseException | None = None,
        flush_result=None,
        drop_error: BaseException | None = None,
        recorder: list | None = None,
        pending_index_ops: bool = False,
        dropped_upsert_count: int | None = None,
    ):
        self.label = label
        self.namespace = namespace
        if final_namespace is not _UNSET:
            self.final_namespace = final_namespace
        self._flush_error = flush_error
        self._flush_result = flush_result
        self._drop_error = drop_error
        self._recorder = recorder if recorder is not None else []
        self._pending_index_ops = pending_index_ops
        self._dropped_upsert_count = dropped_upsert_count
        self.index_done_calls = 0
        self.drop_calls = 0
        self.drop_upsert_calls = 0
        self.drop_upsert_cache_types: list = []
        self.has_pending_calls = 0

    async def index_done_callback(self):
        self.index_done_calls += 1
        self._recorder.append((self.label, "flush"))
        if self._flush_error is not None:
            raise self._flush_error
        return self._flush_result

    async def drop_pending_index_ops(self):
        self.drop_calls += 1
        self._recorder.append((self.label, "drop"))
        if self._drop_error is not None:
            raise self._drop_error

    async def drop_pending_upserts(self, *, cache_types=None) -> int | None:
        self.drop_upsert_calls += 1
        self.drop_upsert_cache_types.append(cache_types)
        self._recorder.append((self.label, "drop_upserts"))
        if self._drop_error is not None:
            raise self._drop_error
        return self._dropped_upsert_count

    async def has_pending_index_ops(self) -> bool:
        self.has_pending_calls += 1
        return self._pending_index_ops

    async def finalize(self):
        # No-op: keeps finalize_storages() quiet when a spy is bound onto a
        # real storage attribute (full_docs / doc_status / llm_response_cache).
        return None


# ---------------------------------------------------------------------------
# IndexFlushError class
# ---------------------------------------------------------------------------


def test_index_flush_error_attributes_and_message():
    cause = RuntimeError("boom")
    err = IndexFlushError("MilvusVectorDBStorage", "entities", cause)
    assert err.storage_name == "MilvusVectorDBStorage"
    assert err.namespace == "entities"
    assert str(err) == "MilvusVectorDBStorage[entities] index flush failed: boom"
    # __cause__ is NOT set by the constructor — only by `raise ... from e`
    # inside _insert_done. That linkage is asserted in the _insert_done tests.
    assert err.__cause__ is None


# ---------------------------------------------------------------------------
# _index_storages (internal contract — weak assertions)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_index_storages_filters_none_and_relative_order(tmp_path):
    rag = await _make_rag(tmp_path)
    try:
        storages = rag._index_storages()
        # None entries are filtered out.
        assert all(s is not None for s in storages)
        # Relative-order contract: enqueue-owned KVs come before the vdbs,
        # and the LLM cache precedes every vector store. (No fixed count is
        # asserted, so adding a new storage won't break this test.)
        idx = {id(s): i for i, s in enumerate(storages)}
        assert idx[id(rag.full_docs)] < idx[id(rag.entities_vdb)]
        assert idx[id(rag.doc_status)] < idx[id(rag.entities_vdb)]
        assert idx[id(rag.llm_response_cache)] < idx[id(rag.entities_vdb)]
        assert idx[id(rag.llm_response_cache)] < idx[id(rag.chunks_vdb)]

        # Filtering: drop one storage and confirm it disappears from the list.
        rag.text_chunks = None
        assert all(s is not None for s in rag._index_storages())
    finally:
        await rag.finalize_storages()


# ---------------------------------------------------------------------------
# _insert_done — success path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_insert_done_success_updates_pipeline_status(tmp_path, monkeypatch):
    rag = await _make_rag(tmp_path)
    try:
        spies = [_SpyStorage("a"), _SpyStorage("b")]
        monkeypatch.setattr(rag, "_index_storages", lambda: spies)
        status = {"latest_message": "", "history_messages": []}
        lock = asyncio.Lock()

        await rag._insert_done(pipeline_status=status, pipeline_status_lock=lock)

        assert all(s.index_done_calls == 1 for s in spies)
        assert status["latest_message"] == "In memory DB persist to disk"
        assert "In memory DB persist to disk" in status["history_messages"]
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_insert_done_success_without_status_no_update(tmp_path, monkeypatch):
    rag = await _make_rag(tmp_path)
    try:
        spies = [_SpyStorage("a")]
        monkeypatch.setattr(rag, "_index_storages", lambda: spies)
        # Should not raise and not require a status dict.
        await rag._insert_done()
        await rag._insert_done(pipeline_status={"history_messages": []})  # lock None
        assert spies[0].index_done_calls == 2
    finally:
        await rag.finalize_storages()


# ---------------------------------------------------------------------------
# _insert_done — failure wrapping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_insert_done_single_failure_wraps_index_flush_error(
    tmp_path, monkeypatch
):
    rag = await _make_rag(tmp_path)
    try:
        cause = RuntimeError("flush boom")
        spies = [_SpyStorage("ok"), _SpyStorage("bad", flush_error=cause)]
        monkeypatch.setattr(rag, "_index_storages", lambda: spies)

        with pytest.raises(IndexFlushError) as ei:
            await rag._insert_done()
        assert ei.value.storage_name == "_SpyStorage"
        # __cause__ linkage is established by `raise ... from e`.
        assert ei.value.__cause__ is cause
    finally:
        await rag.finalize_storages()


@pytest.mark.parametrize(
    "final_namespace, namespace, expected",
    [
        ("fns", "ns", "fns"),  # final_namespace wins
        (_UNSET, "ns", "ns"),  # falls back to namespace
        (_UNSET, "", ""),  # neither -> empty string
    ],
)
@pytest.mark.asyncio
async def test_insert_done_namespace_resolution(
    tmp_path, monkeypatch, final_namespace, namespace, expected
):
    rag = await _make_rag(tmp_path)
    try:
        spy = _SpyStorage(
            "bad",
            namespace=namespace,
            final_namespace=final_namespace,
            flush_error=RuntimeError("x"),
        )
        monkeypatch.setattr(rag, "_index_storages", lambda: [spy])
        with pytest.raises(IndexFlushError) as ei:
            await rag._insert_done()
        assert ei.value.namespace == expected
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_insert_done_multiple_failures_raises_first_logs_rest(
    tmp_path, monkeypatch, caplog
):
    rag = await _make_rag(tmp_path)
    try:
        spies = [
            _SpyStorage("a", flush_error=RuntimeError("first")),
            _SpyStorage("b", flush_error=RuntimeError("second")),
            _SpyStorage("c"),
        ]
        monkeypatch.setattr(rag, "_index_storages", lambda: spies)

        with caplog.at_level("ERROR", logger="lightrag"):
            with pytest.raises(IndexFlushError):
                await rag._insert_done()

        # gather(return_exceptions=True) runs ALL flushes to completion before
        # raising — no detached coroutines.
        assert all(s.index_done_calls == 1 for s in spies)
        # The non-first failure is logged, not raised.
        assert any(
            "Additional index flush failure" in rec.message for rec in caplog.records
        )
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_insert_done_cancelled_error_takes_priority(tmp_path, monkeypatch):
    rag = await _make_rag(tmp_path)
    try:
        spies = [
            _SpyStorage("normal", flush_error=RuntimeError("normal fail")),
            _SpyStorage("cancel", flush_error=asyncio.CancelledError()),
        ]
        monkeypatch.setattr(rag, "_index_storages", lambda: spies)
        # CancelledError must propagate as-is, not be wrapped in IndexFlushError.
        with pytest.raises(asyncio.CancelledError):
            await rag._insert_done()
    finally:
        await rag.finalize_storages()


# ---------------------------------------------------------------------------
# _discard_pending_index_ops
# ---------------------------------------------------------------------------


def _bind_enqueue_owned_spies(rag, recorder):
    """Bind spies onto the identity-checked attributes and return the list
    _index_storages should yield. ``other`` stands in for a regenerable vdb."""
    full = _SpyStorage("full_docs", recorder=recorder)
    status = _SpyStorage("doc_status", recorder=recorder)
    cache = _SpyStorage("llm_cache", recorder=recorder)
    other = _SpyStorage("other_vdb", recorder=recorder)
    rag.full_docs = full
    rag.doc_status = status
    rag.llm_response_cache = cache
    return full, status, cache, other


@pytest.mark.asyncio
async def test_discard_skip_enqueue_owned_true(tmp_path, monkeypatch):
    rag = await _make_rag(tmp_path)
    try:
        rec: list = []
        full, status, cache, other = _bind_enqueue_owned_spies(rag, rec)
        monkeypatch.setattr(
            rag, "_index_storages", lambda: [full, status, cache, other]
        )

        await rag._discard_pending_index_ops()  # skip_enqueue_owned=True default

        # full_docs / doc_status are skipped.
        assert full.drop_calls == 0
        assert status.drop_calls == 0
        # cache + other are dropped — the cache through the upserts-only
        # drop, which is the only one it ever takes (its buffered deletes are
        # tombstones an already-returned deletion promised).
        assert cache.drop_upsert_calls == 1
        assert cache.drop_calls == 0
        assert other.drop_calls == 1
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_discard_skip_enqueue_owned_false(tmp_path, monkeypatch):
    rag = await _make_rag(tmp_path)
    try:
        rec: list = []
        full, status, cache, other = _bind_enqueue_owned_spies(rag, rec)
        monkeypatch.setattr(
            rag, "_index_storages", lambda: [full, status, cache, other]
        )

        await rag._discard_pending_index_ops(skip_enqueue_owned=False)

        # Now full_docs / doc_status are ALSO dropped.
        assert full.drop_calls == 1
        assert status.drop_calls == 1
        assert cache.drop_upsert_calls == 1
        assert other.drop_calls == 1
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_discard_llm_cache_flush_before_drop(tmp_path, monkeypatch):
    rag = await _make_rag(tmp_path)
    try:
        rec: list = []
        full, status, cache, other = _bind_enqueue_owned_spies(rag, rec)
        monkeypatch.setattr(
            rag, "_index_storages", lambda: [full, status, cache, other]
        )

        await rag._discard_pending_index_ops()

        # The LLM cache is flushed (index_done_callback) BEFORE its buffer is
        # dropped — expensive cached results are persisted maximally first.
        assert ("llm_cache", "flush") in rec
        assert rec.index(("llm_cache", "flush")) < rec.index(
            ("llm_cache", "drop_upserts")
        )
        # Non-cache storages are only dropped, never flushed here.
        assert ("other_vdb", "flush") not in rec
        assert cache.index_done_calls == 1
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_discard_best_effort_swallows_drop_error(tmp_path, monkeypatch, caplog):
    rag = await _make_rag(tmp_path)
    try:
        rec: list = []
        full, status, cache, _ = _bind_enqueue_owned_spies(rag, rec)
        boom = _SpyStorage(
            "boom_vdb", drop_error=RuntimeError("drop boom"), recorder=rec
        )
        after = _SpyStorage("after_vdb", recorder=rec)
        monkeypatch.setattr(rag, "_index_storages", lambda: [cache, boom, after])

        with caplog.at_level("ERROR", logger="lightrag"):
            # Must NOT raise — cleanup is best-effort and never masks the
            # original abort cause.
            await rag._discard_pending_index_ops()

        assert boom.drop_calls == 1
        # A later storage is still processed despite the earlier drop error.
        assert after.drop_calls == 1
        assert any(
            "Failed to discard pending ops" in rec_.message for rec_ in caplog.records
        )
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_discard_llm_cache_flush_error_swallowed_still_drops(
    tmp_path, monkeypatch, caplog
):
    rag = await _make_rag(tmp_path)
    try:
        rec: list = []
        cache = _SpyStorage(
            "llm_cache", flush_error=RuntimeError("cache flush boom"), recorder=rec
        )
        rag.llm_response_cache = cache
        monkeypatch.setattr(rag, "_index_storages", lambda: [cache])

        with caplog.at_level("ERROR", logger="lightrag"):
            await rag._discard_pending_index_ops()

        # Flush failed (logged), but the drop still ran so a poisoned cache
        # item cannot wedge the next batch.
        assert cache.index_done_calls == 1
        assert cache.drop_upsert_calls == 1
        assert any(
            "Failed to persist LLM cache on abort" in r.message for r in caplog.records
        )
    finally:
        await rag.finalize_storages()


# ---------------------------------------------------------------------------
# _insert_done_with_cleanup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_insert_done_with_cleanup_success_no_discard(tmp_path, monkeypatch):
    rag = await _make_rag(tmp_path)
    try:
        monkeypatch.setattr(rag, "_index_storages", lambda: [_SpyStorage("a")])
        discard_calls = 0

        async def spy_discard(*, skip_enqueue_owned=True):
            nonlocal discard_calls
            discard_calls += 1

        monkeypatch.setattr(rag, "_discard_pending_index_ops", spy_discard)
        await rag._insert_done_with_cleanup()
        assert discard_calls == 0
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_insert_done_with_cleanup_index_flush_error_discards_and_reraises(
    tmp_path, monkeypatch
):
    rag = await _make_rag(tmp_path)
    try:
        spy = _SpyStorage("bad", flush_error=RuntimeError("boom"))
        monkeypatch.setattr(rag, "_index_storages", lambda: [spy])
        seen_kwargs = {}

        async def spy_discard(*, skip_enqueue_owned=True):
            seen_kwargs["skip_enqueue_owned"] = skip_enqueue_owned

        monkeypatch.setattr(rag, "_discard_pending_index_ops", spy_discard)

        with pytest.raises(IndexFlushError):
            await rag._insert_done_with_cleanup()
        # Direct callers clear full_docs too.
        assert seen_kwargs == {"skip_enqueue_owned": False}
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_insert_done_with_cleanup_cancelled_propagates_no_discard(
    tmp_path, monkeypatch
):
    rag = await _make_rag(tmp_path)
    try:
        spy = _SpyStorage("cancel", flush_error=asyncio.CancelledError())
        monkeypatch.setattr(rag, "_index_storages", lambda: [spy])
        discard_calls = 0

        async def spy_discard(*, skip_enqueue_owned=True):
            nonlocal discard_calls
            discard_calls += 1

        monkeypatch.setattr(rag, "_discard_pending_index_ops", spy_discard)

        # CancelledError is not an IndexFlushError, so cleanup must NOT run.
        with pytest.raises(asyncio.CancelledError):
            await rag._insert_done_with_cleanup()
        assert discard_calls == 0
    finally:
        await rag.finalize_storages()


# ---------------------------------------------------------------------------
# A DECLINED commit is not a successful flush (issue #3854)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_insert_done_rejects_a_declined_commit(tmp_path, monkeypatch):
    """``False`` means the backend refused to write and DISCARDED the pending
    mutation to converge on a newer file (``NetworkXStorage``'s decline
    branch). The flush used to ignore the return value, so the document was
    marked PROCESSED with its graph writes dropped and nothing left to recover
    them from. It has to fail, so the FAILED path's reprocessing re-extracts
    and re-writes the work."""
    rag = await _make_rag(tmp_path)
    try:
        spies = [_SpyStorage("ok"), _SpyStorage("declined", flush_result=False)]
        monkeypatch.setattr(rag, "_index_storages", lambda: spies)

        with pytest.raises(IndexFlushError) as ei:
            await rag._insert_done()
        assert "declined the commit" in str(ei.value)
        # Every flush still ran to completion before the raise.
        assert spies[0].index_done_calls == 1
    finally:
        await rag.finalize_storages()


@pytest.mark.parametrize("flush_result", [None, True])
@pytest.mark.asyncio
async def test_insert_done_accepts_none_and_true(tmp_path, monkeypatch, flush_result):
    """Identity, not truthiness: ``BaseGraphStorage.index_done_callback`` is
    declared ``-> None`` and most backends return nothing, so only an explicit
    ``False`` is an answer. Testing ``None`` for falsiness would fail every
    flush in the project."""
    rag = await _make_rag(tmp_path)
    try:
        spies = [_SpyStorage("plain", flush_result=flush_result)]
        monkeypatch.setattr(rag, "_index_storages", lambda: spies)

        await rag._insert_done()
        assert spies[0].index_done_calls == 1
    finally:
        await rag.finalize_storages()


# ---------------------------------------------------------------------------
# Reference-before-row at the commit layer
#
# An extract cache row is reachable only through the owning chunk's
# llm_cache_list, so text_chunks is chained AHEAD of llm_response_cache instead
# of gathered beside it. See *LLM extraction cache reachability* in
# docs/design/PurgeRecoveryContract.md.
# ---------------------------------------------------------------------------


def _bind_cache_pair_spies(rag, recorder, *, chunks_flush_error=None):
    chunks = _SpyStorage(
        "text_chunks", recorder=recorder, flush_error=chunks_flush_error
    )
    cache = _SpyStorage("llm_cache", recorder=recorder)
    other = _SpyStorage("other_vdb", recorder=recorder)
    rag.text_chunks = chunks
    rag.llm_response_cache = cache
    return chunks, cache, other


@pytest.mark.asyncio
async def test_insert_done_commits_chunk_references_before_cache_rows(
    tmp_path, monkeypatch
):
    """The pair is ordered; everything else still flushes concurrently."""
    rag = await _make_rag(tmp_path)
    try:
        rec: list = []
        # Cache first in the list, to prove the ordering comes from the chain
        # rather than from _index_storages' declaration order.
        chunks, cache, other = _bind_cache_pair_spies(rag, rec)
        monkeypatch.setattr(rag, "_index_storages", lambda: [cache, other, chunks])

        await rag._insert_done()

        assert rec.index(("text_chunks", "flush")) < rec.index(("llm_cache", "flush"))
        assert other.index_done_calls == 1
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_insert_done_skips_the_cache_flush_when_chunks_fail(
    tmp_path, monkeypatch
):
    """A failed chunk flush must not be followed by the rows it would strand.

    On OpenSearch a permanent bulk failure DROPS the chunk operation before
    raising, so a cache row committed beside it has no reference left at all.
    """
    rag = await _make_rag(tmp_path)
    try:
        rec: list = []
        chunks, cache, other = _bind_cache_pair_spies(
            rag, rec, chunks_flush_error=RuntimeError("permanent bulk failure")
        )
        monkeypatch.setattr(rag, "_index_storages", lambda: [chunks, cache, other])

        with pytest.raises(IndexFlushError):
            await rag._insert_done()

        assert cache.index_done_calls == 0, (
            "the cache rows were committed behind a chunk flush that failed"
        )
        # Unrelated namespaces are unaffected: only the pair is serialised.
        assert other.index_done_calls == 1
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_discard_flushes_chunk_references_before_the_cache(tmp_path, monkeypatch):
    """The aborting-batch cleanup obeys the same order — and it must, here more
    than anywhere: the loop only DROPS text_chunks, so a reference not committed
    before the cache flush is discarded outright."""
    rag = await _make_rag(tmp_path)
    try:
        rec: list = []
        chunks, cache, other = _bind_cache_pair_spies(rag, rec)
        monkeypatch.setattr(rag, "_index_storages", lambda: [chunks, cache, other])

        await rag._discard_pending_index_ops()

        assert rec.index(("text_chunks", "flush")) < rec.index(("llm_cache", "flush"))
        assert rec.index(("text_chunks", "flush")) < rec.index(("text_chunks", "drop"))
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_discard_skips_the_cache_flush_when_references_fail(
    tmp_path, monkeypatch, caplog
):
    """Losing the cached results is the accepted side of this trade.

    They are recomputed on the next run. A cache row published here behind a
    reference that just failed to commit — with both buffers dropped on the
    very next lines — is an unreachable row holding document text, which is
    permanent.
    """
    rag = await _make_rag(tmp_path)
    try:
        rec: list = []
        chunks, cache, other = _bind_cache_pair_spies(
            rag, rec, chunks_flush_error=RuntimeError("chunk store is down")
        )
        monkeypatch.setattr(rag, "_index_storages", lambda: [chunks, cache, other])

        with caplog.at_level("ERROR", logger="lightrag"):
            await rag._discard_pending_index_ops()

        assert cache.index_done_calls == 0, (
            "cache rows were published behind references that did not commit, "
            "and the next lines drop the buffer"
        )
        # Upserts only: the buffered DELETES are tombstones a completed
        # deletion promised, covered by its own test below.
        assert cache.drop_upsert_calls == 1
        assert cache.drop_calls == 0
        assert any(
            "Failed to persist chunk cache references on abort" in rec_.message
            for rec_ in caplog.records
        )
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_a_failed_chunk_commit_is_remembered_across_the_cleanup(
    tmp_path, monkeypatch, caplog
):
    """The cleanup must not re-read a buffer the backend already drained.

    OpenSearch removes a permanently-failed operation before raising, so the
    cleanup's own retry of text_chunks returns normally over a reference that
    is gone. Without the recorded failure it would then publish the cache rows.
    """
    rag = await _make_rag(tmp_path)
    try:
        rec: list = []
        chunks, cache, other = _bind_cache_pair_spies(
            rag, rec, chunks_flush_error=RuntimeError("permanent bulk failure")
        )
        monkeypatch.setattr(rag, "_index_storages", lambda: [chunks, cache, other])

        with pytest.raises(IndexFlushError):
            await rag._insert_done()

        # The backend dropped the failed op: a retry now reports success.
        chunks._flush_error = None

        with caplog.at_level("ERROR", logger="lightrag"):
            await rag._discard_pending_index_ops()

        assert cache.index_done_calls == 0, (
            "the cleanup trusted its own retry of a drained buffer and "
            "published cache rows behind a dropped reference"
        )
        assert any(
            "this batch's chunk references are not on disk" in rec_.message.lower()
            for rec_ in caplog.records
        )
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_the_chunk_failure_is_reported_over_an_unrelated_one(
    tmp_path, monkeypatch
):
    """gather appends the chained pair last, so the chunk failure would rank
    second on its own — and it is the one an operator has to act on."""
    rag = await _make_rag(tmp_path)
    try:
        rec: list = []
        chunks, cache, other = _bind_cache_pair_spies(
            rag, rec, chunks_flush_error=RuntimeError("chunk store is down")
        )
        chunks.namespace = "text_chunks"
        other._flush_error = RuntimeError("an unrelated vdb is down")
        monkeypatch.setattr(rag, "_index_storages", lambda: [chunks, cache, other])

        with pytest.raises(IndexFlushError) as excinfo:
            await rag._insert_done()

        assert excinfo.value.namespace == "text_chunks", excinfo.value
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_an_ordered_pair_commit_retires_the_recorded_failure(
    tmp_path, monkeypatch
):
    """Only both flushes landing in order proves the namespaces agree again."""
    rag = await _make_rag(tmp_path)
    try:
        rec: list = []
        chunks, cache, other = _bind_cache_pair_spies(
            rag, rec, chunks_flush_error=RuntimeError("transient outage")
        )
        monkeypatch.setattr(rag, "_index_storages", lambda: [chunks, cache, other])

        with pytest.raises(IndexFlushError):
            await rag._insert_done()
        assert rag._chunk_reference_commit_failed is True

        chunks._flush_error = None
        await rag._insert_done()

        assert rag._chunk_reference_commit_failed is False
        assert cache.index_done_calls == 1
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_a_concurrent_writer_cannot_land_between_the_two_flushes(
    tmp_path, monkeypatch
):
    """Chaining orders the commits; the fence keeps writers out of the gap.

    On a backend that publishes a snapshot taken at commit time, a document
    attaching and writing between the two flushes gets its cache row into the
    cache snapshot while the chunk snapshot predates its reference. The writer
    here stands for another in-flight document: it must be observed to finish
    only after BOTH flushes, never between them.
    """
    from lightrag.utils import get_extract_cache_fence

    rag = await _make_rag(tmp_path)
    try:
        rec: list = []
        chunks, cache, other = _bind_cache_pair_spies(rag, rec)
        monkeypatch.setattr(rag, "_index_storages", lambda: [chunks, cache, other])

        in_the_gap = asyncio.Event()

        original_chunk_flush = chunks.index_done_callback

        async def _flush_then_yield():
            await original_chunk_flush()
            # The gap: hand the loop to the writer before the cache flush.
            in_the_gap.set()
            await asyncio.sleep(0)
            await asyncio.sleep(0)

        chunks.index_done_callback = _flush_then_yield

        async def _concurrent_writer():
            # Bounded: without the fix the gap never opens, and an unbounded
            # wait would HANG the run instead of failing it.
            try:
                await asyncio.wait_for(in_the_gap.wait(), timeout=2)
            except asyncio.TimeoutError:
                pass
            async with get_extract_cache_fence(rag.text_chunks):
                rec.append(("writer", "attach+write"))

        writer = asyncio.create_task(_concurrent_writer())
        await rag._insert_done()
        await writer

        assert ("llm_cache", "flush") in rec, rec
        assert rec.index(("writer", "attach+write")) > rec.index(
            ("llm_cache", "flush")
        ), (
            "a writer attached and wrote inside the commit pair: its cache row "
            "is in the cache snapshot, its reference is not in the chunk one"
        )
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_a_permanent_chunk_failure_quarantines_the_buffered_cache_rows(
    tmp_path, monkeypatch
):
    """Deferring is not enough when the reference can never land.

    A per-item backend raises only for PERMANENT failures, and removes the
    operation from its buffer first. The cache rows naming that reference can
    therefore never become reachable, so the next successful pair commit would
    publish orphans. They are dropped instead.
    """
    rag = await _make_rag(tmp_path)
    try:
        rec: list = []
        chunks, cache, other = _bind_cache_pair_spies(
            rag, rec, chunks_flush_error=RuntimeError("permanent bulk failure")
        )
        monkeypatch.setattr(rag, "_index_storages", lambda: [chunks, cache, other])

        with pytest.raises(IndexFlushError):
            await rag._insert_done()

        assert cache.drop_upsert_calls == 1, (
            "the orphaned cache rows stayed buffered for the next pair commit "
            "to publish"
        )
        assert cache.drop_calls == 0, (
            "the quarantine took the buffered cache DELETES with the upserts"
        )
        assert cache.index_done_calls == 0
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_the_epilogue_records_its_own_failed_chunk_commit(tmp_path, monkeypatch):
    """The epilogue's standalone flush is a chunk commit like any other.

    Without recording it, the aborting-batch cleanup that follows reads an
    unset flag, retries the drained buffer, and publishes what the epilogue
    had just withheld.
    """
    rag = await _make_rag(tmp_path)
    try:
        rec: list = []
        chunks, cache, other = _bind_cache_pair_spies(
            rag, rec, chunks_flush_error=RuntimeError("chunk store is down")
        )
        monkeypatch.setattr(rag, "_index_storages", lambda: [chunks, cache, other])

        committed = await rag._persist_chunk_cache_references_best_effort(
            stage_label="extract failure", doc_id="doc-epilogue"
        )

        assert committed is False
        assert rag._chunk_reference_commit_failed is True
        assert cache.drop_upsert_calls == 1
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_a_declined_epilogue_commit_is_recorded_too(tmp_path, monkeypatch):
    """A DECLINED commit discarded the mutation; the references are not on disk."""
    rag = await _make_rag(tmp_path)
    try:
        rec: list = []
        chunks, cache, other = _bind_cache_pair_spies(rag, rec)
        chunks._flush_result = False
        monkeypatch.setattr(rag, "_index_storages", lambda: [chunks, cache, other])

        committed = await rag._persist_chunk_cache_references_best_effort(
            stage_label="extract failure", doc_id="doc-declined"
        )

        assert committed is False
        assert rag._chunk_reference_commit_failed is True
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_clearing_the_cache_holds_the_fence_across_the_drop(tmp_path):
    """``drop`` commits after releasing its own namespace lock, so a writer's
    attach+write fits in the gap and the drop's commit publishes the row."""
    from lightrag.utils import get_extract_cache_fence

    rag = await _make_rag(tmp_path)
    try:
        order: list = []
        in_the_gap = asyncio.Event()

        class _DroppableCache:
            namespace = "llm_response_cache"

            async def drop(self):
                # Stands for drop's lock-released window before its commit.
                in_the_gap.set()
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                order.append("drop")
                return {"status": "success"}

            async def index_done_callback(self):
                order.append("drop_commit")

            async def finalize(self):
                return None

        rag.llm_response_cache = _DroppableCache()

        async def _concurrent_writer():
            # Bounded: without the fix the gap never opens, and an unbounded
            # wait would HANG the run instead of failing it.
            try:
                await asyncio.wait_for(in_the_gap.wait(), timeout=2)
            except asyncio.TimeoutError:
                pass
            async with get_extract_cache_fence(rag.text_chunks):
                order.append("writer")

        writer = asyncio.create_task(_concurrent_writer())
        await rag.aclear_cache()
        await writer

        assert "drop_commit" in order, order
        assert order.index("writer") > order.index("drop_commit"), order
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_the_quarantine_keeps_buffered_cache_deletions(tmp_path, monkeypatch):
    """A cache tombstone is a promise an already-returned deletion made.

    ``adelete_by_doc_id(delete_llm_cache=True)`` buffers the deletes and
    flushes with a plain ``_insert_done`` precisely so they are not discarded;
    that path reports success after merely LOGGING a flush error. A quarantine
    that took the deletes with the upserts would leave the cache rows holding
    the document prompt on disk, with the chunk rows that name them already
    gone and no retry anywhere.
    """
    rag = await _make_rag(tmp_path)
    try:
        rec: list = []
        chunks, cache, other = _bind_cache_pair_spies(
            rag, rec, chunks_flush_error=RuntimeError("index refresh failed")
        )
        monkeypatch.setattr(rag, "_index_storages", lambda: [chunks, cache, other])

        with pytest.raises(IndexFlushError):
            await rag._insert_done()

        assert ("llm_cache", "drop") not in rec, (
            "the buffered cache deletions were discarded along with the upserts"
        )
        assert ("llm_cache", "drop_upserts") in rec
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_abort_cleanup_keeps_cache_tombstones_when_references_fail(
    tmp_path, monkeypatch
):
    """The second unconditional drop site, and the one with no recovery path.

    A completed ``adelete_by_doc_id(delete_llm_cache=True)`` can leave its
    tombstone buffered when the final ``_insert_done`` hit a ``text_chunks``
    flush error — that path reports success after merely logging it. If an
    aborting batch then discards the tombstone, the document, its status and
    its chunks are already gone, so the cache row holding the prompt has
    nothing left that can ever reach it.
    """
    rag = await _make_rag(tmp_path)
    try:
        rec: list = []
        chunks, cache, other = _bind_cache_pair_spies(
            rag, rec, chunks_flush_error=RuntimeError("chunk store is down")
        )
        monkeypatch.setattr(rag, "_index_storages", lambda: [chunks, cache, other])

        await rag._discard_pending_index_ops()

        assert cache.index_done_calls == 0, "the cache was flushed behind failed refs"
        assert cache.drop_calls == 0, (
            "the buffered cache DELETES were discarded on an aborting batch, "
            "and a completed deletion has no way to reissue them"
        )
        assert cache.drop_upsert_calls == 1
        # Every other namespace still gets the full drop.
        assert other.drop_calls == 1
    finally:
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_abort_cleanup_drops_every_upsert_when_references_commit(
    tmp_path, monkeypatch
):
    """The cache-type narrowing applies only to the failed-reference case.

    The upserts-only narrowing applies to both: a buffered cache delete is
    never this cleanup's to discard.
    """
    rag = await _make_rag(tmp_path)
    try:
        rec: list = []
        chunks, cache, other = _bind_cache_pair_spies(rag, rec)
        monkeypatch.setattr(rag, "_index_storages", lambda: [chunks, cache, other])

        await rag._discard_pending_index_ops()

        assert cache.index_done_calls == 1
        # Upserts only, untyped: the batch is abandoned so every buffered
        # upsert goes, but the deletes never do.
        assert cache.drop_upsert_calls == 1
        assert cache.drop_upsert_cache_types == [None]
        assert cache.drop_calls == 0
    finally:
        await rag.finalize_storages()


# ---------------------------------------------------------------------------
# Reference-before-row: a successful chunk commit is not always proof
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_discard_skips_the_cache_flush_when_chunk_ops_stay_buffered(
    tmp_path, monkeypatch
):
    """A per-item backend retains retryable failures and returns normally.

    Everywhere else that residue is accepted because the retained operations
    replay on the next flush. This cleanup is the exception: it DROPS the
    chunk buffer a few lines later, so nothing ever replays them. Flushing the
    cache on that successful-looking return publishes extract rows whose only
    reference is in the buffer about to be discarded.
    """
    rag = await _make_rag(tmp_path)
    try:
        rec: list = []
        chunks = _SpyStorage("text_chunks", recorder=rec, pending_index_ops=True)
        cache = _SpyStorage("llm_cache", recorder=rec, dropped_upsert_count=3)
        rag.text_chunks = chunks
        rag.llm_response_cache = cache
        monkeypatch.setattr(rag, "_index_storages", lambda: [chunks, cache])

        await rag._discard_pending_index_ops()

        # The chunk commit was attempted and reported success, but operations
        # stayed buffered -- so the cache is NOT published. Asserted first, so
        # a regression fails on the published row rather than on a missing
        # call to the mechanism that prevents it.
        assert cache.index_done_calls == 0
        # Only the upserts go, so an already-promised tombstone survives.
        assert cache.drop_upsert_calls == 1
        assert cache.drop_calls == 0
        assert chunks.index_done_calls == 1
        assert chunks.has_pending_calls == 1
    finally:
        rag.text_chunks = None
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_discard_still_flushes_the_cache_when_nothing_stays_buffered(
    tmp_path, monkeypatch
):
    """Stability: the gate must not suppress the ordinary abort path.

    Cached LLM results are expensive, so a healthy chunk commit still buys the
    cache its final flush before the buffers are dropped.
    """
    rag = await _make_rag(tmp_path)
    try:
        rec: list = []
        chunks = _SpyStorage("text_chunks", recorder=rec, pending_index_ops=False)
        cache = _SpyStorage("llm_cache", recorder=rec)
        rag.text_chunks = chunks
        rag.llm_response_cache = cache
        monkeypatch.setattr(rag, "_index_storages", lambda: [chunks, cache])

        await rag._discard_pending_index_ops()

        assert cache.index_done_calls == 1
        assert rec.index(("llm_cache", "flush")) < rec.index(
            ("llm_cache", "drop_upserts")
        )
        assert cache.drop_upsert_calls == 1
        assert cache.drop_calls == 0
    finally:
        rag.text_chunks = None
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_quarantine_names_only_the_reference_carrying_cache_types(tmp_path):
    """The shared buffer also holds the query answers nobody's chunk names.

    An untyped discard on a chunk-reference failure would take them too — on
    the query path, the answer the very query that triggered it just paid an
    LLM call for.
    """
    rag = await _make_rag(tmp_path)
    try:
        cache = _SpyStorage("llm_cache", dropped_upsert_count=2)
        rag.llm_response_cache = cache

        await rag._record_chunk_reference_commit_failure("unit test")

        assert rag._chunk_reference_commit_failed is True
        assert cache.drop_upsert_cache_types == [{"extract"}]
        assert cache.drop_calls == 0
    finally:
        await rag.finalize_storages()


# ---------------------------------------------------------------------------
# finalize_storages — the one commit site with no next run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_finalize_commits_the_ordered_pair_before_the_loop(tmp_path):
    """``JsonKVStorage.finalize`` flushes ``*_cache`` and nothing else.

    So without an ordered commit first, shutdown publishes the whole cache
    namespace while a dirty ``text_chunks`` is never written at all — an
    orphan on the default backend with no failure involved.
    """
    rag = await _make_rag(tmp_path)
    rec: list = []
    chunks = _SpyStorage("text_chunks", recorder=rec)
    cache = _SpyStorage("llm_cache", recorder=rec)
    rag.text_chunks = chunks
    rag.llm_response_cache = cache

    await rag.finalize_storages()

    flushes = [entry for entry in rec if entry[1] == "flush"]
    assert flushes == [("text_chunks", "flush"), ("llm_cache", "flush")]
    # No quarantine: the references landed.
    assert cache.drop_upsert_calls == 0


@pytest.mark.asyncio
async def test_finalize_quarantines_extract_rows_when_the_chunk_commit_failed(
    tmp_path,
):
    """The process is exiting, so a published orphan is permanent.

    The cache's ``finalize`` flushes its buffer, which publishes the whole
    namespace; the extract rows it holds are discarded first when the
    references did not reach disk.
    """
    rag = await _make_rag(tmp_path)
    rec: list = []
    chunks = _SpyStorage(
        "text_chunks", recorder=rec, flush_error=RuntimeError("chunks down")
    )
    cache = _SpyStorage("llm_cache", recorder=rec, dropped_upsert_count=4)
    rag.text_chunks = chunks
    rag.llm_response_cache = cache

    await rag.finalize_storages()

    # Recorded once by the failed ordered pair, once by the finalize gate;
    # every one of them names the reference-carrying types only.
    assert cache.drop_upsert_calls >= 1
    assert all(names == {"extract"} for names in cache.drop_upsert_cache_types)
    # Quarantined BEFORE the cache's own finalize, which is what publishes.
    last_drop = len(rec) - 1 - rec[::-1].index(("llm_cache", "drop_upserts"))
    assert ("llm_cache", "flush") not in rec[last_drop:]


@pytest.mark.asyncio
async def test_finalize_quarantines_when_chunk_ops_are_merely_retained(tmp_path):
    """A successful chunk commit that retained operations is not proof here.

    Quarantined twice: once by the strict pair before the loop, once by the
    gate before the cache's finalize. Both are needed — the pair's is the one
    that runs before anything could be published.
    """
    rag = await _make_rag(tmp_path)
    chunks = _SpyStorage("text_chunks", pending_index_ops=True)
    cache = _SpyStorage("llm_cache", dropped_upsert_count=1)
    rag.text_chunks = chunks
    rag.llm_response_cache = cache

    await rag.finalize_storages()

    assert cache.drop_upsert_calls >= 1


@pytest.mark.asyncio
async def test_finalize_reports_the_residue_a_snapshot_backend_cannot_close(
    tmp_path, caplog
):
    """On a snapshot backend the discard is a base-class no-op.

    Nothing can be separated out of the shared dict, the process is exiting,
    and no later run can publish the references — the one site with no heal
    path, so it says so instead of logging a discard that did not happen.
    """
    rag = await _make_rag(tmp_path)
    chunks = _SpyStorage("text_chunks", flush_error=RuntimeError("chunks down"))
    # None, not 0: the backend cannot separate the upserts out at all.
    cache = _SpyStorage("llm_cache", dropped_upsert_count=None)
    rag.text_chunks = chunks
    rag.llm_response_cache = cache

    with caplog.at_level("ERROR", logger="lightrag"):
        await rag.finalize_storages()

    messages = [r.message for r in caplog.records]
    assert any("nothing will heal them" in m for m in messages)
    # The finalize itself still ran: it releases the backend's client.
    assert cache.drop_upsert_calls >= 1


@pytest.mark.asyncio
async def test_finalize_does_not_cry_residue_over_an_already_emptied_buffer(
    tmp_path, caplog
):
    """An empty buffer is not an unclosable residue.

    The ordered pair before the loop quarantines on its own chunk failure, so
    by the time the finalize gate runs the per-item backend has nothing left
    to discard and answers 0. Reading that as the base-class no-op would raise
    a permanent-orphan alarm over rows that were already quarantined.
    """
    rag = await _make_rag(tmp_path)
    chunks = _SpyStorage("text_chunks", flush_error=RuntimeError("chunks down"))
    cache = _SpyStorage("llm_cache", dropped_upsert_count=0)
    rag.text_chunks = chunks
    rag.llm_response_cache = cache

    with caplog.at_level("ERROR", logger="lightrag"):
        await rag.finalize_storages()

    messages = [r.message for r in caplog.records]
    assert not any("nothing will heal them" in m for m in messages)
    assert any("none is left unreachable" in m for m in messages)


class _BufferedCacheStorage:
    """A cache storage with a real per-item buffer, like OpenSearchKVStorage.

    Enough of one to observe WHICH rows a discard takes: ``drop_pending_upserts``
    filters on the ``{mode}:{cache_type}:{hash}`` key and keeps the deletes,
    and the flush publishes whatever is still buffered.
    """

    def __init__(
        self,
        rows: dict[str, dict],
        deletes: set[str] | None = None,
        *,
        retain_deletes: bool = False,
    ):
        self.namespace = "llm_response_cache"
        self.pending_upserts = dict(rows)
        self.pending_deletes = set(deletes or set())
        self.published: dict[str, dict] = {}
        # ``retain_deletes`` models a retryable per-item delete failure: the
        # flush keeps it buffered and returns NORMALLY, which is what makes a
        # returning flush no proof that the tombstone landed.
        self._retain_deletes = retain_deletes

    async def index_done_callback(self):
        self.published.update(self.pending_upserts)
        self.pending_upserts.clear()
        if not self._retain_deletes:
            self.pending_deletes.clear()

    async def drop_pending_index_ops(self):
        self.pending_upserts.clear()
        self.pending_deletes.clear()

    async def drop_pending_upserts(self, *, cache_types=None) -> int | None:
        if cache_types is None:
            dropped = len(self.pending_upserts)
            self.pending_upserts.clear()
            return dropped
        doomed = [
            key
            for key in self.pending_upserts
            if len(key.split(":", 2)) == 3 and key.split(":", 2)[1] in cache_types
        ]
        for key in doomed:
            self.pending_upserts.pop(key)
        return len(doomed)

    async def has_pending_index_ops(self) -> bool:
        return bool(self.pending_upserts)

    async def finalize(self):
        return None


@pytest.mark.asyncio
async def test_the_aborting_cleanup_keeps_the_answer_rows_it_cannot_orphan(
    tmp_path, monkeypatch
):
    """The shared buffer also holds query answers, which name no chunk.

    When the references did not land, this cleanup withholds the cache flush
    and discards the buffer instead of publishing it. Discarding it untyped
    takes the answer rows too — full LLM responses that nothing can orphan and
    that the next cache commit would have published for free.
    """
    rag = await _make_rag(tmp_path)
    try:
        chunks = _SpyStorage("text_chunks", pending_index_ops=True)
        cache = _BufferedCacheStorage(
            {
                "default:extract:aaa": {"return": "e1"},
                "default:query:bbb": {"return": "an expensive answer"},
                "default:keywords:ccc": {"return": "kw"},
            },
            deletes={"default:extract:gone"},
        )
        rag.text_chunks = chunks
        rag.llm_response_cache = cache
        monkeypatch.setattr(rag, "_index_storages", lambda: [chunks, cache])

        await rag._discard_pending_index_ops()

        # The unreachable rows are gone, the rest survive for the next commit.
        assert set(cache.pending_upserts) == {
            "default:query:bbb",
            "default:keywords:ccc",
        }
        # Nothing was published behind a reference that is not on disk.
        assert cache.published == {}
        # The tombstone an already-returned deletion promised is still there.
        assert cache.pending_deletes == {"default:extract:gone"}
    finally:
        rag.text_chunks = None
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_the_shutdown_pair_withholds_the_cache_over_retained_chunk_ops(
    tmp_path,
):
    """A returning chunk commit that kept operations buffered is not proof here.

    A per-item backend retains its retryable failures and returns normally.
    Everywhere with a next commit that is an accepted residue: the retained
    operations replay and the pair converges. Shutdown has no next commit, and
    the gate before the cache's own finalize cannot undo this — a row this
    pair publishes is already on disk, where no quarantine reaches it.
    """
    rag = await _make_rag(tmp_path)
    rec: list = []
    chunks = _SpyStorage("text_chunks", recorder=rec, pending_index_ops=True)
    cache = _SpyStorage("llm_cache", recorder=rec, dropped_upsert_count=3)
    rag.text_chunks = chunks
    rag.llm_response_cache = cache

    await rag.finalize_storages()

    # Asserted first: a regression fails on the published row, not on a
    # missing call to the mechanism that prevents it.
    assert ("llm_cache", "flush") not in rec
    assert cache.index_done_calls == 0
    # The still-buffered extract rows are quarantined instead.
    assert cache.drop_upsert_calls >= 1
    assert all(names == {"extract"} for names in cache.drop_upsert_cache_types)
    assert ("text_chunks", "flush") in rec


@pytest.mark.asyncio
async def test_the_runtime_pair_still_accepts_retained_chunk_ops(tmp_path):
    """Stability: the strict gate must not leak into the runtime path.

    There the retained operations replay on the next flush, and withholding
    the cache would make this ordering stricter than the pipeline's own
    PROCESSED write, which acknowledges the identical buffered flush.
    """
    rag = await _make_rag(tmp_path)
    try:
        rec: list = []
        chunks = _SpyStorage("text_chunks", recorder=rec, pending_index_ops=True)
        cache = _SpyStorage("llm_cache", recorder=rec)
        rag.text_chunks = chunks
        rag.llm_response_cache = cache

        await rag._flush_storages([chunks, cache])

        assert cache.index_done_calls == 1
        assert cache.drop_upsert_calls == 0
        assert rag._chunk_reference_commit_failed is False
    finally:
        rag.text_chunks = None
        await rag.finalize_storages()


@pytest.mark.asyncio
async def test_the_aborting_cleanup_keeps_a_tombstone_a_healthy_flush_retained(
    tmp_path, monkeypatch
):
    """A returning cache flush does not prove the tombstone landed.

    ``adelete_by_doc_id(delete_llm_cache=True)`` buffers the delete, verifies
    through a read that is buffer-aware (so a merely-buffered tombstone reads
    as gone), and flushes with a plain ``_insert_done`` that RETURNS when the
    backend retains a retryable delete. The deletion is reported successful
    with the tombstone still buffered, and the document, its status and its
    chunks are already gone. A later abort must not discard it: nothing can
    reissue it, and the row holds the document prompt.
    """
    rag = await _make_rag(tmp_path)
    try:
        chunks = _SpyStorage("text_chunks")
        cache = _BufferedCacheStorage(
            {"default:extract:aaa": {"return": "e1"}},
            deletes={"default:extract:promised"},
            retain_deletes=True,
        )
        rag.text_chunks = chunks
        rag.llm_response_cache = cache
        monkeypatch.setattr(rag, "_index_storages", lambda: [chunks, cache])

        await rag._discard_pending_index_ops()

        # The promise survives the abort.
        assert cache.pending_deletes == {"default:extract:promised"}
        # The references landed, so the flush ran and the upserts still go.
        assert set(cache.published) == {"default:extract:aaa"}
        assert cache.pending_upserts == {}
    finally:
        rag.text_chunks = None
        await rag.finalize_storages()
