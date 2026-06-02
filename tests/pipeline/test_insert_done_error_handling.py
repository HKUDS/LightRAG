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
        drop_error: BaseException | None = None,
        recorder: list | None = None,
    ):
        self.label = label
        self.namespace = namespace
        if final_namespace is not _UNSET:
            self.final_namespace = final_namespace
        self._flush_error = flush_error
        self._drop_error = drop_error
        self._recorder = recorder if recorder is not None else []
        self.index_done_calls = 0
        self.drop_calls = 0

    async def index_done_callback(self):
        self.index_done_calls += 1
        self._recorder.append((self.label, "flush"))
        if self._flush_error is not None:
            raise self._flush_error

    async def drop_pending_index_ops(self):
        self.drop_calls += 1
        self._recorder.append((self.label, "drop"))
        if self._drop_error is not None:
            raise self._drop_error

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
        # cache + other are dropped.
        assert cache.drop_calls == 1
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
        assert cache.drop_calls == 1
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
        assert rec.index(("llm_cache", "flush")) < rec.index(("llm_cache", "drop"))
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
        assert cache.drop_calls == 1
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
