"""Unit tests for ``chunking_by_semantic_vector`` (process_options=V)."""

import asyncio
import contextlib
import logging
import re
import threading

import numpy as np
import pytest

pytest.importorskip("langchain_experimental")

from lightrag.chunker import chunking_by_semantic_vector  # noqa: E402
from lightrag.constants import (  # noqa: E402
    DEFAULT_EMBEDDING_BATCH_NUM,
    DEFAULT_SENTENCE_SPLIT_REGEX,
)
from lightrag.utils import EmbeddingFunc, Tokenizer, TokenizerInterface  # noqa: E402


class _CharTokenizer(TokenizerInterface):
    """1 char ≈ 1 token."""

    def encode(self, content: str):
        return [ord(ch) for ch in content]

    def decode(self, tokens):
        return "".join(chr(t) for t in tokens)


def _tok() -> Tokenizer:
    return Tokenizer("char-tokenizer", _CharTokenizer())


def _make_deterministic_embedding(dim: int = 8) -> EmbeddingFunc:
    """A toy async embedding func that hashes each input text into a
    stable unit vector — enough to drive SemanticChunker without needing
    a real model."""

    async def _embed(texts, **kwargs):
        rng = np.random.default_rng(seed=0)
        # Use a simple hash → seeded rng to get reproducible vectors per text.
        rows = []
        for text in texts:
            seed = abs(hash(text)) % (2**32)
            rng = np.random.default_rng(seed=seed)
            vec = rng.normal(size=dim).astype(np.float32)
            vec /= np.linalg.norm(vec) or 1.0
            rows.append(vec)
        return np.vstack(rows)

    return EmbeddingFunc(embedding_dim=dim, max_token_size=4096, func=_embed)


@pytest.mark.offline
def test_v_chunker_runs_with_stub_embedding():
    """Async chunker should split a multi-sentence body into ≥1 chunk
    when given a working embedding func."""
    body = (
        "Quantum mechanics describes nature at small scales. "
        "It contradicts classical intuition. "
        "Bread is baked from flour. "
        "Sourdough requires a long fermentation. "
    )

    async def _run():
        chunks = await chunking_by_semantic_vector(
            _tok(),
            body,
            chunk_token_size=200,
            embedding_func=_make_deterministic_embedding(),
        )
        return chunks

    chunks = asyncio.run(_run())

    assert len(chunks) >= 1
    # Each chunk dict has the canonical schema.
    assert all({"tokens", "content", "chunk_order_index"} <= set(c) for c in chunks)
    # chunk_order_index is contiguous starting at 0.
    assert [c["chunk_order_index"] for c in chunks] == list(range(len(chunks)))
    # No empty content rows.
    assert all(c["content"].strip() for c in chunks)
    for chunk in chunks:
        span = chunk["_source_span"]
        assert body[span["start"] : span["end"]] == chunk["content"]


class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.mark.offline
def test_v_chunker_falls_back_to_recursive_when_no_embedding():
    """When ``embedding_func`` is None, V must log a warning and route
    to chunking_by_recursive_character (R) — V's only differentiator
    is embeddings, so without them R is the closest neighbour."""
    body = "Para A.\n\nPara B for fallback test.\n\nPara C."

    lightrag_logger = logging.getLogger("lightrag")
    handler = _ListHandler()
    handler.setLevel(logging.WARNING)
    lightrag_logger.addHandler(handler)
    try:

        async def _run():
            return await chunking_by_semantic_vector(
                _tok(),
                body,
                chunk_token_size=20,
                embedding_func=None,
            )

        chunks = asyncio.run(_run())
    finally:
        lightrag_logger.removeHandler(handler)

    assert len(chunks) >= 1
    assert any(
        "embedding_func is None" in rec.getMessage()
        for rec in handler.records
        if rec.levelno == logging.WARNING
    )


@pytest.mark.offline
def test_v_chunker_empty_input_returns_empty_list():
    async def _run():
        return await chunking_by_semantic_vector(_tok(), "")

    assert asyncio.run(_run()) == []


@pytest.mark.offline
def test_v_chunker_spans_repeated_sentences_and_number_of_chunks():
    body = (
        "Repeat sentence.\n"
        "Repeat sentence. "
        "A distant topic appears. "
        "Another distant topic appears."
    )

    async def _run():
        return await chunking_by_semantic_vector(
            _tok(),
            body,
            chunk_token_size=400,
            embedding_func=_make_deterministic_embedding(),
            number_of_chunks=2,
        )

    chunks = asyncio.run(_run())

    assert chunks
    for chunk in chunks:
        span = chunk["_source_span"]
        assert body[span["start"] : span["end"]] == chunk["content"]


@pytest.mark.offline
def test_v_oversized_group_resplit_preserves_child_source_spans():
    body = " ".join(f"word{i:02d}" for i in range(30)) + "."

    async def _run():
        return await chunking_by_semantic_vector(
            _tok(),
            body,
            chunk_token_size=35,
            embedding_func=_make_deterministic_embedding(),
        )

    chunks = asyncio.run(_run())

    assert len(chunks) > 1
    for chunk in chunks:
        span = chunk["_source_span"]
        assert body[span["start"] : span["end"]] == chunk["content"]
        assert chunk["tokens"] <= 35


@pytest.mark.offline
def test_semantic_groups_mirror_matches_upstream_split_text():
    """Drift guard: ``_semantic_groups_with_spans`` re-implements the private body
    of ``SemanticChunker.split_text`` to keep verbatim source spans. If an upstream
    langchain-experimental release changes that grouping (or a private member it
    relies on), this catches it — the mirror must yield the same group count and the
    same content modulo whitespace (it keeps the verbatim slice; upstream reflows
    sentences with single spaces)."""
    from langchain_core.embeddings import Embeddings
    from langchain_experimental.text_splitter import SemanticChunker

    from lightrag.chunker.semantic_vector import _semantic_groups_with_spans
    from lightrag.constants import DEFAULT_SENTENCE_SPLIT_REGEX

    class _SyncEmbeddings(Embeddings):
        """Deterministic per-text vectors, mirroring _make_deterministic_embedding
        but synchronous so SemanticChunker can be driven directly."""

        def embed_documents(self, texts):
            rows = []
            for text in texts:
                seed = abs(hash(text)) % (2**32)
                rng = np.random.default_rng(seed=seed)
                vec = rng.normal(size=8).astype(np.float64)
                vec /= np.linalg.norm(vec) or 1.0
                rows.append(vec.tolist())
            return rows

        def embed_query(self, text):
            return self.embed_documents([text])[0]

    text = (
        "Quantum mechanics describes nature at small scales. "
        "It contradicts classical intuition. "
        "Bread is baked from flour. "
        "Sourdough requires a long fermentation."
    )
    splitter = SemanticChunker(
        _SyncEmbeddings(),
        buffer_size=1,
        breakpoint_threshold_type="percentile",
        sentence_split_regex=DEFAULT_SENTENCE_SPLIT_REGEX,
    )

    # Same process → identical deterministic embeddings on both calls → identical
    # distances/breakpoints, so any divergence is a real grouping drift.
    upstream = splitter.split_text(text)
    mirror = _semantic_groups_with_spans(splitter, text)

    assert len(mirror) == len(upstream)
    for (grp_text, start, end), up in zip(mirror, upstream):
        assert grp_text == text[start:end]  # mirror keeps the verbatim source slice
        assert "".join(grp_text.split()) == "".join(up.split())


# ---------------------------------------------------------------------------
# Sentence-embedding request bounding
# ---------------------------------------------------------------------------
#
# Upstream SemanticChunker embeds the WHOLE document in one
# ``embed_documents`` call, and no provider binding slices that list, so the
# request grew with the document instead of with the chunk size. The adapter
# now batches by item count and truncates each window; these tests pin both,
# plus the failure/cancellation behavior the batching introduces.
#
# A note that shapes every test below: a single-sentence input never reaches
# the adapter at all. ``_semantic_groups_with_spans`` returns the lone group
# before ``_calculate_sentence_distances`` is called, so a truncation test
# built on one long sentence would observe zero embedding calls. Everything
# here uses >= 3 sentences.


def _vec_for(text: str, dim: int = 8) -> np.ndarray:
    """Same deterministic per-text vector as ``_make_deterministic_embedding``.

    Determinism is what makes the batched/unbatched equivalence test
    meaningful: identical inputs must yield identical grouping regardless of
    how the calls were split up.
    """
    seed = abs(hash(text)) % (2**32)
    rng = np.random.default_rng(seed=seed)
    vec = rng.normal(size=dim).astype(np.float32)
    vec /= np.linalg.norm(vec) or 1.0
    return vec


class _RecordingEmbedding:
    """Deterministic embedding func that records how it was called."""

    def __init__(self, *, dim: int = 8, max_token_size: int | None = 4096) -> None:
        self.dim = dim
        self.max_token_size = max_token_size
        self.calls: list[list[str]] = []
        self.inflight = 0
        self.peak_inflight = 0

    async def __call__(self, texts, **kwargs):
        self.calls.append(list(texts))
        self.inflight += 1
        self.peak_inflight = max(self.peak_inflight, self.inflight)
        try:
            # Yield so concurrent batches actually overlap; without it the
            # in-flight peak would always read 1.
            await asyncio.sleep(0)
            rows = [_vec_for(text, self.dim) for text in texts]
        finally:
            self.inflight -= 1
        return np.vstack(rows)

    def as_func(self) -> EmbeddingFunc:
        return EmbeddingFunc(
            embedding_dim=self.dim,
            max_token_size=self.max_token_size,
            func=self,
        )

    @property
    def sizes(self) -> list[int]:
        return [len(call) for call in self.calls]

    @property
    def all_texts(self) -> list[str]:
        return [text for call in self.calls for text in call]


def _multi_sentence_body(count: int) -> str:
    return " ".join(
        f"Sentence number {i:02d} carries some filler words." for i in range(count)
    )


def _body_with_long_sentence(pad: int = 200) -> str:
    """Three sentences, the middle one far over any small token budget."""
    return "Alpha beta gamma. " + "X" * pad + ". Delta epsilon zeta."


def _sentence_count(body: str) -> int:
    return len(re.split(DEFAULT_SENTENCE_SPLIT_REGEX, body))


@contextlib.contextmanager
def _lightrag_warnings():
    """Collect WARNING records from the ``lightrag`` logger.

    A handler attached directly to that logger, not ``caplog``: the logger
    sets ``propagate=False``, so records never reach the root handler caplog
    installs.
    """
    lightrag_logger = logging.getLogger("lightrag")
    handler = _ListHandler()
    handler.setLevel(logging.WARNING)
    lightrag_logger.addHandler(handler)
    try:
        yield handler.records
    finally:
        lightrag_logger.removeHandler(handler)


def _warning_messages(records) -> list[str]:
    return [rec.getMessage() for rec in records if rec.levelno == logging.WARNING]


@pytest.mark.offline
def test_v_batches_sentence_embedding_by_default():
    """Without any new kwarg, V must still split the request.

    The default has to bound the request too — a direct SDK caller gets no
    say in ``EMBEDDING_BATCH_NUM``. Before the fix this was one call carrying
    every sentence window in the document.
    """
    rec = _RecordingEmbedding()
    body = _multi_sentence_body(25)

    async def _run():
        return await chunking_by_semantic_vector(
            _tok(), body, 4000, embedding_func=rec.as_func()
        )

    chunks = asyncio.run(_run())

    assert chunks
    assert rec.calls, "the adapter was never reached"
    assert max(rec.sizes) <= DEFAULT_EMBEDDING_BATCH_NUM
    assert len(rec.calls) >= 2
    # Nothing dropped or duplicated by the split: one window per sentence.
    assert sum(rec.sizes) == _sentence_count(body) == 25


@pytest.mark.offline
def test_v_honors_explicit_embedding_batch_num():
    rec = _RecordingEmbedding()
    body = _multi_sentence_body(25)

    async def _run():
        return await chunking_by_semantic_vector(
            _tok(), body, 4000, embedding_func=rec.as_func(), embedding_batch_num=4
        )

    asyncio.run(_run())

    assert max(rec.sizes) <= 4
    assert len(rec.calls) == 7  # ceil(25 / 4)
    assert sum(rec.sizes) == 25


@pytest.mark.offline
def test_v_batching_does_not_change_the_chunking():
    """Batched and unbatched runs must agree exactly.

    Deterministic per-text vectors mean any divergence is the batching
    layer reordering or mismatching results, not embedding noise. This is
    the only guard that catches a future reordering of batch results.
    """
    body = _multi_sentence_body(25)

    async def _run(batch_num):
        rec = _RecordingEmbedding()
        chunks = await chunking_by_semantic_vector(
            _tok(),
            body,
            4000,
            embedding_func=rec.as_func(),
            embedding_batch_num=batch_num,
        )
        return chunks, rec

    batched, batched_rec = asyncio.run(_run(4))
    single, single_rec = asyncio.run(_run(0))

    assert len(batched_rec.calls) == 7
    assert len(single_rec.calls) == 1  # batching explicitly disabled
    assert [c["content"] for c in batched] == [c["content"] for c in single]
    assert [c["_source_span"] for c in batched] == [c["_source_span"] for c in single]


@pytest.mark.offline
def test_v_truncates_oversized_sentence_windows_and_warns_once():
    """An over-budget window is truncated, loudly but harmlessly.

    ``_CharTokenizer`` makes 1 char == 1 token, so the budget is directly
    observable in the texts handed to the provider. The point of the last
    assertion: truncation must not cost chunk content — the emitted chunks
    are original source spans, so the 200-character sentence survives whole.
    """
    body = _body_with_long_sentence(200)
    rec = _RecordingEmbedding(max_token_size=32)

    async def _run():
        return await chunking_by_semantic_vector(
            _tok(), body, 4000, embedding_func=rec.as_func()
        )

    with _lightrag_warnings() as records:
        chunks = asyncio.run(_run())

    assert rec.calls, "the adapter was never reached"
    assert all(len(text) <= 32 for text in rec.all_texts)

    truncation_warnings = [m for m in _warning_messages(records) if "truncated" in m]
    assert len(truncation_warnings) == 1
    message = truncation_warnings[0]
    assert "truncated 3/3 sentence windows" in message
    assert "budget=32" in message
    assert "source=embedding_func.max_token_size" in message
    assert "longest_original=" in message
    assert "NOT affected" in message

    # Content integrity: full text preserved, spans still verbatim.
    assert any("X" * 200 in chunk["content"] for chunk in chunks)
    for chunk in chunks:
        span = chunk["_source_span"]
        assert body[span["start"] : span["end"]] == chunk["content"]


@pytest.mark.offline
def test_v_does_not_warn_when_every_window_fits():
    """The aggregated warning is gated on there being something to report."""
    rec = _RecordingEmbedding(max_token_size=4096)
    body = _multi_sentence_body(8)

    async def _run():
        return await chunking_by_semantic_vector(
            _tok(), body, 4000, embedding_func=rec.as_func()
        )

    with _lightrag_warnings() as records:
        asyncio.run(_run())

    assert rec.calls
    assert [m for m in _warning_messages(records) if "truncated" in m] == []


@pytest.mark.offline
@pytest.mark.parametrize("declared", [None, 0])
def test_v_falls_back_to_chunk_token_size_when_no_budget_declared(declared):
    """``None`` and ``0`` both fall back to ``chunk_token_size``.

    Treating ``0`` as "unset" diverges from ``openai_embed``, where it means
    "disable truncation"; V needs a budget of its own, and only a direct
    ``EmbeddingFunc(max_token_size=0)`` caller can reach this branch.
    """
    body = _body_with_long_sentence(200)
    rec = _RecordingEmbedding(max_token_size=declared)

    async def _run():
        return await chunking_by_semantic_vector(
            _tok(), body, 40, embedding_func=rec.as_func()
        )

    with _lightrag_warnings() as records:
        chunks = asyncio.run(_run())

    assert chunks
    assert rec.calls, "the adapter was never reached"
    assert all(len(text) <= 40 for text in rec.all_texts)
    message = next(m for m in _warning_messages(records) if "truncated" in m)
    assert "source=chunk_token_size fallback" in message
    assert "budget=40" in message


@pytest.mark.offline
@pytest.mark.parametrize("bad", [0, -1])
def test_v_rejects_non_positive_embedding_max_async(bad):
    """A non-positive worker count must fail fast, not hang.

    Zero workers would leave every batch waiting forever; the short
    ``wait_for`` is what distinguishes "raised" from "hung".
    """
    rec = _RecordingEmbedding()

    async def _run():
        return await asyncio.wait_for(
            chunking_by_semantic_vector(
                _tok(),
                _multi_sentence_body(6),
                4000,
                embedding_func=rec.as_func(),
                embedding_max_async=bad,
            ),
            timeout=5,
        )

    with pytest.raises(ValueError, match="embedding_max_async"):
        asyncio.run(_run())
    assert rec.calls == []


@pytest.mark.offline
def test_v_stops_dispatching_after_a_batch_failure():
    """Once one batch's failure is observed, no new batch may start.

    The window this guards is narrow and was open in the first design: with
    ``asyncio.gather``, the failing batch's exception only cancels siblings
    once it reaches the awaiter, and in the meantime the other workers pull
    fresh batches from the cursor. So the assertion is not "few calls" — it
    is "zero calls after the failure instant", which is why the failing
    batch is gated open until several siblings have succeeded.
    """
    body = _multi_sentence_body(30)
    state: dict[str, object] = {"log": [], "at_failure": None}

    class _FailingSecondBatch:
        def __init__(self) -> None:
            self.release = asyncio.Event()

        async def __call__(self, texts, **kwargs):
            log = state["log"]
            index = len(log)
            log.append(index)
            if index == 1:
                await self.release.wait()
                state["at_failure"] = len(log)
                raise RuntimeError("boom")
            if len(log) >= 8:
                self.release.set()
            await asyncio.sleep(0)
            return np.vstack([_vec_for(text) for text in texts])

    async def _run():
        func = EmbeddingFunc(
            embedding_dim=8, max_token_size=4096, func=_FailingSecondBatch()
        )
        with pytest.raises(RuntimeError, match="boom"):
            await chunking_by_semantic_vector(
                _tok(),
                body,
                4000,
                embedding_func=func,
                embedding_batch_num=1,
                embedding_max_async=3,
            )
        calls_when_raised = len(state["log"])
        # Give any straggler worker a chance to dispatch one more batch.
        await asyncio.sleep(0.05)
        return calls_when_raised, len(state["log"])

    calls_when_raised, calls_later = asyncio.run(_run())

    assert state["at_failure"] is not None, "the designated batch never failed"
    assert calls_when_raised == state["at_failure"]
    assert calls_later == calls_when_raised
    assert calls_when_raised < 30, "the whole document was embedded despite failing"


@pytest.mark.offline
def test_v_cancellation_reaches_the_in_flight_provider_call():
    """Cancelling the chunker must tear down the bounced work.

    ``asyncio.to_thread`` returns on cancellation but the worker thread
    keeps running — blocked on the bridging future. Without cancelling that
    future, the pool would run the whole document to completion and discard
    the result.
    """
    body = _multi_sentence_body(12)
    seen = {"cancelled": 0}

    async def _run():
        started = asyncio.Event()

        async def _blocking(texts, **kwargs):
            started.set()
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                seen["cancelled"] += 1
                raise
            return np.vstack([_vec_for(text) for text in texts])

        func = EmbeddingFunc(embedding_dim=8, max_token_size=4096, func=_blocking)
        task = asyncio.create_task(
            chunking_by_semantic_vector(
                _tok(), body, 4000, embedding_func=func, embedding_batch_num=4
            )
        )
        await asyncio.wait_for(started.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # Let the cancellation reach the loop-side worker pool.
        await asyncio.sleep(0.1)
        return len(asyncio.all_tasks()) - 1  # minus the task running _run

    leftover = asyncio.run(_run())

    assert seen["cancelled"] >= 1, "the provider coroutine never saw the cancellation"
    assert leftover == 0, "adapter worker tasks outlived the cancellation"


@pytest.mark.offline
def test_v_degrades_instead_of_failing_when_no_code_point_fits():
    """The one case where the adapter's own budget cannot be met.

    A budget smaller than a single code point makes
    ``truncate_by_token_limit`` raise ``TokenBudgetError``. Failing the whole
    document over a boundary-quality nicety is the wrong trade, so the
    window degrades to its first code point and the aggregated warning says
    so.
    """

    class _TwoTokensPerCharImpl(TokenizerInterface):
        """Every character costs 2 tokens, so nothing fits a budget of 1."""

        def encode(self, content: str):
            tokens: list[int] = []
            for ch in content:
                tokens.extend((ord(ch), ord(ch)))
            return tokens

        def decode(self, tokens):
            return "".join(chr(t) for t in tokens[::2])

    body = _body_with_long_sentence(60)
    rec = _RecordingEmbedding(max_token_size=1)
    tokenizer = Tokenizer("two-tokens-per-char", _TwoTokensPerCharImpl())

    async def _run():
        return await chunking_by_semantic_vector(
            tokenizer, body, 4000, embedding_func=rec.as_func()
        )

    with _lightrag_warnings() as records:
        chunks = asyncio.run(_run())

    assert chunks, "the document must not fail over an unmeetable budget"
    assert rec.calls, "the adapter was never reached"
    assert all(len(text) == 1 for text in rec.all_texts)
    message = next(m for m in _warning_messages(records) if "truncated" in m)
    assert "did not fit even one code point" in message
    for chunk in chunks:
        span = chunk["_source_span"]
        assert body[span["start"] : span["end"]] == chunk["content"]
    assert any("X" * 60 in chunk["content"] for chunk in chunks)


@pytest.mark.offline
def test_v_concurrency_peak_is_bounded():
    rec = _RecordingEmbedding()
    body = _multi_sentence_body(30)

    async def _run():
        return await chunking_by_semantic_vector(
            _tok(),
            body,
            4000,
            embedding_func=rec.as_func(),
            embedding_batch_num=2,
            embedding_max_async=3,
        )

    asyncio.run(_run())

    assert len(rec.calls) == 15
    assert rec.peak_inflight <= 3


@pytest.mark.offline
def test_v_token_budget_is_adapter_side_only():
    """Pin the budget as BEST EFFORT, not a service-side guarantee.

    Two things outside the adapter's reach make the per-item token count
    unknowable here: the truncation uses LightRAG's general-purpose
    tokenizer, which need not be the embedding model's, and providers may
    rewrite the input afterwards (``openai_embed`` prepends
    ``document_prefix`` before running its own truncation). This test asserts
    the adapter-side bound holds and that the provider-side count can still
    exceed it — so nobody later restates this as a strong guarantee.
    """
    body = _body_with_long_sentence(200)
    budget = 32
    rec = _RecordingEmbedding(max_token_size=budget)

    async def _run():
        return await chunking_by_semantic_vector(
            _tok(), body, 4000, embedding_func=rec.as_func()
        )

    asyncio.run(_run())

    assert rec.calls
    # Adapter side, under the tokenizer the adapter was given.
    assert all(len(_tok().encode(text)) <= budget for text in rec.all_texts)

    # A provider whose tokenizer is denser than the adapter's sees more.
    denser = [2 * len(text) for text in rec.all_texts]
    assert max(denser) > budget

    # A provider that prepends a document prefix sees more, too.
    prefixed = [len("search_document: " + text) for text in rec.all_texts]
    assert max(prefixed) > budget


@pytest.mark.offline
def test_v_cancellation_before_the_bridge_future_exists_blocks_every_batch():
    """Cancelling before a future is published must still stick.

    Cancelling only "whatever is in flight" leaves a window open: the worker
    thread can be running the grouping, or the truncation pass, with no
    future published yet — so the cancellation is dropped, and the thread
    then publishes a fresh one and dispatches every batch of a document
    whose task is already gone.

    The gated tokenizer parks the thread inside ``_truncate``, which is
    reached before publication, making that window deterministic. (The
    narrower case of a still-QUEUED ``to_thread`` needs no handling of its
    own: ``ThreadPoolExecutor.cancel`` succeeds while a work item is queued,
    so the callable never runs.)
    """
    body = _multi_sentence_body(12)
    rec = _RecordingEmbedding()

    class _GatedTokenizerImpl(TokenizerInterface):
        """1 char == 1 token, but the first encode parks the caller."""

        def __init__(self, entered: threading.Event, hold: threading.Event) -> None:
            self.entered = entered
            self.hold = hold
            self.first = True

        def encode(self, content: str):
            if self.first:
                self.first = False
                self.entered.set()
                self.hold.wait(10)
            return [ord(ch) for ch in content]

        def decode(self, tokens):
            return "".join(chr(t) for t in tokens)

    async def _run():
        entered = threading.Event()
        hold = threading.Event()
        tokenizer = Tokenizer("gated", _GatedTokenizerImpl(entered, hold))

        task = asyncio.create_task(
            chunking_by_semantic_vector(
                tokenizer,
                body,
                4000,
                embedding_func=rec.as_func(),
                embedding_batch_num=4,
            )
        )
        while not entered.is_set():
            await asyncio.sleep(0.01)
        # Inside _truncate: the grouping is running, nothing published yet.
        assert rec.calls == [], "no batch should have been dispatched yet"

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        hold.set()
        # Loop stays alive on purpose: this is where the batches would
        # complete if the thread were still allowed to schedule work.
        await asyncio.sleep(0.3)

    asyncio.run(_run())

    assert rec.calls == [], "batches were dispatched after the task was cancelled"
