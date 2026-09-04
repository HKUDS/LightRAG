"""Semantic vector chunking — the ``"V"`` strategy.

Wraps LangChain's :class:`SemanticChunker` (from ``langchain-experimental``)
which splits text by sentence embeddings: it first segments the input into
sentences, embeds each sentence (in adjacent windows of ``buffer_size``),
and finds breakpoints where the cosine distance between consecutive
windows crosses a threshold derived from the chosen distribution
(``percentile`` / ``standard_deviation`` / ``interquartile`` /
``gradient``).

The chunker exposed here is ``async`` because LightRAG's
:class:`EmbeddingFunc` is async.  Internally we call SemanticChunker
synchronously inside :func:`asyncio.to_thread` and bridge the embedding
calls back to the main event loop via
:func:`asyncio.run_coroutine_threadsafe`.

Caveats:
  - SemanticChunker does NOT enforce a maximum chunk size; the caller's
    ``chunk_token_size`` is *advisory* here.  Oversized chunks will be
    hard-split before embedding by
    :func:`lightrag.utils.enforce_chunk_token_limit_before_embedding`.
    That function only covers the chunks this module *emits*; the
    sentence-window embeddings V performs while chunking are bounded
    separately, see below.
  - When ``embedding_func`` is ``None`` we log a warning and fall back to
    :func:`lightrag.chunker.chunking_by_recursive_character` — V's only
    differentiator is embeddings, and R is the closest structural-only
    alternative.

Embedding request bounding:
    Upstream ``SemanticChunker`` embeds the *whole document* in a single
    ``embed_documents`` call — one item per sentence window, so the request
    carries ``N = sentence count`` items and roughly
    ``(2 * buffer_size + 1) x document`` tokens.  No provider binding in
    ``lightrag/llm`` slices that list, so a large enough document breaks
    the service's per-request limits.  :class:`_AsyncEmbeddingFuncAdapter`
    therefore batches by item count and truncates each window.  What is
    guaranteed, per ``chunking_by_semantic_vector`` call:

      * every provider request carries at most ``embedding_batch_num``
        items (``<= 0`` deliberately disables batching and gives up the
        bound entirely);
      * at most ``embedding_max_async`` requests from this one adapter are
        in flight at a time — cross-instance concurrency is the priority
        wrapper's job, not this pool's;
      * once one batch's failure is observed, no further batch is started.

    Per-item token counts are **best effort only**: the truncation uses
    LightRAG's general-purpose tokenizer, which need not match the
    embedding model's, and providers may rewrite the input afterwards
    (``openai_embed`` prepends ``document_prefix`` before its own
    truncation).  ``batch_size * budget`` is a logical bound under the
    adapter's tokenizer, never a promise about what the service receives.
    Truncation cannot lose chunk content: the emitted chunks are original
    source spans, so only the boundary distance computation degrades.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import re
from typing import TYPE_CHECKING, Any

from lightrag.constants import (
    DEFAULT_EMBEDDING_BATCH_NUM,
    DEFAULT_EMBEDDING_FUNC_MAX_ASYNC,
    DEFAULT_SENTENCE_SPLIT_REGEX,
)
from lightrag.utils import (
    EmbeddingFunc,
    TokenBudgetError,
    Tokenizer,
    logger,
    run_in_chunking_executor,
)

if TYPE_CHECKING:
    from langchain_experimental.text_splitter import (
        SemanticChunker as SemanticChunkerType,
    )
else:
    SemanticChunkerType = Any

try:
    from langchain_core.embeddings import Embeddings
    from langchain_experimental.text_splitter import SemanticChunker

    _LANGCHAIN_EXPERIMENTAL_AVAILABLE = True
except ImportError:
    _LANGCHAIN_EXPERIMENTAL_AVAILABLE = False
    Embeddings = object  # type: ignore[assignment,misc]
    SemanticChunker = None  # type: ignore[assignment]


class _AsyncEmbeddingFuncAdapter(Embeddings):
    """Bridge a LightRAG :class:`EmbeddingFunc` (async) to LangChain's
    sync :class:`Embeddings` interface used by ``SemanticChunker``.

    The adapter must be constructed inside the running event loop so it
    can capture the loop reference; the blocking ``embed_documents`` /
    ``embed_query`` calls are then made from a worker thread (via
    :func:`asyncio.to_thread` in the public chunker) and bounce back to
    the captured loop with :func:`asyncio.run_coroutine_threadsafe`.

    Every call truncates each text to ``max_token_size`` and splits the
    list into ``batch_size``-item provider requests, running at most
    ``max_concurrency`` of them at a time.  See the module docstring for
    exactly what that does and does not guarantee.

    .. warning::
        Never invoke this adapter from inside the chunking executor.  That
        pool has a single worker (see ``run_in_chunking_executor``), and a
        blocking ``future.result()`` there would occupy it — while
        ``chunking_by_semantic_vector``'s own fallback path submits to the
        same pool.  Today the adapter only runs on the *default* executor
        via ``asyncio.to_thread``; ``_size_and_resplit`` is the part that
        runs in the chunking pool, and it never touches the adapter.
        Moving an embedding bounce into it would deadlock.

        Related, unchanged by the batching: an ``embedding_func`` that
        wraps a *synchronous* SDK in ``asyncio.to_thread`` competes with
        V's own blocking thread for default-executor workers (one thread
        per document either way).
    """

    def __init__(
        self,
        embedding_func: EmbeddingFunc,
        loop: asyncio.AbstractEventLoop,
        *,
        tokenizer: Tokenizer | None = None,
        max_token_size: int | None = None,
        batch_size: int = DEFAULT_EMBEDDING_BATCH_NUM,
        max_concurrency: int = DEFAULT_EMBEDDING_FUNC_MAX_ASYNC,
        budget_source: str = "unspecified",
    ) -> None:
        # Rejected here rather than at the concurrency primitive: zero
        # workers would leave every batch waiting forever, which surfaces
        # as a hung document instead of an error.
        if max_concurrency <= 0:
            raise ValueError(f"max_concurrency must be positive, got {max_concurrency}")
        self._embedding_func = embedding_func
        self._loop = loop
        self._tokenizer = tokenizer
        self._max_token_size = max_token_size
        self._batch_size = batch_size
        self._max_concurrency = max_concurrency
        self._budget_source = budget_source
        self._inflight: concurrent.futures.Future | None = None

    # -- truncation ----------------------------------------------------

    def _truncate(self, texts: list[str]) -> list[str]:
        """Cap each text at ``max_token_size`` tokens, warning once.

        Truncation here can never lose chunk content: the emitted chunks
        are original source spans, so an over-budget sentence window only
        degrades the boundary distance it contributes to.
        """
        tokenizer = self._tokenizer
        budget = self._max_token_size
        if tokenizer is None or not budget or budget <= 0:
            return texts

        out: list[str] = []
        truncated = 0
        longest_original = 0
        degraded = 0
        for text in texts:
            if not text:
                out.append(text)
                continue
            # Encode first instead of calling truncate_by_token_limit
            # unconditionally: the common case (text fits) costs the same
            # single encode, and the over-budget case is the only one that
            # pays twice — while giving us the ORIGINAL token count for the
            # warning.  ``TokenSpan.token_count`` is the count that
            # SURVIVED truncation, so reporting it as "longest" would
            # mislead.  A character-length pre-check is not an option: cl100k
            # emits more than one token per CJK character, so
            # ``len(text) <= budget`` does not imply the token count fits.
            total = len(tokenizer.encode(text))
            if total <= budget:
                out.append(text)
                continue
            truncated += 1
            longest_original = max(longest_original, total)
            try:
                span = tokenizer.truncate_by_token_limit(text, budget)
                out.append(text[span.start : span.end])
            except TokenBudgetError:
                # Not even the first code point fits.  Degrade to that
                # single code point rather than failing the document — a
                # boundary-quality nicety must not turn into a FAILED doc.
                degraded += 1
                out.append(text[:1])

        if truncated:
            message = (
                f"[semantic_vector] truncated {truncated}/{len(texts)} sentence "
                f"windows to budget={budget} tokens (source={self._budget_source}, "
                f"longest_original={longest_original} tokens) before embedding. "
                "Chunk content and source spans are NOT affected — only the "
                "semantic boundary distances for those windows degrade."
            )
            if degraded:
                message += (
                    f" {degraded} window(s) did not fit even one code point and "
                    "were degraded to their first code point."
                )
            logger.warning(message)
        return out

    # -- batching ------------------------------------------------------

    async def _embed_all(self, batches: list[list[str]], context: str) -> list[Any]:
        """Run ``batches`` through ``embedding_func``, at most
        ``max_concurrency`` at a time, results in batch order.

        A bounded worker pool rather than ``asyncio.gather`` over every
        batch: ``gather`` without ``return_exceptions`` raises the first
        error to its awaiter but does NOT cancel the remaining awaitables,
        so one 429 would leave thousands of batches hammering the provider
        after the document had already failed.  It also materialises one
        Task per batch up front.  ``asyncio.TaskGroup`` would do, but the
        project floor is Python 3.10.

        The pool also bounds how many queue slots V occupies in
        ``priority_limit_async_func_call``: that wrapper enqueues at CALL
        time, so keeping calls in flight bounded keeps queued tasks
        bounded.  It is NOT here to limit execution concurrency — the
        wrapper already caps that at ``embedding_func_max_async``; it is
        here so V cannot park thousands of coroutines behind the (1000
        slot, blocking-on-full) queue and add minutes of latency to
        queries.
        """
        results: list[Any] = [None] * len(batches)
        cursor = iter(range(len(batches)))
        # Single-threaded event loop: next() on the shared iterator cannot
        # interleave, so no lock is needed.
        stopped = asyncio.Event()

        async def _worker() -> None:
            for i in cursor:
                # Checked before every batch, and set in the except below
                # rather than relying on the outer gather's cancellation:
                # between one worker's failure and that error reaching the
                # awaiter, the other workers would otherwise have pulled
                # and dispatched fresh batches, breaking the "no new batch
                # after a failure" guarantee.  CancelledError is a
                # BaseException too, so an outer cancel also stops intake.
                if stopped.is_set():
                    return
                try:
                    results[i] = await self._embedding_func(batches[i], context=context)
                except BaseException:
                    stopped.set()
                    raise

        tasks = [
            asyncio.create_task(_worker())
            for _ in range(min(self._max_concurrency, len(batches)))
        ]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        return results

    def cancel_inflight(self) -> None:
        """Cancel the in-flight bounced call, if any.

        Called from the event loop when the outer chunking task is
        cancelled.  The bridging future is never ``set_running``, so
        ``cancel()`` both wakes the blocked worker thread and — through
        ``_chain_future``'s cancellation callback — cancels the loop-side
        task, and with it the worker pool.
        """
        future = self._inflight
        if future is not None:
            future.cancel()

    def _run(self, texts: list[str], context: str) -> list[list[float]]:
        texts = self._truncate(texts)
        batch_size = self._batch_size
        if batch_size and batch_size > 0:
            batches = [
                texts[i : i + batch_size] for i in range(0, len(texts), batch_size)
            ]
        else:
            # Explicitly opted out of batching: no item or token bound.
            batches = [texts] if texts else []

        # One bounce per call, not one per batch: bouncing each batch and
        # blocking on its result would serialize them.  Every asyncio
        # object lives inside the bounced coroutine — this method runs on a
        # worker thread with no running loop.
        # Chained so the cancellation handle is published as early as
        # possible: a cancel arriving before the store would find no
        # in-flight future and leave this thread blocked until the call
        # finished on its own (today's behavior, but no reason to widen it).
        self._inflight = future = asyncio.run_coroutine_threadsafe(
            self._embed_all(batches, context),
            self._loop,
        )
        try:
            batch_results = future.result()
        finally:
            self._inflight = None

        rows: list[list[float]] = []
        for result in batch_results:
            rows.extend(list(map(float, vec)) for vec in result)
        return rows

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._run(list(texts), context="document")

    def embed_query(self, text: str) -> list[float]:
        return self._run([text], context="query")[0]


def _sentence_spans(text: str, sentences: list[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    for sentence in sentences:
        if not sentence:
            spans.append((cursor, cursor))
            continue
        start = text.find(sentence, cursor)
        if start < 0:
            start = text.find(sentence)
        if start < 0:
            start = cursor
        end = start + len(sentence)
        spans.append((start, end))
        cursor = end
    return spans


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    start = max(0, min(start, len(text)))
    end = max(start, min(end, len(text)))
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _semantic_groups_with_spans(
    splitter: SemanticChunkerType,
    text: str,
) -> list[tuple[str, int, int]]:
    """Mirror SemanticChunker grouping while keeping original source spans.

    .. warning::
        This re-implements the body of ``SemanticChunker.split_text`` so each group
        carries its exact source span (``text[start:end]``) instead of the upstream
        ``" ".join(sentences)`` reflow. It relies on **private** members
        (``sentence_split_regex``, ``breakpoint_threshold_type``, ``min_chunk_size``,
        ``number_of_chunks``, ``_calculate_sentence_distances``,
        ``_threshold_from_clusters``, ``_calculate_breakpoint_threshold``). Verified
        byte-for-byte against ``langchain-experimental`` 0.3.2–0.4.x (the range pinned
        in ``pyproject.toml``: ``langchain-experimental>=0.3.2,<1``). If that pin is
        widened, re-verify against the new upstream ``split_text`` —
        ``tests/chunker/test_chunker_semantic_vector.py`` has a drift guard that
        compares this mirror's grouping to the live ``splitter.split_text`` output.
    """
    single_sentences_list = re.split(splitter.sentence_split_regex, text)
    spans = _sentence_spans(text, single_sentences_list)

    def _group(start_index: int, end_index: int) -> tuple[str, int, int] | None:
        start, _ = spans[start_index]
        _, end = spans[end_index]
        start, end = _trim_span(text, start, end)
        if start >= end:
            return None
        return text[start:end], start, end

    if len(single_sentences_list) == 1:
        group = _group(0, 0)
        return [group] if group else []
    if (
        splitter.breakpoint_threshold_type == "gradient"
        and len(single_sentences_list) == 2
    ):
        return [g for i in range(2) if (g := _group(i, i)) is not None]

    distances, sentences = splitter._calculate_sentence_distances(single_sentences_list)
    if splitter.number_of_chunks is not None:
        breakpoint_distance_threshold = splitter._threshold_from_clusters(distances)
        breakpoint_array = distances
    else:
        breakpoint_distance_threshold, breakpoint_array = (
            splitter._calculate_breakpoint_threshold(distances)
        )

    indices_above_thresh = [
        i for i, x in enumerate(breakpoint_array) if x > breakpoint_distance_threshold
    ]

    chunks: list[tuple[str, int, int]] = []
    start_index = 0
    for index in indices_above_thresh:
        end_index = index
        group_sentences = sentences[start_index : end_index + 1]
        combined_text = " ".join([d["sentence"] for d in group_sentences])
        if (
            splitter.min_chunk_size is not None
            and len(combined_text) < splitter.min_chunk_size
        ):
            continue
        group = _group(start_index, end_index)
        if group is not None:
            chunks.append(group)
        start_index = index + 1

    if start_index < len(sentences):
        group = _group(start_index, len(sentences) - 1)
        if group is not None:
            chunks.append(group)
    return chunks


async def chunking_by_semantic_vector(
    tokenizer: Tokenizer,
    content: str,
    chunk_token_size: int = 1200,
    *,
    embedding_func: EmbeddingFunc | None = None,
    breakpoint_threshold_type: str = "percentile",
    breakpoint_threshold_amount: float | None = None,
    buffer_size: int = 1,
    sentence_split_regex: str = DEFAULT_SENTENCE_SPLIT_REGEX,
    number_of_chunks: int | None = None,
    min_chunk_size: int | None = None,
    embedding_batch_num: int | None = None,
    embedding_max_async: int | None = None,
) -> list[dict[str, Any]]:
    """Semantic vector chunker — the ``"V"`` chunking strategy.

    Args:
        tokenizer: LightRAG tokenizer (used for output token counts, and
            for the best-effort truncation of sentence windows before
            embedding).
        content: Text to split.
        chunk_token_size: Hard upper bound (tokens). SemanticChunker does
            NOT enforce a maximum natively, so any piece that exceeds
            this value is re-split via
            :func:`chunking_by_recursive_character` before being emitted.
        embedding_func: LightRAG :class:`EmbeddingFunc`. When ``None``
            this chunker logs a warning and falls back to
            :func:`chunking_by_recursive_character`.
        breakpoint_threshold_type: ``percentile`` | ``standard_deviation``
            | ``interquartile`` | ``gradient`` (LangChain default:
            ``percentile``).
        breakpoint_threshold_amount: Threshold magnitude. ``None`` lets
            LangChain pick the per-type default (e.g. 95 for percentile).
        buffer_size: Number of adjacent sentences combined when computing
            distances (LangChain default: 1).
        sentence_split_regex: Pattern fed to LangChain's
            :class:`SemanticChunker` for the initial sentence split.
            Default extends the upstream English-only pattern with
            Chinese sentence terminators ``。？！`` so mixed-language and
            pure-Chinese inputs split correctly.
        number_of_chunks: Optional target chunk count (LangChain SemanticChunker).
        min_chunk_size: Optional minimum character size for semantic groups.
        embedding_batch_num: Items per sentence-embedding provider request.
            ``None`` uses :data:`DEFAULT_EMBEDDING_BATCH_NUM`; ``<= 0``
            disables batching, which gives up the request bound entirely
            (both item count and tokens become unbounded again).
        embedding_max_async: How many of this call's own sentence-embedding
            requests may be in flight at once. ``None`` uses
            :data:`DEFAULT_EMBEDDING_FUNC_MAX_ASYNC`; a non-positive value
            raises :class:`ValueError` rather than hanging. This bounds one
            chunker call, not the deployment — cross-instance concurrency
            stays with the priority wrapper that ``LightRAG`` applies.

    The per-window token budget is ``embedding_func.max_token_size``, or
    the effective ``chunk_token_size`` when that is ``None`` **or** ``0``.
    Treating ``0`` as "unset" is a deliberate divergence from
    :func:`lightrag.llm.openai.openai_embed`, where ``max_token_size=0``
    documents "disable truncation": V needs a budget of its own, and the
    server path cannot even produce ``0`` here (``embedding_token_limit or
    provider_max_token_size`` replaces it with the provider default), so
    only a direct ``EmbeddingFunc(max_token_size=0)`` SDK caller reaches
    this branch. See the module docstring for what the resulting bound
    does and does not guarantee.

    Returns:
        Ordered list of ``{"tokens", "content", "chunk_order_index"}``
        dicts.
    """
    if not content or not content.strip():
        return []

    if embedding_func is None:
        # V's only differentiator is embeddings — without them the
        # closest neighbour is R's structural splitting.  V chunks are
        # non-overlapping by design (semantic boundaries), so the
        # fallback uses ``chunk_overlap_token_size=0`` to preserve that
        # semantic and avoid LangChain's "overlap > chunk_size" guard
        # for very small ``chunk_token_size``.
        logger.warning(
            "[semantic_vector] embedding_func is None; falling back to "
            "recursive-character chunking."
        )
        from lightrag.chunker.recursive_character import (
            chunking_by_recursive_character,
        )

        # Off the loop like every other R run. Without this the ``await
        # asyncio.to_thread`` further down is never reached, so V's "already
        # offloaded" reputation does not hold for a deployment without an
        # embedding function — the R attack surface reappears in full.
        return await run_in_chunking_executor(
            chunking_by_recursive_character,
            tokenizer,
            content,
            chunk_token_size,
            chunk_overlap_token_size=0,
        )

    if not _LANGCHAIN_EXPERIMENTAL_AVAILABLE:
        raise ImportError(
            "langchain-experimental is required for the 'V' chunking "
            "strategy; install with `pip install langchain-experimental>=0.3.2`."
        )

    # Hoisted above the adapter: it is both R's re-split target below and
    # the fallback budget for sentence-window truncation.
    target_max = max(int(chunk_token_size), 1)

    declared_budget = embedding_func.max_token_size
    if declared_budget is None or int(declared_budget) == 0:
        # See the "0 is treated as unset" note in this function's docstring.
        budget = target_max
        budget_source = "chunk_token_size fallback"
    else:
        budget = int(declared_budget)
        budget_source = "embedding_func.max_token_size"
    # ``truncate_by_token_limit`` raises ValueError on a non-positive
    # budget, and a direct SDK caller can pass a negative chunk_token_size.
    budget = max(budget, 1)

    resolved_batch_num = (
        DEFAULT_EMBEDDING_BATCH_NUM
        if embedding_batch_num is None
        else int(embedding_batch_num)
    )
    resolved_max_async = (
        DEFAULT_EMBEDDING_FUNC_MAX_ASYNC
        if embedding_max_async is None
        else int(embedding_max_async)
    )
    if resolved_max_async <= 0:
        raise ValueError(
            f"embedding_max_async must be positive, got {resolved_max_async}"
        )

    loop = asyncio.get_running_loop()
    adapter = _AsyncEmbeddingFuncAdapter(
        embedding_func,
        loop,
        tokenizer=tokenizer,
        max_token_size=budget,
        batch_size=resolved_batch_num,
        max_concurrency=resolved_max_async,
        budget_source=budget_source,
    )

    chunker_kwargs: dict[str, Any] = {
        "embeddings": adapter,
        "buffer_size": int(buffer_size),
        "breakpoint_threshold_type": breakpoint_threshold_type,
        "sentence_split_regex": sentence_split_regex,
        "number_of_chunks": number_of_chunks,
        "min_chunk_size": min_chunk_size,
    }
    if breakpoint_threshold_amount is not None:
        chunker_kwargs["breakpoint_threshold_amount"] = float(
            breakpoint_threshold_amount
        )

    splitter = SemanticChunker(**chunker_kwargs)
    try:
        pieces = await asyncio.to_thread(_semantic_groups_with_spans, splitter, content)
    except asyncio.CancelledError:
        # ``to_thread`` returns on cancellation but the worker thread keeps
        # running — and it is blocked on a bounced embedding call.  Cancel
        # that bridging future so the thread is woken and the loop-side
        # worker pool is torn down instead of running to completion with
        # its results discarded.
        adapter.cancel_inflight()
        raise

    # SemanticChunker has no internal size cap; oversized pieces here
    # would otherwise rely on the embedding-time hard fallback (which
    # uses ``embedding_token_limit``, not ``chunk_token_size``) to split
    # them.  Enforce ``chunk_token_size`` directly via R for any piece
    # that exceeds it so the user-configured size is actually honored.
    # Lazy import dodges the recursive_character ↔ semantic_vector
    # circular dependency (same pattern as the embedding-None fallback
    # above).
    from lightrag.chunker.recursive_character import (
        chunking_by_recursive_character,
    )

    def _size_and_resplit() -> list[dict[str, Any]]:
        """Encode every semantic piece and re-split the oversized ones.

        ``await asyncio.to_thread`` above covers the embedding-driven grouping
        and nothing else: this loop encodes each piece and, for anything over
        budget, runs a whole R pass on it. On the event loop that is comparable
        in cost to the grouping it follows, which is why the whole loop — not
        each piece — moves off it in one hop.

        ``chunking_by_recursive_character`` is called DIRECTLY here: this
        function already runs inside the chunking executor, and a nested
        submission into a single-worker pool whose permit it holds would
        deadlock.
        """
        results: list[dict[str, Any]] = []
        for piece, source_start, source_end in pieces:
            body = piece.strip()
            if not body:
                continue
            piece_tokens = len(tokenizer.encode(body))
            if piece_tokens <= target_max:
                results.append(
                    {
                        "tokens": piece_tokens,
                        "content": body,
                        "chunk_order_index": len(results),
                        "_source_span": {
                            "start": source_start,
                            "end": source_end,
                        },
                    }
                )
                continue
            # Oversized semantic piece: re-split via R while preserving the
            # surrounding chunk order.  ``chunk_overlap_token_size=0`` keeps
            # V's non-overlapping semantics.
            sub_pieces = chunking_by_recursive_character(
                tokenizer,
                body,
                target_max,
                chunk_overlap_token_size=0,
            )
            for sub in sub_pieces:
                sub_body = sub.get("content", "")
                if not sub_body:
                    continue
                sub_span = sub.get("_source_span")
                source_span = None
                if isinstance(sub_span, dict):
                    try:
                        source_span = {
                            "start": source_start + int(sub_span["start"]),
                            "end": source_start + int(sub_span["end"]),
                        }
                    except (KeyError, TypeError, ValueError):
                        source_span = None
                results.append(
                    {
                        "tokens": sub.get("tokens", len(tokenizer.encode(sub_body))),
                        "content": sub_body,
                        "chunk_order_index": len(results),
                        **({"_source_span": source_span} if source_span else {}),
                    }
                )
        return results

    return await run_in_chunking_executor(_size_and_resplit)
