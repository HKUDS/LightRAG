"""Synchronous LLM bridge for parser extract hooks running in worker threads.

Native parser ``extract`` hooks are synchronous and run in a worker thread,
while the LLM role funcs (and their cache) are async and bound to the
LightRAG owning loop. :class:`SyncLLMBridge` crosses that boundary: it is
constructed ON the loop thread (capturing the loop), and called FROM the
worker thread, where it submits the coroutine via
``run_coroutine_threadsafe`` and waits with a short-interval poll so a
cancellation can interrupt the wait promptly.

The poll timeout is NOT an LLM timeout — total LLM timeout/retry semantics
belong to the role wrapper (``llm_timeout`` etc.); the bridge waits
indefinitely, checking its cancel events once per interval. Engine-agnostic
on purpose: future xlsx/pptx native engines reuse it as-is.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from collections.abc import Callable, Coroutine
from typing import Any


class LLMBridgeCancelled(RuntimeError):
    """The parse was cancelled (or the rag shut down) during a blocking wait.

    Named for its first consumer (the LLM bridge), but the hierarchy is the
    repo-wide "a blocking parse wait was cancelled" signal: ``pipeline.py``
    catches :class:`LLMBridgePipelineCancelled` to mark a document cancelled
    rather than failed, so any blocking parse path that wants that treatment
    must raise from this family.
    """


class LLMBridgePipelineCancelled(LLMBridgeCancelled):
    """The active document pipeline was cancelled by its caller."""


class LLMBridgeShutdown(LLMBridgeCancelled):
    """The RAG parser executor is shutting down."""


def normalize_cancel_events(
    cancel_events: tuple[
        threading.Event | tuple[threading.Event, type[LLMBridgeCancelled]] | None, ...
    ],
) -> tuple[tuple[threading.Event, type[LLMBridgeCancelled]], ...]:
    """Normalize a cancel-event spec into ``(event, exception_type)`` pairs.

    An entry may be a bare Event (mapped to :class:`LLMBridgeCancelled`) or an
    ``(Event, exception_type)`` pair when the caller needs the cancellation
    source preserved. ``None`` events are dropped so callers can pass optional
    events positionally without filtering first.
    """
    normalized: list[tuple[threading.Event, type[LLMBridgeCancelled]]] = []
    for entry in cancel_events:
        if isinstance(entry, tuple):
            event, exception_type = entry
        else:
            event, exception_type = entry, LLMBridgeCancelled
        if event is not None:
            normalized.append((event, exception_type))
    return tuple(normalized)


def first_cancellation(
    cancel_events: tuple[tuple[threading.Event, type[LLMBridgeCancelled]], ...],
    message: str,
) -> LLMBridgeCancelled | None:
    """Return the exception for the first set event, or ``None`` if none is set."""
    for event, exception_type in cancel_events:
        if event.is_set():
            return exception_type(message)
    return None


class SyncLLMBridge:
    """Call an async LLM submit function from a synchronous worker thread.

    Args:
        loop: The event loop the coroutine must run on. Capture it on the
            loop thread (``asyncio.get_running_loop()``) before entering the
            worker thread.
        submit: Async callable ``(prompt, *, system_prompt=None) -> str``
            executed on ``loop`` (typically wrapping
            ``use_llm_func_with_cache``).
        cancel_events: Events polled between waits. An entry may be an Event,
            or an ``(Event, exception_type)`` pair when callers need the
            cancellation source preserved. Any set event aborts the wait after
            at most one poll interval.
        poll_interval: Seconds per ``future.result`` wait slice.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        submit: Callable[..., Coroutine[Any, Any, str]],
        *,
        cancel_events: tuple[
            threading.Event | tuple[threading.Event, type[LLMBridgeCancelled]], ...
        ] = (),
        poll_interval: float = 1.0,
    ) -> None:
        self._loop = loop
        self._submit = submit
        self._cancel_events = normalize_cancel_events(cancel_events)
        self._poll_interval = max(0.01, float(poll_interval))
        # The loop thread id at construction time. Calling the bridge from
        # that thread would block the loop the coroutine needs — a guaranteed
        # deadlock — so __call__ turns it into an immediate error.
        self._loop_thread_id = threading.get_ident()

    def _cancellation_exception(self, message: str) -> LLMBridgeCancelled | None:
        return first_cancellation(self._cancel_events, message)

    def __call__(self, prompt: str, *, system_prompt: str | None = None) -> str:
        if threading.get_ident() == self._loop_thread_id:
            raise RuntimeError(
                "SyncLLMBridge called from the event-loop thread; this would "
                "deadlock waiting on the loop it is blocking. Call it only "
                "from a worker thread (parser extract runs in one)."
            )
        cancellation = self._cancellation_exception(
            "parse cancelled before the LLM call"
        )
        if cancellation is not None:
            raise cancellation
        future = asyncio.run_coroutine_threadsafe(
            self._submit(prompt, system_prompt=system_prompt), self._loop
        )
        while True:
            cancellation = self._cancellation_exception(
                "parse cancelled while awaiting the LLM"
            )
            if cancellation is not None:
                # Propagates cancellation into the loop-side coroutine chain;
                # whether the underlying provider request truly aborts is up
                # to its implementation — the worker thread exits regardless.
                if not future.cancel() and future.done():
                    # Already finished (perhaps with an exception): consume the
                    # result so asyncio does not log "exception was never
                    # retrieved" for the abandoned future.
                    try:
                        future.exception()
                    except Exception:
                        pass
                raise cancellation
            try:
                return future.result(timeout=self._poll_interval)
            except concurrent.futures.TimeoutError:
                continue
            except concurrent.futures.CancelledError as exc:
                raise LLMBridgeCancelled("LLM call cancelled") from exc
