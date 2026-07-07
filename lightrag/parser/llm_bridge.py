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
    """The parse was cancelled (or the rag shut down) during an LLM wait."""


class SyncLLMBridge:
    """Call an async LLM submit function from a synchronous worker thread.

    Args:
        loop: The event loop the coroutine must run on. Capture it on the
            loop thread (``asyncio.get_running_loop()``) before entering the
            worker thread.
        submit: Async callable ``(prompt, *, system_prompt=None) -> str``
            executed on ``loop`` (typically wrapping
            ``use_llm_func_with_cache``).
        cancel_events: Events polled between waits — per-parse cancel and/or
            the rag-level shutdown event. Any set event aborts the wait with
            :class:`LLMBridgeCancelled` after at most one poll interval.
        poll_interval: Seconds per ``future.result`` wait slice.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        submit: Callable[..., Coroutine[Any, Any, str]],
        *,
        cancel_events: tuple[threading.Event, ...] = (),
        poll_interval: float = 1.0,
    ) -> None:
        self._loop = loop
        self._submit = submit
        self._cancel_events = tuple(e for e in cancel_events if e is not None)
        self._poll_interval = max(0.01, float(poll_interval))
        # The loop thread id at construction time. Calling the bridge from
        # that thread would block the loop the coroutine needs — a guaranteed
        # deadlock — so __call__ turns it into an immediate error.
        self._loop_thread_id = threading.get_ident()

    def _cancelled(self) -> bool:
        return any(event.is_set() for event in self._cancel_events)

    def __call__(self, prompt: str, *, system_prompt: str | None = None) -> str:
        if threading.get_ident() == self._loop_thread_id:
            raise RuntimeError(
                "SyncLLMBridge called from the event-loop thread; this would "
                "deadlock waiting on the loop it is blocking. Call it only "
                "from a worker thread (parser extract runs in one)."
            )
        if self._cancelled():
            raise LLMBridgeCancelled("parse cancelled before the LLM call")
        future = asyncio.run_coroutine_threadsafe(
            self._submit(prompt, system_prompt=system_prompt), self._loop
        )
        while True:
            if self._cancelled():
                # Propagates cancellation into the loop-side coroutine chain;
                # whether the underlying provider request truly aborts is up
                # to its implementation — the worker thread exits regardless.
                future.cancel()
                raise LLMBridgeCancelled("parse cancelled while awaiting the LLM")
            try:
                return future.result(timeout=self._poll_interval)
            except concurrent.futures.TimeoutError:
                continue
            except concurrent.futures.CancelledError as exc:
                raise LLMBridgeCancelled("LLM call cancelled") from exc
