"""Shared retry classification for bindings that call HTTP APIs via aiohttp.

Used by ``lightrag.llm.jina``, ``lightrag.llm.lollms`` and ``lightrag.rerank``.
An ``aiohttp.ClientResponseError`` is raised for every non-200 response alike,
so a retry policy keyed on the exception type alone retries a bad API key or a
malformed request exactly as eagerly as a rate limit.
"""

from __future__ import annotations

import asyncio

import aiohttp

# 408 / 409 / 429: the client-error statuses a later attempt can succeed at.
# Same set as the zhipu binding; the openai binding retries 408 / 409 next to
# its RateLimitError handling.
TRANSIENT_STATUS_CODES = frozenset({408, 409, 429})


def is_transient_aiohttp_error(exc: BaseException) -> bool:
    """tenacity predicate: retry connection failures and transient statuses.

    An ``aiohttp.ClientResponseError`` is retried only for 408 / 409 / 429 and
    5xx; every other status (bad API key, exhausted balance, malformed or
    oversized request) fails on the first attempt.

    Without a response, only ``ClientConnectionError`` (refused, reset or
    timed-out connections, DNS failures), ``ClientPayloadError`` (a body cut
    off mid-transfer) and ``TimeoutError`` are retried. The last is what an
    expired ``aiohttp.ClientTimeout(total=...)`` raises: a bare
    ``asyncio.TimeoutError``, not a ``ClientError``. An outer cancellation
    (the LightRAG queue's own cap) is ``CancelledError`` and never retried.
    ``aiohttp.ClientError`` itself is too broad: it also covers permanent
    local failures such as ``InvalidURL`` (a malformed ``base_url``), which no
    retry can fix.

    Pair it with ``reraise=True`` so callers receive the last attempt's own
    exception rather than ``tenacity.RetryError``.
    """
    if isinstance(exc, aiohttp.ClientResponseError):
        return exc.status in TRANSIENT_STATUS_CODES or exc.status >= 500
    return isinstance(
        exc,
        (
            aiohttp.ClientConnectionError,
            aiohttp.ClientPayloadError,
            asyncio.TimeoutError,
        ),
    )
