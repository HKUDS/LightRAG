"""Shared classification of OpenAI-style API errors.

Lives outside ``lightrag/llm/openai.py`` so consumers that only need the
classification -- e.g. ``lightrag.parser.docx.utils``, which treats the
``openai`` package as an optional dependency -- can import it without pulling
in the binding's own imports (``tiktoken``, ``numpy``, and the ``pipmaster``
auto-install of ``openai`` at module import time).

Deliberately dependency-free: it inspects only the parsed error envelope and
the rendered message, never an exception class, so the caller decides which
exception types the check applies to.
"""

from __future__ import annotations

# HTTP 429 has two distinct meanings and only one of them is transient.
#
# - LiteLLM Proxy reports an exhausted spend budget as 429 with
#   ``type == "budget_exceeded"`` (``litellm.proxy._types.ProxyErrorTypes``).
# - OpenAI reports an exhausted account quota as 429 with
#   ``type == "insufficient_quota"``.
#
# Neither clears on its own -- an operator has to raise or reset the budget, or
# top up billing -- so backing off and retrying only burns the retry window and
# then buries the one actionable message behind a ``tenacity.RetryError``.
PERMANENT_RATE_LIMIT_ERROR_TYPES = frozenset(
    {
        "budget_exceeded",
        "insufficient_quota",
    }
)


def is_permanent_rate_limit_error(error: BaseException) -> bool:
    """Report whether a 429 is a permanent spend stop rather than a throttle.

    The OpenAI SDK unwraps the JSON error envelope before constructing the
    exception (``data = body.get("error", body)`` in ``_make_status_error``),
    so ``error.body`` is normally the inner object -- but proxies and SDK
    versions differ, so both shapes are probed, with a substring check on the
    rendered message as a last resort. The type slugs are specific enough that
    a false positive would have to name one of them verbatim.
    """
    body = getattr(error, "body", None)
    candidates = [body]
    if isinstance(body, dict):
        candidates.append(body.get("error"))
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        error_type = candidate.get("type")
        if (
            isinstance(error_type, str)
            and error_type.strip().lower() in PERMANENT_RATE_LIMIT_ERROR_TYPES
        ):
            return True
    rendered = str(error).lower()
    return any(slug in rendered for slug in PERMANENT_RATE_LIMIT_ERROR_TYPES)
