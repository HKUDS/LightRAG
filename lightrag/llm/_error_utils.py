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


# Provider content-moderation verdict codes, matched case-insensitively on the
# error CODE (never error.type: DashScope models that as invalid_request_error
# while the verdict lives in code). The endswith rule covers the Algo-prefixed
# native variant (InternalError.Algo.DataInspectionFailed) while excluding
# InternalError.DataInspection / InvalidParameter.DataInspection, which are a
# moderation-service outage and a media decode failure, not content verdicts.
_CONTENT_POLICY_ERROR_CODES = frozenset(
    {
        "data_inspection_failed",
        "datainspectionfailed",
        "content_filter",
    }
)

# Distinctive DashScope slugs for the message fallbacks. "content_filter" is
# deliberately NOT among them: that word appears in Azure annotation payloads
# (prompt_filter_results) on choice-less chunks, which are not block verdicts.
_DASHSCOPE_POLICY_SLUGS = ("data_inspection_failed", "datainspectionfailed")


def is_content_policy_error(error: BaseException) -> bool:
    """Report whether an API error is a provider content-moderation verdict.

    Probes ``error.body`` (the OpenAI SDK unwraps the JSON envelope via
    ``body.get("error", body)``, so both shapes are checked) for a verdict
    code: DashScope ``data_inspection_failed`` / ``DataInspectionFailed``,
    including Algo-prefixed variants, and Azure ``content_filter``. Messages
    are not stable (URL suffixes, ``<400>`` prefixes), so a message-substring
    fallback covers proxies that flatten the body; with no dict body at all,
    the rendered message is the last resort, matched only on the two DashScope
    slugs.

    NOT verdicts (must keep their own retry/failure behaviour):
    ``InternalError.DataInspection`` (moderation service unavailable),
    ``InvalidParameter.DataInspection`` (media download/decode failure),
    ``Contain.Forbidden.Label.Error`` (account/billing) and ``FaqRuleBlocked``
    (FAQ rule). The suffix rule excludes the first two by construction.
    """
    body = getattr(error, "body", None)
    candidates = [body]
    if isinstance(body, dict):
        candidates.append(body.get("error"))
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        code = str(candidate.get("code") or "").strip().lower()
        if code in _CONTENT_POLICY_ERROR_CODES or code.endswith("datainspectionfailed"):
            return True
        message = str(candidate.get("message") or "").lower()
        if any(slug in message for slug in _DASHSCOPE_POLICY_SLUGS):
            return True
    if isinstance(body, dict):
        return False
    rendered = str(error).lower()
    return any(slug in rendered for slug in _DASHSCOPE_POLICY_SLUGS)
