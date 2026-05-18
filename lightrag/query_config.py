"""Query configuration for LightRAG.

`QueryParam` is a pure value object — no environment reads at definition time.

For production use, call `default_query_param(**overrides)` which reads env vars
once at call time. For tests, use `ConfigResolver(env={...}).query_param(**overrides)`
to inject config without touching os.environ.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping

from .constants import (
    DEFAULT_CHUNK_TOP_K,
    DEFAULT_HISTORY_TURNS,
    DEFAULT_MAX_ENTITY_TOKENS,
    DEFAULT_MAX_RELATION_TOKENS,
    DEFAULT_MAX_TOTAL_TOKENS,
    DEFAULT_TOP_K,
)


@dataclass
class QueryParam:
    """Configuration parameters for a Knowledge Graph query."""

    mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"] = "mix"
    """Retrieval mode. See CONTEXT.md § Retrieval Mode."""

    only_need_context: bool = False
    """If True, return retrieved context without LLM synthesis."""

    only_need_prompt: bool = False
    """If True, return the generated prompt without producing a response."""

    response_type: str = "Multiple Paragraphs"
    """Response format hint for the LLM. E.g. 'Bullet Points', 'Single Paragraph'."""

    stream: bool = False
    """If True, stream output tokens as they are produced."""

    top_k: int = DEFAULT_TOP_K
    """Entities (local mode) or relationships (global mode) to retrieve."""

    chunk_top_k: int = DEFAULT_CHUNK_TOP_K
    """Text chunks to retrieve and keep after reranking."""

    max_entity_tokens: int = DEFAULT_MAX_ENTITY_TOKENS
    """Token budget for entity context."""

    max_relation_tokens: int = DEFAULT_MAX_RELATION_TOKENS
    """Token budget for relationship context."""

    max_total_tokens: int = DEFAULT_MAX_TOTAL_TOKENS
    """Total token budget (entities + relations + chunks + system prompt)."""

    hl_keywords: list[str] = field(default_factory=list)
    """High-level keywords to prioritise in retrieval."""

    ll_keywords: list[str] = field(default_factory=list)
    """Low-level keywords to refine retrieval focus."""

    conversation_history: list[dict[str, str]] = field(default_factory=list)
    """Past conversation turns for context. Format: [{"role": "user/assistant", "content": "..."}]."""

    history_turns: int = DEFAULT_HISTORY_TURNS
    """Deprecated — kept for backwards compatibility. All conversation_history is sent to LLM."""

    model_func: Callable[..., object] | None = None
    """Optional per-query LLM override. If None, the global model function is used."""

    user_prompt: str | None = None
    """Additional instructions injected into the LLM prompt template."""

    enable_rerank: bool = True
    """Enable reranking for retrieved chunks. Requires a configured rerank model."""

    include_references: bool = False
    """If True, include citation references in the response."""


class ConfigResolver:
    """Resolves QueryParam defaults from an environment mapping.

    Use this instead of reading os.environ directly in QueryParam field defaults.
    Tests inject a plain dict; production code passes os.environ.

    Example (test)::

        param = ConfigResolver(env={"TOP_K": "5"}).query_param(mode="hybrid")

    Example (production)::

        param = default_query_param(mode="hybrid")
    """

    def __init__(self, env: Mapping[str, str]) -> None:
        self._env = env

    def _get(self, key: str, default: Any, cast: type = str) -> Any:
        raw = self._env.get(key)
        if raw is None:
            return default
        try:
            return cast(raw)
        except (ValueError, TypeError):
            return default

    def query_param(self, **overrides: Any) -> QueryParam:
        """Return a QueryParam with env-resolved defaults, overridden by any kwargs."""
        resolved: dict[str, Any] = {
            "top_k": self._get("TOP_K", DEFAULT_TOP_K, int),
            "chunk_top_k": self._get("CHUNK_TOP_K", DEFAULT_CHUNK_TOP_K, int),
            "max_entity_tokens": self._get(
                "MAX_ENTITY_TOKENS", DEFAULT_MAX_ENTITY_TOKENS, int
            ),
            "max_relation_tokens": self._get(
                "MAX_RELATION_TOKENS", DEFAULT_MAX_RELATION_TOKENS, int
            ),
            "max_total_tokens": self._get(
                "MAX_TOTAL_TOKENS", DEFAULT_MAX_TOTAL_TOKENS, int
            ),
            "history_turns": self._get("HISTORY_TURNS", DEFAULT_HISTORY_TURNS, int),
            "enable_rerank": self._get("RERANK_BY_DEFAULT", "true").lower() == "true",
        }
        resolved.update(overrides)
        return QueryParam(**resolved)


def default_query_param(**overrides: Any) -> QueryParam:
    """Return a QueryParam with defaults drawn from os.environ.

    This is the production convenience wrapper over ConfigResolver.
    Use ConfigResolver(env={...}).query_param() in tests.
    """
    return ConfigResolver(env=os.environ).query_param(**overrides)
