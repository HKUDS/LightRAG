"""OpenAI-compatible API bindings for the bench's LLM and embedding roles.

Rules:
- Endpoints and credentials are runtime values only: the host comes from the
  bench config, then ``LLM_BINDING_HOST`` / ``EMBEDDING_BINDING_HOST``; the
  API key from ``LLM_BINDING_API_KEY`` / ``EMBEDDING_BINDING_API_KEY``, then
  ``OPENAI_API_KEY``. No concrete provider endpoint or key lives in code.
- The product binding owns retry and rate-limit handling (tenacity over an
  SDK client with ``max_retries=0``); the bench never wraps or swallows it.
- ``llm_num_predict`` maps to ``max_tokens`` on this path.
- ``llm_enable_thinking`` rides in ``extra_body``, the OpenAI SDK's generic
  passthrough for provider-specific request-body fields; a thinking model
  used for extraction should set it False. None sends nothing.
- Embedding dimensions are requested explicitly (``dimensions`` parameter)
  so a provider default can never silently disagree with the vector tables.
"""

from __future__ import annotations

import os
from functools import partial
from typing import TYPE_CHECKING

from lightrag.utils import EmbeddingFunc

if TYPE_CHECKING:
    from lightrag.evaluation.hybrid_bench.rag_setup import BenchConfig


def _resolve_host(configured: str, env_var: str) -> str | None:
    return configured or os.environ.get(env_var) or None


def _resolve_api_key(binding_env_var: str) -> str | None:
    return os.environ.get(binding_env_var) or os.environ.get("OPENAI_API_KEY")


def build_openai_llm_func(config: BenchConfig):
    """Partial of the product's openai completion binding, model bound.

    The model rides POSITIONALLY: callers pass the prompt as the first
    positional argument, exactly as the ollama binding expects it.
    """
    from lightrag.llm.openai import openai_complete_if_cache

    kwargs: dict = {
        "base_url": _resolve_host(config.llm_host, "LLM_BINDING_HOST"),
        "api_key": _resolve_api_key("LLM_BINDING_API_KEY"),
        "timeout": config.request_timeout,
    }
    if config.llm_num_predict is not None:
        kwargs["max_tokens"] = config.llm_num_predict
    if config.llm_enable_thinking is not None:
        kwargs["extra_body"] = {"enable_thinking": config.llm_enable_thinking}
    return partial(openai_complete_if_cache, config.llm_model, **kwargs)


def build_openai_embedding_func(config: BenchConfig) -> EmbeddingFunc:
    from lightrag.llm.openai import openai_embed

    return EmbeddingFunc(
        embedding_dim=config.embedding_dim,
        max_token_size=8192,
        func=partial(
            openai_embed.func,
            model=config.embedding_model,
            base_url=_resolve_host(
                config.embedding_host, "EMBEDDING_BINDING_HOST"
            ),
            api_key=_resolve_api_key("EMBEDDING_BINDING_API_KEY"),
            embedding_dim=config.embedding_dim,
            timeout=config.request_timeout,
        ),
        model_name=config.embedding_model_name,
    )
