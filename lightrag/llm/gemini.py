"""
Gemini LLM binding for LightRAG.

This module provides asynchronous helpers that adapt Google's Gemini models
to the same interface used by the rest of the LightRAG LLM bindings. The
implementation mirrors the OpenAI helpers while relying on the official
``google-genai`` client under the hood.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from functools import lru_cache
from typing import Any

import numpy as np
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from lightrag.utils import (
    logger,
    remove_think_tags,
    safe_unicode_decode,
    wrap_embedding_func_with_attrs,
)

import pipmaster as pm

# Install the Google Gemini client and its dependencies on demand
if not pm.is_installed("google-genai"):
    pm.install("google-genai")
if not pm.is_installed("google-api-core"):
    pm.install("google-api-core")

from google import genai  # type: ignore
from google.genai import types  # type: ignore
from google.api_core import exceptions as google_api_exceptions  # type: ignore

DEFAULT_GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com"

LOG = logging.getLogger(__name__)


class InvalidResponseError(Exception):
    """Custom exception class for triggering retry mechanism when Gemini returns empty responses"""

    pass


@lru_cache(maxsize=8)
def _get_gemini_client(
    api_key: str, base_url: str | None, timeout: int | None = None
) -> genai.Client:
    """
    Create (or fetch cached) Gemini client.

    Args:
        api_key: Google Gemini API key.
        base_url: Optional custom API endpoint.
        timeout: Optional request timeout in milliseconds.

    Returns:
        genai.Client: Configured Gemini client instance.
    """
    client_kwargs: dict[str, Any] = {"api_key": api_key}

    if base_url and base_url != DEFAULT_GEMINI_ENDPOINT or timeout is not None:
        try:
            http_options_kwargs = {}
            if base_url and base_url != DEFAULT_GEMINI_ENDPOINT:
                http_options_kwargs["api_endpoint"] = base_url
            if timeout is not None:
                http_options_kwargs["timeout"] = timeout

            client_kwargs["http_options"] = types.HttpOptions(**http_options_kwargs)
        except Exception as exc:  # pragma: no cover - defensive
            LOG.warning("Failed to apply custom Gemini http_options: %s", exc)

    try:
        return genai.Client(**client_kwargs)
    except TypeError:
        # Older google-genai releases don't accept http_options; retry without it.
        client_kwargs.pop("http_options", None)
        return genai.Client(**client_kwargs)


def _ensure_api_key(api_key: str | None) -> str:
    key = api_key or os.getenv("LLM_BINDING_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError(
            "Gemini API key not provided. "
            "Set LLM_BINDING_API_KEY or GEMINI_API_KEY in the environment."
        )
    return key


def _build_generation_config(
    base_config: dict[str, Any] | None,
    system_prompt: str | None,
    keyword_extraction: bool,
) -> types.GenerateContentConfig | None:
    config_data = dict(base_config or {})

    if system_prompt:
        if config_data.get("system_instruction"):
            config_data["system_instruction"] = (
                f"{config_data['system_instruction']}\n{system_prompt}"
            )
        else:
            config_data["system_instruction"] = system_prompt

    if keyword_extraction and not config_data.get("response_mime_type"):
        config_data["response_mime_type"] = "application/json"

    # Remove entries that are explicitly set to None to avoid type errors
    sanitized = {
        key: value
        for key, value in config_data.items()
        if value is not None and value != ""
    }

    if not sanitized:
        return None

    return types.GenerateContentConfig(**sanitized)


def _format_history_messages(history_messages: list[dict[str, Any]] | None) -> str:
    if not history_messages:
        return ""

    history_lines: list[str] = []
    for message in history_messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        history_lines.append(f"[{role}] {content}")

    return "\n".join(history_lines)


def _extract_response_text(
    response: Any, extract_thoughts: bool = False
) -> tuple[str, str]:
    """
    Extract text content from Gemini response, separating regular content from thoughts.

    Args:
        response: Gemini API response object
        extract_thoughts: Whether to extract thought content separately

    Returns:
        Tuple of (regular_text, thought_text)
    """
    candidates = getattr(response, "candidates", None)
    if not candidates:
        return ("", "")

    regular_parts: list[str] = []
    thought_parts: list[str] = []

    for candidate in candidates:
        if not getattr(candidate, "content", None):
            continue
        # Use 'or []' to handle None values from parts attribute
        for part in getattr(candidate.content, "parts", None) or []:
            text = getattr(part, "text", None)
            if not text:
                continue

            # Check if this part is thought content using the 'thought' attribute
            is_thought = getattr(part, "thought", False)

            if is_thought and extract_thoughts:
                thought_parts.append(text)
            elif not is_thought:
                regular_parts.append(text)

    return ("\n".join(regular_parts), "\n".join(thought_parts))


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=(
        retry_if_exception_type(google_api_exceptions.InternalServerError)
        | retry_if_exception_type(google_api_exceptions.ServiceUnavailable)
        | retry_if_exception_type(google_api_exceptions.ResourceExhausted)
        | retry_if_exception_type(google_api_exceptions.GatewayTimeout)
        | retry_if_exception_type(google_api_exceptions.BadGateway)
        | retry_if_exception_type(google_api_exceptions.DeadlineExceeded)
        | retry_if_exception_type(google_api_exceptions.Aborted)
        | retry_if_exception_type(google_api_exceptions.Unknown)
        | retry_if_exception_type(InvalidResponseError)
    ),
)
async def gemini_complete_if_cache(
    model: str,
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    enable_cot: bool = False,
    base_url: str | None = None,
    api_key: str | None = None,
    token_tracker: Any | None = None,
    stream: bool | None = None,
    keyword_extraction: bool = False,
    generation_config: dict[str, Any] | None = None,
    timeout: int | None = None,
    **_: Any,
) -> str | AsyncIterator[str]:
    """
    Complete a prompt using Gemini's API with Chain of Thought (COT) support.

    This function supports automatic integration of reasoning content from Gemini models
    that provide Chain of Thought capabilities via the thinking_config API feature.

    COT Integration:
    - When enable_cot=True: Thought content is wrapped in <think>...</think> tags
    - When enable_cot=False: Thought content is filtered out, only regular content returned
    - Thought content is identified by the 'thought' attribute on response parts
    - Requires thinking_config to be enabled in generation_config for API to return thoughts

    Args:
        model: The Gemini model to use.
        prompt: The prompt to complete.
        system_prompt: Optional system prompt to include.
        history_messages: Optional list of previous messages in the conversation.
        api_key: Optional Gemini API key. If None, uses environment variable.
        base_url: Optional custom API endpoint.
        generation_config: Optional generation configuration dict.
        keyword_extraction: Whether to use JSON response format.
        token_tracker: Optional token usage tracker for monitoring API usage.
        stream: Whether to stream the response.
        hashing_kv: Storage interface (for interface parity with other bindings).
        enable_cot: Whether to include Chain of Thought content in the response.
        timeout: Request timeout in seconds (will be converted to milliseconds for Gemini API).
        **_: Additional keyword arguments (ignored).

    Returns:
        The completed text (with COT content if enable_cot=True) or an async iterator
        of text chunks if streaming. COT content is wrapped in <think>...</think> tags.

    Raises:
        RuntimeError: If the response from Gemini is empty.
        ValueError: If API key is not provided or configured.
    """
    loop = asyncio.get_running_loop()

    key = _ensure_api_key(api_key)
    # Convert timeout from seconds to milliseconds for Gemini API
    timeout_ms = timeout * 1000 if timeout else None
    client = _get_gemini_client(key, base_url, timeout_ms)

    history_block = _format_history_messages(history_messages)
    prompt_sections = []
    if history_block:
        prompt_sections.append(history_block)
    prompt_sections.append(f"[user] {prompt}")
    combined_prompt = "\n".join(prompt_sections)

    config_obj = _build_generation_config(
        generation_config,
        system_prompt=system_prompt,
        keyword_extraction=keyword_extraction,
    )

    request_kwargs: dict[str, Any] = {
        "model": model,
        "contents": [combined_prompt],
    }
    if config_obj is not None:
        request_kwargs["config"] = config_obj

    def _call_model():
        return client.models.generate_content(**request_kwargs)

    if stream:
        queue: asyncio.Queue[Any] = asyncio.Queue()
        usage_container: dict[str, Any] = {}

        def _stream_model() -> None:
            # COT state tracking for streaming
            cot_active = False
            cot_started = False
            initial_content_seen = False

            try:
                stream_kwargs = dict(request_kwargs)
                stream_iterator = client.models.generate_content_stream(**stream_kwargs)
                for chunk in stream_iterator:
                    usage = getattr(chunk, "usage_metadata", None)
                    if usage is not None:
                        usage_container["usage"] = usage

                    # Extract both regular and thought content
                    regular_text, thought_text = _extract_response_text(
                        chunk, extract_thoughts=True
                    )

                    if enable_cot:
                        # Process regular content
                        if regular_text:
                            if not initial_content_seen:
                                initial_content_seen = True

                            # Close COT section if it was active
                            if cot_active:
                                loop.call_soon_threadsafe(queue.put_nowait, "</think>")
                                cot_active = False

                            # Send regular content
                            loop.call_soon_threadsafe(queue.put_nowait, regular_text)

                        # Process thought content
                        if thought_text:
                            if not initial_content_seen and not cot_started:
                                # Start COT section
                                loop.call_soon_threadsafe(queue.put_nowait, "<think>")
                                cot_active = True
                                cot_started = True

                            # Send thought content if COT is active
                            if cot_active:
                                loop.call_soon_threadsafe(
                                    queue.put_nowait, thought_text
                                )
                    else:
                        # COT disabled - only send regular content
                        if regular_text:
                            loop.call_soon_threadsafe(queue.put_nowait, regular_text)

                # Ensure COT is properly closed if still active
                if cot_active:
                    loop.call_soon_threadsafe(queue.put_nowait, "</think>")

                loop.call_soon_threadsafe(queue.put_nowait, None)
            except Exception as exc:  # pragma: no cover - surface runtime issues
                # Try to close COT tag before reporting error
                if cot_active:
                    try:
                        loop.call_soon_threadsafe(queue.put_nowait, "</think>")
                    except Exception:
                        pass
                loop.call_soon_threadsafe(queue.put_nowait, exc)

        loop.run_in_executor(None, _stream_model)

        async def _async_stream() -> AsyncIterator[str]:
            try:
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    if isinstance(item, Exception):
                        raise item

                    chunk_text = str(item)
                    if "\\u" in chunk_text:
                        chunk_text = safe_unicode_decode(chunk_text.encode("utf-8"))

                    # Yield the chunk directly without filtering
                    # COT filtering is already handled in _stream_model()
                    yield chunk_text
            finally:
                usage = usage_container.get("usage")
                if token_tracker and usage:
                    token_tracker.add_usage(
                        {
                            "prompt_tokens": getattr(usage, "prompt_token_count", 0),
                            "completion_tokens": getattr(
                                usage, "candidates_token_count", 0
                            ),
                            "total_tokens": getattr(usage, "total_token_count", 0),
                        }
                    )

        return _async_stream()

    response = await asyncio.to_thread(_call_model)

    # Extract both regular text and thought text
    regular_text, thought_text = _extract_response_text(response, extract_thoughts=True)

    # Apply COT filtering logic based on enable_cot parameter
    if enable_cot:
        # Include thought content wrapped in <think> tags
        if thought_text and thought_text.strip():
            if not regular_text or regular_text.strip() == "":
                # Only thought content available
                final_text = f"<think>{thought_text}</think>"
            else:
                # Both content types present: prepend thought to regular content
                final_text = f"<think>{thought_text}</think>{regular_text}"
        else:
            # No thought content, use regular content only
            final_text = regular_text or ""
    else:
        # Filter out thought content, return only regular content
        final_text = regular_text or ""

    if not final_text:
        raise InvalidResponseError("Gemini response did not contain any text content.")

    if "\\u" in final_text:
        final_text = safe_unicode_decode(final_text.encode("utf-8"))

    final_text = remove_think_tags(final_text)

    usage = getattr(response, "usage_metadata", None)
    if token_tracker and usage:
        token_tracker.add_usage(
            {
                "prompt_tokens": getattr(usage, "prompt_token_count", 0),
                "completion_tokens": getattr(usage, "candidates_token_count", 0),
                "total_tokens": getattr(usage, "total_token_count", 0),
            }
        )

    logger.debug("Gemini response length: %s", len(final_text))
    return final_text


async def gemini_model_complete(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    keyword_extraction: bool = False,
    **kwargs: Any,
) -> str | AsyncIterator[str]:
    hashing_kv = kwargs.get("hashing_kv")
    model_name = None
    if hashing_kv is not None:
        model_name = hashing_kv.global_config.get("llm_model_name")
    if model_name is None:
        model_name = kwargs.pop("model_name", None)
    if model_name is None:
        raise ValueError("Gemini model name not provided in configuration.")

    return await gemini_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        keyword_extraction=keyword_extraction,
        **kwargs,
    )


@wrap_embedding_func_with_attrs(
    embedding_dim=1536, max_token_size=2048, model_name="gemini-embedding-001"
)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=(
        retry_if_exception_type(google_api_exceptions.InternalServerError)
        | retry_if_exception_type(google_api_exceptions.ServiceUnavailable)
        | retry_if_exception_type(google_api_exceptions.ResourceExhausted)
        | retry_if_exception_type(google_api_exceptions.GatewayTimeout)
        | retry_if_exception_type(google_api_exceptions.BadGateway)
        | retry_if_exception_type(google_api_exceptions.DeadlineExceeded)
        | retry_if_exception_type(google_api_exceptions.Aborted)
        | retry_if_exception_type(google_api_exceptions.Unknown)
    ),
)
async def gemini_embed(
    texts: list[str],
    model: str = "gemini-embedding-001",
    base_url: str | None = None,
    api_key: str | None = None,
    embedding_dim: int | None = None,
    task_type: str = "RETRIEVAL_DOCUMENT",
    timeout: int | None = None,
    token_tracker: Any | None = None,
) -> np.ndarray:
    """Generate embeddings for a list of texts using Gemini's API.

    This function uses Google's Gemini embedding model to generate text embeddings.
    It supports dynamic dimension control and automatic normalization for dimensions
    less than 3072.

    Args:
        texts: List of texts to embed.
        model: The Gemini embedding model to use. Default is "gemini-embedding-001".
        base_url: Optional custom API endpoint.
        api_key: Optional Gemini API key. If None, uses environment variables.
        embedding_dim: Optional embedding dimension for dynamic dimension reduction.
            **IMPORTANT**: This parameter is automatically injected by the EmbeddingFunc wrapper.
            Do NOT manually pass this parameter when calling the function directly.
            The dimension is controlled by the @wrap_embedding_func_with_attrs decorator
            or the EMBEDDING_DIM environment variable.
            Supported range: 128-3072. Recommended values: 768, 1536, 3072.
        task_type: Task type for embedding optimization. Default is "RETRIEVAL_DOCUMENT".
            Supported types: SEMANTIC_SIMILARITY, CLASSIFICATION, CLUSTERING,
            RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY, CODE_RETRIEVAL_QUERY,
            QUESTION_ANSWERING, FACT_VERIFICATION.
        timeout: Request timeout in seconds (will be converted to milliseconds for Gemini API).
        token_tracker: Optional token usage tracker for monitoring API usage.

    Returns:
        A numpy array of embeddings, one per input text. For dimensions < 3072,
        the embeddings are L2-normalized to ensure optimal semantic similarity performance.

    Raises:
        ValueError: If API key is not provided or configured.
        RuntimeError: If the response from Gemini is invalid or empty.

    Note:
        - For dimension 3072: Embeddings are already normalized by the API
        - For dimensions < 3072: Embeddings are L2-normalized after retrieval
        - Normalization ensures accurate semantic similarity via cosine distance
    """
    loop = asyncio.get_running_loop()

    key = _ensure_api_key(api_key)
    # Convert timeout from seconds to milliseconds for Gemini API
    timeout_ms = timeout * 1000 if timeout else None
    client = _get_gemini_client(key, base_url, timeout_ms)

    # Prepare embedding configuration
    config_kwargs: dict[str, Any] = {}

    # Add task_type to config
    if task_type:
        config_kwargs["task_type"] = task_type

    # Add output_dimensionality if embedding_dim is provided
    if embedding_dim is not None:
        config_kwargs["output_dimensionality"] = embedding_dim

    # Create config object if we have parameters
    config_obj = types.EmbedContentConfig(**config_kwargs) if config_kwargs else None

    def _call_embed() -> Any:
        """Call Gemini embedding API in executor thread."""
        request_kwargs: dict[str, Any] = {
            "model": model,
            "contents": texts,
        }
        if config_obj is not None:
            request_kwargs["config"] = config_obj

        return client.models.embed_content(**request_kwargs)

    # Execute API call in thread pool
    response = await loop.run_in_executor(None, _call_embed)

    # Extract embeddings from response
    if not hasattr(response, "embeddings") or not response.embeddings:
        raise RuntimeError("Gemini response did not contain embeddings.")

    # Convert embeddings to numpy array
    embeddings = np.array(
        [np.array(e.values, dtype=np.float32) for e in response.embeddings]
    )

    # Apply L2 normalization for dimensions < 3072
    # The 3072 dimension embedding is already normalized by Gemini API
    if embedding_dim and embedding_dim < 3072:
        # Normalize each embedding vector to unit length
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        embeddings = embeddings / norms
        logger.debug(
            f"Applied L2 normalization to {len(embeddings)} embeddings of dimension {embedding_dim}"
        )

    # Track token usage if tracker is provided
    # Note: Gemini embedding API may not provide usage metadata
    if token_tracker and hasattr(response, "usage_metadata"):
        usage = response.usage_metadata
        token_counts = {
            "prompt_tokens": getattr(usage, "prompt_token_count", 0),
            "total_tokens": getattr(usage, "total_token_count", 0),
        }
        token_tracker.add_usage(token_counts)

    logger.debug(
        f"Generated {len(embeddings)} Gemini embeddings with dimension {embeddings.shape[1]}"
    )

    return embeddings


__all__ = [
    "gemini_complete_if_cache",
    "gemini_model_complete",
    "gemini_embed",
]
