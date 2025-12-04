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

from lightrag.utils import logger, remove_think_tags, safe_unicode_decode

import pipmaster as pm

# Install the Google Gemini client on demand
if not pm.is_installed("google-genai"):
    pm.install("google-genai")

from google import genai  # type: ignore
from google.genai import types  # type: ignore

DEFAULT_GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com"

LOG = logging.getLogger(__name__)


@lru_cache(maxsize=8)
def _get_gemini_client(api_key: str, base_url: str | None) -> genai.Client:
    """
    Create (or fetch cached) Gemini client.

    Args:
        api_key: Google Gemini API key.
        base_url: Optional custom API endpoint.

    Returns:
        genai.Client: Configured Gemini client instance.
    """
    client_kwargs: dict[str, Any] = {"api_key": api_key}

    if base_url and base_url != DEFAULT_GEMINI_ENDPOINT:
        try:
            client_kwargs["http_options"] = types.HttpOptions(api_endpoint=base_url)
        except Exception as exc:  # pragma: no cover - defensive
            LOG.warning("Failed to apply custom Gemini endpoint %s: %s", base_url, exc)

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


def _extract_response_text(response: Any) -> str:
    if getattr(response, "text", None):
        return response.text

    candidates = getattr(response, "candidates", None)
    if not candidates:
        return ""

    parts: list[str] = []
    for candidate in candidates:
        if not getattr(candidate, "content", None):
            continue
        for part in getattr(candidate.content, "parts", []):
            text = getattr(part, "text", None)
            if text:
                parts.append(text)

    return "\n".join(parts)


async def gemini_complete_if_cache(
    model: str,
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    generation_config: dict[str, Any] | None = None,
    keyword_extraction: bool = False,
    token_tracker: Any | None = None,
    hashing_kv: Any | None = None,  # noqa: ARG001 - present for interface parity
    stream: bool | None = None,
    enable_cot: bool = False,  # noqa: ARG001 - not supported by Gemini currently
    timeout: float | None = None,  # noqa: ARG001 - handled by caller if needed
    **_: Any,
) -> str | AsyncIterator[str]:
    loop = asyncio.get_running_loop()

    key = _ensure_api_key(api_key)
    client = _get_gemini_client(key, base_url)

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
            try:
                stream_kwargs = dict(request_kwargs)
                stream_iterator = client.models.generate_content_stream(**stream_kwargs)
                for chunk in stream_iterator:
                    usage = getattr(chunk, "usage_metadata", None)
                    if usage is not None:
                        usage_container["usage"] = usage
                    text_piece = getattr(chunk, "text", None) or _extract_response_text(
                        chunk
                    )
                    if text_piece:
                        loop.call_soon_threadsafe(queue.put_nowait, text_piece)
                loop.call_soon_threadsafe(queue.put_nowait, None)
            except Exception as exc:  # pragma: no cover - surface runtime issues
                loop.call_soon_threadsafe(queue.put_nowait, exc)

        loop.run_in_executor(None, _stream_model)

        async def _async_stream() -> AsyncIterator[str]:
            accumulated = ""
            emitted = ""
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

                    accumulated += chunk_text
                    sanitized = remove_think_tags(accumulated)
                    if sanitized.startswith(emitted):
                        delta = sanitized[len(emitted) :]
                    else:
                        delta = sanitized
                    emitted = sanitized

                    if delta:
                        yield delta
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

    text = _extract_response_text(response)
    if not text:
        raise RuntimeError("Gemini response did not contain any text content.")

    if "\\u" in text:
        text = safe_unicode_decode(text.encode("utf-8"))

    text = remove_think_tags(text)

    usage = getattr(response, "usage_metadata", None)
    if token_tracker and usage:
        token_tracker.add_usage(
            {
                "prompt_tokens": getattr(usage, "prompt_token_count", 0),
                "completion_tokens": getattr(usage, "candidates_token_count", 0),
                "total_tokens": getattr(usage, "total_token_count", 0),
            }
        )

    logger.debug("Gemini response length: %s", len(text))
    return text


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


__all__ = [
    "gemini_complete_if_cache",
    "gemini_model_complete",
]
