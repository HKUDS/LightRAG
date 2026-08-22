from __future__ import annotations

import json
import os
from typing import Any, Union
from collections.abc import AsyncIterator

import httpx
import numpy as np
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from lightrag.utils import (
    logger,
    safe_unicode_decode,
    verbose_debug,
    wrap_embedding_func_with_attrs,
)
from lightrag.api import __api_version__


class CloudflareAPIError(Exception):
    """Generic Cloudflare Workers AI API error."""

    pass


class InvalidResponseError(Exception):
    """Exception raised when Cloudflare returns an invalid response payload."""

    pass


def _get_cloudflare_credentials(
    api_key: str | None = None,
    account_id: str | None = None,
) -> tuple[str, str]:
    """Retrieve and validate Cloudflare API Token and Account ID."""
    token = api_key or os.environ.get("CLOUDFLARE_API_TOKEN") or os.environ.get("CLOUDFLARE_API_KEY")
    acc_id = account_id or os.environ.get("CLOUDFLARE_ACCOUNT_ID")

    if not token:
        raise ValueError(
            "Cloudflare API token is required. Pass api_key or set CLOUDFLARE_API_TOKEN environment variable."
        )
    if not acc_id:
        raise ValueError(
            "Cloudflare Account ID is required. Pass account_id or set CLOUDFLARE_ACCOUNT_ID environment variable."
        )
    return token, acc_id


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
    retry=retry_if_exception_type(
        (
            httpx.HTTPStatusError,
            httpx.RequestError,
            httpx.TimeoutException,
            InvalidResponseError,
        )
    ),
)
async def cloudflare_complete_if_cache(
    model: str = "@cf/meta/llama-3.3-70b-instruct",
    prompt: Union[str, list[dict[str, str]]] = "",
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    enable_cot: bool = False,
    api_key: str | None = None,
    account_id: str | None = None,
    base_url: str | None = None,
    **kwargs: Any,
) -> Union[str, AsyncIterator[str]]:
    """Call Cloudflare Workers AI text generation endpoint with LightRAG interface.

    Args:
        model: Cloudflare model identifier (e.g. '@cf/meta/llama-3.3-70b-instruct').
        prompt: User query string or message history.
        system_prompt: Optional system prompt to instruct the model.
        history_messages: Optional prior conversation history.
        enable_cot: Whether to include reasoning trace if present.
        api_key: Cloudflare API token.
        account_id: Cloudflare account identifier.
        base_url: Optional custom base URL.
        **kwargs: Additional generation parameters (e.g. temperature, max_tokens, stream).

    Returns:
        Generated text string, or an async iterator of text chunks if stream=True.
    """
    token, acc_id = _get_cloudflare_credentials(api_key, account_id)
    endpoint = base_url or f"https://api.cloudflare.com/client/v4/accounts/{acc_id}/ai/run"
    url = f"{endpoint.rstrip('/')}/{model.lstrip('/')}"

    # Prepare messages
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)

    if isinstance(prompt, str):
        if prompt:
            messages.append({"role": "user", "content": prompt})
    elif isinstance(prompt, list):
        messages.extend(prompt)

    # Clean kwargs and assemble payload
    kwargs.pop("hashing_kv", None)
    kwargs.pop("keyword_extraction", None)
    kwargs.pop("entity_extraction", None)
    stream = kwargs.pop("stream", False)

    payload: dict[str, Any] = {
        "messages": messages,
        "stream": stream,
    }
    for key in ("max_tokens", "temperature", "top_p", "top_k", "seed", "repetition_penalty"):
        if key in kwargs and kwargs[key] is not None:
            payload[key] = kwargs[key]

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "User-Agent": f"LightRAG/{__api_version__}",
    }

    timeout = httpx.Timeout(60.0, connect=10.0)

    logger.debug(f"===== Query Input to Cloudflare Workers AI ({model}) =====")
    verbose_debug(f"Payload: {payload}")

    if stream:
        async def inner_stream() -> AsyncIterator[str]:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream("POST", url, headers=headers, json=payload) as response:
                    if response.status_code != 200:
                        content = await response.aread()
                        raise httpx.HTTPStatusError(
                            f"Cloudflare stream error {response.status_code}: {content.decode('utf-8', errors='replace')}",
                            request=response.request,
                            response=response,
                        )
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        if line.startswith("data:"):
                            data_str = line[len("data:"):].strip()
                            if data_str == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                chunk = data.get("response", "")
                                if chunk:
                                    if r"\u" in chunk:
                                        chunk = safe_unicode_decode(chunk.encode("utf-8"))
                                    yield chunk
                            except json.JSONDecodeError:
                                continue

        return inner_stream()

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, headers=headers, json=payload)
        if response.status_code != 200:
            raise httpx.HTTPStatusError(
                f"Cloudflare API error {response.status_code}: {response.text}",
                request=response.request,
                response=response,
            )

        resp_json = response.json()
        if not resp_json.get("success", False) and "result" not in resp_json:
            errors = resp_json.get("errors", [])
            raise CloudflareAPIError(f"Cloudflare API returned error: {errors}")

        result = resp_json.get("result", {})
        content = result.get("response", "")
        if not content and isinstance(result, str):
            content = result

        if not content:
            raise InvalidResponseError(f"Empty response from Cloudflare model {model}")

        if r"\u" in content:
            content = safe_unicode_decode(content.encode("utf-8"))

        return content


@wrap_embedding_func_with_attrs(
    embedding_dim=1024,
    max_token_size=8192,
    supports_asymmetric=False,
    model_name="@cf/baai/bge-m3",
)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
    retry=retry_if_exception_type(
        (
            httpx.HTTPStatusError,
            httpx.RequestError,
            httpx.TimeoutException,
            InvalidResponseError,
        )
    ),
)
async def cloudflare_embed(
    texts: list[str],
    model: str = "@cf/baai/bge-m3",
    api_key: str | None = None,
    account_id: str | None = None,
    base_url: str | None = None,
    **kwargs: Any,
) -> np.ndarray:
    """Generate embeddings for a list of texts using Cloudflare Workers AI.

    Args:
        texts: List of text strings to embed.
        model: Cloudflare embedding model identifier (default: '@cf/baai/bge-m3').
        api_key: Cloudflare API token.
        account_id: Cloudflare account identifier.
        base_url: Optional custom base URL.
        **kwargs: Additional parameters.

    Returns:
        numpy.ndarray: 2D array of shape (len(texts), embedding_dim).
    """
    if not texts:
        return np.empty((0, 1024), dtype=np.float32)

    token, acc_id = _get_cloudflare_credentials(api_key, account_id)
    endpoint = base_url or f"https://api.cloudflare.com/client/v4/accounts/{acc_id}/ai/run"
    url = f"{endpoint.rstrip('/')}/{model.lstrip('/')}"

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "User-Agent": f"LightRAG/{__api_version__}",
    }

    payload = {"text": texts}
    timeout = httpx.Timeout(60.0, connect=10.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, headers=headers, json=payload)
        if response.status_code != 200:
            raise httpx.HTTPStatusError(
                f"Cloudflare embedding error {response.status_code}: {response.text}",
                request=response.request,
                response=response,
            )

        resp_json = response.json()
        if not resp_json.get("success", False) and "result" not in resp_json:
            errors = resp_json.get("errors", [])
            raise CloudflareAPIError(f"Cloudflare embedding API returned error: {errors}")

        result = resp_json.get("result", {})
        data = result.get("data")
        if data is None:
            raise InvalidResponseError(f"Cloudflare embedding response missing 'data' field: {resp_json}")

        return np.array(data, dtype=np.float32)
