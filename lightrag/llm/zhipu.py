import asyncio
import os
import sys
import warnings
from ..utils import verbose_debug

if sys.version_info < (3, 9):
    pass
else:
    pass
import pipmaster as pm  # Pipmaster for dynamic library install

# install specific modules
if not pm.is_installed("zai-sdk"):
    pm.install("zai-sdk")

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
)

from lightrag.utils import (
    TruncatedResponse,
    wrap_embedding_func_with_attrs,
    logger,
)

import numpy as np
from typing import Any, Union, List, Optional, Dict


def _build_client(client_cls, api_key: Optional[str]):
    """Build a zai-sdk client whose retries are owned by the tenacity decorators.

    zai-sdk reads ZAI_* variables; the old zhipuai ones are still honored.
    ``max_retries=0`` turns off the SDK's own retry loop, which would otherwise
    multiply with the decorators (3 SDK attempts per decorator attempt).
    """
    kwargs = {"max_retries": 0}
    api_key = (
        api_key or os.environ.get("ZAI_API_KEY") or os.environ.get("ZHIPUAI_API_KEY")
    )
    if api_key:
        kwargs["api_key"] = api_key
    if not os.environ.get("ZAI_BASE_URL"):
        base_url = os.environ.get("ZHIPUAI_BASE_URL")
        if base_url:
            kwargs["base_url"] = base_url
    return client_cls(**kwargs)


# The statuses zai-sdk's own ``_should_retry`` retries; with that loop turned
# off (see _build_client) the decorators have to cover them: 408 request
# timeout, 409 lock timeout, 429 rate limit, and every 5xx.
_RETRYABLE_STATUS_CODES = frozenset({408, 409, 429})


def _is_retryable_zhipu_error(error: BaseException) -> bool:
    """tenacity predicate: retry zai-sdk connection errors and transient statuses.

    zai-sdk raises its own exception types, never openai's. ``APITimeoutError``
    subclasses ``APIConnectionError``; ``APIConnectionError`` is exported only
    from ``zai.core._errors``, not from ``zai.core``.
    """
    try:
        from zai.core._errors import APIConnectionError, APIStatusError
    except ImportError:
        return False
    if isinstance(error, APIConnectionError):
        return True
    if isinstance(error, APIStatusError):
        status_code = getattr(error, "status_code", None)
        return isinstance(status_code, int) and (
            status_code in _RETRYABLE_STATUS_CODES or status_code >= 500
        )
    return False


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception(_is_retryable_zhipu_error),
    reraise=True,
)
async def zhipu_complete_if_cache(
    prompt: Union[str, List[Dict[str, str]]],
    model: str = "glm-4-flashx",  # The most cost/performance balance model in glm-4 series
    api_key: Optional[str] = None,
    system_prompt: Optional[str] = None,
    history_messages: List[Dict[str, str]] = [],
    enable_cot: bool = False,  # LightRAG output switch: include reasoning_content as <think>...</think>
    thinking: Optional[
        Dict[str, object]
    ] = None,  # Zhipu request param: use {"type": "enabled"} to enable thinking
    token_tracker: Any | None = None,
    **kwargs,
) -> str:
    """Call Zhipu chat completions with optional official thinking support.

    Parameter roles:
    - `thinking`: forwarded to the Zhipu API as-is. To enable thinking output,
      pass a config such as `{"type": "enabled"}`.
    - `enable_cot`: LightRAG-only formatting switch. When True and the API
      returns `reasoning_content`, it is preserved in the final string as
      `<think>...</think>`.
    - `response_format`: forwarded as Zhipu's OpenAI-compatible structured
      output parameter when supplied by callers.
    - `token_tracker`: optional token usage tracker, recorded from the
      response's `usage` field.
    - Deprecated `keyword_extraction` and `entity_extraction` booleans are
      compatibility shims; when no explicit `response_format` is supplied,
      they are mapped to `{"type": "json_object"}`.
    """
    # dynamically load ZhipuAI
    try:
        from zai import ZhipuAiClient
    except ImportError:
        raise ImportError("Please install zai-sdk before initialize zhipuai backend.")

    client = _build_client(ZhipuAiClient, api_key)

    messages = []

    if not system_prompt:
        system_prompt = "You are a helpful assistant. Note that sensitive words in the content should be replaced with ***"

    # Add system prompt if provided
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    # Add debug logging
    logger.debug("===== Query Input to LLM =====")
    logger.debug(f"Query: {prompt}")
    verbose_debug(f"System prompt: {system_prompt}")

    # Deprecation shims: map legacy extraction booleans to response_format only
    # when an explicit response_format was not supplied by the caller. The
    # legacy path also forces enable_cot=False so reasoning_content cannot
    # corrupt the JSON payload expected by callers relying on it.
    keyword_extraction = kwargs.pop("keyword_extraction", False)
    entity_extraction = kwargs.pop("entity_extraction", False)
    if kwargs.get("response_format") is None:
        if entity_extraction:
            warnings.warn(
                "zhipu_complete_if_cache(entity_extraction=True) is deprecated; "
                "pass response_format={'type': 'json_object'} instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            kwargs["response_format"] = {"type": "json_object"}
            enable_cot = False
        elif keyword_extraction:
            warnings.warn(
                "zhipu_complete_if_cache(keyword_extraction=True) is deprecated; "
                "pass response_format={'type': 'json_object'} instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            kwargs["response_format"] = {"type": "json_object"}
            enable_cot = False

    # Structured output and COT are mutually exclusive here because
    # reasoning_content would corrupt the JSON payload expected by callers.
    if kwargs.get("response_format") is not None:
        enable_cot = False

    # Remove unsupported kwargs
    kwargs = {
        k: v
        for k, v in kwargs.items()
        if k not in ["hashing_kv", "keyword_extraction", "entity_extraction"]
    }
    # `thinking` is an official Zhipu request field. Example:
    # {"type": "enabled"} enables reasoning output on supported models.
    if thinking is not None:
        kwargs["thinking"] = thinking

    # ZhipuAI's client wraps a synchronous httpx.Client, not an async one --
    # calling it directly here would block the whole event loop for the
    # duration of the HTTP request, stalling every other concurrent task
    # (other LLM calls, embeddings, storage I/O) sharing the loop.
    #
    # Cancelling this await (e.g. an outer execution timeout) only cancels
    # the asyncio wrapper: CPython cannot forcibly stop a running thread, so
    # the HTTP call keeps running in the background until it finishes on its
    # own. This is an inherent limit of bridging a synchronous SDK through
    # asyncio.to_thread, not something fixable at this call site.
    try:
        response = await asyncio.to_thread(
            client.chat.completions.create, model=model, messages=messages, **kwargs
        )
    except asyncio.CancelledError:
        logger.warning(
            "zhipu_complete_if_cache: cancelled while awaiting the SDK call; "
            "the request keeps running in the background thread until it "
            "completes on its own"
        )
        raise

    # Record usage before extracting content: the API call is already billed
    # and response.usage is already populated at this point, regardless of
    # whether the response has a usable choice below.
    if token_tracker and getattr(response, "usage", None):
        token_tracker.add_usage(
            {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                "total_tokens": getattr(response.usage, "total_tokens", 0),
            }
        )

    if not response.choices or response.choices[0].message is None:
        return ""
    choice = response.choices[0]
    message = choice.message
    content = message.content or ""
    reasoning_content = getattr(message, "reasoning_content", "") or ""

    if enable_cot and reasoning_content.strip():
        if content:
            content = f"<think>{reasoning_content}</think>{content}"
        else:
            content = f"<think>{reasoning_content}</think>"

    if getattr(choice, "finish_reason", None) == "length":
        return TruncatedResponse(content)
    return content


async def zhipu_complete(
    prompt,
    system_prompt=None,
    history_messages=[],
    keyword_extraction=False,
    entity_extraction=False,
    enable_cot: bool = False,
    **kwargs,
):
    """Zhipu completion wrapper with LightRAG structured-output shims.

    Structured output note:
    - This adapter accepts OpenAI-style ``response_format`` and forwards it to
      Zhipu's compatible chat-completions API.
    - Deprecated ``keyword_extraction`` and ``entity_extraction`` booleans are
      compatibility shims; when no explicit ``response_format`` is supplied,
      they are mapped to ``{"type": "json_object"}``.
    """
    # Pop legacy extraction flags from kwargs to avoid passing them downstream.
    keyword_extraction = kwargs.pop("keyword_extraction", keyword_extraction)
    entity_extraction = kwargs.pop("entity_extraction", entity_extraction)

    # Deprecation shims: map legacy boolean flags to response_format only when
    # an explicit response_format was not supplied by the caller. The legacy
    # path also forces enable_cot=False so that reasoning_content cannot
    # corrupt the JSON payload expected by callers that were relying on it.
    if kwargs.get("response_format") is None:
        if entity_extraction:
            warnings.warn(
                "zhipu_complete(entity_extraction=True) is deprecated; "
                "pass response_format={'type': 'json_object'} instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            kwargs["response_format"] = {"type": "json_object"}
            enable_cot = False
        elif keyword_extraction:
            warnings.warn(
                "zhipu_complete(keyword_extraction=True) is deprecated; "
                "pass response_format={'type': 'json_object'} instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            kwargs["response_format"] = {"type": "json_object"}
            enable_cot = False

    return await zhipu_complete_if_cache(
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        enable_cot=enable_cot,
        **kwargs,
    )


@wrap_embedding_func_with_attrs(
    embedding_dim=1024, max_token_size=8192, model_name="embedding-3"
)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception(_is_retryable_zhipu_error),
    reraise=True,
)
async def zhipu_embedding(
    texts: list[str],
    model: str = "embedding-3",
    api_key: str = None,
    embedding_dim: int | None = None,
    **kwargs,
) -> np.ndarray:
    # dynamically load ZhipuAI
    try:
        from zai import ZhipuAiClient
    except ImportError:
        raise ImportError("Please install zai-sdk before initialize zhipuai backend.")
    client = _build_client(ZhipuAiClient, api_key)

    # Convert single text to list if needed
    if isinstance(texts, str):
        texts = [texts]

    embeddings = []
    for text in texts:
        try:
            request_kwargs = dict(kwargs)
            if embedding_dim is not None:
                request_kwargs["dimensions"] = embedding_dim
            # Same blocking-client concern as zhipu_complete_if_cache, same
            # cancellation caveat: run each synchronous HTTP call off the
            # event loop thread, but a timeout here cannot stop it early.
            response = await asyncio.to_thread(
                client.embeddings.create,
                model=model,
                input=[text],
                **request_kwargs,
            )
            embeddings.append(response.data[0].embedding)
        except Exception as e:
            # A retryable error must reach the decorator with its own type;
            # wrapping it in a bare Exception would make it unretryable.
            if _is_retryable_zhipu_error(e):
                raise
            raise Exception(f"Error calling ChatGLM Embedding API: {str(e)}") from e

    return np.array(embeddings)
