from collections.abc import Iterable
import os
import pipmaster as pm  # Pipmaster for dynamic library install

# install specific modules
if not pm.is_installed("openai"):
    pm.install("openai")

from openai import (
    AsyncAzureOpenAI,
    APIConnectionError,
    RateLimitError,
    APITimeoutError,
)
from openai.types.chat import ChatCompletionMessageParam

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from lightrag.utils import (
    wrap_embedding_func_with_attrs,
    safe_unicode_decode,
    logger,
)

import numpy as np


def _normalize_openai_kwargs_for_model(model: str, kwargs: dict) -> None:
    """
    Normalize OpenAI API parameters based on the model being used.
    
    This function handles model-specific parameter requirements:
    - gpt-5-nano uses 'max_completion_tokens' instead of 'max_tokens'
    - gpt-5-nano uses reasoning tokens which consume from the token budget
    - gpt-5-nano doesn't support custom temperature values
    - Other models support both parameters
    
    Args:
        model: The model name (e.g., 'gpt-5-nano', 'gpt-4o', 'gpt-4o-mini')
        kwargs: The API parameters dict to normalize (modified in-place)
    """
    # Handle max_tokens vs max_completion_tokens conversion for gpt-5 models
    if model.startswith("gpt-5"):
        # gpt-5-nano and variants use max_completion_tokens
        if "max_tokens" in kwargs and "max_completion_tokens" not in kwargs:
            # If only max_tokens is set, move it to max_completion_tokens
            max_tokens = kwargs.pop("max_tokens")
            # For gpt-5-nano, we need to account for reasoning tokens
            # Increase buffer to ensure actual content is generated
            # Reasoning typically uses 1.5-2x the actual content tokens needed
            kwargs["max_completion_tokens"] = int(max(max_tokens * 2.5, 300))
        else:
            # If both are set, remove max_tokens (it's not supported)
            max_tokens = kwargs.pop("max_tokens", None)
            if max_tokens and "max_completion_tokens" in kwargs:
                # If max_completion_tokens is already set and seems too small, increase it
                if kwargs["max_completion_tokens"] < 300:
                    kwargs["max_completion_tokens"] = int(max(kwargs["max_completion_tokens"] * 2.5, 300))
        
        # Ensure a minimum token budget for gpt-5-nano due to reasoning overhead
        if "max_completion_tokens" in kwargs:
            if kwargs["max_completion_tokens"] < 300:
                # Minimum 300 tokens to account for reasoning (reasoning can be expensive)
                original = kwargs["max_completion_tokens"]
                kwargs["max_completion_tokens"] = 300
                logger.debug(f"Increased max_completion_tokens from {original} to 300 for {model} (reasoning overhead)")
    
    # Handle temperature constraint for gpt-5 models
    if model.startswith("gpt-5"):
        # gpt-5-nano requires default temperature (doesn't support custom values)
        # Remove any custom temperature setting
        if "temperature" in kwargs:
            kwargs.pop("temperature")
            logger.debug(f"Removed custom temperature for {model}: uses default")
    
    logger.debug(f"Normalized parameters for {model}: {kwargs}")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APIConnectionError)
    ),
)
async def azure_openai_complete_if_cache(
    model,
    prompt,
    system_prompt: str | None = None,
    history_messages: Iterable[ChatCompletionMessageParam] | None = None,
    enable_cot: bool = False,
    base_url: str | None = None,
    api_key: str | None = None,
    api_version: str | None = None,
    **kwargs,
):
    if enable_cot:
        logger.debug(
            "enable_cot=True is not supported for the Azure OpenAI API and will be ignored."
        )
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT") or model or os.getenv("LLM_MODEL")
    base_url = (
        base_url or os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("LLM_BINDING_HOST")
    )
    api_key = (
        api_key or os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("LLM_BINDING_API_KEY")
    )
    api_version = (
        api_version
        or os.getenv("AZURE_OPENAI_API_VERSION")
        or os.getenv("OPENAI_API_VERSION")
    )

    kwargs.pop("hashing_kv", None)
    kwargs.pop("keyword_extraction", None)
    timeout = kwargs.pop("timeout", None)

    openai_async_client = AsyncAzureOpenAI(
        azure_endpoint=base_url,
        azure_deployment=deployment,
        api_key=api_key,
        api_version=api_version,
        timeout=timeout,
    )
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    if prompt is not None:
        messages.append({"role": "user", "content": prompt})

    # Normalize API parameters based on model requirements
    _normalize_openai_kwargs_for_model(model, kwargs)

    if "response_format" in kwargs:
        response = await openai_async_client.beta.chat.completions.parse(
            model=model, messages=messages, **kwargs
        )
    else:
        response = await openai_async_client.chat.completions.create(
            model=model, messages=messages, **kwargs
        )

    if hasattr(response, "__aiter__"):

        async def inner():
            async for chunk in response:
                if len(chunk.choices) == 0:
                    continue
                content = chunk.choices[0].delta.content
                if content is None:
                    continue
                if r"\u" in content:
                    content = safe_unicode_decode(content.encode("utf-8"))
                yield content

        return inner()
    else:
        content = response.choices[0].message.content
        if r"\u" in content:
            content = safe_unicode_decode(content.encode("utf-8"))
        return content


async def azure_openai_complete(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    kwargs.pop("keyword_extraction", None)
    result = await azure_openai_complete_if_cache(
        os.getenv("LLM_MODEL", "gpt-4o-mini"),
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )
    return result


@wrap_embedding_func_with_attrs(embedding_dim=1536)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError)
    ),
)
async def azure_openai_embed(
    texts: list[str],
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    api_version: str | None = None,
) -> np.ndarray:
    deployment = (
        os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
        or model
        or os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    )
    base_url = (
        base_url
        or os.getenv("AZURE_EMBEDDING_ENDPOINT")
        or os.getenv("EMBEDDING_BINDING_HOST")
    )
    api_key = (
        api_key
        or os.getenv("AZURE_EMBEDDING_API_KEY")
        or os.getenv("EMBEDDING_BINDING_API_KEY")
    )
    api_version = (
        api_version
        or os.getenv("AZURE_EMBEDDING_API_VERSION")
        or os.getenv("OPENAI_API_VERSION")
    )

    openai_async_client = AsyncAzureOpenAI(
        azure_endpoint=base_url,
        azure_deployment=deployment,
        api_key=api_key,
        api_version=api_version,
    )

    response = await openai_async_client.embeddings.create(
        model=model, input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])
