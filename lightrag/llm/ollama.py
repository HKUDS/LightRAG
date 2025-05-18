import sys

if sys.version_info < (3, 9):
    from typing import AsyncIterator
else:
    from collections.abc import AsyncIterator

import pipmaster as pm  # Pipmaster for dynamic library install

# install specific modules
if not pm.is_installed("ollama"):
    pm.install("ollama")

import ollama

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from lightrag.exceptions import (
    APIConnectionError,
    RateLimitError,
    APITimeoutError,
)
from lightrag.api import __api_version__

import numpy as np
from typing import Union
from lightrag.utils import logger


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError)
    ),
)
async def _ollama_model_if_cache(
    model,
    prompt,
    system_prompt=None,
    history_messages=[],
    **kwargs,
) -> Union[str, AsyncIterator[str]]:
    stream = True if kwargs.get("stream") else False

    kwargs.pop("max_tokens", None)
    # kwargs.pop("response_format", None) # allow json
    host = kwargs.pop("host", None)
    timeout = kwargs.pop("timeout", None) or 300  # Default timeout 300s
    kwargs.pop("hashing_kv", None)
    api_key = kwargs.pop("api_key", None)
    headers = {
        "Content-Type": "application/json",
        "User-Agent": f"LightRAG/{__api_version__}",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    ollama_client = ollama.AsyncClient(host=host, timeout=timeout, headers=headers)

    try:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})

        response = await ollama_client.chat(model=model, messages=messages, **kwargs)
        if stream:
            """cannot cache stream response and process reasoning"""

            async def inner():
                try:
                    async for chunk in response:
                        yield chunk["message"]["content"]
                except Exception as e:
                    logger.error(f"Error in stream response: {str(e)}")
                    raise
                finally:
                    try:
                        await ollama_client._client.aclose()
                        logger.debug("Successfully closed Ollama client for streaming")
                    except Exception as close_error:
                        logger.warning(f"Failed to close Ollama client: {close_error}")

            return inner()
        else:
            model_response = response["message"]["content"]

            """
            If the model also wraps its thoughts in a specific tag,
            this information is not needed for the final
            response and can simply be trimmed.
            """

            return model_response
    except Exception as e:
        try:
            await ollama_client._client.aclose()
            logger.debug("Successfully closed Ollama client after exception")
        except Exception as close_error:
            logger.warning(
                f"Failed to close Ollama client after exception: {close_error}"
            )
        raise e
    finally:
        if not stream:
            try:
                await ollama_client._client.aclose()
                logger.debug(
                    "Successfully closed Ollama client for non-streaming response"
                )
            except Exception as close_error:
                logger.warning(
                    f"Failed to close Ollama client in finally block: {close_error}"
                )


async def ollama_model_complete(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> Union[str, AsyncIterator[str]]:
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    if keyword_extraction:
        kwargs["format"] = "json"
    model_name = kwargs["hashing_kv"].global_config["llm_model_name"]
    return await _ollama_model_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def ollama_embed(texts: list[str], embed_model, **kwargs) -> np.ndarray:
    api_key = kwargs.pop("api_key", None)
    headers = {
        "Content-Type": "application/json",
        "User-Agent": f"LightRAG/{__api_version__}",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    host = kwargs.pop("host", None)
    timeout = kwargs.pop("timeout", None) or 90  # Default time out 90s

    ollama_client = ollama.AsyncClient(host=host, timeout=timeout, headers=headers)

    try:
        data = await ollama_client.embed(model=embed_model, input=texts)
        return np.array(data["embeddings"])
    except Exception as e:
        logger.error(f"Error in ollama_embed: {str(e)}")
        try:
            await ollama_client._client.aclose()
            logger.debug("Successfully closed Ollama client after exception in embed")
        except Exception as close_error:
            logger.warning(
                f"Failed to close Ollama client after exception in embed: {close_error}"
            )
        raise e
    finally:
        try:
            await ollama_client._client.aclose()
            logger.debug("Successfully closed Ollama client after embed")
        except Exception as close_error:
            logger.warning(f"Failed to close Ollama client after embed: {close_error}")
