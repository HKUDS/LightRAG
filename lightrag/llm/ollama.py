from collections.abc import AsyncIterator
import os
import re

import pipmaster as pm

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
from typing import Optional, Union
from lightrag.utils import (
    wrap_embedding_func_with_attrs,
    logger,
)


_OLLAMA_CLOUD_HOST = "https://ollama.com"
_CLOUD_MODEL_SUFFIX_PATTERN = re.compile(r"(?:-cloud|:cloud)$")


def _coerce_host_for_cloud_model(host: Optional[str], model: object) -> Optional[str]:
    if host:
        return host
    try:
        model_name_str = str(model) if model is not None else ""
    except (TypeError, ValueError, AttributeError) as e:
        logger.warning(f"Failed to convert model to string: {e}, using empty string")
        model_name_str = ""
    if _CLOUD_MODEL_SUFFIX_PATTERN.search(model_name_str):
        logger.debug(
            f"Detected cloud model '{model_name_str}', using Ollama Cloud host"
        )
        return _OLLAMA_CLOUD_HOST
    return host


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
    enable_cot: bool = False,
    **kwargs,
) -> Union[str, AsyncIterator[str]]:
    if enable_cot:
        logger.debug("enable_cot=True is not supported for ollama and will be ignored.")
    stream = True if kwargs.get("stream") else False

    kwargs.pop("max_tokens", None)
    # kwargs.pop("response_format", None) # allow json
    host = kwargs.pop("host", None)
    timeout = kwargs.pop("timeout", None)
    if timeout == 0:
        timeout = None
    kwargs.pop("hashing_kv", None)
    api_key = kwargs.pop("api_key", None)
    # fallback to environment variable when not provided explicitly
    if not api_key:
        api_key = os.getenv("OLLAMA_API_KEY")
    headers = {
        "Content-Type": "application/json",
        "User-Agent": f"LightRAG/{__api_version__}",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    host = _coerce_host_for_cloud_model(host, model)

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
    prompt,
    system_prompt=None,
    history_messages=[],
    enable_cot: bool = False,
    keyword_extraction=False,
    **kwargs,
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
        enable_cot=enable_cot,
        **kwargs,
    )


@wrap_embedding_func_with_attrs(embedding_dim=1024, max_token_size=8192)
async def ollama_embed(texts: list[str], embed_model, **kwargs) -> np.ndarray:
    api_key = kwargs.pop("api_key", None)
    if not api_key:
        api_key = os.getenv("OLLAMA_API_KEY")
    headers = {
        "Content-Type": "application/json",
        "User-Agent": f"LightRAG/{__api_version__}",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    host = kwargs.pop("host", None)
    timeout = kwargs.pop("timeout", None)

    host = _coerce_host_for_cloud_model(host, embed_model)

    ollama_client = ollama.AsyncClient(host=host, timeout=timeout, headers=headers)
    try:
        options = kwargs.pop("options", {})
        data = await ollama_client.embed(
            model=embed_model, input=texts, options=options
        )
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
