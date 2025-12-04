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


async def ollama_embed(texts: list[str], embed_model, **kwargs) -> np.ndarray:
    """
    Generate embeddings using Ollama API.
    
    Uses httpx directly instead of ollama.AsyncClient to work around a bug in ollama SDK v0.6.1
    where the host parameter is not properly used for the embed endpoint.
    """
    import httpx
    import json
    
    api_key = kwargs.pop("api_key", None)
    headers = {
        "Content-Type": "application/json",
        "User-Agent": f"LightRAG/{__api_version__}",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    host = kwargs.pop("host", None)
    timeout = kwargs.pop("timeout", None)
    
    # Ensure host has proper format
    if host and not host.startswith("http"):
        host = f"http://{host}"
    if not host:
        host = "http://localhost:11434"
    
    # Validate host format to catch any corruption
    if not isinstance(host, str) or not host.startswith("http"):
        logger.error(f"Invalid host format for Ollama embed: {host} (type: {type(host).__name__})")
        raise ValueError(f"Invalid host format for Ollama: {host}")

    logger.info(f"Ollama embed called with host: {host}, model: {embed_model}")

    # Use httpx directly to avoid ollama SDK bug with embed endpoint
    async with httpx.AsyncClient(timeout=timeout if timeout else 120.0) as client:
        try:
            options = kwargs.pop("options", {})
            
            # Construct the embed API endpoint
            embed_url = f"{host}/api/embed"
            
            # Prepare request payload
            payload = {
                "model": embed_model,
                "input": texts,
            }
            if options:
                payload["options"] = options
            
            logger.debug(f"Sending embed request to {embed_url}")
            
            # Make the request
            response = await client.post(
                embed_url,
                json=payload,
                headers=headers
            )
            
            # Check for errors
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            
            if "embeddings" not in data:
                raise ValueError(f"Invalid response from Ollama: {data}")
            
            return np.array(data["embeddings"])
            
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP error from Ollama: {e.response.status_code} - {e.response.text}"
            logger.error(error_msg)
            raise Exception(error_msg) from e
        except httpx.RequestError as e:
            error_msg = f"Connection error to Ollama at {host}: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg) from e
        except Exception as e:
            logger.error(f"Error in ollama_embed: {str(e)}")
            raise
