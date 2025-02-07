"""
Ollama LLM Interface Module
==========================

This module provides interfaces for interacting with Ollama's language models,
including text generation and embedding capabilities.

Author: Lightrag team
Created: 2024-01-24
License: MIT License

Copyright (c) 2024 Lightrag

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

Version: 1.0.0

Change Log:
- 1.0.0 (2024-01-24): Initial release
    * Added async chat completion support
    * Added embedding generation
    * Added stream response capability

Dependencies:
    - ollama
    - numpy
    - pipmaster
    - Python >= 3.10

Usage:
    from llm_interfaces.ollama_interface import ollama_model_complete, ollama_embed
"""

__version__ = "1.0.0"
__author__ = "lightrag Team"
__status__ = "Production"

import sys

if sys.version_info < (3, 9):
    from typing import AsyncIterator
else:
    from collections.abc import AsyncIterator
import pipmaster as pm  # Pipmaster for dynamic library install

# install specific modules
if not pm.is_installed("ollama"):
    pm.install("ollama")
if not pm.is_installed("tenacity"):
    pm.install("tenacity")

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
from lightrag.utils import extract_reasoning
import numpy as np
from typing import Union


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError)
    ),
)
async def ollama_model_if_cache(
    model,
    prompt,
    system_prompt=None,
    history_messages=[],
    **kwargs,
) -> Union[str, AsyncIterator[str]]:
    stream = True if kwargs.get("stream") else False
    reasoning_tag = kwargs.pop("reasoning_tag", None)
    kwargs.pop("max_tokens", None)
    # kwargs.pop("response_format", None) # allow json
    host = kwargs.pop("host", None)
    timeout = kwargs.pop("timeout", None)
    kwargs.pop("hashing_kv", None)
    api_key = kwargs.pop("api_key", None)
    headers = {
        "Content-Type": "application/json",
        "User-Agent": f"LightRAG/{__api_version__}",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    ollama_client = ollama.AsyncClient(host=host, timeout=timeout, headers=headers)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    response = await ollama_client.chat(model=model, messages=messages, **kwargs)
    if stream:
        """cannot cache stream response and process reasoning"""

        async def inner():
            async for chunk in response:
                yield chunk["message"]["content"]

        return inner()
    else:
        model_response = response["message"]["content"]

        """
        If the model also wraps its thoughts in a specific tag,
        this information is not needed for the final
        response and can simply be trimmed.
        """

        return (
            model_response
            if reasoning_tag is None
            else extract_reasoning(model_response, reasoning_tag).response_content
        )


async def ollama_model_complete(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> Union[str, AsyncIterator[str]]:
    keyword_extraction = kwargs.pop("keyword_extraction", None)
    if keyword_extraction:
        kwargs["format"] = "json"
    model_name = kwargs["hashing_kv"].global_config["llm_model_name"]
    return await ollama_model_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def ollama_embedding(texts: list[str], embed_model, **kwargs) -> np.ndarray:
    """
    Deprecated in favor of `embed`.
    """
    embed_text = []
    ollama_client = ollama.Client(**kwargs)
    for text in texts:
        data = ollama_client.embeddings(model=embed_model, prompt=text)
        embed_text.append(data["embedding"])

    return embed_text


async def ollama_embed(texts: list[str], embed_model, **kwargs) -> np.ndarray:
    api_key = kwargs.pop("api_key", None)
    headers = {
        "Content-Type": "application/json",
        "User-Agent": f"LightRAG/{__api_version__}",
    }
    if api_key:
        headers["Authorization"] = api_key
    kwargs["headers"] = headers
    ollama_client = ollama.Client(**kwargs)
    data = ollama_client.embed(model=embed_model, input=texts)
    return data["embeddings"]
