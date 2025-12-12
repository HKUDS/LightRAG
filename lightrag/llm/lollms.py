import asyncio
from collections.abc import AsyncIterator

import pipmaster as pm  # Pipmaster for dynamic library install

if not pm.is_installed('aiohttp'):
    pm.install('aiohttp')


import aiohttp
import numpy as np
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from lightrag.exceptions import (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
)
from lightrag.utils import (
    wrap_embedding_func_with_attrs,
)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, APITimeoutError)),
)
async def lollms_model_if_cache(
    model,
    prompt,
    system_prompt=None,
    history_messages=None,
    enable_cot: bool = False,
    base_url='http://localhost:9600',
    **kwargs,
) -> str | AsyncIterator[str]:
    """Client implementation for lollms generation."""
    if history_messages is None:
        history_messages = []
    if enable_cot:
        from lightrag.utils import logger

        logger.debug('enable_cot=True is not supported for lollms and will be ignored.')

    stream = bool(kwargs.get('stream'))
    api_key = kwargs.pop('api_key', None)
    headers = (
        {'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'}
        if api_key
        else {'Content-Type': 'application/json'}
    )

    # Extract lollms specific parameters
    request_data = {
        'prompt': prompt,
        'model_name': model,
        'personality': kwargs.get('personality', -1),
        'n_predict': kwargs.get('n_predict'),
        'stream': stream,
        'temperature': kwargs.get('temperature', 1.0),
        'top_k': kwargs.get('top_k', 50),
        'top_p': kwargs.get('top_p', 0.95),
        'repeat_penalty': kwargs.get('repeat_penalty', 0.8),
        'repeat_last_n': kwargs.get('repeat_last_n', 40),
        'seed': kwargs.get('seed'),
        'n_threads': kwargs.get('n_threads', 8),
    }

    # Prepare the full prompt including history
    prompt_parts = []
    if system_prompt:
        prompt_parts.append(f'{system_prompt}\n')
    for msg in history_messages:
        prompt_parts.append(f'{msg["role"]}: {msg["content"]}\n')
    prompt_parts.append(prompt)
    full_prompt = ''.join(prompt_parts)

    request_data['prompt'] = full_prompt
    timeout = aiohttp.ClientTimeout(total=kwargs.get('timeout', 300))

    async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
        if stream:

            async def inner():
                async with session.post(f'{base_url}/lollms_generate', json=request_data) as response:
                    if response.status < 200 or response.status >= 300:
                        body = await response.text()
                        raise RuntimeError(f'lollms_generate failed: {response.status} {body}')
                    async for line in response.content:
                        yield line.decode().strip()

            return inner()
        else:
            async with session.post(f'{base_url}/lollms_generate', json=request_data) as response:
                if response.status < 200 or response.status >= 300:
                    body = await response.text()
                    raise RuntimeError(f'lollms_generate failed: {response.status} {body}')
                return await response.text()


async def lollms_model_complete(
    prompt,
    system_prompt=None,
    history_messages=None,
    enable_cot: bool = False,
    keyword_extraction=False,
    **kwargs,
) -> str | AsyncIterator[str]:
    """Complete function for lollms model generation."""

    if history_messages is None:
        history_messages = []

    # Get model name from config
    try:
        model_name = kwargs['hashing_kv'].global_config['llm_model_name']
    except (KeyError, AttributeError) as exc:
        raise ValueError('Missing required configuration: hashing_kv.global_config.llm_model_name') from exc

    # If keyword extraction is needed, we might need to modify the prompt
    # or add specific parameters for JSON output (if lollms supports it)
    if keyword_extraction:
        # Note: You might need to adjust this based on how lollms handles structured output
        pass

    return await lollms_model_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        enable_cot=enable_cot,
        **kwargs,
    )


@wrap_embedding_func_with_attrs(embedding_dim=1024, max_token_size=8192)
async def lollms_embed(texts: list[str], embed_model=None, base_url='http://localhost:9600', **kwargs) -> np.ndarray:
    """
    Generate embeddings for a list of texts using lollms server.

    Args:
        texts: List of strings to embed
        embed_model: Model name (not used directly as lollms uses configured vectorizer)
        base_url: URL of the lollms server
        **kwargs: Additional arguments passed to the request

    Returns:
        np.ndarray: Array of embeddings
    """
    api_key = kwargs.pop('api_key', None)
    headers = (
        {'Content-Type': 'application/json', 'Authorization': f'Bearer {api_key}'}
        if api_key
        else {'Content-Type': 'application/json'}
    )
    async with aiohttp.ClientSession(headers=headers) as session:

        async def fetch_embedding(text: str):
            request_data = {'text': text}
            async with session.post(f'{base_url}/lollms_embed', json=request_data) as response:
                if response.status < 200 or response.status >= 300:
                    body = await response.text()
                    raise RuntimeError(f'lollms_embed failed: {response.status} {body}')
                result = await response.json()
                if 'vector' not in result:
                    raise ValueError(f'Unexpected embedding response format: {result}')
                return result['vector']

        embeddings = await asyncio.gather(*[fetch_embedding(text) for text in texts])

        return np.array(embeddings)
