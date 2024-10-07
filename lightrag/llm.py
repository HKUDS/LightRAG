import os
import numpy as np
from openai import AsyncOpenAI, APIConnectionError, RateLimitError, Timeout
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from .base import BaseKVStorage
from .utils import compute_args_hash, wrap_embedding_func_with_attrs

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)
async def openai_complete_if_cache(
    model, prompt, api_key='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'
, system_prompt=None, history_messages=[], **kwargs
) -> str:
    openai_async_client = AsyncOpenAI(api_key=api_key)
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]

    response = await openai_async_client.chat.completions.create(
        model=model, messages=messages, **kwargs
    )

    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": model}}
        )
    return response.choices[0].message.content

async def gpt_4o_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        "gpt-4o",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def gpt_4o_mini_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        "gpt-4o-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )

@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)
async def openai_embedding(texts: list[str]) -> np.ndarray:
    api_key = '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'
    openai_async_client = AsyncOpenAI(api_key=api_key)
    response = await openai_async_client.embeddings.create(
        model="text-embedding-3-small", input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])

async def moonshot_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        "moonshot-v1-128k",
        prompt,
        api_key='sk-OsvLvHgFFH3tz6Yhym3OAhcTfZ9y7rHEgQ3JDLmnuLpTw9C0',
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )

if __name__ == "__main__":
    import asyncio

    async def main():
        result = await gpt_4o_mini_complete('How are you?')
        print(result)

    asyncio.run(main())
