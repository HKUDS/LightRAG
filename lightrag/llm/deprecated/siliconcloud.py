pass
import pipmaster as pm  # Pipmaster for dynamic library install

# install specific modules
if not pm.is_installed('lmdeploy'):
    pm.install('lmdeploy')

import base64
import struct

import aiohttp
import numpy as np
from openai import (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, APITimeoutError)),
)
async def siliconcloud_embedding(
    texts: list[str],
    model: str = 'netease-youdao/bce-embedding-base_v1',
    base_url: str = 'https://api.siliconflow.cn/v1/embeddings',
    max_token_size: int = 8192,
    api_key: str | None = None,
) -> np.ndarray:
    if api_key and not api_key.startswith('Bearer '):
        api_key = 'Bearer ' + api_key

    headers = {'Authorization': api_key, 'Content-Type': 'application/json'}

    truncate_texts = [text[0:max_token_size] for text in texts]

    payload = {'model': model, 'input': truncate_texts, 'encoding_format': 'base64'}

    base64_strings = []
    async with (
        aiohttp.ClientSession() as session,
        session.post(base_url, headers=headers, json=payload) as response,
    ):
        content = await response.json()
        if 'code' in content:
            raise ValueError(content)
        base64_strings = [item['embedding'] for item in content['data']]

    embeddings = []
    for string in base64_strings:
        decode_bytes = base64.b64decode(string)
        n = len(decode_bytes) // 4
        float_array = struct.unpack('<' + 'f' * n, decode_bytes)
        embeddings.append(float_array)
    return np.array(embeddings)
