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

from lightrag.utils import logger


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
    encoding_format: str = 'base64',
) -> np.ndarray:
    logger.debug(f'siliconcloud_embedding called with {len(texts)} texts, model={model}, encoding={encoding_format}')
    if api_key and not api_key.startswith('Bearer '):
        api_key = 'Bearer ' + api_key

    headers = {'Authorization': api_key, 'Content-Type': 'application/json'}

    truncate_texts = [text[0:max_token_size] for text in texts]

    payload = {'model': model, 'input': truncate_texts, 'encoding_format': encoding_format}

    async with (
        aiohttp.ClientSession() as session,
        session.post(base_url, headers=headers, json=payload) as response,
    ):
        try:
            content = await response.json()
        except Exception as exc:
            logger.error(f'Failed to parse siliconcloud response: {exc}')
            raise
        if 'code' in content:
            logger.error(f'API error response: {content}')
            raise ValueError(content)

        if encoding_format == 'base64':
            base64_strings = [item['embedding'] for item in content['data']]
            embeddings = []
            for string in base64_strings:
                decode_bytes = base64.b64decode(string)
                n = len(decode_bytes) // 4
                float_array = struct.unpack('<' + 'f' * n, decode_bytes)
                embeddings.append(float_array)
            logger.debug(f'Decoded {len(embeddings)} embeddings from base64')
            return np.array(embeddings)

        embeddings = np.array([item['embedding'] for item in content['data']])
        logger.debug(f'Returned {len(embeddings)} embeddings (raw format)')
        return embeddings
