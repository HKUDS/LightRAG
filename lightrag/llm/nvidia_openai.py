import base64
import os
from typing import Literal

import pipmaster as pm  # Pipmaster for dynamic library install

# install specific modules
if not pm.is_installed('openai'):
    pm.install('openai')

import numpy as np
from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from lightrag.utils import (
    wrap_embedding_func_with_attrs,
)


@wrap_embedding_func_with_attrs(embedding_dim=2048, max_token_size=8192)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, APITimeoutError)),
)
async def nvidia_openai_embed(
    texts: list[str],
    model: str = 'nvidia/llama-3.2-nv-embedqa-1b-v1',
    # refer to https://build.nvidia.com/nim?filters=usecase%3Ausecase_text_to_embedding
    base_url: str = 'https://integrate.api.nvidia.com/v1',
    api_key: str | None = None,
    input_type: str = 'passage',  # query for retrieval, passage for embedding
    trunc: str = 'NONE',  # NONE or START or END
    encode: Literal['float', 'base64'] = 'float',  # float or base64
) -> np.ndarray:
    if api_key:
        os.environ['OPENAI_API_KEY'] = api_key

    openai_async_client = AsyncOpenAI(base_url=base_url)
    response = await openai_async_client.embeddings.create(
        model=model,
        input=texts,
        encoding_format=encode,
        extra_body={'input_type': input_type, 'truncate': trunc},
    )
    embeddings = []
    for dp in response.data:
        emb = dp.embedding
        if encode == 'base64':
            if isinstance(emb, str):
                emb_bytes = base64.b64decode(emb)
                emb_arr = np.frombuffer(emb_bytes, dtype=np.float32)
            else:
                emb_arr = np.array(emb, dtype=np.float32)
        else:
            emb_arr = np.array(emb, dtype=np.float32)
        embeddings.append(emb_arr)

    return np.vstack(embeddings) if embeddings else np.empty((0, 0), dtype=np.float32)
