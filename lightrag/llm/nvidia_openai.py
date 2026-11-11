import sys

if sys.version_info < (3, 9):
    pass
else:
    pass

import pipmaster as pm  # Pipmaster for dynamic library install

# install specific modules
if not pm.is_installed("openai"):
    pm.install("openai")

from openai import (
    AsyncOpenAI,
    APIConnectionError,
    RateLimitError,
    APITimeoutError,
)
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from lightrag.utils import (
    wrap_embedding_func_with_attrs,
)


import numpy as np


@wrap_embedding_func_with_attrs(
    embedding_dim=2048, max_token_size=8192, model_name="nvidia_embedding_model"
)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError)
    ),
)
async def nvidia_openai_embed(
    texts: list[str],
    model: str = "nvidia/llama-3.2-nv-embedqa-1b-v1",
    # refer to https://build.nvidia.com/nim?filters=usecase%3Ausecase_text_to_embedding
    base_url: str = "https://integrate.api.nvidia.com/v1",
    api_key: str = None,
    input_type: str | None = None,  # "query" or "passage"; None defers to context
    trunc: str = "NONE",  # NONE or START or END
    encode: str = "float",  # float or base64
    context: str | None = None,
) -> np.ndarray:
    """Generate embeddings with an NVIDIA NIM embedding model.

    NVIDIA's embedqa models are asymmetric: a search query and an indexed
    document embed into different regions of the vector space, distinguished
    by `input_type`. Without an explicit `input_type`, this maps LightRAG's
    own `context` ("query" / "document") onto NVIDIA's "query" / "passage"
    values, so a query embedded via a LightRAG query path actually uses
    NVIDIA's query mode instead of always defaulting to "passage".
    """
    if input_type is None:
        input_type = "query" if context == "query" else "passage"

    client_kwargs = {}
    if base_url is not None:
        client_kwargs["base_url"] = base_url
    if api_key:
        client_kwargs["api_key"] = api_key

    openai_async_client = AsyncOpenAI(**client_kwargs)
    # Hold the client in an async-with so its httpx connection pool is
    # released on every exit path (success, error, and each @retry attempt),
    # instead of leaking one pool per call until GC. Mirrors ``openai_embed``.
    async with openai_async_client:
        response = await openai_async_client.embeddings.create(
            model=model,
            input=texts,
            encoding_format=encode,
            extra_body={"input_type": input_type, "truncate": trunc},
        )
        return np.array([dp.embedding for dp in response.data])
