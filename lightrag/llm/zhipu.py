import sys
import warnings
from ..utils import verbose_debug

if sys.version_info < (3, 9):
    pass
else:
    pass
import pipmaster as pm  # Pipmaster for dynamic library install

# install specific modules
if not pm.is_installed("zhipuai"):
    pm.install("zhipuai")

from openai import (
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
    logger,
)

import numpy as np
from typing import Union, List, Optional, Dict


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError)
    ),
)
async def zhipu_complete_if_cache(
    prompt: Union[str, List[Dict[str, str]]],
    model: str = "glm-4-flashx",  # The most cost/performance balance model in glm-4 series
    api_key: Optional[str] = None,
    system_prompt: Optional[str] = None,
    history_messages: List[Dict[str, str]] = [],
    enable_cot: bool = False,  # LightRAG output switch: include reasoning_content as <think>...</think>
    thinking: Optional[
        Dict[str, object]
    ] = None,  # Zhipu request param: use {"type": "enabled"} to enable thinking
    **kwargs,
) -> str:
    """Call Zhipu chat completions with optional official thinking support.

    Parameter roles:
    - `thinking`: forwarded to the Zhipu API as-is. To enable thinking output,
      pass a config such as `{"type": "enabled"}`.
    - `enable_cot`: LightRAG-only formatting switch. When True and the API
      returns `reasoning_content`, it is preserved in the final string as
      `<think>...</think>`.
    """
    # dynamically load ZhipuAI
    try:
        from zhipuai import ZhipuAI
    except ImportError:
        raise ImportError("Please install zhipuai before initialize zhipuai backend.")

    if api_key:
        client = ZhipuAI(api_key=api_key)
    else:
        # please set ZHIPUAI_API_KEY in your environment
        # os.environ["ZHIPUAI_API_KEY"]
        client = ZhipuAI()

    messages = []

    if not system_prompt:
        system_prompt = "You are a helpful assistant. Note that sensitive words in the content should be replaced with ***"

    # Add system prompt if provided
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    # Add debug logging
    logger.debug("===== Query Input to LLM =====")
    logger.debug(f"Query: {prompt}")
    verbose_debug(f"System prompt: {system_prompt}")

    # Remove unsupported kwargs
    kwargs = {
        k: v for k, v in kwargs.items() if k not in ["hashing_kv", "keyword_extraction"]
    }
    # `thinking` is an official Zhipu request field. Example:
    # {"type": "enabled"} enables reasoning output on supported models.
    if thinking is not None:
        kwargs["thinking"] = thinking

    response = client.chat.completions.create(model=model, messages=messages, **kwargs)
    message = response.choices[0].message
    content = message.content or ""
    reasoning_content = getattr(message, "reasoning_content", "") or ""

    if enable_cot and reasoning_content.strip():
        if content:
            return f"<think>{reasoning_content}</think>{content}"
        return f"<think>{reasoning_content}</think>"

    return content


async def zhipu_complete(
    prompt,
    system_prompt=None,
    history_messages=[],
    keyword_extraction=False,
    entity_extraction=False,
    enable_cot: bool = False,
    **kwargs,
):
    # Pop legacy extraction flags from kwargs to avoid passing them downstream.
    keyword_extraction = kwargs.pop("keyword_extraction", keyword_extraction)
    entity_extraction = kwargs.pop("entity_extraction", entity_extraction)

    # Deprecation shims: map legacy boolean flags to response_format only when
    # an explicit response_format was not supplied by the caller. The legacy
    # path also forces enable_cot=False so that reasoning_content cannot
    # corrupt the JSON payload expected by callers that were relying on it.
    if kwargs.get("response_format") is None:
        if entity_extraction:
            warnings.warn(
                "zhipu_complete(entity_extraction=True) is deprecated; "
                "pass response_format={'type': 'json_object'} instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            kwargs["response_format"] = {"type": "json_object"}
            enable_cot = False
        elif keyword_extraction:
            warnings.warn(
                "zhipu_complete(keyword_extraction=True) is deprecated; "
                "pass response_format={'type': 'json_object'} instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            kwargs["response_format"] = {"type": "json_object"}
            enable_cot = False

    return await zhipu_complete_if_cache(
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        enable_cot=enable_cot,
        **kwargs,
    )


@wrap_embedding_func_with_attrs(
    embedding_dim=1024, max_token_size=8192, model_name="embedding-3"
)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError)
    ),
)
async def zhipu_embedding(
    texts: list[str],
    model: str = "embedding-3",
    api_key: str = None,
    embedding_dim: int | None = None,
    **kwargs,
) -> np.ndarray:
    # dynamically load ZhipuAI
    try:
        from zhipuai import ZhipuAI
    except ImportError:
        raise ImportError("Please install zhipuai before initialize zhipuai backend.")
    if api_key:
        client = ZhipuAI(api_key=api_key)
    else:
        # please set ZHIPUAI_API_KEY in your environment
        # os.environ["ZHIPUAI_API_KEY"]
        client = ZhipuAI()

    # Convert single text to list if needed
    if isinstance(texts, str):
        texts = [texts]

    embeddings = []
    for text in texts:
        try:
            request_kwargs = dict(kwargs)
            if embedding_dim is not None:
                request_kwargs["dimensions"] = embedding_dim
            response = client.embeddings.create(
                model=model, input=[text], **request_kwargs
            )
            embeddings.append(response.data[0].embedding)
        except Exception as e:
            raise Exception(f"Error calling ChatGLM Embedding API: {str(e)}")

    return np.array(embeddings)
