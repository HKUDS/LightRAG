import json
import re

from lightrag.utils import verbose_debug

pass
import pipmaster as pm  # Pipmaster for dynamic library install

# install specific modules
if not pm.is_installed('zhipuai'):
    pm.install('zhipuai')


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

from lightrag.types import GPTKeywordExtractionFormat
from lightrag.utils import (
    logger,
    wrap_embedding_func_with_attrs,
)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, APITimeoutError)),
)
async def zhipu_complete_if_cache(
    prompt: str | list[dict[str, str]],
    model: str = 'glm-4-flashx',  # The most cost/performance balance model in glm-4 series
    api_key: str | None = None,
    system_prompt: str | None = None,
    history_messages: list[dict[str, str]] | None = None,
    enable_cot: bool = False,
    **kwargs,
) -> str:
    if history_messages is None:
        history_messages = []
    if enable_cot:
        logger.debug('enable_cot=True is not supported for ZhipuAI and will be ignored.')
    # dynamically load ZhipuAI
    try:
        from zhipuai import ZhipuAI
    except ImportError as e:
        raise ImportError('Please install zhipuai before initialize zhipuai backend.') from e

    # please set ZHIPUAI_API_KEY in your environment
    # os.environ["ZHIPUAI_API_KEY"]
    client = ZhipuAI(api_key=api_key) if api_key else ZhipuAI()

    messages = []

    if not system_prompt:
        system_prompt = (
            'You are a helpful assistant. Note that sensitive words in the content should be replaced with ***'
        )

    # Add system prompt if provided
    if system_prompt:
        messages.append({'role': 'system', 'content': system_prompt})
    messages.extend(history_messages)
    messages.append({'role': 'user', 'content': prompt})

    # Add debug logging
    logger.debug('===== Query Input to LLM =====')
    logger.debug(f'Query: {prompt}')
    verbose_debug(f'System prompt: {system_prompt}')

    # Remove unsupported kwargs
    kwargs = {k: v for k, v in kwargs.items() if k not in ['hashing_kv', 'keyword_extraction']}

    response = client.chat.completions.create(model=model, messages=messages, **kwargs)

    return response.choices[0].message.content


async def zhipu_complete(
    prompt,
    system_prompt=None,
    history_messages=None,
    keyword_extraction=False,
    enable_cot: bool = False,
    **kwargs,
):
    if history_messages is None:
        history_messages = []
    # Pop keyword_extraction from kwargs to avoid passing it to zhipu_complete_if_cache
    keyword_extraction = kwargs.pop('keyword_extraction', None)

    if keyword_extraction:
        # Add a system prompt to guide the model to return JSON format
        extraction_prompt = """You are a helpful assistant that extracts keywords from text.
        Please analyze the content and extract two types of keywords:
        1. High-level keywords: Important concepts and main themes
        2. Low-level keywords: Specific details and supporting elements

        Return your response in this exact JSON format:
        {
            "high_level_keywords": ["keyword1", "keyword2"],
            "low_level_keywords": ["keyword1", "keyword2", "keyword3"]
        }

        Only return the JSON, no other text."""

        # Combine with existing system prompt if any
        system_prompt = f'{system_prompt}\n\n{extraction_prompt}' if system_prompt else extraction_prompt

        try:
            response = await zhipu_complete_if_cache(
                prompt=prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                enable_cot=enable_cot,
                **kwargs,
            )

            # Try to parse as JSON
            try:
                data = json.loads(response)
                return GPTKeywordExtractionFormat(
                    high_level_keywords=data.get('high_level_keywords', []),
                    low_level_keywords=data.get('low_level_keywords', []),
                )
            except json.JSONDecodeError:
                # If direct JSON parsing fails, try to extract JSON from text
                match = re.search(r'\{[\s\S]*\}', response)
                if match:
                    try:
                        data = json.loads(match.group())
                        return GPTKeywordExtractionFormat(
                            high_level_keywords=data.get('high_level_keywords', []),
                            low_level_keywords=data.get('low_level_keywords', []),
                        )
                    except json.JSONDecodeError:
                        pass

                # If all parsing fails, log warning and return empty format
                logger.warning(f'Failed to parse keyword extraction response: {response}')
                return GPTKeywordExtractionFormat(high_level_keywords=[], low_level_keywords=[])
        except Exception as e:
            logger.error(f'Error during keyword extraction: {e!s}')
            return GPTKeywordExtractionFormat(high_level_keywords=[], low_level_keywords=[])
    else:
        # For non-keyword-extraction, just return the raw response string
        return await zhipu_complete_if_cache(
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            enable_cot=enable_cot,
            **kwargs,
        )


@wrap_embedding_func_with_attrs(embedding_dim=1024)
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, APITimeoutError)),
)
async def zhipu_embedding(
    texts: list[str], model: str = 'embedding-3', api_key: str | None = None, **kwargs
) -> np.ndarray:
    # dynamically load ZhipuAI
    try:
        from zhipuai import ZhipuAI
    except ImportError as e:
        raise ImportError('Please install zhipuai before initialize zhipuai backend.') from e
    # please set ZHIPUAI_API_KEY in your environment
    # os.environ["ZHIPUAI_API_KEY"]
    client = ZhipuAI(api_key=api_key) if api_key else ZhipuAI()

    # Convert single text to list if needed
    if isinstance(texts, str):
        texts = [texts]

    embeddings = []
    for text in texts:
        try:
            response = client.embeddings.create(model=model, input=[text], **kwargs)
            embeddings.append(response.data[0].embedding)
        except Exception as e:
            raise Exception(f'Error calling ChatGLM Embedding API: {e!s}') from e

    return np.array(embeddings)
