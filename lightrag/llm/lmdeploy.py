"""
LMDeploy LLM Interface Module
==========================

This module provides interfaces for interacting with LMDeploy's language models,
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
    - tenacity
    - numpy
    - pipmaster
    - Python >= 3.10

Usage:
    from llm_interfaces.lmdeploy import lmdeploy_model_complete, lmdeploy_embed
"""

__version__ = "1.0.0"
__author__ = "lightrag Team"
__status__ = "Production"

import pipmaster as pm  # Pipmaster for dynamic library install

# install specific modules
if not pm.is_installed("lmdeploy"):
    pm.install("lmdeploy[all]")
if not pm.is_installed("tenacity"):
    pm.install("tenacity")

from lightrag.exceptions import (
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


from functools import lru_cache


@lru_cache(maxsize=1)
def initialize_lmdeploy_pipeline(
    model,
    tp=1,
    chat_template=None,
    log_level="WARNING",
    model_format="hf",
    quant_policy=0,
):
    from lmdeploy import pipeline, ChatTemplateConfig, TurbomindEngineConfig

    lmdeploy_pipe = pipeline(
        model_path=model,
        backend_config=TurbomindEngineConfig(
            tp=tp, model_format=model_format, quant_policy=quant_policy
        ),
        chat_template_config=(
            ChatTemplateConfig(model_name=chat_template) if chat_template else None
        ),
        log_level="WARNING",
    )
    return lmdeploy_pipe


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(
        (RateLimitError, APIConnectionError, APITimeoutError)
    ),
)
async def lmdeploy_model_if_cache(
    model,
    prompt,
    system_prompt=None,
    history_messages=[],
    chat_template=None,
    model_format="hf",
    quant_policy=0,
    **kwargs,
) -> str:
    """
    Args:
        model (str): The path to the model.
            It could be one of the following options:
                    - i) A local directory path of a turbomind model which is
                        converted by `lmdeploy convert` command or download
                        from ii) and iii).
                    - ii) The model_id of a lmdeploy-quantized model hosted
                        inside a model repo on huggingface.co, such as
                        "InternLM/internlm-chat-20b-4bit",
                        "lmdeploy/llama2-chat-70b-4bit", etc.
                    - iii) The model_id of a model hosted inside a model repo
                        on huggingface.co, such as "internlm/internlm-chat-7b",
                        "Qwen/Qwen-7B-Chat ", "baichuan-inc/Baichuan2-7B-Chat"
                        and so on.
        chat_template (str): needed when model is a pytorch model on
            huggingface.co, such as "internlm-chat-7b",
            "Qwen-7B-Chat ", "Baichuan2-7B-Chat" and so on,
            and when the model name of local path did not match the original model name in HF.
        tp (int): tensor parallel
        prompt (Union[str, List[str]]): input texts to be completed.
        do_preprocess (bool): whether pre-process the messages. Default to
            True, which means chat_template will be applied.
        skip_special_tokens (bool): Whether or not to remove special tokens
            in the decoding. Default to be True.
        do_sample (bool): Whether or not to use sampling, use greedy decoding otherwise.
            Default to be False, which means greedy decoding will be applied.
    """
    try:
        import lmdeploy
        from lmdeploy import version_info, GenerationConfig
    except Exception:
        raise ImportError("Please install lmdeploy before initialize lmdeploy backend.")
    kwargs.pop("hashing_kv", None)
    kwargs.pop("response_format", None)
    max_new_tokens = kwargs.pop("max_tokens", 512)
    tp = kwargs.pop("tp", 1)
    skip_special_tokens = kwargs.pop("skip_special_tokens", True)
    do_preprocess = kwargs.pop("do_preprocess", True)
    do_sample = kwargs.pop("do_sample", False)
    gen_params = kwargs

    version = version_info
    if do_sample is not None and version < (0, 6, 0):
        raise RuntimeError(
            "`do_sample` parameter is not supported by lmdeploy until "
            f"v0.6.0, but currently using lmdeloy {lmdeploy.__version__}"
        )
    else:
        do_sample = True
        gen_params.update(do_sample=do_sample)

    lmdeploy_pipe = initialize_lmdeploy_pipeline(
        model=model,
        tp=tp,
        chat_template=chat_template,
        model_format=model_format,
        quant_policy=quant_policy,
        log_level="WARNING",
    )

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    gen_config = GenerationConfig(
        skip_special_tokens=skip_special_tokens,
        max_new_tokens=max_new_tokens,
        **gen_params,
    )

    response = ""
    async for res in lmdeploy_pipe.generate(
        messages,
        gen_config=gen_config,
        do_preprocess=do_preprocess,
        stream_response=False,
        session_id=1,
    ):
        response += res.response
    return response
