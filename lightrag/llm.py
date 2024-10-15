import os
import numpy as np
from openai import AsyncOpenAI, APIConnectionError, RateLimitError, Timeout
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from transformers import AutoModel,AutoTokenizer, AutoModelForCausalLM
import torch
from .base import BaseKVStorage
from .utils import compute_args_hash, wrap_embedding_func_with_attrs
import copy
os.environ["TOKENIZERS_PARALLELISM"] = "false"
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
)
async def openai_complete_if_cache(
    model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    openai_async_client = AsyncOpenAI()
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

async def hf_model_if_cache(
    model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    model_name = model
    hf_tokenizer = AutoTokenizer.from_pretrained(model_name,device_map = 'auto')
    if hf_tokenizer.pad_token == None:
        # print("use eos token")
        hf_tokenizer.pad_token = hf_tokenizer.eos_token
    hf_model = AutoModelForCausalLM.from_pretrained(model_name,device_map = 'auto')
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
    input_prompt = ''
    try:
        input_prompt = hf_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)    
    except:
        try:
            ori_message = copy.deepcopy(messages)
            if messages[0]['role'] == "system":
                messages[1]['content'] = "<system>" + messages[0]['content'] + "</system>\n" + messages[1]['content']
                messages = messages[1:]
                input_prompt = hf_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)    
        except:      
            len_message = len(ori_message)
            for msgid in range(len_message):
                input_prompt =input_prompt+ '<'+ori_message[msgid]['role']+'>'+ori_message[msgid]['content']+'</'+ori_message[msgid]['role']+'>\n'
    
    input_ids = hf_tokenizer(input_prompt, return_tensors='pt', padding=True, truncation=True).to("cuda")
    output = hf_model.generate(**input_ids, max_new_tokens=200, num_return_sequences=1,early_stopping = True)
    response_text = hf_tokenizer.decode(output[0], skip_special_tokens=True)
    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response_text, "model": model}}
        )
    return response_text


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



async def hf_model_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    model_name = kwargs['hashing_kv'].global_config['llm_model_name']
    return await hf_model_if_cache(
        model_name,
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
    openai_async_client = AsyncOpenAI()
    response = await openai_async_client.embeddings.create(
        model="text-embedding-3-small", input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])



@wrap_embedding_func_with_attrs(
    embedding_dim=384,
    max_token_size=5000,
)
async def hf_embedding(texts: list[str], tokenizer, embed_model) -> np.ndarray:
    input_ids = tokenizer(texts, return_tensors='pt', padding=True, truncation=True).input_ids
    with torch.no_grad():
        outputs = embed_model(input_ids)
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.detach().numpy()


if __name__ == "__main__":
    import asyncio

    async def main():
        result = await gpt_4o_mini_complete('How are you?')
        print(result)

    asyncio.run(main())
