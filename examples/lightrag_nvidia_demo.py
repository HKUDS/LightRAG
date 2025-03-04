import os
import asyncio
import nest_asyncio

nest_asyncio.apply()

from lightrag import LightRAG, QueryParam
from lightrag.llm import (
    openai_complete_if_cache,
    nvidia_openai_embed,
)
from lightrag.utils import EmbeddingFunc
import numpy as np
from lightrag.kg.shared_storage import initialize_pipeline_status

# for custom llm_model_func
from lightrag.utils import locate_json_string_body_from_string


WORKING_DIR = "./dickens"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# some method to use your API key (choose one)
# NVIDIA_OPENAI_API_KEY = os.getenv("NVIDIA_OPENAI_API_KEY")
NVIDIA_OPENAI_API_KEY = "nvapi-xxxx"  # your api key

# using pre-defined function for nvidia LLM API. OpenAI compatible
# llm_model_func = nvidia_openai_complete


# If you trying to make custom llm_model_func to use llm model on NVIDIA API like other example:
async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    result = await openai_complete_if_cache(
        "nvidia/llama-3.1-nemotron-70b-instruct",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=NVIDIA_OPENAI_API_KEY,
        base_url="https://integrate.api.nvidia.com/v1",
        **kwargs,
    )
    if keyword_extraction:
        return locate_json_string_body_from_string(result)
    return result


# custom embedding
nvidia_embed_model = "nvidia/nv-embedqa-e5-v5"


async def indexing_embedding_func(texts: list[str]) -> np.ndarray:
    return await nvidia_openai_embed(
        texts,
        model=nvidia_embed_model,  # maximum 512 token
        # model="nvidia/llama-3.2-nv-embedqa-1b-v1",
        api_key=NVIDIA_OPENAI_API_KEY,
        base_url="https://integrate.api.nvidia.com/v1",
        input_type="passage",
        trunc="END",  # handling on server side if input token is longer than maximum token
        encode="float",
    )


async def query_embedding_func(texts: list[str]) -> np.ndarray:
    return await nvidia_openai_embed(
        texts,
        model=nvidia_embed_model,  # maximum 512 token
        # model="nvidia/llama-3.2-nv-embedqa-1b-v1",
        api_key=NVIDIA_OPENAI_API_KEY,
        base_url="https://integrate.api.nvidia.com/v1",
        input_type="query",
        trunc="END",  # handling on server side if input token is longer than maximum token
        encode="float",
    )


# dimension are same
async def get_embedding_dim():
    test_text = ["This is a test sentence."]
    embedding = await indexing_embedding_func(test_text)
    embedding_dim = embedding.shape[1]
    return embedding_dim


# function test
async def test_funcs():
    result = await llm_model_func("How are you?")
    print("llm_model_func: ", result)

    result = await indexing_embedding_func(["How are you?"])
    print("embedding_func: ", result)


# asyncio.run(test_funcs())


async def initialize_rag():
    embedding_dimension = await get_embedding_dim()
    print(f"Detected embedding dimension: {embedding_dimension}")

    # lightRAG class during indexing
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        # llm_model_name="meta/llama3-70b-instruct", #un comment if
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dimension,
            max_token_size=512,  # maximum token size, somehow it's still exceed maximum number of token
            # so truncate (trunc) parameter on embedding_func will handle it and try to examine the tokenizer used in LightRAG
            # so you can adjust to be able to fit the NVIDIA model (future work)
            func=indexing_embedding_func,
        ),
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


async def main():
    try:
        # Initialize RAG instance
        rag = await initialize_rag()

        # reading file
        with open("./book.txt", "r", encoding="utf-8") as f:
            await rag.ainsert(f.read())

        # Perform naive search
        print("==============Naive===============")
        print(
            await rag.aquery(
                "What are the top themes in this story?", param=QueryParam(mode="naive")
            )
        )

        # Perform local search
        print("==============local===============")
        print(
            await rag.aquery(
                "What are the top themes in this story?", param=QueryParam(mode="local")
            )
        )

        # Perform global search
        print("==============global===============")
        print(
            await rag.aquery(
                "What are the top themes in this story?",
                param=QueryParam(mode="global"),
            )
        )

        # Perform hybrid search
        print("==============hybrid===============")
        print(
            await rag.aquery(
                "What are the top themes in this story?",
                param=QueryParam(mode="hybrid"),
            )
        )
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())
