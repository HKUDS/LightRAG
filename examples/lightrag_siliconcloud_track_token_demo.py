import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.llm.siliconcloud import siliconcloud_embedding
from lightrag.utils import EmbeddingFunc
from lightrag.utils import TokenTracker
import numpy as np
from lightrag.kg.shared_storage import initialize_pipeline_status
from dotenv import load_dotenv

load_dotenv()

token_tracker = TokenTracker()
WORKING_DIR = "./dickens"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        "Qwen/Qwen2.5-7B-Instruct",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("SILICONFLOW_API_KEY"),
        base_url="https://api.siliconflow.cn/v1/",
        token_tracker=token_tracker,
        **kwargs,
    )


async def embedding_func(texts: list[str]) -> np.ndarray:
    return await siliconcloud_embedding(
        texts,
        model="BAAI/bge-m3",
        api_key=os.getenv("SILICONFLOW_API_KEY"),
        max_token_size=512,
    )


# function test
async def test_funcs():
    # Context Manager Method
    with token_tracker:
        result = await llm_model_func("How are you?")
        print("llm_model_func: ", result)


asyncio.run(test_funcs())


async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=1024, max_token_size=512, func=embedding_func
        ),
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


def main():
    # Initialize RAG instance
    rag = asyncio.run(initialize_rag())

    # Reset tracker before processing queries
    token_tracker.reset()

    with open("./book.txt", "r", encoding="utf-8") as f:
        rag.insert(f.read())

    print(
        rag.query(
            "What are the top themes in this story?", param=QueryParam(mode="naive")
        )
    )

    print(
        rag.query(
            "What are the top themes in this story?", param=QueryParam(mode="local")
        )
    )

    print(
        rag.query(
            "What are the top themes in this story?", param=QueryParam(mode="global")
        )
    )

    print(
        rag.query(
            "What are the top themes in this story?", param=QueryParam(mode="hybrid")
        )
    )

    # Display final token usage after main query
    print("Token usage:", token_tracker.get_usage())


if __name__ == "__main__":
    main()
