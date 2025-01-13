import sys
import os
from pathlib import Path
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm import openai_complete_if_cache, openai_embedding
from lightrag.utils import EmbeddingFunc
import numpy as np
from lightrag.kg.oracle_impl import OracleDB

print(os.getcwd())
script_directory = Path(__file__).resolve().parent.parent
sys.path.append(os.path.abspath(script_directory))

WORKING_DIR = "./dickens"

# We use OpenAI compatible API to call LLM on Oracle Cloud
# More docs here https://github.com/jin38324/OCI_GenAI_access_gateway
BASE_URL = "http://localhost:8088/v1/"
APIKEY = "ocigenerativeai"
CHATMODEL = "cohere.command-r-plus"
EMBEDMODEL = "cohere.embed-multilingual-v3.0"


if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        CHATMODEL,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=APIKEY,
        base_url=BASE_URL,
        **kwargs,
    )


async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embedding(
        texts,
        model=EMBEDMODEL,
        api_key=APIKEY,
        base_url=BASE_URL,
    )


async def get_embedding_dim():
    test_text = ["This is a test sentence."]
    embedding = await embedding_func(test_text)
    embedding_dim = embedding.shape[1]
    return embedding_dim


async def main():
    # try:
        # Detect embedding dimension
        embedding_dimension = await get_embedding_dim()
        print(f"Detected embedding dimension: {embedding_dimension}")

        # You can add `addon_params={"example_number": 1, "language": "Simplfied Chinese"}` to control the prompt
        rag = LightRAG(
            enable_llm_cache=False,
            working_dir=WORKING_DIR,
            chunk_token_size=512,
            llm_model_func=llm_model_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=embedding_dimension,
                max_token_size=512,
                func=embedding_func,
            ),
            # graph_storage="OracleGraphStorage",
            # kv_storage="OracleKVStorage",
            # vector_storage="OracleVectorDBStorage",
        )

        # add embedding_func for graph database, it's deleted in commit 5661d76860436f7bf5aef2e50d9ee4a59660146c
        rag.chunk_entity_relation_graph.embedding_func = rag.embedding_func

        # Extract and Insert into LightRAG storage
        with open("./dickens/demo.txt", "r", encoding="utf-8") as f:
            await rag.ainsert(f.read())

        # Perform search in different modes
        modes = ["naive", "local", "global", "hybrid"]
        for mode in modes:
            print("=" * 20, mode, "=" * 20)
            print(
                await rag.aquery(
                    "What makes the main character change their mind?",
                    param=QueryParam(mode=mode),
                )
            )
            print("-" * 100, "\n")

    # except Exception as e:
    #     print(f"An error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())
