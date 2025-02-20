import asyncio
import os

import numpy as np

from lightrag import LightRAG, QueryParam
from lightrag.llm import siliconcloud_embedding, openai_complete_if_cache
from lightrag.utils import EmbeddingFunc

WORKING_DIR = "./dickens"

# We use SiliconCloud API to call LLM on Oracle Cloud
# More docs here https://docs.siliconflow.cn/introduction
BASE_URL = "https://api.siliconflow.cn/v1/"
APIKEY = ""
CHATMODEL = ""
EMBEDMODEL = ""

os.environ["TIDB_HOST"] = ""
os.environ["TIDB_PORT"] = ""
os.environ["TIDB_USER"] = ""
os.environ["TIDB_PASSWORD"] = ""
os.environ["TIDB_DATABASE"] = "lightrag"

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
    return await siliconcloud_embedding(
        texts,
        # model=EMBEDMODEL,
        api_key=APIKEY,
    )


async def get_embedding_dim():
    test_text = ["This is a test sentence."]
    embedding = await embedding_func(test_text)
    embedding_dim = embedding.shape[1]
    return embedding_dim


async def main():
    try:
        # Detect embedding dimension
        embedding_dimension = await get_embedding_dim()
        print(f"Detected embedding dimension: {embedding_dimension}")

        # Initialize LightRAG
        # We use TiDB DB as the KV/vector
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
            kv_storage="TiDBKVStorage",
            vector_storage="TiDBVectorDBStorage",
            graph_storage="TiDBGraphStorage",
        )

        # Extract and Insert into LightRAG storage
        with open("./dickens/demo.txt", "r", encoding="utf-8") as f:
            await rag.ainsert(f.read())

        # Perform search in different modes
        modes = ["naive", "local", "global", "hybrid"]
        for mode in modes:
            print("=" * 20, mode, "=" * 20)
            print(
                await rag.aquery(
                    "What are the top themes in this story?",
                    param=QueryParam(mode=mode),
                )
            )
            print("-" * 100, "\n")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())
