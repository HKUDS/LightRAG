import numpy as np
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from lightrag.llm import jina_embedding, openai_complete_if_cache
import os
import asyncio


async def embedding_func(texts: list[str]) -> np.ndarray:
    return await jina_embedding(texts, api_key="YourJinaAPIKey")


WORKING_DIR = "./dickens"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        "solar-mini",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("UPSTAGE_API_KEY"),
        base_url="https://api.upstage.ai/v1/solar",
        **kwargs,
    )


rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=llm_model_func,
    embedding_func=EmbeddingFunc(
        embedding_dim=1024, max_token_size=8192, func=embedding_func
    ),
)


async def lightraginsert(file_path, semaphore):
    async with semaphore:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            # If UTF-8 decoding fails, try other encodings
            with open(file_path, "r", encoding="gbk") as f:
                content = f.read()
        await rag.ainsert(content)


async def process_files(directory, concurrency_limit):
    semaphore = asyncio.Semaphore(concurrency_limit)
    tasks = []
    for root, dirs, files in os.walk(directory):
        for f in files:
            file_path = os.path.join(root, f)
            if f.startswith("."):
                continue
            tasks.append(lightraginsert(file_path, semaphore))
    await asyncio.gather(*tasks)


async def main():
    try:
        rag = LightRAG(
            working_dir=WORKING_DIR,
            llm_model_func=llm_model_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=1024,
                max_token_size=8192,
                func=embedding_func,
            ),
        )

        asyncio.run(process_files(WORKING_DIR, concurrency_limit=4))

        # Perform naive search
        print(
            await rag.aquery(
                "What are the top themes in this story?", param=QueryParam(mode="naive")
            )
        )

        # Perform local search
        print(
            await rag.aquery(
                "What are the top themes in this story?", param=QueryParam(mode="local")
            )
        )

        # Perform global search
        print(
            await rag.aquery(
                "What are the top themes in this story?",
                param=QueryParam(mode="global"),
            )
        )

        # Perform hybrid search
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
