import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm import openai_complete_if_cache, openai_embedding
from lightrag.utils import EmbeddingFunc
import numpy as np
from dotenv import load_dotenv

load_dotenv()
WORKING_DIR = "./dickens"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def llm_model_func(
        prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        "glm-4-flash",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("ZHIPU_API_KEY"),
        base_url="https://open.bigmodel.cn/api/paas/v4",
        **kwargs,
    )


async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embedding(
        texts,
        model="BAAI/bge-m3",
        api_key=os.getenv("SILICON_API_KEY"),
        base_url="https://api.siliconflow.cn/v1",
    )

async def get_embedding_dim():
    test_text = ["This is a test sentence."]
    embedding = await embedding_func(test_text)
    embedding_dim = embedding.shape[1]
    return embedding_dim


# function test
async def test_funcs():
    result = await llm_model_func("How are you?")
    print("llm_model_func: ", result)

    result = await embedding_func(["How are you?"])
    print("embedding_func: ", result)


# asyncio.run(test_funcs())


async def main():
    try:
        embedding_dimension = await get_embedding_dim()
        print(f"Detected embedding dimension: {embedding_dimension}")

        rag = LightRAG(
            working_dir=WORKING_DIR,
            llm_model_func=llm_model_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=1024,
                max_token_size=8192,
                func=embedding_func,
            ),
        )

        # with open("./book.txt", "r", encoding="utf-8") as f:
        #     await rag.ainsert(f.read())
        # # with open("./models.txt", "r", encoding="utf-8") as f:
        # #     await rag.ainsert(f.read())
        # with open("./testmodels.txt", "r", encoding="utf-8") as f:
        #     await rag.ainsert(f.read())

        # Perform naive search
        print(
            await rag.aquery(
                "反航母作战中有哪些行动？", param=QueryParam(mode="global")
            )
        )

        # # Perform local search
        # print(
        #     await rag.aquery(
        #         "What are the top themes in this story?", param=QueryParam(mode="local")
        #     )
        # )
        #
        # # Perform global search
        # print(
        #     await rag.aquery(
        #         "What are the top themes in this story?",
        #         param=QueryParam(mode="global"),
        #     )
        # )
        #
        # # Perform hybrid search
        # print(
        #     await rag.aquery(
        #         "What are the top themes in this story?",
        #         param=QueryParam(mode="hybrid"),
        #     )
        # )
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
