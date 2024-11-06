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


# asyncio.run(test_funcs())


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


        # # Perform naive search
        # print(
        #     await rag.aquery(
        #         "作战中怎么组织地方战斗机进攻，要使用哪些模型?", param=QueryParam(mode="naive")
        #     )
        # )
        #
        # # Perform local search
        # print(
        #     await rag.aquery(
        #         "作战中每一步行动需要使用哪些模型?", param=QueryParam(mode="local")
        #     )
        # )
        #
        # # Perform global search
        # print(
        #     await rag.aquery(
        #         "作战中怎么组织地方战斗机进攻?",
        #         param=QueryParam(mode="global"),
        #     )
        # )

        # Perform hybrid search
        print(
            await rag.aquery(
                "我要专门对敌方战斗机进行拦截，请给我推荐一个最合适的模型?",
                param=QueryParam(mode="local"),
            )
        )
        print(
            await rag.aquery(
                "反航母作战中需要用到哪些模型？",
                param=QueryParam(mode="hybrid"),
            )
        )
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
