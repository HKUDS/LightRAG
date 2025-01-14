import os
# import asyncio
# import inspect
import logging
from dotenv import load_dotenv
from lightrag import LightRAG, QueryParam
from lightrag.llm import openai_complete_if_cache, ollama_embedding
from lightrag.utils import EmbeddingFunc

load_dotenv()

WORKING_DIR = "./examples/output"

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        "deepseek-chat",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_ENDPOINT"),
        **kwargs,
    )

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=llm_model_func,
    embedding_func=EmbeddingFunc(
        embedding_dim=1024,
        max_token_size=8192,
        func=lambda texts: ollama_embedding(
            texts, embed_model="bge-m3:latest", host="http://m4.lan.znipower.com:11434"
        ),
    ),
)

with open("./examples/input/book.txt", "r", encoding="utf-8") as f:
    rag.insert(f.read())

# Perform naive search
print(
    rag.query("What are the top themes in this story?", param=QueryParam(mode="naive"))
)

# Perform local search
print(
    rag.query("What are the top themes in this story?", param=QueryParam(mode="local"))
)

# Perform global search
print(
    rag.query("What are the top themes in this story?", param=QueryParam(mode="global"))
)

# Perform hybrid search
print(
    rag.query("What are the top themes in this story?", param=QueryParam(mode="hybrid"))
)

# # stream response
# resp = rag.query(
#     "What are the top themes in this story?",
#     param=QueryParam(mode="hybrid", stream=True),
# )


# async def print_stream(stream):
#     async for chunk in stream:
#         print(chunk, end="", flush=True)


# if inspect.isasyncgen(resp):
#     asyncio.run(print_stream(resp))
# else:
#     print(resp)
