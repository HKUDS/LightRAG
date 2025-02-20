import inspect
import os
from lightrag import LightRAG
from lightrag.llm import openai_complete, openai_embed
from lightrag.utils import EmbeddingFunc, always_get_an_event_loop
from lightrag import QueryParam

# WorkingDir
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKING_DIR = os.path.join(ROOT_DIR, "dickens")
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)
print(f"WorkingDir: {WORKING_DIR}")

api_key = "empty"
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=openai_complete,
    llm_model_name="qwen2.5-14b-instruct@4bit",
    llm_model_max_async=4,
    llm_model_max_token_size=32768,
    llm_model_kwargs={"base_url": "http://127.0.0.1:1234/v1", "api_key": api_key},
    embedding_func=EmbeddingFunc(
        embedding_dim=1024,
        max_token_size=8192,
        func=lambda texts: openai_embed(
            texts=texts,
            model="text-embedding-bge-m3",
            base_url="http://127.0.0.1:1234/v1",
            api_key=api_key,
        ),
    ),
)

with open("./book.txt", "r", encoding="utf-8") as f:
    rag.insert(f.read())

resp = rag.query(
    "What are the top themes in this story?",
    param=QueryParam(mode="hybrid", stream=True),
)


async def print_stream(stream):
    async for chunk in stream:
        if chunk:
            print(chunk, end="", flush=True)


loop = always_get_an_event_loop()
if inspect.isasyncgen(resp):
    loop.run_until_complete(print_stream(resp))
else:
    print(resp)
