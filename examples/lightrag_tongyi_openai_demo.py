import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
import numpy as np
from dotenv import load_dotenv
import logging
from openai import OpenAI
from lightrag.kg.shared_storage import initialize_pipeline_status

logging.basicConfig(level=logging.INFO)

load_dotenv()

LLM_MODEL = os.environ.get("LLM_MODEL", "qwen-turbo-latest")
LLM_BINDING_HOST = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_BINDING_API_KEY = os.getenv("LLM_BINDING_API_KEY")

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-v3")
EMBEDDING_BINDING_HOST = os.getenv("EMBEDDING_BINDING_HOST", LLM_BINDING_HOST)
EMBEDDING_BINDING_API_KEY = os.getenv("EMBEDDING_BINDING_API_KEY", LLM_BINDING_API_KEY)
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", 1024))
EMBEDDING_MAX_TOKEN_SIZE = int(os.environ.get("EMBEDDING_MAX_TOKEN_SIZE", 8192))
EMBEDDING_MAX_BATCH_SIZE = int(os.environ.get("EMBEDDING_MAX_BATCH_SIZE", 10))

print(f"LLM_MODEL: {LLM_MODEL}")
print(f"EMBEDDING_MODEL: {EMBEDDING_MODEL}")

WORKING_DIR = "./dickens"

if os.path.exists(WORKING_DIR):
    import shutil

    shutil.rmtree(WORKING_DIR)

os.mkdir(WORKING_DIR)


async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    client = OpenAI(
        api_key=LLM_BINDING_API_KEY,
        base_url=LLM_BINDING_HOST,
    )

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    chat_completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=kwargs.get("temperature", 0),
        top_p=kwargs.get("top_p", 1),
        n=kwargs.get("n", 1),
        extra_body={"enable_thinking": False},
    )
    return chat_completion.choices[0].message.content


async def embedding_func(texts: list[str]) -> np.ndarray:
    client = OpenAI(
        api_key=EMBEDDING_BINDING_API_KEY,
        base_url=EMBEDDING_BINDING_HOST,
    )

    print("##### embedding: texts: %d #####" % len(texts))
    max_batch_size = EMBEDDING_MAX_BATCH_SIZE
    embeddings = []
    for i in range(0, len(texts), max_batch_size):
        batch = texts[i : i + max_batch_size]
        embedding = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        embeddings += [item.embedding for item in embedding.data]

    return np.array(embeddings)


async def test_funcs():
    result = await llm_model_func("How are you?")
    print("Resposta do llm_model_func: ", result)

    result = await embedding_func(["How are you?"])
    print("Resultado do embedding_func: ", result.shape)
    print("Dimensão da embedding: ", result.shape[1])


asyncio.run(test_funcs())


async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=EMBEDDING_MAX_TOKEN_SIZE,
            func=embedding_func,
        ),
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


def main():
    rag = asyncio.run(initialize_rag())

    with open("./book.txt", "r", encoding="utf-8") as f:
        rag.insert(f.read())

    query_text = "What are the main themes?"

    print("Result (Naive):")
    print(rag.query(query_text, param=QueryParam(mode="naive")))

    print("\nResult (Local):")
    print(rag.query(query_text, param=QueryParam(mode="local")))

    print("\nResult (Global):")
    print(rag.query(query_text, param=QueryParam(mode="global")))

    print("\nResult (Hybrid):")
    print(rag.query(query_text, param=QueryParam(mode="hybrid")))

    print("\nResult (mix):")
    print(rag.query(query_text, param=QueryParam(mode="mix")))


if __name__ == "__main__":
    main()
