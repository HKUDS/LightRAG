import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
import numpy as np
from dotenv import load_dotenv
import aiohttp
import logging

logging.basicConfig(level=logging.INFO)

load_dotenv()

AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
AZURE_EMBEDDING_API_VERSION = os.getenv("AZURE_EMBEDDING_API_VERSION")

WORKING_DIR = "./dickens"

if os.path.exists(WORKING_DIR):
    import shutil

    shutil.rmtree(WORKING_DIR)

os.mkdir(WORKING_DIR)


async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_API_KEY,
    }
    endpoint = f"{AZURE_OPENAI_ENDPOINT}openai/deployments/{AZURE_OPENAI_DEPLOYMENT}/chat/completions?api-version={AZURE_OPENAI_API_VERSION}"

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    payload = {
        "messages": messages,
        "temperature": kwargs.get("temperature", 0),
        "top_p": kwargs.get("top_p", 1),
        "n": kwargs.get("n", 1),
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(endpoint, headers=headers, json=payload) as response:
            if response.status != 200:
                raise ValueError(
                    f"Request failed with status {response.status}: {await response.text()}"
                )
            result = await response.json()
            return result["choices"][0]["message"]["content"]


async def embedding_func(texts: list[str]) -> np.ndarray:
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_OPENAI_API_KEY,
    }
    endpoint = f"{AZURE_OPENAI_ENDPOINT}openai/deployments/{AZURE_EMBEDDING_DEPLOYMENT}/embeddings?api-version={AZURE_EMBEDDING_API_VERSION}"

    payload = {"input": texts}

    async with aiohttp.ClientSession() as session:
        async with session.post(endpoint, headers=headers, json=payload) as response:
            if response.status != 200:
                raise ValueError(
                    f"Request failed with status {response.status}: {await response.text()}"
                )
            result = await response.json()
            embeddings = [item["embedding"] for item in result["data"]]
            return np.array(embeddings)


async def test_funcs():
    result = await llm_model_func("How are you?")
    print("Resposta do llm_model_func: ", result)

    result = await embedding_func(["How are you?"])
    print("Resultado do embedding_func: ", result.shape)
    print("Dimensão da embedding: ", result.shape[1])


asyncio.run(test_funcs())

embedding_dimension = 3072

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=llm_model_func,
    embedding_func=EmbeddingFunc(
        embedding_dim=embedding_dimension,
        max_token_size=8192,
        func=embedding_func,
    ),
)

book1 = open("./book_1.txt", encoding="utf-8")
book2 = open("./book_2.txt", encoding="utf-8")

rag.insert([book1.read(), book2.read()])

query_text = "What are the main themes?"

print("Result (Naive):")
print(rag.query(query_text, param=QueryParam(mode="naive")))

print("\nResult (Local):")
print(rag.query(query_text, param=QueryParam(mode="local")))

print("\nResult (Global):")
print(rag.query(query_text, param=QueryParam(mode="global")))

print("\nResult (Hybrid):")
print(rag.query(query_text, param=QueryParam(mode="hybrid")))
