import asyncio
import logging
import os

import numpy as np
from dotenv import load_dotenv
from openai import AzureOpenAI

from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc

logging.basicConfig(level=logging.INFO)

load_dotenv()

AZURE_OPENAI_API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION')
AZURE_OPENAI_DEPLOYMENT = os.getenv('AZURE_OPENAI_DEPLOYMENT')
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')

AZURE_EMBEDDING_DEPLOYMENT = os.getenv('AZURE_EMBEDDING_DEPLOYMENT')
AZURE_EMBEDDING_API_VERSION = os.getenv('AZURE_EMBEDDING_API_VERSION')

WORKING_DIR = './dickens'

if os.path.exists(WORKING_DIR):
    import shutil

    shutil.rmtree(WORKING_DIR)

os.mkdir(WORKING_DIR)


async def llm_model_func(prompt, system_prompt=None, history_messages=None, keyword_extraction=False, **kwargs) -> str:
    if history_messages is None:
        history_messages = []
    client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )

    messages = []
    if system_prompt:
        messages.append({'role': 'system', 'content': system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({'role': 'user', 'content': prompt})

    chat_completion = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,  # model = "deployment_name".
        messages=messages,
        temperature=kwargs.get('temperature', 0),
        top_p=kwargs.get('top_p', 1),
        n=kwargs.get('n', 1),
    )
    return chat_completion.choices[0].message.content


async def embedding_func(texts: list[str]) -> np.ndarray:
    client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_EMBEDDING_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )
    embedding = client.embeddings.create(model=AZURE_EMBEDDING_DEPLOYMENT, input=texts)

    embeddings = [item.embedding for item in embedding.data]
    return np.array(embeddings)


async def test_funcs():
    result = await llm_model_func('How are you?')
    print('Resposta do llm_model_func: ', result)

    result = await embedding_func(['How are you?'])
    print('Resultado do embedding_func: ', result.shape)
    print('Dimensão da embedding: ', result.shape[1])


asyncio.run(test_funcs())

embedding_dimension = 3072


async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dimension,
            max_token_size=8192,
            func=embedding_func,
        ),
    )

    await rag.initialize_storages()  # Auto-initializes pipeline_status
    return rag


def main():
    rag = asyncio.run(initialize_rag())

    with (
        open('./book_1.txt', encoding='utf-8') as book1,
        open('./book_2.txt', encoding='utf-8') as book2,
    ):
        rag.insert([book1.read(), book2.read()])

    query_text = 'What are the main themes?'

    print('Result (Naive):')
    print(rag.query(query_text, param=QueryParam(mode='naive')))

    print('\nResult (Local):')
    print(rag.query(query_text, param=QueryParam(mode='local')))

    print('\nResult (Global):')
    print(rag.query(query_text, param=QueryParam(mode='global')))

    print('\nResult (Hybrid):')
    print(rag.query(query_text, param=QueryParam(mode='hybrid')))


if __name__ == '__main__':
    main()
