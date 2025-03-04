import asyncio
import logging
import os
import time
from dotenv import load_dotenv

from lightrag import LightRAG, QueryParam
from lightrag.llm.zhipu import zhipu_complete
from lightrag.llm.ollama import ollama_embedding
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status

load_dotenv()
ROOT_DIR = os.environ.get("ROOT_DIR")
WORKING_DIR = f"{ROOT_DIR}/dickens-pg"

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# AGE
os.environ["AGE_GRAPH_NAME"] = "dickens"

os.environ["POSTGRES_HOST"] = "localhost"
os.environ["POSTGRES_PORT"] = "15432"
os.environ["POSTGRES_USER"] = "rag"
os.environ["POSTGRES_PASSWORD"] = "rag"
os.environ["POSTGRES_DATABASE"] = "rag"


async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=zhipu_complete,
        llm_model_name="glm-4-flashx",
        llm_model_max_async=4,
        llm_model_max_token_size=32768,
        enable_llm_cache_for_entity_extract=True,
        embedding_func=EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=8192,
            func=lambda texts: ollama_embedding(
                texts, embed_model="bge-m3", host="http://localhost:11434"
            ),
        ),
        kv_storage="PGKVStorage",
        doc_status_storage="PGDocStatusStorage",
        graph_storage="PGGraphStorage",
        vector_storage="PGVectorStorage",
        auto_manage_storages_states=False,
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


async def main():
    # Initialize RAG instance
    rag = await initialize_rag()

    # add embedding_func for graph database, it's deleted in commit 5661d76860436f7bf5aef2e50d9ee4a59660146c
    rag.chunk_entity_relation_graph.embedding_func = rag.embedding_func

    with open(f"{ROOT_DIR}/book.txt", "r", encoding="utf-8") as f:
        await rag.ainsert(f.read())

    print("==== Trying to test the rag queries ====")
    print("**** Start Naive Query ****")
    start_time = time.time()
    # Perform naive search
    print(
        await rag.aquery(
            "What are the top themes in this story?", param=QueryParam(mode="naive")
        )
    )
    print(f"Naive Query Time: {time.time() - start_time} seconds")
    # Perform local search
    print("**** Start Local Query ****")
    start_time = time.time()
    print(
        await rag.aquery(
            "What are the top themes in this story?", param=QueryParam(mode="local")
        )
    )
    print(f"Local Query Time: {time.time() - start_time} seconds")
    # Perform global search
    print("**** Start Global Query ****")
    start_time = time.time()
    print(
        await rag.aquery(
            "What are the top themes in this story?", param=QueryParam(mode="global")
        )
    )
    print(f"Global Query Time: {time.time() - start_time}")
    # Perform hybrid search
    print("**** Start Hybrid Query ****")
    print(
        await rag.aquery(
            "What are the top themes in this story?", param=QueryParam(mode="hybrid")
        )
    )
    print(f"Hybrid Query Time: {time.time() - start_time} seconds")


if __name__ == "__main__":
    asyncio.run(main())
