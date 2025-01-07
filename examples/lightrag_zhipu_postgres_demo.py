import asyncio
import logging
import os
import time
from dotenv import load_dotenv

from lightrag import LightRAG, QueryParam
from lightrag.kg.postgres_impl import PostgreSQLDB
from lightrag.llm import ollama_embedding, zhipu_complete
from lightrag.utils import EmbeddingFunc

load_dotenv()
ROOT_DIR = os.environ.get("ROOT_DIR")
WORKING_DIR = f"{ROOT_DIR}/dickens-pg"

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# AGE
os.environ["AGE_GRAPH_NAME"] = "dickens"

postgres_db = PostgreSQLDB(
    config={
        "host": "localhost",
        "port": 15432,
        "user": "rag",
        "password": "rag",
        "database": "rag",
    }
)


async def main():
    await postgres_db.initdb()
    # Check if PostgreSQL DB tables exist, if not, tables will be created
    await postgres_db.check_tables()

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=zhipu_complete,
        llm_model_name="glm-4-flashx",
        llm_model_max_async=4,
        llm_model_max_token_size=32768,
        enable_llm_cache_for_entity_extract=True,
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
            func=lambda texts: ollama_embedding(
                texts, embed_model="nomic-embed-text", host="http://localhost:11434"
            ),
        ),
        kv_storage="PGKVStorage",
        doc_status_storage="PGDocStatusStorage",
        graph_storage="PGGraphStorage",
        vector_storage="PGVectorStorage",
    )
    # Set the KV/vector/graph storage's `db` property, so all operation will use same connection pool
    rag.doc_status.db = postgres_db
    rag.full_docs.db = postgres_db
    rag.text_chunks.db = postgres_db
    rag.llm_response_cache.db = postgres_db
    rag.key_string_value_json_storage_cls.db = postgres_db
    rag.chunks_vdb.db = postgres_db
    rag.relationships_vdb.db = postgres_db
    rag.entities_vdb.db = postgres_db
    rag.graph_storage_cls.db = postgres_db
    rag.chunk_entity_relation_graph.db = postgres_db
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
