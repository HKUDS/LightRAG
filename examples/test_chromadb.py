import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.utils import EmbeddingFunc
import numpy as np
from lightrag.kg.shared_storage import initialize_pipeline_status

#########
# Uncomment the below two lines if running in a jupyter notebook to handle the async nature of rag.insert()
# import nest_asyncio
# nest_asyncio.apply()
#########
WORKING_DIR = "./chromadb_test_dir"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# ChromaDB Configuration
CHROMADB_USE_LOCAL_PERSISTENT = False
# Local PersistentClient Configuration
CHROMADB_LOCAL_PATH = os.environ.get(
    "CHROMADB_LOCAL_PATH", os.path.join(WORKING_DIR, "chroma_data")
)
# Remote HttpClient Configuration
CHROMADB_HOST = os.environ.get("CHROMADB_HOST", "localhost")
CHROMADB_PORT = int(os.environ.get("CHROMADB_PORT", 8000))
CHROMADB_AUTH_TOKEN = os.environ.get("CHROMADB_AUTH_TOKEN", "secret-token")
CHROMADB_AUTH_PROVIDER = os.environ.get(
    "CHROMADB_AUTH_PROVIDER", "chromadb.auth.token_authn.TokenAuthClientProvider"
)
CHROMADB_AUTH_HEADER = os.environ.get("CHROMADB_AUTH_HEADER", "X-Chroma-Token")

# Embedding Configuration and Functions
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-large")
EMBEDDING_MAX_TOKEN_SIZE = int(os.environ.get("EMBEDDING_MAX_TOKEN_SIZE", 8192))

# ChromaDB requires knowing the dimension of embeddings upfront when
# creating a collection. The embedding dimension is model-specific
# (e.g. text-embedding-3-large uses 3072 dimensions)
# we dynamically determine it by running a test embedding
# and then pass it to the ChromaDBStorage class


async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embed(
        texts,
        model=EMBEDDING_MODEL,
    )


async def get_embedding_dimension():
    test_text = ["This is a test sentence."]
    embedding = await embedding_func(test_text)
    return embedding.shape[1]


async def create_embedding_function_instance():
    # Get embedding dimension
    embedding_dimension = await get_embedding_dimension()
    # Create embedding function instance
    return EmbeddingFunc(
        embedding_dim=embedding_dimension,
        max_token_size=EMBEDDING_MAX_TOKEN_SIZE,
        func=embedding_func,
    )


async def initialize_rag():
    embedding_func_instance = await create_embedding_function_instance()
    if CHROMADB_USE_LOCAL_PERSISTENT:
        rag = LightRAG(
            working_dir=WORKING_DIR,
            llm_model_func=gpt_4o_mini_complete,
            embedding_func=embedding_func_instance,
            vector_storage="ChromaVectorDBStorage",
            log_level="DEBUG",
            embedding_batch_num=32,
            vector_db_storage_cls_kwargs={
                "local_path": CHROMADB_LOCAL_PATH,
                "collection_settings": {
                    "hnsw:space": "cosine",
                    "hnsw:construction_ef": 128,
                    "hnsw:search_ef": 128,
                    "hnsw:M": 16,
                    "hnsw:batch_size": 100,
                    "hnsw:sync_threshold": 1000,
                },
            },
        )
    else:
        rag = LightRAG(
            working_dir=WORKING_DIR,
            llm_model_func=gpt_4o_mini_complete,
            embedding_func=embedding_func_instance,
            vector_storage="ChromaVectorDBStorage",
            log_level="DEBUG",
            embedding_batch_num=32,
            vector_db_storage_cls_kwargs={
                "host": CHROMADB_HOST,
                "port": CHROMADB_PORT,
                "auth_token": CHROMADB_AUTH_TOKEN,
                "auth_provider": CHROMADB_AUTH_PROVIDER,
                "auth_header_name": CHROMADB_AUTH_HEADER,
                "collection_settings": {
                    "hnsw:space": "cosine",
                    "hnsw:construction_ef": 128,
                    "hnsw:search_ef": 128,
                    "hnsw:M": 16,
                    "hnsw:batch_size": 100,
                    "hnsw:sync_threshold": 1000,
                },
            },
        )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


def main():
    # Initialize RAG instance
    rag = asyncio.run(initialize_rag())

    with open("./book.txt", "r", encoding="utf-8") as f:
        rag.insert(f.read())

    # Perform naive search
    print(
        rag.query(
            "What are the top themes in this story?", param=QueryParam(mode="naive")
        )
    )

    # Perform local search
    print(
        rag.query(
            "What are the top themes in this story?", param=QueryParam(mode="local")
        )
    )

    # Perform global search
    print(
        rag.query(
            "What are the top themes in this story?", param=QueryParam(mode="global")
        )
    )

    # Perform hybrid search
    print(
        rag.query(
            "What are the top themes in this story?", param=QueryParam(mode="hybrid")
        )
    )


if __name__ == "__main__":
    main()
