import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.utils import EmbeddingFunc
import numpy as np

#########
# Uncomment the below two lines if running in a jupyter notebook to handle the async nature of rag.insert()
# import nest_asyncio
# nest_asyncio.apply()
#########
WORKING_DIR = "./mongodb_test_dir"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


os.environ["OPENAI_API_KEY"] = "sk-"
os.environ["MONGO_URI"] = "mongodb://0.0.0.0:27017/?directConnection=true"
os.environ["MONGO_DATABASE"] = "LightRAG"
os.environ["MONGO_KG_COLLECTION"] = "MDB_KG"

# Embedding Configuration and Functions
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-large")
EMBEDDING_MAX_TOKEN_SIZE = int(os.environ.get("EMBEDDING_MAX_TOKEN_SIZE", 8192))


async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embed.func(
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

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=gpt_4o_mini_complete,
        embedding_func=embedding_func_instance,
        graph_storage="MongoGraphStorage",
        log_level="DEBUG",
    )

    await rag.initialize_storages()  # Auto-initializes pipeline_status
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
