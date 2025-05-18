import os
from lightrag import LightRAG, QueryParam
from lightrag.llm.llama_index_impl import (
    llama_index_complete_if_cache,
    llama_index_embed,
)
from lightrag.utils import EmbeddingFunc
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
import asyncio
import nest_asyncio

nest_asyncio.apply()

from lightrag.kg.shared_storage import initialize_pipeline_status

# Configure working directory
WORKING_DIR = "./index_default"
print(f"WORKING_DIR: {WORKING_DIR}")

# Model configuration
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4")
print(f"LLM_MODEL: {LLM_MODEL}")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-large")
print(f"EMBEDDING_MODEL: {EMBEDDING_MODEL}")
EMBEDDING_MAX_TOKEN_SIZE = int(os.environ.get("EMBEDDING_MAX_TOKEN_SIZE", 8192))
print(f"EMBEDDING_MAX_TOKEN_SIZE: {EMBEDDING_MAX_TOKEN_SIZE}")

# OpenAI configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "your-api-key-here")

if not os.path.exists(WORKING_DIR):
    print(f"Creating working directory: {WORKING_DIR}")
    os.mkdir(WORKING_DIR)


# Initialize LLM function
async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    try:
        # Initialize OpenAI if not in kwargs
        if "llm_instance" not in kwargs:
            llm_instance = OpenAI(
                model=LLM_MODEL,
                api_key=OPENAI_API_KEY,
                temperature=0.7,
            )
            kwargs["llm_instance"] = llm_instance

        response = await llama_index_complete_if_cache(
            kwargs["llm_instance"],
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs,
        )
        return response
    except Exception as e:
        print(f"LLM request failed: {str(e)}")
        raise


# Initialize embedding function
async def embedding_func(texts):
    try:
        embed_model = OpenAIEmbedding(
            model=EMBEDDING_MODEL,
            api_key=OPENAI_API_KEY,
        )
        return await llama_index_embed(texts, embed_model=embed_model)
    except Exception as e:
        print(f"Embedding failed: {str(e)}")
        raise


# Get embedding dimension
async def get_embedding_dim():
    test_text = ["This is a test sentence."]
    embedding = await embedding_func(test_text)
    embedding_dim = embedding.shape[1]
    print(f"embedding_dim={embedding_dim}")
    return embedding_dim


async def initialize_rag():
    embedding_dimension = await get_embedding_dim()

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dimension,
            max_token_size=EMBEDDING_MAX_TOKEN_SIZE,
            func=embedding_func,
        ),
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


def main():
    # Initialize RAG instance
    rag = asyncio.run(initialize_rag())

    # Insert example text
    with open("./book.txt", "r", encoding="utf-8") as f:
        rag.insert(f.read())

    # Test different query modes
    print("\nNaive Search:")
    print(
        rag.query(
            "What are the top themes in this story?", param=QueryParam(mode="naive")
        )
    )

    print("\nLocal Search:")
    print(
        rag.query(
            "What are the top themes in this story?", param=QueryParam(mode="local")
        )
    )

    print("\nGlobal Search:")
    print(
        rag.query(
            "What are the top themes in this story?", param=QueryParam(mode="global")
        )
    )

    print("\nHybrid Search:")
    print(
        rag.query(
            "What are the top themes in this story?", param=QueryParam(mode="hybrid")
        )
    )


if __name__ == "__main__":
    main()
