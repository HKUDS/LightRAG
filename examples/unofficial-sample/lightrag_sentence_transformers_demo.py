import os

from lightrag import LightRAG, QueryParam
from lightrag.llm.hf import hf_model_complete
from lightrag.llm.sentence_transformers import sentence_transformers_embed
from lightrag.utils import EmbeddingFunc
from sentence_transformers import SentenceTransformer

import asyncio
import nest_asyncio

nest_asyncio.apply()

WORKING_DIR = "./dickens"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=hf_model_complete,
        llm_model_name="meta-llama/Llama-3.1-8B-Instruct",
        embedding_func=EmbeddingFunc(
            embedding_dim=384,
            max_token_size=512,
            func=lambda texts: sentence_transformers_embed(
                texts,
                model=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2"),
            ),
        ),
    )

    await rag.initialize_storages()  # Auto-initializes pipeline_status
    return rag


def main():
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
