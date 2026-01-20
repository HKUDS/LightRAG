import os

from lightrag import LightRAG, QueryParam
from modelscope import AutoModel, AutoTokenizer
from lightrag.llm.modelscope import ms_embed, ms_model_complete
from lightrag.utils import EmbeddingFunc

import asyncio
import nest_asyncio

nest_asyncio.apply()

WORKING_DIR = "./dickens"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)
import torch
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
torch.backends.mps.enabled = False

async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ms_model_complete,
        llm_model_name="Qwen/Qwen2.5-0.5B-Instruct",
        embedding_func=EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=5000,
            func=lambda texts: ms_embed(
                texts,
                tokenizer=AutoTokenizer.from_pretrained(
                    "Qwen/Qwen3-Embedding-0.6B"
                ),
                embed_model=AutoModel.from_pretrained(
                    "Qwen/Qwen3-Embedding-0.6B"
                ).to("cpu"),
            ),
        ),
    )

    await rag.initialize_storages()  # Auto-initializes pipeline_status
    return rag


def main():
    import shutil
    if os.path.exists(WORKING_DIR):
        shutil.rmtree(WORKING_DIR)
    os.mkdir(WORKING_DIR)

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
