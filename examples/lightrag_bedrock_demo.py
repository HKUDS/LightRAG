"""
LightRAG meets Amazon Bedrock ⛰️
"""

import os
import logging

from lightrag import LightRAG, QueryParam
from lightrag.llm.bedrock import bedrock_complete, bedrock_embed
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status

import asyncio
import nest_asyncio

nest_asyncio.apply()

logging.getLogger("aiobotocore").setLevel(logging.WARNING)

WORKING_DIR = "./dickens"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=bedrock_complete,
        llm_model_name="Anthropic Claude 3 Haiku // Amazon Bedrock",
        embedding_func=EmbeddingFunc(
            embedding_dim=1024, max_token_size=8192, func=bedrock_embed
        ),
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


def main():
    rag = asyncio.run(initialize_rag())

    with open("./book.txt", "r", encoding="utf-8") as f:
        rag.insert(f.read())

    for mode in ["naive", "local", "global", "hybrid"]:
        print("\n+-" + "-" * len(mode) + "-+")
        print(f"| {mode.capitalize()} |")
        print("+-" + "-" * len(mode) + "-+\n")
        print(
            rag.query(
                "What are the top themes in this story?", param=QueryParam(mode=mode)
            )
        )


if __name__ == "__main__":
    main()
