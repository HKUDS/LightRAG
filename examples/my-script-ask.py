import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.utils import setup_logger

setup_logger("lightrag", level="INFO")

WORKING_DIR = "./rag_storage"  # Use the same directory as before

async def main():
    # Initialize LightRAG with the same settings as before
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
    )
    await rag.initialize_storages()  # Loads the existing indexed data

    # Ask a question on the already-indexed data
    question = "需要用什么材料烤羊排"
    result = await rag.aquery(
        question,
        param=QueryParam(
            mode="hybrid",  # or "local", "global", etc.
            only_need_context=False,  # Set to True if you only want the retrieved context
        )
    )
    print("Answer:", result)

    await rag.finalize_storages()

asyncio.run(main())
