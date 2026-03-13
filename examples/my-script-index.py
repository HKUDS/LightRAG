import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.utils import setup_logger
from examples.rag_texts import text_to_rag

setup_logger("lightrag", level="INFO")

WORKING_DIR = "./rag_storage"
os.makedirs(WORKING_DIR, exist_ok=True)

async def main():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        embedding_func=openai_embed,
        llm_model_func=gpt_4o_mini_complete,
    )
    await rag.initialize_storages()  # Required!

    # Insert text
    await rag.ainsert(text_to_rag)

    # Query
    result = await rag.aquery(
        "What are the top themes in this story?",
        param=QueryParam(mode="hybrid")
    )
    print(result)

    await rag.finalize_storages()

asyncio.run(main())
