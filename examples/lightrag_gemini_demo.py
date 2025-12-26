"""
LightRAG Demo with Google Gemini Models

This example demonstrates how to use LightRAG with Google's Gemini 2.0 Flash model
for text generation and the text-embedding-004 model for embeddings.

Prerequisites:
    1. Set GEMINI_API_KEY environment variable:
       export GEMINI_API_KEY='your-actual-api-key'
    
    2. Prepare a text file named 'book.txt' in the current directory
       (or modify BOOK_FILE constant to point to your text file)

Usage:
    python examples/lightrag_gemini_demo.py
"""
import os
import asyncio
import nest_asyncio
import numpy as np

from lightrag import LightRAG, QueryParam
from lightrag.llm.gemini import gemini_model_complete, gemini_embed
from lightrag.utils import wrap_embedding_func_with_attrs

nest_asyncio.apply()

WORKING_DIR = "./rag_storage"
BOOK_FILE = "./book.txt"

# Validate API key
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY environment variable is not set. "
        "Please set it with: export GEMINI_API_KEY='your-api-key'"
    )

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


# --------------------------------------------------
# LLM function
# --------------------------------------------------
async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
    return await gemini_model_complete(
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=GEMINI_API_KEY,
        model_name="gemini-2.0-flash",
        **kwargs,
    )


# --------------------------------------------------
# Embedding function
# --------------------------------------------------
@wrap_embedding_func_with_attrs(
    embedding_dim=768,
    send_dimensions=True,
    max_token_size=2048,
    model_name="models/text-embedding-004",
)
async def embedding_func(texts: list[str]) -> np.ndarray:
    return await gemini_embed.func(
        texts, api_key=GEMINI_API_KEY, model="models/text-embedding-004"
    )


# --------------------------------------------------
# Initialize RAG
# --------------------------------------------------
async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
        llm_model_name="gemini-2.0-flash",
    )

    # 🔑 REQUIRED
    await rag.initialize_storages()
    return rag


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    # Validate book file exists
    if not os.path.exists(BOOK_FILE):
        raise FileNotFoundError(
            f"'{BOOK_FILE}' not found. "
            "Please provide a text file to index in the current directory."
        )
    
    rag = asyncio.run(initialize_rag())

    # Insert text
    with open(BOOK_FILE, "r", encoding="utf-8") as f:
        rag.insert(f.read())

    query = "What are the top themes?"

    print("\nNaive Search:")
    print(rag.query(query, param=QueryParam(mode="naive")))

    print("\nLocal Search:")
    print(rag.query(query, param=QueryParam(mode="local")))

    print("\nGlobal Search:")
    print(rag.query(query, param=QueryParam(mode="global")))

    print("\nHybrid Search:")
    print(rag.query(query, param=QueryParam(mode="hybrid")))


if __name__ == "__main__":
    main()
