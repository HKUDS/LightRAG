"""LightRAG Demo with Cloudflare Workers AI Models.

This example demonstrates how to use LightRAG with Cloudflare Workers AI for
both text generation (e.g. Llama 3.3 70B Instruct) and text embeddings (BGE-M3).

Prerequisites:
    1. Set CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID environment variables:
       export CLOUDFLARE_API_TOKEN='your-api-token'
       export CLOUDFLARE_ACCOUNT_ID='your-account-id'

    2. Prepare a text file named 'book.txt' in the current directory
       (or modify BOOK_FILE constant to point to your text file)

Usage:
    python examples/lightrag_cloudflare_demo.py
"""

import os
import asyncio
import nest_asyncio

from lightrag import LightRAG, QueryParam
from lightrag.llm.cloudflare import cloudflare_complete_if_cache, cloudflare_embed
from lightrag.utils import wrap_embedding_func_with_attrs

nest_asyncio.apply()

WORKING_DIR = "./rag_storage"
BOOK_FILE = "./book.txt"

# Validate Cloudflare credentials
CLOUDFLARE_API_TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN") or os.environ.get("CLOUDFLARE_API_KEY")
CLOUDFLARE_ACCOUNT_ID = os.environ.get("CLOUDFLARE_ACCOUNT_ID")

if not CLOUDFLARE_API_TOKEN or not CLOUDFLARE_ACCOUNT_ID:
    raise ValueError(
        "CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID environment variables must be set.\n"
        "Export them with:\n"
        "  export CLOUDFLARE_API_TOKEN='your-api-token'\n"
        "  export CLOUDFLARE_ACCOUNT_ID='your-account-id'"
    )

if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR, exist_ok=True)


# --------------------------------------------------
# LLM Function
# --------------------------------------------------
async def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
    return await cloudflare_complete_if_cache(
        model="@cf/meta/llama-3.3-70b-instruct",
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages or [],
        api_key=CLOUDFLARE_API_TOKEN,
        account_id=CLOUDFLARE_ACCOUNT_ID,
        **kwargs,
    )


# --------------------------------------------------
# Embedding Function
# --------------------------------------------------
@wrap_embedding_func_with_attrs(
    embedding_dim=1024,
    max_token_size=8192,
    supports_asymmetric=False,
    model_name="@cf/baai/bge-m3",
)
async def embedding_func(texts: list[str]) -> list[list[float]]:
    return await cloudflare_embed(
        texts,
        model="@cf/baai/bge-m3",
        api_key=CLOUDFLARE_API_TOKEN,
        account_id=CLOUDFLARE_ACCOUNT_ID,
    )


async def main():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
    )

    await rag.initialize_storages()

    sample_text = (
        "Artificial Intelligence (AI) is intelligence demonstrated by machines, "
        "unlike the natural intelligence displayed by humans and animals. "
        "LightRAG is an innovative dual-level retrieval-augmented generation system "
        "that combines graph-based knowledge indexing with vector search."
    )

    if os.path.exists(BOOK_FILE):
        with open(BOOK_FILE, "r", encoding="utf-8") as f:
            content = f.read()
    else:
        content = sample_text

    print("Indexing document...")
    await rag.ainsert(content)
    print("Indexing completed!")

    print("\n--- Naive Search ---")
    res_naive = await rag.aquery("What is LightRAG?", param=QueryParam(mode="naive"))
    print(res_naive)

    print("\n--- Hybrid Search ---")
    res_hybrid = await rag.aquery("What is LightRAG?", param=QueryParam(mode="hybrid"))
    print(res_hybrid)


if __name__ == "__main__":
    asyncio.run(main())
