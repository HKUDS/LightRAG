import os
import asyncio
import nest_asyncio
import numpy as np

from lightrag import LightRAG, QueryParam
from lightrag.llm.gemini import gemini_model_complete, gemini_embed
from lightrag.utils import wrap_embedding_func_with_attrs

nest_asyncio.apply()

WORKING_DIR = "./rag_storage"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "your-api-key-here")

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
        **kwargs
    )

# --------------------------------------------------
# Embedding function
# --------------------------------------------------
@wrap_embedding_func_with_attrs(
    embedding_dim=768,
    max_token_size=2048,
    model_name="models/text-embedding-004"
)
async def embedding_func(texts: list[str]) -> np.ndarray:
    return await gemini_embed.func(
        texts,
        api_key=GEMINI_API_KEY,
        model="models/text-embedding-004"
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
    rag = asyncio.run(initialize_rag())

    # Insert text
    with open("./book.txt", "r", encoding="utf-8") as f:
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
