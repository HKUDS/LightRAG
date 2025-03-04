import sys
import os
from pathlib import Path
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
import numpy as np
from lightrag.kg.shared_storage import initialize_pipeline_status

print(os.getcwd())
script_directory = Path(__file__).resolve().parent.parent
sys.path.append(os.path.abspath(script_directory))

WORKING_DIR = "./dickens"

# We use OpenAI compatible API to call LLM on Oracle Cloud
# More docs here https://github.com/jin38324/OCI_GenAI_access_gateway
BASE_URL = "http://xxx.xxx.xxx.xxx:8088/v1/"
APIKEY = "ocigenerativeai"
CHATMODEL = "cohere.command-r-plus"
EMBEDMODEL = "cohere.embed-multilingual-v3.0"
CHUNK_TOKEN_SIZE = 1024
MAX_TOKENS = 4000

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

os.environ["ORACLE_USER"] = "username"
os.environ["ORACLE_PASSWORD"] = "xxxxxxxxx"
os.environ["ORACLE_DSN"] = "xxxxxxx_medium"
os.environ["ORACLE_CONFIG_DIR"] = "path_to_config_dir"
os.environ["ORACLE_WALLET_LOCATION"] = "path_to_wallet_location"
os.environ["ORACLE_WALLET_PASSWORD"] = "wallet_password"
os.environ["ORACLE_WORKSPACE"] = "company"


async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        CHATMODEL,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=APIKEY,
        base_url=BASE_URL,
        **kwargs,
    )


async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embed(
        texts,
        model=EMBEDMODEL,
        api_key=APIKEY,
        base_url=BASE_URL,
    )


async def get_embedding_dim():
    test_text = ["This is a test sentence."]
    embedding = await embedding_func(test_text)
    embedding_dim = embedding.shape[1]
    return embedding_dim


async def initialize_rag():
    # Detect embedding dimension
    embedding_dimension = await get_embedding_dim()
    print(f"Detected embedding dimension: {embedding_dimension}")

    # Initialize LightRAG
    # We use Oracle DB as the KV/vector/graph storage
    # You can add `addon_params={"example_number": 1, "language": "Simplfied Chinese"}` to control the prompt
    rag = LightRAG(
        # log_level="DEBUG",
        working_dir=WORKING_DIR,
        entity_extract_max_gleaning=1,
        enable_llm_cache=True,
        enable_llm_cache_for_entity_extract=True,
        embedding_cache_config=None,  # {"enabled": True,"similarity_threshold": 0.90},
        chunk_token_size=CHUNK_TOKEN_SIZE,
        llm_model_max_token_size=MAX_TOKENS,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dimension,
            max_token_size=500,
            func=embedding_func,
        ),
        graph_storage="OracleGraphStorage",
        kv_storage="OracleKVStorage",
        vector_storage="OracleVectorDBStorage",
        addon_params={
            "example_number": 1,
            "language": "Simplfied Chinese",
            "entity_types": ["organization", "person", "geo", "event"],
            "insert_batch_size": 2,
        },
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


async def main():
    try:
        # Initialize RAG instance
        rag = await initialize_rag()

        # Extract and Insert into LightRAG storage
        with open(WORKING_DIR + "/docs.txt", "r", encoding="utf-8") as f:
            all_text = f.read()
            texts = [x for x in all_text.split("\n") if x]

        # New mode use pipeline
        await rag.apipeline_enqueue_documents(texts)
        await rag.apipeline_process_enqueue_documents()

        # Old method use ainsert
        # await rag.ainsert(texts)

        # Perform search in different modes
        modes = ["naive", "local", "global", "hybrid"]
        for mode in modes:
            print("=" * 20, mode, "=" * 20)
            print(
                await rag.aquery(
                    "What are the top themes in this story?",
                    param=QueryParam(mode=mode),
                )
            )
            print("-" * 100, "\n")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())
