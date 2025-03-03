import asyncio
import inspect
import os

# Uncomment these lines below to filter out somewhat verbose INFO level
# logging prints (the default loglevel is INFO).
# This has to go before the lightrag imports to work,
# which triggers linting errors, so we keep it commented out:
# import logging
# logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.WARN)

from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_embed, ollama_model_complete
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status

WORKING_DIR = "./dickens_gremlin"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# Gremlin
os.environ["GREMLIN_HOST"] = "localhost"
os.environ["GREMLIN_PORT"] = "8182"
os.environ["GREMLIN_GRAPH"] = "dickens"

# Creating a non-default source requires manual
# configuration and a restart on the server: use the dafault "g"
os.environ["GREMLIN_TRAVERSE_SOURCE"] = "g"

# No authorization by default on docker tinkerpop/gremlin-server
os.environ["GREMLIN_USER"] = ""
os.environ["GREMLIN_PASSWORD"] = ""


async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name="llama3.1:8b",
        llm_model_max_async=4,
        llm_model_max_token_size=32768,
        llm_model_kwargs={
            "host": "http://localhost:11434",
            "options": {"num_ctx": 32768},
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
            func=lambda texts: ollama_embed(
                texts, embed_model="nomic-embed-text", host="http://localhost:11434"
            ),
        ),
        graph_storage="GremlinStorage",
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


async def print_stream(stream):
    async for chunk in stream:
        print(chunk, end="", flush=True)


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

    # stream response
    resp = rag.query(
        "What are the top themes in this story?",
        param=QueryParam(mode="hybrid", stream=True),
    )

    if inspect.isasyncgen(resp):
        asyncio.run(print_stream(resp))
    else:
        print(resp)


if __name__ == "__main__":
    main()
