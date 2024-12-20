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
from lightrag.llm import ollama_embedding, ollama_model_complete
from lightrag.utils import EmbeddingFunc

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

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=ollama_model_complete,
    llm_model_name="llama3.1:8b",
    llm_model_max_async=4,
    llm_model_max_token_size=32768,
    llm_model_kwargs={"host": "http://localhost:11434", "options": {"num_ctx": 32768}},
    embedding_func=EmbeddingFunc(
        embedding_dim=768,
        max_token_size=8192,
        func=lambda texts: ollama_embedding(
            texts, embed_model="nomic-embed-text", host="http://localhost:11434"
        ),
    ),
    graph_storage="GremlinStorage",
)

with open("./book.txt", "r", encoding="utf-8") as f:
    rag.insert(f.read())

# Perform naive search
print(
    rag.query("What are the top themes in this story?", param=QueryParam(mode="naive"))
)

# Perform local search
print(
    rag.query("What are the top themes in this story?", param=QueryParam(mode="local"))
)

# Perform global search
print(
    rag.query("What are the top themes in this story?", param=QueryParam(mode="global"))
)

# Perform hybrid search
print(
    rag.query("What are the top themes in this story?", param=QueryParam(mode="hybrid"))
)

# stream response
resp = rag.query(
    "What are the top themes in this story?",
    param=QueryParam(mode="hybrid", stream=True),
)


async def print_stream(stream):
    async for chunk in stream:
        print(chunk, end="", flush=True)


if inspect.isasyncgen(resp):
    asyncio.run(print_stream(resp))
else:
    print(resp)
