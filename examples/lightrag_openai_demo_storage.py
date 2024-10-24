import os

from lightrag import LightRAG, QueryParam
from lightrag.llm import gpt_4o_mini_complete
from lightrag.storage import Neo4jKVStorage
# WORKING_DIR = "./dickens"

# if not os.path.exists(WORKING_DIR):
#     os.mkdir(WORKING_DIR)

rag = LightRAG (
    working_dir="./dickens",
    llm_model_func=gpt_4o_mini_complete,
    neo4j_config={
        "uri": "neo4j://localhost:7687",
        "username": "neo4j",
        "password": "admin12345"
    },
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
