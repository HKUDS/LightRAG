"""
LightRAG meets Amazon Bedrock ⛰️
"""

import os

from lightrag import LightRAG, QueryParam
from lightrag.llm import bedrock_complete, bedrock_embedding
from lightrag.utils import EmbeddingFunc

WORKING_DIR = "./dickens"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=bedrock_complete,
    llm_model_name="anthropic.claude-3-haiku-20240307-v1:0",
    node2vec_params = {
        'dimensions': 1024,
        'num_walks': 10,
        'walk_length': 40,
        'window_size': 2,
        'iterations': 3,
        'random_seed': 3
    },
    embedding_func=EmbeddingFunc(
        embedding_dim=1024,
        max_token_size=8192,
        func=lambda texts: bedrock_embedding(texts)
    )
)

with open("./book.txt") as f:
    rag.insert(f.read())

# Naive search
print(rag.query("What are the top themes in this story?", param=QueryParam(mode="naive")))

# Local search
print(rag.query("What are the top themes in this story?", param=QueryParam(mode="local")))

# Global search
print(rag.query("What are the top themes in this story?", param=QueryParam(mode="global")))

# Hybrid search
print(rag.query("What are the top themes in this story?", param=QueryParam(mode="hybrid")))
