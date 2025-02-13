import os
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_embed, openai_complete_if_cache
from lightrag.utils import EmbeddingFunc

# WorkingDir
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKING_DIR = os.path.join(ROOT_DIR, "myKG")
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)
print(f"WorkingDir: {WORKING_DIR}")

# redis
os.environ["REDIS_URI"] = "redis://localhost:6379"

# neo4j
BATCH_SIZE_NODES = 500
BATCH_SIZE_EDGES = 100
os.environ["NEO4J_URI"] = "bolt://117.50.173.35:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "12345678"

# milvus
os.environ["MILVUS_URI"] = "http://117.50.173.35:19530"
os.environ["MILVUS_USER"] = "root"
os.environ["MILVUS_PASSWORD"] = "Milvus"
os.environ["MILVUS_DB_NAME"] = "lightrag"


async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        "deepseek-chat",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key="sk-91d0b59f25554251aa813ed756d79a6d",
        base_url="https://api.deepseek.com",
        **kwargs,
    )


embedding_func = EmbeddingFunc(
    embedding_dim=768,
    max_token_size=512,
    func=lambda texts: ollama_embed(
        texts, embed_model="shaw/dmeta-embedding-zh", host="http://117.50.173.35:11434"
    ),
)

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=llm_model_func,
    llm_model_max_token_size=32768,
    embedding_func=embedding_func,
    chunk_token_size=512,
    chunk_overlap_token_size=256,
    kv_storage="RedisKVStorage",
    graph_storage="Neo4JStorage",
    vector_storage="MilvusVectorDBStorage",
    doc_status_storage="RedisKVStorage",
)

file = "../book.txt"
with open(file, "r", encoding="utf-8") as f:
    rag.insert(f.read())


print(rag.query("谁会3D建模 ？", param=QueryParam(mode="mix")))
