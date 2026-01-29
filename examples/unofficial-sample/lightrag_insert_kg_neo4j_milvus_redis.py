import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKING_DIR = os.path.join(ROOT_DIR, "myKG")
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)
print(f"WorkingDir: {WORKING_DIR}")

# redis
os.environ["REDIS_URI"] = "redis://localhost:6379"

# neo4j
os.environ["NEO4J_URI"] = "neo4j://localhost:7687"
os.environ["NEO4J_USERNAME"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "12345678"

# milvus
os.environ["MILVUS_URI"] = "http://localhost:19530"
os.environ["MILVUS_USER"] = "root"
os.environ["MILVUS_PASSWORD"] = "Milvus"
os.environ["MILVUS_DB_NAME"] = "lightrag"

embedding_func = EmbeddingFunc(
    embedding_dim=1024,
    max_token_size=8192,
    func=lambda texts: ollama_embed(
        texts, embed_model="bge-m3:latest", host="http://localhost:11434"
    ),
)


async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name="deepseek-r1:32b",
        llm_model_kwargs={
            "host": "http://localhost:11434",
            "options": {"num_ctx": 65536},
            "timeout": 300,
        },
        embedding_func=embedding_func,
        chunk_token_size=8192,
        chunk_overlap_token_size=256,
        kv_storage="RedisKVStorage",
        graph_storage="Neo4JStorage",
        vector_storage="MilvusVectorDBStorage",
        doc_status_storage="RedisDocStatusStorage",
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag


async def main():
    rag = await initialize_rag()
    ##       relaton_types: keywords
    ##       Entity and relationship attributes can be placed in their respective description fields.
    custom_kg = {
        "entities": [
            {
                "entity_name": "CompanyA",
                "entity_type": "Organization",
                "description": "A major technology company",
            },
            {
                "entity_name": "ProductX",
                "entity_type": "Product",
                "description": "A popular product developed by CompanyA",
            },
            {
                "entity_name": "PersonA",
                "entity_type": "Person",
                "description": "A renowned researcher in AI",
            },
            {
                "entity_name": "UniversityB",
                "entity_type": "Organization",
                "description": "A leading university specializing in technology and sciences",
            },
            {
                "entity_name": "CityC",
                "entity_type": "Location",
                "description": "A large metropolitan city known for its culture and economy",
            },
            {
                "entity_name": "EventY",
                "entity_type": "Event",
                "description": "An annual technology conference held in CityC",
            },
        ],
        "relationships": [
            {
                "src_id": "CompanyA",
                "tgt_id": "ProductX",
                "description": "CompanyA develops ProductX",
                "keywords": "develop",
            },
            {
                "src_id": "PersonA",
                "tgt_id": "UniversityB",
                "description": "PersonA works at UniversityB",
                "keywords": "employment",
            },
            {
                "src_id": "CityC",
                "tgt_id": "EventY",
                "description": "EventY is hosted in CityC",
                "keywords": "host",
            },
        ],
    }

    print("Inserting custom Knowledge Graph...")
    await rag.ainsert_custom_kg(custom_kg)
    print("知识图谱已导入！")
    output_file = os.path.join(WORKING_DIR, "query_results.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        for mode in ["bypass", "local_graph", "global_graph", "hybrid_graph"]:
            f.write("\n=====================\n")
            f.write(f"Query mode: {mode}\n")
            f.write("=====================\n")
            resp = await rag.aquery(
                "What product did CompanyA create?",
                param=QueryParam(mode=mode, stream=False),
            )
            f.write(str(resp) + "\n")
    print(f"所有查询结果已写入: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
