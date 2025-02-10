
import os
from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI
from langchain_community.embeddings.oci_generative_ai import OCIGenAIEmbeddings
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from lightrag.kg.oracle_impl import OracleDB

from llm import OCICohereCommandRLLM
from llm import llm_model_ociCohereRLLM_multithread


print(os.getcwd())

WORKING_DIR = "./dickens"
AUTH_TYPE = os.getenv("OCI_AUTH_TYPE", "API_KEY")
OCI_PROFILE = os.getenv("OCI_PROFILE")
REGION = os.getenv("REGION", "us-ashburn-1")
COMPARTMENT_ID = os.getenv("COMPARTMENT_ID")
ENVIRONMENT = os.getenv("ENVIRONMENT")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
ORACLE_DB_DSN = os.getenv("ORACLE_DB_DSN")
ORACLE_DB_USER = os.getenv("ORACLE_DB_USER")
ORACLE_DB_PASSWORD = os.getenv("ORACLE_DB_PASSWORD")
CHATMODEL = "cohere.command-r-plus"
EMBEDMODEL = "cohere.embed-multilingual-v3.0"



async def embeddings():
    return await OCIGenAIEmbeddings(
    model_id="cohere.embed-multilingual-v3.0",          # EMBEDMODEL
    service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
    compartment_id=COMPARTMENT_ID
    )

llm_model = llm_model_ociCohereRLLM_multithread()

# embeddings = OCIGenAIEmbeddings(
#     model_id="cohere.embed-multilingual-v3.0",          # EMBEDMODEL
#     service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
#     compartment_id=COMPARTMENT_ID
# )


# llm = LLMModel().llm

async def main():

    oracle_db = OracleDB(
        config = {
            "user": ORACLE_DB_USER,
            "password": ORACLE_DB_PASSWORD,
            "dsn": ORACLE_DB_DSN,
        })
    await oracle_db.check_tables()

    rag = LightRAG(
                enable_llm_cache=False,
                working_dir=WORKING_DIR,
                chunk_token_size=512,
                llm_model_func=llm_model,
                embedding_func=EmbeddingFunc(max_token_size=512, embedding_dim=512, func=embeddings),
                graph_storage="OracleGraphStorage",
                kv_storage="OracleKVStorage",
                vector_storage="OracleVectorDBStorage",
            )

    # Setthe KV/vector/graph storage's `db` property, so all operation will use same connection pool
    rag.graph_storage_cls.db = oracle_db
    rag.key_string_value_json_storage_cls.db = oracle_db
    rag.vector_db_storage_cls.db = oracle_db
    # add embedding_func for graph database, it's deleted in commit 5661d76860436f7bf5aef2e50d9ee4a59660146c
    rag.chunk_entity_relation_graph.embedding_func = rag.embedding_func

    with open("./book.txt", "r", encoding="utf-8") as f:
        await rag.insert(f.read())

    modes = ["naive", "local", "global", "hybrid"]
    for mode in modes:
        print("=" * 20, mode, "=" * 20)
        print(await rag.aquery("What are the top themes in this story?", param=QueryParam(mode=mode),))
        print("-" * 100, "\n")




