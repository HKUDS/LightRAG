import os
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete, openai_embed
from lightrag.utils import EmbeddingFunc

# WorkingDir
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKING_DIR = os.path.join(ROOT_DIR, "myKG")
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)
print(f"WorkingDir: {WORKING_DIR}")

# mongo
os.environ["MONGO_URI"] = "mongodb://thilak-d4748^bigbrain:"+os.environ["S2DB_PASSWORD"]+"@svc-452cc4b1-df20-4130-9e2f-e72ba79e3d46-shared-mongo.aws-virginia-hd2.svc.singlestore.com:27017/?authMechanism=PLAIN&tls=true&loadBalanced=true"
os.environ["MONGO_DATABASE"] = "bigbrain"

# neo4j
BATCH_SIZE_NODES = 500
BATCH_SIZE_EDGES = 100
#os.environ["NEO4J_URI"] = "bolt://localhost:7687"
#os.environ["NEO4J_USERNAME"] = "neo4j"
#os.environ["NEO4J_PASSWORD"] = "neo4j"

# milvus
#os.environ["MILVUS_URI"] = "http://localhost:19530"
#os.environ["MILVUS_USER"] = "root"
#os.environ["MILVUS_PASSWORD"] = "root"
#os.environ["MILVUS_DB_NAME"] = "lightrag"


rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=openai_complete,
    llm_model_name="deepseek-ai/deepseek-llm-7b-chat",
    llm_model_max_async=4,
    llm_model_max_token_size=32768,
    llm_model_kwargs={"base_url": "https://apps.aws-virginia-nbstg1.svc.singlestore.com:8000/modelasaservice/9c985ce0-421b-40a3-ae50-03486a5c8fc3/v1", "api_key": os.environ["LLM_API_KEY"]},
    embedding_func=EmbeddingFunc(
        embedding_dim=1024,
        max_token_size=8192,
        func=lambda texts: openai_embed(
            texts=texts, model="mixedbread-ai/mxbai-embed-xsmall-v1", base_url="https://apps.aws-virginia-nbstg1.svc.singlestore.com:8000/modelasaservice/c0d90767-ec36-489c-a38c-c9d11364cde8", api_key=os.environ["EMBEDDINGS_API_KEY"]
        ),
    ),
    kv_storage="MongoKVStorage",
    graph_storage="MongoGraphStorage",
    vector_storage="SingleStoreVectorDBStorage",
)

file = "./book.txt"
with open(file, "r") as f:
    rag.insert(f.read())

print(
    rag.query("What are the top themes in this story?", param=QueryParam(mode="hybrid"))
)
