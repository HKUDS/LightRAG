import os
import shutil  # Add this import
from dataclasses import dataclass
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete


#########
# Uncomment the below two lines if running in a jupyter notebook to handle the async nature of rag.insert()
# import nest_asyncio
# nest_asyncio.apply()
#########

@dataclass
class GPTEmbedder:
    embedding_dim: int = 1536  # OpenAI's ada-002 model dimension

    def __call__(self, text):
        from lightrag.llm.openai import openai_embed
        return openai_embed(text)  # This will use the default ada-002 model

WORKING_DIR = "./local_neo4jWorkDir"

# Delete the existing working directory if it exists
if os.path.exists(WORKING_DIR):
    shutil.rmtree(WORKING_DIR)

# Create fresh directory
os.mkdir(WORKING_DIR)

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=gpt_4o_mini_complete,  # Use gpt_4o_mini_complete LLM model
    embedding_func=GPTEmbedder(),  # Add the embedding function wrapper
    graph_storage="Neo4JStorage",
    log_level="INFO"
    # llm_model_func=gpt_4o_complete  # Optionally, use a stronger model
)

with open("/mnt/ebsdata/SANDBOX/data/example1/Transpower,ICT Strategy 2021_p011_20250213_125728.md", "r") as f:
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
