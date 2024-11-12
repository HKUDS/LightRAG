import os
from lightrag import LightRAG, QueryParam
from lightrag.llm import hf_model_complete, hf_embedding
from lightrag.utils import EmbeddingFunc
from transformers import AutoModel, AutoTokenizer


#########
# Uncomment the below two lines if running in a jupyter notebook to handle the async nature of rag.insert()
# import nest_asyncio
# nest_asyncio.apply()
#########

WORKING_DIR = "./local_neo4jWorkDir"
system_prompt="What can be logically inferred from the information presented in the text?"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=hf_model_complete(
        system_prompt=system_prompt,

    ),
    llm_model_name="meta-llama/Llama-3.1-8B-Instruct",
    embedding_func=EmbeddingFunc(
        embedding_dim=384,
        max_token_size=5000,
        func=lambda texts: hf_embedding(
            texts,
            tokenizer=AutoTokenizer.from_pretrained(
                "sentence-transformers/all-MiniLM-L6-v2"
            ),
            embed_model=AutoModel.from_pretrained(
                "sentence-transformers/all-MiniLM-L6-v2"
            ),
        ),
    ),  # Use gpt_4o_mini_complete LLM model
    kg="Neo4JStorage",
    log_level="INFO",
)

with open("./book.txt") as f:
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
