from lightrag import LightRAG, QueryParam
from lightrag.llm import gpt_4o_mini_complete, gpt_4o_complete
import os
WORKING_DIR = "./ragtest"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=gpt_4o_mini_complete  # Use gpt_4o_mini_complete LLM model
    # llm_model_func=gpt_4o_complete  # Optionally, use a stronger model
)

query = "What is the purpose of this project?"

# Perform naive search
print("\033[94m" + rag.query(query, param=QueryParam(mode="naive")) + "\033[0m")
# Perform local search
print("\033[92m" + rag.query(query, param=QueryParam(mode="local")) + "\033[0m")
# Perform global search
print("\033[90m" + rag.query(query, param=QueryParam(mode="global")) + "\033[0m")
# Perform hybrid search
print("\033[93m" + rag.query(query, param=QueryParam(mode="hybrid")) + "\033[0m")
