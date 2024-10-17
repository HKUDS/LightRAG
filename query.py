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
# Perform naive search
print(rag.query("what happens when i click the palmier tab button?", param=QueryParam(mode="naive")))
# Perform local search
print(rag.query("what happens when i click the palmier tab button?", param=QueryParam(mode="local")))
# Perform global search
print(rag.query("what happens when i click the palmier tab button?", param=QueryParam(mode="global")))
# Perform hybrid search
print(rag.query("what happens when i click the palmier tab button?", param=QueryParam(mode="hybrid")))