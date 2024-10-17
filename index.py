from lightrag import LightRAG
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
# Load all .txt files from the input folder
input_folder = os.path.join(WORKING_DIR, "input")
texts_to_insert = []
for filename in os.listdir(input_folder):
    if filename.endswith(".txt"):
        file_path = os.path.join(input_folder, filename)
        with open(file_path, "r") as f:
            texts_to_insert.append(f.read())
# Batch insert all loaded texts
rag.insert(texts_to_insert)