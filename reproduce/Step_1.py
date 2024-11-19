import os
import json
import time

from lightrag import LightRAG


def insert_text(rag, file_path):
    with open(file_path, mode="r") as f:
        unique_contexts = json.load(f)

    retries = 0
    max_retries = 3
    while retries < max_retries:
        try:
            rag.insert(unique_contexts)
            break
        except Exception as e:
            retries += 1
            print(f"Insertion failed, retrying ({retries}/{max_retries}), error: {e}")
            time.sleep(10)
    if retries == max_retries:
        print("Insertion failed after exceeding the maximum number of retries")


cls = "agriculture"
WORKING_DIR = f"../{cls}"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = LightRAG(working_dir=WORKING_DIR)

insert_text(rag, f"../datasets/unique_contexts/{cls}_unique_contexts.json")
