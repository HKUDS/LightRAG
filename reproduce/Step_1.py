import os
import json
import time
import asyncio

from lightrag import LightRAG
from lightrag.kg.shared_storage import initialize_pipeline_status


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


async def initialize_rag():
    rag = LightRAG(working_dir=WORKING_DIR)

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


def main():
    # Initialize RAG instance
    rag = asyncio.run(initialize_rag())
    insert_text(rag, f"../datasets/unique_contexts/{cls}_unique_contexts.json")


if __name__ == "__main__":
    main()
