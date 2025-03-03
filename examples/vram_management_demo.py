import os
import time
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status

# Working directory and the directory path for text files
WORKING_DIR = "./dickens"
TEXT_FILES_DIR = "/llm/mt"

# Create the working directory if it doesn't exist
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def initialize_rag():
    # Initialize LightRAG
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name="qwen2.5:3b-instruct-max-context",
        embedding_func=EmbeddingFunc(
            embedding_dim=768,
            max_token_size=8192,
            func=lambda texts: ollama_embed(texts, embed_model="nomic-embed-text"),
        ),
    )
    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


# Read all .txt files from the TEXT_FILES_DIR directory
texts = []
for filename in os.listdir(TEXT_FILES_DIR):
    if filename.endswith(".txt"):
        file_path = os.path.join(TEXT_FILES_DIR, filename)
        with open(file_path, "r", encoding="utf-8") as file:
            texts.append(file.read())


# Batch insert texts into LightRAG with a retry mechanism
def insert_texts_with_retry(rag, texts, retries=3, delay=5):
    for _ in range(retries):
        try:
            rag.insert(texts)
            return
        except Exception as e:
            print(
                f"Error occurred during insertion: {e}. Retrying in {delay} seconds..."
            )
            time.sleep(delay)
    raise RuntimeError("Failed to insert texts after multiple retries.")


def main():
    # Initialize RAG instance
    rag = asyncio.run(initialize_rag())

    insert_texts_with_retry(rag, texts)

    # Perform different types of queries and handle potential errors
    try:
        print(
            rag.query(
                "What are the top themes in this story?", param=QueryParam(mode="naive")
            )
        )
    except Exception as e:
        print(f"Error performing naive search: {e}")

    try:
        print(
            rag.query(
                "What are the top themes in this story?", param=QueryParam(mode="local")
            )
        )
    except Exception as e:
        print(f"Error performing local search: {e}")

    try:
        print(
            rag.query(
                "What are the top themes in this story?",
                param=QueryParam(mode="global"),
            )
        )
    except Exception as e:
        print(f"Error performing global search: {e}")

    try:
        print(
            rag.query(
                "What are the top themes in this story?",
                param=QueryParam(mode="hybrid"),
            )
        )
    except Exception as e:
        print(f"Error performing hybrid search: {e}")

    # Function to clear VRAM resources
    def clear_vram():
        os.system("sudo nvidia-smi --gpu-reset")

    # Regularly clear VRAM to prevent overflow
    clear_vram_interval = 3600  # Clear once every hour
    start_time = time.time()

    while True:
        current_time = time.time()
        if current_time - start_time > clear_vram_interval:
            clear_vram()
            start_time = current_time
        time.sleep(60)  # Check the time every minute


if __name__ == "__main__":
    main()
