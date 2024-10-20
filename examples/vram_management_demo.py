import os
import time
from lightrag import LightRAG, QueryParam
from lightrag.llm import ollama_model_complete, ollama_embedding
from lightrag.utils import EmbeddingFunc

# 工作目录和文本文件目录路径
WORKING_DIR = "./dickens"
TEXT_FILES_DIR = "/llm/mt"

# 如果工作目录不存在，则创建该目录
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# 初始化 LightRAG
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=ollama_model_complete,
    llm_model_name="qwen2.5:3b-instruct-max-context",
    embedding_func=EmbeddingFunc(
        embedding_dim=768,
        max_token_size=8192,
        func=lambda texts: ollama_embedding(texts, embed_model="nomic-embed-text"),
    ),
)

# 读取 TEXT_FILES_DIR 目录下所有的 .txt 文件
texts = []
for filename in os.listdir(TEXT_FILES_DIR):
    if filename.endswith('.txt'):
        file_path = os.path.join(TEXT_FILES_DIR, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            texts.append(file.read())

# 批量插入文本到 LightRAG，带有重试机制
def insert_texts_with_retry(rag, texts, retries=3, delay=5):
    for _ in range(retries):
        try:
            rag.insert(texts)
            return
        except Exception as e:
            print(f"Error occurred during insertion: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
    raise RuntimeError("Failed to insert texts after multiple retries.")

insert_texts_with_retry(rag, texts)

# 执行不同类型的查询，并处理潜在的错误
try:
    print(rag.query("What are the top themes in this story?", param=QueryParam(mode="naive")))
except Exception as e:
    print(f"Error performing naive search: {e}")

try:
    print(rag.query("What are the top themes in this story?", param=QueryParam(mode="local")))
except Exception as e:
    print(f"Error performing local search: {e}")

try:
    print(rag.query("What are the top themes in this story?", param=QueryParam(mode="global")))
except Exception as e:
    print(f"Error performing global search: {e}")

try:
    print(rag.query("What are the top themes in this story?", param=QueryParam(mode="hybrid")))
except Exception as e:
    print(f"Error performing hybrid search: {e}")

# 清理 VRAM 资源的函数
def clear_vram():
    os.system("sudo nvidia-smi --gpu-reset")

# 定期清理 VRAM 以防止溢出
clear_vram_interval = 3600  # 每小时清理一次
start_time = time.time()

while True:
    current_time = time.time()
    if current_time - start_time > clear_vram_interval:
        clear_vram()
        start_time = current_time
    time.sleep(60)  # 每分钟检查一次时间