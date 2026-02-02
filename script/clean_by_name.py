import os
import asyncio
import sys
import numpy as np
from loguru import logger
from dotenv import load_dotenv

# === Load Env ===
load_dotenv()

# LightRAG Imports
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from lightrag.llm.openai import azure_openai_complete, openai_embed
from lightrag.utils import DocStatus # 引入狀態枚舉

# === Path Setup ===
current_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_path))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# === Config ===
WORKING_DIR = "./data/rag_storage"
TARGET_DOC_ID = "doc-1a240209386d418c61ec3d1ae4a8a738" # 🔥 填入那個頑固的 ID

# === Helper ===
def get_clean_env(key, default=None):
    val = os.getenv(key, default)
    return val.strip() if val else val

async def main():
    logger.info(f"🚑 啟動 LightRAG 修復程序 (Fixer Mode)...")
    
    # 1. 初始化 LightRAG (為了連接 Storage)
    embed_model_name = get_clean_env("EMBEDDING_MODEL")
    embed_api_key = get_clean_env("EMBEDDING_BINDING_API_KEY") 
    embed_base_url = get_clean_env("EMBEDDING_BINDING_HOST")
    embed_dim = int(get_clean_env("EMBEDDING_DIM", "1024"))

    async def embedding_func_wrapper(texts: list[str]) -> np.ndarray:
        return await openai_embed.func(
            texts=texts,
            model=embed_model_name,
            api_key=embed_api_key,     
            base_url=embed_base_url
        )

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=azure_openai_complete,
        embedding_func=EmbeddingFunc(
            embedding_dim=embed_dim,
            max_token_size=8192,
            func=embedding_func_wrapper
        ),
    )
    await rag.initialize_storages()

    # 2. 嘗試掃描所有 Chunks，找出屬於目標文檔的孤兒
    logger.info("🔍 正在掃描 text_chunks 尋找孤兒碎片...")
    
    # 存取內部數據 (Private Access)
    if not hasattr(rag.text_chunks, "_data"):
        logger.error("❌ 無法存取 text_chunks，操作中止。")
        return

    all_chunks = rag.text_chunks._data
    found_chunk_ids = []

    for chunk_id, chunk_data in all_chunks.items():
        # 檢查這個 chunk 是否屬於我們的目標文檔
        # 通常 chunk_data 會有 'doc_id' 欄位
        if chunk_data.get("doc_id") == TARGET_DOC_ID:
            found_chunk_ids.append(chunk_id)

    logger.info(f"📊 找到 {len(found_chunk_ids)} 個屬於 {TARGET_DOC_ID} 的孤兒碎片。")

    # 3. 偽造 doc_status (Mocking the Status)
    logger.info("🛠️ 正在偽造 doc_status 記錄...")
    
    fake_status = {
        "status": DocStatus.PROCESSED, # 告訴系統這是處理好的
        "file_path": "force_restored_file", # 隨便填，唔重要
        "chunks_list": found_chunk_ids, # 🔥 關鍵：把找到的 ID 塞進去
        "chunks_count": len(found_chunk_ids),
        "create_time": 0,
        "update_time": 0
    }

    # 強制寫入 KV Store
    await rag.doc_status.upsert({TARGET_DOC_ID: fake_status})
    logger.success("✅ doc_status 已成功修復！")

    # 4. 現在可以執行正規刪除了！
    logger.info("🗑️ 執行正規刪除 (adelete_by_doc_id)...")
    try:
        # 因為 doc_status 存在了，這次它會乖乖刪除 vector, graph 和 text chunks
        await rag.adelete_by_doc_id(TARGET_DOC_ID)
        logger.success("🎉 完美！文檔及其所有碎片已被徹底清除。")
    except Exception as e:
        logger.error(f"❌ 刪除失敗: {e}")

if __name__ == "__main__":
    asyncio.run(main())