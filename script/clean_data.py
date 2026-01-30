import os
import asyncio
import numpy as np
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from lightrag.llm.ollama import ollama_embed
from lightrag.llm.openai import azure_openai_complete 
from functools import partial
from dotenv import load_dotenv

load_dotenv()

# === 設定 ===
WORKING_DIR = "./data/rag_storage"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBEDDING_MODEL = "bge-m3"
EMBEDDING_DIM = 1024

async def main():
    print(f"🚀 正在連接 Ollama ({OLLAMA_HOST}) 載入 LightRAG 數據庫...")
    
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=azure_openai_complete, 
        embedding_func=EmbeddingFunc(
            embedding_dim=EMBEDDING_DIM,
            max_token_size=8192,
            func=partial(
                ollama_embed,
                host=OLLAMA_HOST,
                embed_model=EMBEDDING_MODEL
            )
        )
    )

    # 🔥 [關鍵修正] 必須先初始化儲存層，否則刪除操作會報錯
    print("⚙️ 正在初始化 Storage...")
    await rag.initialize_storages()

        # 2. 定義要刪除的壞 ID
    bad_doc_ids = [
        "doc-b4ace52dec7e0f66e2bf9672910f3398",
        "doc-4147cae28d264c7596a59bcf46e8db67",
        "doc-45d100ad9dc1f5149a79d37f796e42e9",
        "doc-0e51218b2ea2fa4ef493c321802fc912"  # 🔥 這是之前被忽略的 0001 文件 ID
    ]

    # 3. 執行刪除
    print(f"🗑️ 準備刪除 {len(bad_doc_ids)} 個損壞的文檔...")
    
    for doc_id in bad_doc_ids:
        try:
            print(f"   ↳ 正在刪除: {doc_id}")
            await rag.adelete_by_doc_id(doc_id)
            print(f"   ✅ 刪除成功: {doc_id}")
        except Exception as e:
            # 如果錯誤訊息包含 "not found"，代表之前可能已經刪除了一部分，可以當作成功
            if "not found" in str(e).lower() and "pipeline" not in str(e).lower():
                print(f"   ⚠️ 文檔已不存在 (視為成功): {doc_id}")
            else:
                print(f"   ❌ 刪除失敗 {doc_id}: {str(e)}")

    print("🏁 清理完成！現在可以重新運行 step3 了。")

if __name__ == "__main__":
    asyncio.run(main())