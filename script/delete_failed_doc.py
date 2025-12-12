import sys
import os
import asyncio 
from loguru import logger
from dotenv import load_dotenv
from functools import partial

# 引入必要的 LightRAG 元件
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from lightrag.llm.openai import azure_openai_complete, openai_embed

# === 強制加入本地路徑 ===
current_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_path)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

load_dotenv()

# 設定路徑 (需與 step3 一致)
WORKING_DIR = "./data/rag_storage"

# 要刪除的壞掉文件 ID
TARGET_DOC_ID = "doc-a0564021f1b1a2d5015f8f9661b52b1f"

# 獲取 SiliconFlow 設定 (與 Step 3 相同)
SF_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("EMBEDDING_BINDING_API_KEY") or os.getenv("SILICONFLOW_API_KEY")
SF_BASE_URL = os.getenv("OPENAI_BASE_URL") or os.getenv("EMBEDDING_BINDING_HOST") or "https://api.siliconflow.cn/v1"
ENV_EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

async def main():
    if not os.path.exists(WORKING_DIR):
        logger.error(f"❌ 找不到資料庫目錄: {WORKING_DIR}")
        return

    logger.info("🚀 初始化 LightRAG (只為了執行刪除)...")
    
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=azure_openai_complete,
        
        embedding_func=EmbeddingFunc(
            embedding_dim=1024, 
            max_token_size=512,
            func=partial(
                openai_embed.func, 
                model=ENV_EMBED_MODEL,
                api_key=SF_API_KEY,      
                base_url=SF_BASE_URL     
            )
        ),
        # 為了安全，這裡也加上限速設定
        embedding_func_max_async=1,
        max_parallel_insert=1
    )

    # 🔥 [關鍵修正] 必須初始化 Storage，否則 pipeline_status 不存在會報錯
    logger.info("⚙️ 正在初始化 Storage...")
    await rag.initialize_storages()

    logger.info(f"🗑️ 正在嘗試刪除文件 ID: {TARGET_DOC_ID}")
    
    try:
        # 呼叫刪除 API
        result = await rag.adelete_by_doc_id(TARGET_DOC_ID)
        
        if result.status == "success":
            logger.success(f"✅ 成功刪除！({result.message})")
            logger.info("👉 現在你可以重新執行 step3.py 了")
        elif result.status == "not_found":
            logger.warning(f"⚠️ 文件未找到 (可能已經刪除過): {result.message}")
        else:
            logger.warning(f"⚠️ 刪除結果: {result.status} - {result.message}")
            
    except Exception as e:
        logger.error(f"❌ 刪除過程發生錯誤: {e}")

if __name__ == "__main__":
    asyncio.run(main())