import sys
import os
import json
import time
import glob
import asyncio 
from loguru import logger
from collections import defaultdict
from dotenv import load_dotenv
from functools import partial

# === 強制加入本地路徑 ===
current_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_path)))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ========================

# 讀取 .env
load_dotenv()

# === 🔥 [新增] 黑名單設定 ===
# 填入不想注入的文件資料夾名稱 (即 doc_label)
# 例如: ["SFC", "Old_Report_2023"]
SKIP_FILES = [
    "SFC",
    "Example_Doc_To_Skip"
]
# ============================

# 引入 LightRAG
try:
    import lightrag 
    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc
    # 引入官方函數
    from lightrag.llm.openai import azure_openai_complete, openai_embed
    
    logger.info("✅ 成功載入 LightRAG")
    logger.info(f"📍 LightRAG 來源: {os.path.dirname(lightrag.__file__)}")
    
except ImportError as e:
    logger.error(f"❌ 找不到 LightRAG 或相關模組: {e}")
    sys.exit(1)

# 設定 Log
LOG_DIR = "./logs"
if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>", level="INFO")
logger.add(os.path.join(LOG_DIR, f"step3_multi_run_{time.strftime('%Y%m%d_%H%M%S')}.log"), rotation="10 MB", encoding="utf-8")

# 設定路徑
WORKING_DIR = "./data/rag_storage"
STEP2_BASE_DIR = "./data/output/step2_output_granular"

# 自動配置讀取
ENV_LLM_MODEL = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini") 
ENV_EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

# 手動獲取 SiliconFlow 的 Key 和 URL
SF_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("EMBEDDING_BINDING_API_KEY") or os.getenv("SILICONFLOW_API_KEY")
SF_BASE_URL = os.getenv("OPENAI_BASE_URL") or os.getenv("EMBEDDING_BINDING_HOST") or "https://api.siliconflow.cn/v1"

if not SF_API_KEY:
    logger.error("❌ 找不到 API Key！請檢查 .env 是否包含 OPENAI_API_KEY 或 SILICONFLOW_API_KEY")
    sys.exit(1)

async def main():
    if not os.path.exists(STEP2_BASE_DIR):
        logger.error(f"❌ 找不到 Step 2 輸出目錄: {STEP2_BASE_DIR}")
        return

    all_json_files = glob.glob(os.path.join(STEP2_BASE_DIR, "*", "granular_content.json"))
    
    if not all_json_files:
        logger.error("❌ 找不到任何 granular_content.json")
        return

    if not os.path.exists(WORKING_DIR): os.makedirs(WORKING_DIR)
    
    logger.info("🚀 初始化 LightRAG (Azure + SiliconFlow)...")
    logger.info(f"📋 LLM: {ENV_LLM_MODEL} | Embedding: {ENV_EMBED_MODEL}")
    logger.info(f"🔌 Embedding Endpoint: {SF_BASE_URL}")

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=azure_openai_complete,
        
        # 明確傳入 api_key 和 base_url
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
        chunk_token_size=512, 
        chunk_overlap_token_size=50
    )

    logger.info("⚙️ 正在初始化 Storage...")
    await rag.initialize_storages()

    logger.info(f"📦 發現 {len(all_json_files)} 份文件，開始批量注入...")

    total_files = len(all_json_files)
    
    for i, json_file_path in enumerate(all_json_files):
        # 取得資料夾名稱作為 doc_label (例如 "SFC")
        doc_label = os.path.basename(os.path.dirname(json_file_path))
        
        # === 🚫 Blacklist Check (新增檢查邏輯) ===
        if doc_label in SKIP_FILES:
            logger.warning(f"🚫 [{i+1}/{total_files}] 跳過黑名單文件: {doc_label}")
            continue
        # ========================================

        logger.info(f"\n📄 [File {i+1}/{total_files}] 處理中: {doc_label}")
        
        try:
            with open(json_file_path, "r", encoding="utf-8") as f:
                blocks = json.load(f)
        except Exception as e:
            logger.error(f"❌ 讀取 JSON 失敗 ({doc_label}): {e}")
            continue

        pages_map = defaultdict(str)
        for block in blocks:
            page_num = block.get('page', 'Unknown')
            content = block.get('content', '').strip()
            if not content: continue
            
            sep = "\n\n" if block.get('type') in ['table', 'image'] else "\n"
            pages_map[page_num] += f"{sep}{content}{sep}"

        sorted_pages = sorted(pages_map.items(), key=lambda x: int(x[0]) if isinstance(x[0], int) or str(x[0]).isdigit() else 9999)
        file_success_count = 0
        total_pages = len(pages_map)

        logger.info(f"   ↳ 共有 {total_pages} 頁，正在寫入 Graph...")

        for page_num, full_content in sorted_pages:
            if len(full_content) < 10: continue
            
            source_id = f"{doc_label} <Page {page_num}>"
            final_text = f"Source: {source_id}\n\n{full_content}"

            try:
                await rag.ainsert(final_text, file_paths=source_id)
                
                file_success_count += 1
                if file_success_count % 10 == 0: 
                    logger.info(f"     ⏳ 已注入 {file_success_count}/{total_pages} 頁...")
            except Exception as e:
                logger.error(f"     ❌ 注入失敗 (Page {page_num}): {e}")

        logger.success(f"✅ 文件 {doc_label} 完成！共注入 {file_success_count} 頁")

    logger.info("\n" + "="*40)
    logger.success(f"🎉 所有文件處理完畢！")
    logger.info(f"💾 知識庫位置: {WORKING_DIR}")
    logger.info("="*40)

if __name__ == "__main__":
    asyncio.run(main())