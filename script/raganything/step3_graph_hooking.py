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

# === 🔥 設定區 ===
SKIP_FILES = ["SFC", "Example_Doc_To_Skip"]

# 定義哪些文件是「高可信度」的 (可以根據 doc_label 判斷)
TRUSTED_KEYWORDS = ["Global Market Insights"] 
# =================

# 引入 LightRAG
try:
    import lightrag 
    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc
    from lightrag.llm.openai import azure_openai_complete, openai_embed
    logger.info("✅ 成功載入 LightRAG")
except ImportError as e:
    logger.error(f"❌ 找不到 LightRAG 或相關模組: {e}")
    sys.exit(1)

# 設定 Log
LOG_DIR = "./logs"
if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>", level="INFO")
logger.add(os.path.join(LOG_DIR, f"step3_graph_hook_{time.strftime('%Y%m%d_%H%M%S')}.log"), rotation="10 MB", encoding="utf-8")

# 設定路徑
WORKING_DIR = "./data/rag_storage"
STEP2_BASE_DIR = "./data/output/step2_output_granular"

# 自動配置讀取
ENV_LLM_MODEL = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini") 
ENV_EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
SF_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("EMBEDDING_BINDING_API_KEY") or os.getenv("SILICONFLOW_API_KEY")
SF_BASE_URL = os.getenv("OPENAI_BASE_URL") or os.getenv("EMBEDDING_BINDING_HOST") or "https://api.siliconflow.cn/v1"

if not SF_API_KEY:
    logger.error("❌ 找不到 API Key！請檢查 .env")
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
    
    logger.info("🚀 初始化 LightRAG (Graph Hook Version)...")
    
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=azure_openai_complete,
        embedding_func=EmbeddingFunc(
            embedding_dim=1024, 
            max_token_size=512,
            func=partial(openai_embed.func, model=ENV_EMBED_MODEL, api_key=SF_API_KEY, base_url=SF_BASE_URL)
        ),
        chunk_token_size=512, 
        chunk_overlap_token_size=50,
        embedding_func_max_async=1, 
        max_parallel_insert=1 
    )

    logger.info("⚙️ 正在初始化 Storage...")
    await rag.initialize_storages()

    logger.info(f"📦 發現 {len(all_json_files)} 份文件，開始批量注入...")
    total_files = len(all_json_files)
    
    for i, json_file_path in enumerate(all_json_files):
        doc_label = os.path.basename(os.path.dirname(json_file_path))
        
        if doc_label in SKIP_FILES:
            continue

        logger.info(f"\n📄 [File {i+1}/{total_files}] 處理中: {doc_label}")
        
        # 判斷是否為可信文件
        is_trusted = any(k in doc_label for k in TRUSTED_KEYWORDS)
        if is_trusted:
            logger.info(f"   ⭐ 識別為可信來源 (Graph Hook Enabled)")

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
        
        # === 動態合併 Buffer 設定 ===
        TARGET_CHUNK_SIZE = 3000 
        buffer_content = ""
        start_page = -1
        end_page = -1
        chunk_count = 0
        total_pages = len(pages_map)

        for j, (page_num, page_content) in enumerate(sorted_pages):
            if len(page_content) < 10: continue
            
            if start_page == -1: start_page = page_num
            
            sep = f"\n\n--- Page {page_num} ---\n\n" if buffer_content else f"--- Page {page_num} ---\n"
            buffer_content += sep + page_content
            end_page = page_num

            is_last_page = (j == len(sorted_pages) - 1)
            
            if len(buffer_content) >= TARGET_CHUNK_SIZE or is_last_page:
                
                if start_page == end_page:
                    page_range_str = f"<Page {start_page}>"
                else:
                    page_range_str = f"<Page {start_page}-{end_page}>"
                
                source_id = f"{doc_label} {page_range_str}"
                final_text = f"Source: {source_id}\n\n{buffer_content}"

                try:
                    # 1. 正常插入內容
                    await rag.ainsert(final_text, file_paths=source_id)
                    chunk_count += 1
                    
                    # === 🔥 [Method 1] Graph Hook 插入 ===
                    # 如果是可信文件，手動插入一條高權重關係
                    if is_trusted:
                        custom_kg = {
                            "entities": [
                                {
                                    "entity_name": "Verified Source",
                                    "entity_type": "QualityIndicator",
                                    "description": "Represents highly credible and verified information sources.",
                                    "source_id": "manual_trust_indicator" 
                                }
                            ],
                            "relationships": [
                                {
                                    "src_id": source_id, # 將這個 Chunk 連接到 Verified Source
                                    "tgt_id": "Verified Source",
                                    "description": "This content is from a verified high-credibility source.",
                                    "keywords": "verified, official, trusted",
                                    "weight": 10.0, # 🚀 超高權重
                                    "source_id": "manual_trust_indicator"
                                }
                            ],
                            "chunks": [] # 內容已經在 ainsert 處理了，這裡留空
                        }
                        # 注意：這裡 source_id 不用太在意，主要是為了建立邊
                        await rag.ainsert_custom_kg(custom_kg)
                        logger.debug(f"      🔗 已建立 Graph Hook: {source_id} -> Verified Source")
                    # ========================================

                    logger.info(f"     ✅ 注入區塊: {source_id} (Size: {len(buffer_content)})")

                except Exception as e:
                    logger.error(f"     ❌ 注入失敗 ({source_id}): {e}")

                # Buffer Reset Logic (Sliding Window)
                OVERLAP_SIZE = 500
                if len(page_content) > OVERLAP_SIZE and not is_last_page:
                    buffer_content = f"...(Context from Page {end_page})\n{page_content[-OVERLAP_SIZE:]}"
                else:
                    buffer_content = ""
                start_page = -1

        logger.success(f"✅ 文件 {doc_label} 完成！共產生 {chunk_count} 個合併區塊")

    logger.info("\n" + "="*40)
    logger.success(f"🎉 所有文件處理完畢！")
    logger.info("="*40)

if __name__ == "__main__":
    asyncio.run(main())