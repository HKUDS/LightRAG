import sys
import os
import json
import glob
import asyncio
import time
import shutil
import numpy as np
from loguru import logger
from dotenv import load_dotenv
from functools import partial
from typing import Any, List

# === 1. Load Env ===
load_dotenv()

# LightRAG Imports
from lightrag import LightRAG
from lightrag.utils import TiktokenTokenizer, EmbeddingFunc
from lightrag.operate import chunking_by_token_size
# 🔥 直接使用 openai.py 中已經處理好 Azure 邏輯的函數
from lightrag.llm.openai import azure_openai_complete, openai_embed

# === Force Local Path ===
current_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_path)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# === Helper ===
def get_clean_env(key, default=None):
    val = os.getenv(key, default)
    return val.strip() if val else val

# === 🔥 Configuration ===
SKIP_FILES = [ "0008_2024", "6823_2024", "Strategic Investment Partners" ]
TRUSTED_KEYWORDS = [ "HSBC_Report_Source_A", "Final" ]
LOW_QUALITY_KEYWORDS = [ "HSBC_Report_Source_B", "Draft", "Preliminary" ]

# === ⚙️ Settings ===
TARGET_RPM = 100 
SECONDS_PER_REQUEST = 60.0 / TARGET_RPM 
WORKING_DIR = "./data/rag_storage"
STEP2_BASE_DIR = "./data/output/step2_output_granular"

# Logger
LOG_DIR = "./logs"
if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>", level="INFO")
logger.add(os.path.join(LOG_DIR, f"step3_graph_{time.strftime('%Y%m%d_%H%M%S')}.log"), rotation="10 MB", encoding="utf-8")

# === Core Logic: Smart Chunking ===
def process_json_smartly(tokenizer, json_data, doc_label, priority_val):
    full_text = ""
    page_map = [] 
    current_char_index = 0
    
    # 1. Reconstruct Text
    for block in json_data:
        content = block.get('content', '').strip()
        if not content: continue
        page_num = block.get('page', 'Unknown')
        sep = "\n\n" if block.get('type') in ['table', 'image'] else "\n"
        text_segment = f"{sep}{content}{sep}"
        length = len(text_segment)
        page_map.append({"page": page_num, "start": current_char_index, "end": current_char_index + length})
        full_text += text_segment
        current_char_index += length
    
    if not full_text: return "", []
    
    # 2. Chunking
    native_chunks = chunking_by_token_size(tokenizer, full_text, chunk_token_size=512, chunk_overlap_token_size=50)
    structured_chunks = []
    
    # 3. Build Payload
    for i, chunk in enumerate(native_chunks):
        chunk_content = chunk["content"]
        start_idx = full_text.find(chunk_content) 
        if start_idx != -1:
            mid_point = start_idx + (len(chunk_content) // 2)
            found_page = "Unknown"
            for p_info in page_map:
                if p_info["start"] <= mid_point < p_info["end"]:
                    found_page = f"Page {p_info['page']}"
                    break
            page_info = found_page
        else:
            page_info = "Unknown"
            
        final_content = f"Source: {doc_label} | {page_info} | {priority_val}\n{chunk_content}"

        chunk_payload = {
            "content": final_content,
            "priority": priority_val,
            "page_info": page_info,
            "file_path": doc_label,
            "chunk_order_index": i
        }
        structured_chunks.append(chunk_payload)
    return full_text, structured_chunks

async def main():
    if not os.path.exists(STEP2_BASE_DIR):
        logger.error("❌ Data directory not found")
        return

    all_json_files = glob.glob(os.path.join(STEP2_BASE_DIR, "*", "granular_content.json"))
    
    doc_mapping = {}
    mapping_file = "doc_mapping.json"
    if os.path.exists(mapping_file):
        try:
            with open(mapping_file, "r", encoding="utf-8") as f: doc_mapping = json.load(f)
        except: pass

    logger.info(f"🚀 Initializing LightRAG... (Files: {len(all_json_files)})")

    # ==============================================================================
    # 🏗️ 初始化 LightRAG (Clean Version)
    # ==============================================================================
    
    # 針對 Alibaba Dashscope Embedding 的 Wrapper
    # 因為 LightRAG 默認的 openai_embed 可能不會自動讀取 Dashscope 的設定，
    # 這裡我們顯式傳入 Key/URL 給 embedding_func 會比較保險，但 LLM 部分可以完全交給 azure_openai_complete
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
        chunk_token_size=512, 
        chunk_overlap_token_size=50,
        # 建議降低並發數，避免 Azure Rate Limit (429) 導致更多的重試和超時
        embedding_func_max_async=4,  
        max_parallel_insert=1,
        
        # 增加超時設定
        default_embedding_timeout=120, # Embedding 通常較快，120s 足夠
        default_llm_timeout=300        # 🔥 改為 300s (5分鐘)，給 Azure 足夠時間思考
    )

    await rag.initialize_storages()
    
    if not rag.tokenizer:
        rag.tokenizer = TiktokenTokenizer(model_name="gpt-4o") 
    tokenizer = rag.tokenizer 

    # ==============================================================================
    # 🔄 Main Loop (Batching & Smart Chunking)
    # ==============================================================================
    for i, json_file_path in enumerate(all_json_files):
        original_label = os.path.basename(os.path.dirname(json_file_path))
        if any(skip in original_label for skip in SKIP_FILES):
            logger.warning(f"⏭️  Skipping: {original_label}")
            continue

        doc_label = doc_mapping.get(original_label, original_label)
        
        check_name = f"{original_label} {doc_label}" 
        is_trusted = any(k in check_name for k in TRUSTED_KEYWORDS)
        is_low_quality = any(k in check_name for k in LOW_QUALITY_KEYWORDS)
        priority_val = "HIGH" if is_trusted else ("LOW" if is_low_quality else "NORMAL")
        
        icon = "⭐" if priority_val == "HIGH" else ("📉" if priority_val == "LOW" else "📄")
        logger.info(f"{icon} Processing: {doc_label} [Priority: {priority_val}]")

        try:
            with open(json_file_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)
            
            full_text, structured_chunks = process_json_smartly(tokenizer, json_data, doc_label, priority_val)
            if not structured_chunks: continue

            # === Rate Limited Upsert (Batching) ===
            BATCH_SIZE = 10 
            total_chunks = len(structured_chunks)
            target_batch_duration = BATCH_SIZE * SECONDS_PER_REQUEST

            for start_idx in range(0, total_chunks, BATCH_SIZE):
                batch_start_time = time.time()
                end_idx = min(start_idx + BATCH_SIZE, total_chunks)
                batch_chunks = structured_chunks[start_idx:end_idx]
                
                logger.info(f"   🧩 Injecting batch {start_idx+1}-{end_idx} / {total_chunks}...")
                
                # 🔥 調用你在 lightrag.py 中修復好的函數
                await rag.ainsert_structured_chunks(full_text=full_text, text_chunks=batch_chunks)
                
                elapsed = time.time() - batch_start_time
                if elapsed < target_batch_duration and elapsed > 0:
                     # 避免 sleep 負數
                    time.sleep(max(0, target_batch_duration - elapsed))
            
            logger.success(f"   ✅ Done: {doc_label}")

        except Exception as e:
            logger.error(f"❌ Failed: {doc_label} - {e}")
            import traceback
            logger.error(traceback.format_exc())

    logger.info("🎉 All files processed!")

if __name__ == "__main__":
    asyncio.run(main())