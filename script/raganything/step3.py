import sys
import os
import json
import glob
import asyncio
import time
from loguru import logger
from dotenv import load_dotenv
from functools import partial
from typing import Any 

# 引入 LightRAG
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from lightrag.llm.ollama import ollama_embed
from lightrag.llm.openai import azure_openai_complete 
from lightrag.operate import chunking_by_token_size
from lightrag.utils import TiktokenTokenizer
from lightrag.utils import compute_mdhash_id, sanitize_text_for_encoding

# === Force Local Path ===
current_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_path)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load .env
load_dotenv()

# ==========================================
# 🛑 [已移除 Hotfix] 
# 因為你已經在 lightrag.py 修復了 ainsert_structured_chunks
# 我們這裡直接用原生的就好！
# ==========================================

# === 🔥 Configuration (Business Rules) ===
SKIP_FILES = [    
    # "0001_2024",
    #   "0775_2024", 
    #   "1038_2024",
    #     "1113_2024",
          "0008_2024",
            "6823_2024",
    # "SFC", 
    "Strategic Investment Partners",
    "Global Market Insights",
]

TRUSTED_KEYWORDS = [ "HSBC_Report_Source_A", "Final" ]
LOW_QUALITY_KEYWORDS = [ "HSBC_Report_Source_B", "Draft", "Preliminary" ]

# === 🦙 Settings ===
OLLAMA_HOST = os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")  
WORKING_DIR = "./data/rag_storage"
STEP2_BASE_DIR = "./data/output/step2_output_granular"

# Logger Setup
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
    search_start_pointer = 0

    # 3. Build Payload
    for i, chunk in enumerate(native_chunks):
        chunk_content = chunk["content"]
        start_idx = full_text.find(chunk_content, search_start_pointer)
        if start_idx != -1:
            search_start_pointer = start_idx + (len(chunk_content) // 2)
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
    
    logger.info(f"🚀 Initializing LightRAG... (Files: {len(all_json_files)})")
    
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=azure_openai_complete,
        embedding_func=EmbeddingFunc(
            embedding_dim=1024, 
            max_token_size=8192, 
            func=partial(ollama_embed, embed_model=OLLAMA_EMBED_MODEL, host=OLLAMA_HOST)
        ),
        chunk_token_size=512, 
        chunk_overlap_token_size=50,
        embedding_func_max_async=1, 
        max_parallel_insert=1,
        # 只傳遞 timeout，其他的交給 lightrag.py 內部處理
        addon_params={"timeout": 600} 
    )

    await rag.initialize_storages()
    
    if not rag.tokenizer:
        rag.tokenizer = TiktokenTokenizer(model_name="gpt-4o-mini")
    
    tokenizer = rag.tokenizer 

    for i, json_file_path in enumerate(all_json_files):
        doc_label = os.path.basename(os.path.dirname(json_file_path))
        
        if any(skip in doc_label for skip in SKIP_FILES):
            logger.warning(f"⏭️  Skipping: {doc_label}")
            continue

        # === Priority Logic ===
        is_trusted = any(k in doc_label for k in TRUSTED_KEYWORDS)
        is_low_quality = any(k in doc_label for k in LOW_QUALITY_KEYWORDS)

        if is_trusted: priority_val = "HIGH"
        elif is_low_quality: priority_val = "LOW"
        else: priority_val = "NORMAL"
        
        icon = "⭐" if priority_val == "HIGH" else ("📉" if priority_val == "LOW" else "📄")
        logger.info(f"{icon} Processing: {doc_label} [Priority: {priority_val}]")

        try:
            with open(json_file_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)
            
            full_text, structured_chunks = process_json_smartly(tokenizer, json_data, doc_label, priority_val)
            
            if not structured_chunks: continue

            # === Upsert ===
            # 🔥 關鍵：直接調用你在 lightrag.py 裡寫好的新函數
            # 注意函數名稱是 ainsert_structured_chunks
            await rag.ainsert_structured_chunks(full_text=full_text, text_chunks=structured_chunks)
            
            logger.success(f"   ✅ Injected {len(structured_chunks)} chunks (Graph Enabled)")

        except Exception as e:
            logger.error(f"❌ Failed: {doc_label} - {e}")
            import traceback
            logger.error(traceback.format_exc())

    logger.info("🎉 All files processed!")

if __name__ == "__main__":
    asyncio.run(main())