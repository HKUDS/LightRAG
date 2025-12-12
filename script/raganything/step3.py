import sys
import os
import json
import time
from loguru import logger
from collections import defaultdict
from dotenv import load_dotenv

# 1. 讀取 .env
load_dotenv()

# 引入 LightRAG
try:
    sys.path.insert(0, os.path.abspath("..")) 
    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc
    # 引入官方函數
    from lightrag.llm import azure_openai_complete, openai_embedding
    logger.info("✅ LightRAG Library 載入成功")
except ImportError:
    logger.error("❌ 找不到 LightRAG")
    sys.exit(1)

# 設定路徑
WORKING_DIR = "./rag_storage"
INPUT_JSON = "./data/output/step2_output_granular/granular_content.json"

# === 自動配置讀取 (Auto-Configuration) ===
# 這裡我們模仿 config.py 的行為，自動從 env 讀取模型名稱
# 如果 .env 無寫，就用 Default 值
ENV_LLM_MODEL = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini") 
ENV_EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")

def main():
    if not os.path.exists(INPUT_JSON): return
    if not os.path.exists(WORKING_DIR): os.makedirs(WORKING_DIR)
    
    logger.info("🚀 初始化 LightRAG (Env-Driven Mode)...")
    logger.info(f"📋 使用 LLM: {ENV_LLM_MODEL}")
    logger.info(f"📋 使用 Embedding: {ENV_EMBED_MODEL}")

    rag = LightRAG(
        working_dir=WORKING_DIR,
        
        # LLM 函數 (Azure): 它會自動讀 AZURE_OPENAI_API_KEY 等環境變數
        llm_model_func=azure_openai_complete,  
        
        # Embedding 函數 (SiliconFlow/OpenAI): 它會自動讀 OPENAI_API_KEY 等環境變數
        embedding_func=EmbeddingFunc(
            embedding_dim=1024, 
            max_token_size=512,  
            func=openai_embedding, 
            
            # 🌟 這裡直接讀取 .env 的 EMBEDDING_MODEL，不用寫死！
            model=ENV_EMBED_MODEL   
        ),
        
        chunk_token_size=512, 
        chunk_overlap_token_size=50
    )

    # === 以下邏輯不變 (讀取 JSON -> 合併 -> 注入) ===
    logger.info(f"📂 讀取 JSON: {INPUT_JSON}")
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        blocks = json.load(f)

    # 簡單估名
    doc_label = "Document"
    try:
        if blocks and blocks[0].get("original_img"):
            p = blocks[0]["original_img"].split(os.sep)
            if len(p) > 1: doc_label = p[-3] if "images" in p else p[0]
    except: pass
    
    logger.info(f"🏷️ 文件標籤: {doc_label}")

    pages_map = defaultdict(str)
    for block in blocks:
        page_num = block.get('page', 'Unknown')
        content = block.get('content', '').strip()
        if not content: continue
        sep = "\n\n" if block.get('type') in ['table', 'image'] else "\n"
        pages_map[page_num] += f"{sep}{content}{sep}"

    sorted_pages = sorted(pages_map.items(), key=lambda x: int(x[0]) if isinstance(x[0], int) or str(x[0]).isdigit() else 9999)
    success_count = 0

    for page_num, full_content in sorted_pages:
        if len(full_content) < 10: continue
        source_id = f"{doc_label} <Page {page_num}>"
        final_text = f"Source: {source_id}\n\n{full_content}"
        try:
            rag.insert(final_text, custom_file_path=source_id)
            success_count += 1
            if success_count % 5 == 0: logger.info(f"⏳ 已注入 {success_count} 頁...")
        except Exception as e:
            logger.error(f"❌ Error: {e}")

    logger.success(f"🎉 完成！共注入 {success_count} 頁")

if __name__ == "__main__":
    main()