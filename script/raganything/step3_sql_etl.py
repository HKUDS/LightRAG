import os
import json
import asyncio
import base64
import sqlite3
import numpy as np
from loguru import logger
from dotenv import load_dotenv
from openai import AzureOpenAI, AsyncAzureOpenAI
from lightrag import LightRAG, QueryParam

# Force load .env
load_dotenv()

# ==========================================
# Part 1: Configuration (Azure & Paths)
# ==========================================
WORKING_DIR = "./rag_storage"
INPUT_JSON_DIR = "./raganything_output"  # 指向 Step 2 的輸出目錄
DB_PATH = "financial_data.db"            # Vanna 用

# Azure Settings
AZURE_API_KEY = os.getenv("LLM_BINDING_API_KEY")
AZURE_ENDPOINT = os.getenv("LLM_BINDING_HOST")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o") # 用於 Chat & Vision
# [注意] LightRAG 需要 Embedding 模型。請在 Azure 部署 'text-embedding-3-small' 或類似模型
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")

if not AZURE_API_KEY or not AZURE_ENDPOINT:
    logger.error("❌ Azure API Key or Endpoint missing in .env")
    exit(1)

# 初始化 Clients
sync_client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    api_version=AZURE_API_VERSION,
    azure_endpoint=AZURE_ENDPOINT
)

async_client = AsyncAzureOpenAI(
    api_key=AZURE_API_KEY,
    api_version=AZURE_API_VERSION,
    azure_endpoint=AZURE_ENDPOINT
)

# ==========================================
# Part 2: Database Setup (For Vanna)
# ==========================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS financial_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_code TEXT,
            report_year INTEGER,
            metric_name TEXT,
            metric_value REAL,
            unit TEXT,
            source_file TEXT,
            page_number INTEGER,
            image_path TEXT,
            extraction_method TEXT,
            original_text TEXT
        )
    ''')
    conn.commit()
    return conn

# ==========================================
# Part 3: LightRAG Bindings (Async Azure)
# ==========================================
async def azure_llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    response = await async_client.chat.completions.create(
        model=AZURE_DEPLOYMENT,
        messages=messages,
        temperature=kwargs.get("temperature", 0.1),
        top_p=kwargs.get("top_p", 1.0),
        n=kwargs.get("n", 1),
    )
    return response.choices[0].message.content

async def azure_embedding_func(texts: list[str]) -> np.ndarray:
    # Azure Embedding Batch logic
    # Note: Azure often has strict batch limits (e.g., 16 or 1 input per request depending on version)
    # Here we send one by one for safety, or you can implement batching
    embeddings = []
    for text in texts:
        response = await async_client.embeddings.create(
            model=AZURE_EMBEDDING_DEPLOYMENT,
            input=text
        )
        embeddings.append(response.data[0].embedding)
    return np.array(embeddings)

# ==========================================
# Part 4: ETL Functions (Sync Azure for SQL)
# ==========================================
def encode_image(image_path):
    if not os.path.exists(image_path): return None
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def is_good_markdown(text):
    if not text: return False
    if len(text) < 50 or "|" not in text: return False
    return True

def azure_extract_from_text(text_content):
    """從 Markdown 表格提取數據"""
    system_prompt = """
    Extract financial metrics from the provided Markdown table.
    Output JSON format: {"metrics": [{"metric_name": "Revenue", "value": 1000, "unit": "HKD", "year": 2023}]}
    Normalize values to absolute numbers (e.g., 50 million -> 50000000).
    """
    try:
        response = sync_client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text_content}
            ],
            response_format={"type": "json_object"},
            temperature=0
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.warning(f"Text Extraction Failed: {e}")
        return {}

def azure_extract_from_image(image_path):
    """從圖片提取數據 (Vision)"""
    base64_img = encode_image(image_path)
    if not base64_img: return {}

    system_prompt = """
    Analyze the image (Chart/Table). Extract financial metrics.
    Output JSON format: {"metrics": [{"metric_name": "Revenue", "value": 1000, "unit": "HKD", "year": 2023}]}
    Identify company name and year from context if possible.
    """
    try:
        response = sync_client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": "Extract structured financial data."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                ]}
            ],
            response_format={"type": "json_object"},
            temperature=0
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.warning(f"Vision Extraction Failed: {e}")
        return {}

# ==========================================
# Part 5: Core Processing Logic
# ==========================================
def process_blocks_to_sql(db_conn, blocks, file_name):
    cursor = db_conn.cursor()
    count = 0
    
    for block in blocks:
        data = {}
        method = ""
        
        # 1. Text Table Path
        if block.get('type') in ['table', 'tabular'] and is_good_markdown(block.get('content', '')):
            logger.info(f"Processing Table (Text): {block.get('unique_id')}")
            data = azure_extract_from_text(block['content'])
            method = "azure_text"

        # 2. Image/Chart Path
        elif block.get('type') in ['image', 'figure']:
            img_path = block.get('original_img') # 這裡假設你已經resolve了絕對路徑
            # 如果原始路徑是相對的，記得要轉成絕對路徑，或者用 find_real_image_path 邏輯
            if img_path and os.path.exists(img_path):
                logger.info(f"Processing Chart (Vision): {block.get('unique_id')}")
                data = azure_extract_from_image(img_path)
                method = "azure_vision"

        # 3. Fallback Path (Bad Markdown)
        elif block.get('type') == 'table' and not is_good_markdown(block.get('content', '')):
             img_path = block.get('original_img')
             if img_path and os.path.exists(img_path):
                 logger.info(f"Fallback to Vision: {block.get('unique_id')}")
                 data = azure_extract_from_image(img_path)
                 method = "azure_vision_fallback"

        # Insert to DB
        if data and "metrics" in data:
            for m in data["metrics"]:
                cursor.execute("""
                    INSERT INTO financial_metrics 
                    (company_code, report_year, metric_name, metric_value, unit, source_file, page_number, image_path, extraction_method)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    m.get('company_code', 'Unknown'),
                    m.get('year', 2024),
                    m.get('metric_name'),
                    m.get('value'),
                    m.get('unit'),
                    file_name,
                    block.get('page', 0),
                    block.get('original_img', ''),
                    method
                ))
            count += len(data["metrics"])
    
    db_conn.commit()
    logger.success(f"Saved {count} metrics to SQL from {file_name}")

# ==========================================
# Part 6: Main Execution
# ==========================================
async def main():
    if not os.path.exists(WORKING_DIR):
        os.mkdir(WORKING_DIR)
    
    # 1. Initialize LightRAG with Azure Functions
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=azure_llm_model_func,  # Custom Azure Binding
        embedding_func=EmbeddingFunc(
            embedding_dim=1536, # text-embedding-3-small is 1536
            max_token_size=8192,
            func=azure_embedding_func
        )
    )
    
    db_conn = init_db()

    # 2. Process Files
    # 假設 input dir 結構是 step2_output_granular/FileName/granular_content.json
    # 這裡你需要根據你的資料夾結構微調 glob
    import glob
    json_files = glob.glob(os.path.join(INPUT_JSON_DIR, "**", "*.json"), recursive=True)
    
    for json_file in json_files:
        logger.info(f"Processing file: {json_file}")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            blocks = json.load(f)
            
        file_name = os.path.basename(os.path.dirname(json_file)) # Assume parent folder is filename
        
        # A. Run SQL ETL (Vanna Prep)
        process_blocks_to_sql(db_conn, blocks, file_name)
        
        # B. Run LightRAG Ingestion (Graph RAG)
        # 組合全文或使用 ainsert_custom_chunks
        full_text = "\n\n".join([
            f"Page {b.get('page')}:\n{b.get('content', '')}" 
            for b in blocks if b.get('content')
        ])
        
        logger.info("Ingesting into LightRAG...")
        await rag.ainsert(full_text)
    
    db_conn.close()
    logger.success("🎉 Hybrid Processing Complete!")

if __name__ == "__main__":
    asyncio.run(main())