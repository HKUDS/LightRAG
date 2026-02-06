import os
import json
import time
import sys
import asyncio
import base64
import sqlite3
import subprocess
import shutil
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from loguru import logger
from functools import partial

# === LightRAG Imports ===
try:
    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc
    from lightrag.llm.openai import azure_openai_complete, openai_embed
except ImportError:
    logger.error("❌ Critical: lightrag.py not found or dependencies missing.")
    raise

# Force load .env
load_dotenv()

# ==============================================================================
# Helper: Async Wrapper for Azure Embedding
# ==============================================================================
async def embedding_func_wrapper(texts: list[str]) -> np.ndarray:
    return await openai_embed.func(
        texts=texts,
        model=os.getenv("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"),
        api_key=os.getenv("LLM_BINDING_API_KEY"),
        base_url=os.getenv("LLM_BINDING_HOST"),
    )

class RagAnythingPipeline:
    def __init__(self, upload_dir: str, output_dir: str, sql_db_path: str, working_dir: str = "./rag_storage"):
        """
        Initialize the RagAnything Pipeline with integrated Logging, OCR, ETL, and LightRAG.
        """
        self.upload_dir = Path(upload_dir)
        self.output_dir = Path(output_dir)
        self.sql_db_path = sql_db_path
        self.working_dir = working_dir
        self.log_dir = Path("./logs")

        # 1. Setup Directories
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 2. Setup Logging
        log_file = self.log_dir / f"rag_pipeline_{time.strftime('%Y%m%d_%H%M%S')}.log"
        logger.remove() 
        logger.add(sys.stderr, level="INFO") 
        logger.add(log_file, rotation="10 MB", level="DEBUG", encoding="utf-8")
        logger.info(f"📝 Pipeline Log file created: {log_file}")

        # 3. Setup Azure OpenAI Client (For Step 2 Vision)
        self.azure_client = None
        self.azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
        self._init_azure_client()

        # 4. Setup Database
        self._init_db()

        # 5. Placeholder for LightRAG
        self.rag = None

    def _init_azure_client(self):
        try:
            from openai import AzureOpenAI
            api_key = os.getenv("LLM_BINDING_API_KEY")
            endpoint = os.getenv("LLM_BINDING_HOST")
            api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

            if api_key and endpoint:
                self.azure_client = AzureOpenAI(
                    api_key=api_key,
                    api_version=api_version,
                    azure_endpoint=endpoint
                )
                logger.info(f"✅ Azure OpenAI Enabled for Vision (Deployment: {self.azure_deployment})")
            else:
                logger.warning("⚠️ Azure Env Vars missing. Vision features disabled.")
        except ImportError:
            logger.warning("⚠️ 'openai' package not found. Vision features disabled.")

    def _init_db(self):
        try:
            conn = sqlite3.connect(self.sql_db_path)
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
                    extraction_method TEXT,
                    original_text TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")

    async def initialize_rag(self):
        """
        Async initialization for LightRAG.
        """
        if self.rag:
            return

        logger.info("🚀 Initializing LightRAG...")
        try:
            self.rag = LightRAG(
                working_dir=self.working_dir,
                llm_model_func=azure_openai_complete, 
                embedding_func=EmbeddingFunc(
                    embedding_dim=int(os.getenv("EMBEDDING_DIM", "1536")),
                    max_token_size=8192,
                    func=embedding_func_wrapper
                ),
                chunk_token_size=512,
                chunk_overlap_token_size=50,
                default_llm_timeout=300
            )
            await self.rag.initialize_storages()
            logger.success("✅ LightRAG Initialized Successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize LightRAG: {e}")
            raise e

    # =========================================================================
    # Helpers
    # =========================================================================

    def _encode_image(self, image_path: str) -> Optional[str]:
        if not os.path.exists(image_path): return None
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _get_safe_content(self, item: Dict) -> str:
        candidates = [item.get("text"), item.get("content"), item.get("table_body")]
        caption = item.get("image_caption") or item.get("table_caption")
        if isinstance(caption, list): caption = "".join(caption)
        if caption: candidates.append(caption)
        
        for c in candidates:
            if c and str(c).strip(): return str(c).strip()
        return ""

    def _find_real_image_path(self, file_stem: str, rel_path: str) -> Optional[str]:
        if not rel_path: return None
        img_filename = os.path.basename(rel_path)
        possible_paths = [
            self.output_dir / file_stem / "auto" / "images" / img_filename,
            self.output_dir / file_stem / "images" / img_filename,
            self.output_dir / file_stem / img_filename
        ]
        for p in possible_paths:
            if p.exists(): return str(p)
        return None

    # =========================================================================
    # Step 1: Async Mineru OCR
    # =========================================================================

    async def run_mineru_extraction(self, file_path: str) -> str:
        file_path_obj = Path(file_path)
        file_stem = file_path_obj.stem
        
        expected_json = self.output_dir / file_stem / "auto" / f"{file_stem}_content_list.json"
        if not expected_json.exists():
            expected_json = self.output_dir / file_stem / "auto" / "content_list.json"

        if expected_json.exists():
            logger.info(f"⚡ Skipping Mineru, output exists: {expected_json}")
            return str(expected_json)

        logger.info(f"🚀 [Step 1] Running Mineru OCR on {file_path_obj.name}...")
        cmd = ["magic-pdf", "-p", str(file_path), "-o", str(self.output_dir), "-m", "auto"]

        def _run_subprocess():
            return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        try:
            result = await asyncio.to_thread(_run_subprocess)
            if result.returncode != 0:
                 logger.error(f"❌ Mineru Error: {result.stderr}")
                 raise Exception(f"Mineru Failed: {result.stderr}")
            
            logger.success(f"✅ Mineru finished processing {file_stem}")
            
            json_files = list((self.output_dir / file_stem).rglob("*content_list.json"))
            if json_files:
                return str(json_files[0])
            else:
                raise FileNotFoundError(f"Mineru ran but generated no JSON for {file_stem}")

        except Exception as e:
            logger.error(f"❌ OCR Step Failed: {e}")
            raise e

    # =========================================================================
    # Step 2: Async ETL (Vision + SQL)
    # =========================================================================

    async def _call_vision_llm(self, img_path: str, mode: str = "table", context_text: str = "") -> Optional[str]:
        if not self.azure_client: return None
        
        base64_image = await asyncio.to_thread(self._encode_image, img_path)
        if not base64_image: return None

        system_prompt = "You are a Financial Data Extractor." if mode == "table" else "Describe this image for RAG."
        user_msg = f"Context: {context_text}\nExtract data."

        try:
            def _api_call():
                return self.azure_client.chat.completions.create(
                    model=self.azure_deployment,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": [
                            {"type": "text", "text": user_msg},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]}
                    ],
                    temperature=0.1, max_tokens=2048,
                    response_format={"type": "json_object"} if mode == "table" else None
                )

            response = await asyncio.to_thread(_api_call)
            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"❌ Vision LLM Failed: {e}")
            return None

    async def run_hybrid_etl(self, json_path: str, file_name: str) -> List[Dict[str, Any]]:
        logger.info(f"🚀 [Step 2] Starting Hybrid ETL for {file_name}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            content_list = json.load(f)

        file_stem = Path(file_name).stem
        structured_chunks = []
        stats = {"sql": 0, "text": 0, "table": 0}

        conn = sqlite3.connect(self.sql_db_path)
        cursor = conn.cursor()

        # 🔥 暫定優先級為預設值，未來可透過參數傳入
        default_priority = "NORMAL"

        for idx, item in enumerate(content_list):
            item_type = item.get('type', 'text')
            page_idx = item.get('page_idx', 0)
            content = self._get_safe_content(item)
            
            context_text = ""
            if idx > 0:
                prev = self._get_safe_content(content_list[idx-1])
                context_text = prev[-200:]

            rel_path = item.get('img_path', '')
            abs_img_path = self._find_real_image_path(file_stem, rel_path)

            final_text_content = content
            
            # Metadata Construction
            chunk_metadata = {
                "source": file_name,
                "page": page_idx,
                "type": item_type,
                "priority": default_priority # 👈 這裡全部統一為 NORMAL，不看類型
            }

            if item_type in ['table', 'tabular', 'image', 'figure']:
                if abs_img_path and self.azure_client:
                    mode = "table" if item_type in ['table', 'tabular'] else "caption"
                    ai_result = await self._call_vision_llm(abs_img_path, mode=mode, context_text=context_text)
                    
                    if ai_result:
                        final_text_content = f"{content}\n\n[AI Analysis]: {ai_result}"
                        if mode == "table" and "metrics" in ai_result:
                            try:
                                data = json.loads(ai_result)
                                if "metrics" in data:
                                    for m in data["metrics"]:
                                        cursor.execute("""
                                            INSERT INTO financial_metrics 
                                            (company_code, report_year, metric_name, metric_value, unit, source_file, page_number, original_text)
                                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                        """, ("Unknown", 2024, m.get('metric_name'), m.get('value'), m.get('unit'), file_name, page_idx, str(m)))
                                        stats["sql"] += 1
                            except: pass
                stats["table"] += 1
            else:
                stats["text"] += 1

            if final_text_content:
                chunk_obj = {
                    "content": final_text_content,
                    "metadata": chunk_metadata 
                }
                structured_chunks.append(chunk_obj)

        conn.commit()
        conn.close()
        logger.success(f"✅ ETL Done. SQL Rows: {stats['sql']} | Chunks: {len(structured_chunks)}")
        return structured_chunks

    # =========================================================================
    # Step 3: Async Ingestion to LightRAG
    # =========================================================================

    async def ingest_to_lightrag(self, chunks: List[Dict[str, Any]], doc_id: str):
        """
        Adapt Dict chunks to LightRAG's custom `ainsert_structured_chunks`.
        """
        if not self.rag:
            await self.initialize_rag()

        logger.info(f"🚀 [Step 3] Ingesting {len(chunks)} chunks into LightRAG...")

        # Transform to format accepted by ainsert_structured_chunks
        formatted_chunks = []
        full_text_buffer = []

        for c in chunks:
            meta = c['metadata']
            
            chunk_data = {
                "content": c['content'],
                # 這裡會吃到 run_hybrid_etl 設定的 "NORMAL"
                "priority": meta.get('priority', 'NORMAL'),
                "page_info": f"Page {meta.get('page', 0)}",
                "file_path": meta.get('source', doc_id)
            }
            
            formatted_chunks.append(chunk_data)
            full_text_buffer.append(c['content'])

        full_text = "\n\n".join(full_text_buffer)

        # Use methods available in your lightrag.py
        await self.rag.ainsert_structured_chunks(
            full_text=full_text,
            text_chunks=formatted_chunks, 
            doc_id=doc_id
        )
        logger.success(f"✅ LightRAG Ingestion Complete for {doc_id}")

    # =========================================================================
    # Main Orchestrator
    # =========================================================================

    async def process_document(self, file_path: str):
        file_name = os.path.basename(file_path)
        doc_id = file_name 

        try:
            # 1. OCR
            json_path = await self.run_mineru_extraction(file_path)
            
            # 2. ETL
            chunks = await self.run_hybrid_etl(json_path, file_name)
            
            # 3. Ingest
            if chunks:
                await self.ingest_to_lightrag(chunks, doc_id)
            else:
                logger.warning(f"⚠️ No chunks generated for {file_name}")

            return {"status": "success", "doc_id": doc_id}

        except Exception as e:
            logger.exception(f"🔥 Critical Pipeline Failure for {file_name}")
            return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    async def main():
        pipeline = RagAnythingPipeline(
            upload_dir="./data/input",
            output_dir="./data/output",
            sql_db_path="./financial.db"
        )
        logger.info("Pipeline initialized. Call process_document() to run.")

    asyncio.run(main())