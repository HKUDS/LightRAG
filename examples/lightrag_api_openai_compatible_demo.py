from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from lightrag import LightRAG, QueryParam
from lightrag.llm import openai_complete_if_cache, openai_embedding
from lightrag.utils import EmbeddingFunc
import numpy as np
from typing import Optional
import asyncio
import nest_asyncio
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

# Apply nest_asyncio to solve event loop issues
nest_asyncio.apply()

# Default directory for RAG index
DEFAULT_RAG_DIR = "index_default"
WORKING_DIR = os.getenv("RAG_DIR", DEFAULT_RAG_DIR)      

# Ensure the working directory exists
if not os.path.exists(WORKING_DIR):
    os.makedirs(WORKING_DIR)

# Initialize FastAPI app
app = FastAPI(title="LightRAG API", description="API for RAG operations")

# CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all domains, can be restricted to specific domains
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# LLM model function
async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        "glm-4-flash",
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("ZHIPU_API_KEY"),
        base_url="https://open.bigmodel.cn/api/paas/v4",
        **kwargs,
    )

# Embedding function
async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embedding(
        texts,
        model="BAAI/bge-m3",
        api_key=os.getenv("SILICON_API_KEY"),
        base_url="https://api.siliconflow.cn/v1",
    )

# Initialize RAG instance
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=llm_model_func,
    embedding_func=EmbeddingFunc(
        embedding_dim=1024, max_token_size=8192, func=embedding_func
    ),
)

# Data models for API requests
class QueryRequest(BaseModel):
    query: str
    mode: str = "local"

class InsertRequest(BaseModel):
    text: str

class InsertFileRequest(BaseModel):
    file_path: str

class Response(BaseModel):
    status: str
    data: Optional[str] = None
    message: Optional[str] = None

# API routes
@app.post("/query", response_model=Response)
async def query_endpoint(request: QueryRequest):
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: rag.query(request.query, param=QueryParam(mode=request.mode))
        )
        return Response(status="success", data=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/insert", response_model=Response)
async def insert_endpoint(request: InsertRequest):
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: rag.insert(request.text))
        return Response(status="success", message="Text inserted successfully")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/insert_file", response_model=Response)
async def insert_file(request: InsertFileRequest):
    try:
        if not os.path.exists(request.file_path):
            raise HTTPException(
                status_code=404, detail=f"File not found: {request.file_path}"
            )

        # Read file content
        try:
            with open(request.file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(request.file_path, "r", encoding="gbk") as f:
                content = f.read()

        # Insert file content into RAG system
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: rag.insert(content))

        return Response(
            status="success",
            message=f"File content from {request.file_path} inserted successfully",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Start the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8020)

# Usage example:
# To run the server, use the following command in your terminal:
# python lightrag_api_openai_compatible_demo.py
