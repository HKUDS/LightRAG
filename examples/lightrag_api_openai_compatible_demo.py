from fastapi import FastAPI, HTTPException, File, UploadFile
from contextlib import asynccontextmanager
from pydantic import BaseModel
import os
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
import numpy as np
from typing import Optional
import asyncio
import nest_asyncio
from lightrag.kg.shared_storage import initialize_pipeline_status

# Apply nest_asyncio to solve event loop issues
nest_asyncio.apply()

DEFAULT_RAG_DIR = "index_default"
app = FastAPI(title="LightRAG API", description="API for RAG operations")

# Configure working directory
WORKING_DIR = os.environ.get("RAG_DIR", f"{DEFAULT_RAG_DIR}")
print(f"WORKING_DIR: {WORKING_DIR}")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
print(f"LLM_MODEL: {LLM_MODEL}")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-large")
print(f"EMBEDDING_MODEL: {EMBEDDING_MODEL}")
EMBEDDING_MAX_TOKEN_SIZE = int(os.environ.get("EMBEDDING_MAX_TOKEN_SIZE", 8192))
print(f"EMBEDDING_MAX_TOKEN_SIZE: {EMBEDDING_MAX_TOKEN_SIZE}")
BASE_URL = os.environ.get("BASE_URL", "https://api.openai.com/v1")
print(f"BASE_URL: {BASE_URL}")
API_KEY = os.environ.get("API_KEY", "xxxxxxxx")
print(f"API_KEY: {API_KEY}")

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


# LLM model function


async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        model=LLM_MODEL,
        prompt=prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        base_url=BASE_URL,
        api_key=API_KEY,
        **kwargs,
    )


# Embedding function


async def embedding_func(texts: list[str]) -> np.ndarray:
    return await openai_embed(
        texts=texts,
        model=EMBEDDING_MODEL,
        base_url=BASE_URL,
        api_key=API_KEY,
    )


async def get_embedding_dim():
    test_text = ["This is a test sentence."]
    embedding = await embedding_func(test_text)
    embedding_dim = embedding.shape[1]
    print(f"{embedding_dim=}")
    return embedding_dim


# Initialize RAG instance
async def init():
    embedding_dimension = await get_embedding_dim()

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dimension,
            max_token_size=EMBEDDING_MAX_TOKEN_SIZE,
            func=embedding_func,
        ),
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag
    rag = await init()
    print("done!")
    yield


app = FastAPI(
    title="LightRAG API", description="API for RAG operations", lifespan=lifespan
)

# Data models


class QueryRequest(BaseModel):
    query: str
    mode: str = "hybrid"
    only_need_context: bool = False


class InsertRequest(BaseModel):
    text: str


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
            None,
            lambda: rag.query(
                request.query,
                param=QueryParam(
                    mode=request.mode, only_need_context=request.only_need_context
                ),
            ),
        )
        return Response(status="success", data=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/insert", response_model=Response)
async def insert_endpoint(request: InsertRequest):
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: rag.insert(request.text))
        return Response(status="success", message="Text inserted successfully")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/insert_file", response_model=Response)
async def insert_file(file: UploadFile = File(...)):
    try:
        file_content = await file.read()
        # Read file content
        try:
            content = file_content.decode("utf-8")
        except UnicodeDecodeError:
            # If UTF-8 decoding fails, try other encodings
            content = file_content.decode("gbk")
        # Insert file content
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: rag.insert(content))

        return Response(
            status="success",
            message=f"File content from {file.filename} inserted successfully",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8020)

# Usage example
# To run the server, use the following command in your terminal:
# python lightrag_api_openai_compatible_demo.py

# Example requests:
# 1. Query:
# curl -X POST "http://127.0.0.1:8020/query" -H "Content-Type: application/json" -d '{"query": "your query here", "mode": "hybrid"}'

# 2. Insert text:
# curl -X POST "http://127.0.0.1:8020/insert" -H "Content-Type: application/json" -d '{"text": "your text here"}'

# 3. Insert file:
# curl -X POST "http://127.0.0.1:8020/insert_file" -H "Content-Type: multipart/form-data" -F "file=@path/to/your/file.txt"

# 4. Health check:
# curl -X GET "http://127.0.0.1:8020/health"
