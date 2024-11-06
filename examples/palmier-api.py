from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import os
from lightrag import LightRAG, QueryParam
from lightrag.llm import gpt_4o_mini_complete, openai_embedding
from typing import Optional
import asyncio
import nest_asyncio
from chunking.code_chunker import get_github_repo, CodeChunker, load_config

# Apply nest_asyncio to solve event loop issues
nest_asyncio.apply()

app = FastAPI(title="Palmier API", description="API for Palmier Code RAG")

# global config
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config = load_config(os.path.join(project_root, 'config.yaml'))

# Configure working directory
WORKING_DIR = config['working_dir']
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)
if not os.path.exists(os.path.join(WORKING_DIR, "input")):
    os.mkdir(os.path.join(WORKING_DIR, "input"))

print(f"WORKING_DIR: {WORKING_DIR}")

# Initialize RAG instance
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=gpt_4o_mini_complete,
    embedding_func=openai_embedding
)

# Data models


class QueryRequest(BaseModel):
    query: str
    mode: str = "hybrid"
    only_need_context: bool = False

class IndexRequest(BaseModel):
    repo: str
    branch: str = "main"

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
            None, lambda: rag.query(request.query, param=QueryParam(mode=request.mode, only_need_context=request.only_need_context))
        )
        return Response(status="success", data=result)
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/index", response_model=Response)
async def index_endpoint(request: IndexRequest, x_github_token: str | None = Header(None, alias="X-Github-Token")):
    try:
        # Download the Github repo to the working directory
        root_dir = get_github_repo(request.repo, request.branch, WORKING_DIR, github_token=x_github_token)

        # Chunk the repo and write the chunks into .txt files
        chunker = CodeChunker(root_dir)
        chunker.process_files()

        input_folder = os.path.join(WORKING_DIR, "input")
        texts_to_insert = []
        for filename in os.listdir(input_folder):
            if filename.endswith(".txt"):
                file_path = os.path.join(input_folder, filename)
                with open(file_path, "r") as f:
                    texts_to_insert.append(f.read())

        # Insert the chunks into lightrag
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: rag.insert(texts_to_insert))
        return Response(status="success", message="Index created successfully")
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8020)

# Usage example
# To run the server, use the following command in your terminal:
# python examples/palmier-api.py

# Example requests:
# 1. Query:
# curl -X POST "http://127.0.0.1:8020/query" -H "Content-Type: application/json" -d '{"query": "your query here", "mode": "hybrid"}'

# 2. Query the context retrieved only (without generating the response)
# curl -X POST "http://127.0.0.1:8020/query" -H "Content-Type: application/json" -d '{"query": "your query here", "mode": "hybrid", "only_need_context": true}'

# 2. Index Github repo: (for public repos, you don't need to provide X-Github-Token)
# curl -X POST "http://127.0.0.1:8020/index" -H "Content-Type: application/json, X-Github-Token: your_github_token" -d '{"repo": "owner/repo", "branch": "main"}'

# 3. Health check:
# curl -X GET "http://127.0.0.1:8020/health"
