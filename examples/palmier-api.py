from fastapi import FastAPI, HTTPException, Header, BackgroundTasks
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
    mode: Optional[str] = "hybrid"
    only_need_context: Optional[bool] = False
    response_type: Optional[str] = "Multiple Paragraphs"
    top_k: Optional[int] = 60

class IndexRequest(BaseModel):
    repo: str
    branch: str = "main"

class Response(BaseModel):
    status: str
    data: Optional[str] = None
    message: Optional[str] = None

class IndexStatus(BaseModel):
    status: str
    job_id: str
    message: Optional[str] = None

# Add this dictionary to store job statuses
indexing_jobs = {}

# Helper functions
async def process_index_request(request: IndexRequest, github_token: str | None, job_id: str):
    try:
        # Download the Github repo to the working directory
        root_dir = get_github_repo(request.repo, request.branch, WORKING_DIR, github_token=github_token)
        indexing_jobs[job_id]["message"] = "Repository downloaded, processing files..."

        # Chunk the repo and write the chunks into .txt files
        chunker = CodeChunker(root_dir)
        chunker.process_files()
        indexing_jobs[job_id]["message"] = "Files processed, inserting into RAG..."

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
        
        indexing_jobs[job_id]["status"] = "completed"
        indexing_jobs[job_id]["message"] = "Indexing completed successfully"
    except Exception as e:
        indexing_jobs[job_id]["status"] = "failed"
        indexing_jobs[job_id]["message"] = f"Indexing failed: {str(e)}"
        print(f"Indexing failed: {str(e)}")

# API routes


@app.post("/query", response_model=Response)
async def query_endpoint(request: QueryRequest):
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: rag.query(
                request.query,
                param=QueryParam(
                    mode=request.mode,
                    only_need_context=request.only_need_context,
                    response_type=request.response_type,
                    top_k=request.top_k
                )
            )
        )
        return Response(status="success", data=result)
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/index", response_model=IndexStatus)
async def index_endpoint(
    request: IndexRequest, 
    background_tasks: BackgroundTasks,
    x_github_token: str | None = Header(None, alias="X-Github-Token")
):
    try:
        job_id = f"index_{len(indexing_jobs) + 1}"  # Simple job ID generation
        indexing_jobs[job_id] = {"status": "pending", "message": "Indexing started"}
        
        # Move the indexing logic to a background task
        background_tasks.add_task(
            process_index_request,
            request,
            x_github_token,
            job_id
        )
        
        return IndexStatus(
            status="accepted",
            job_id=job_id,
            message="Indexing job started"
        )
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Add new endpoint to check indexing status
@app.get("/index/status/{job_id}", response_model=IndexStatus)
async def get_index_status(job_id: str):
    if job_id not in indexing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_status = indexing_jobs[job_id]
    return IndexStatus(
        status=job_status["status"],
        job_id=job_id,
        message=job_status["message"]
    )

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
# curl -X POST "http://127.0.0.1:8020/query" -H "Content-Type: application/json" -d '{"query": "your query here"}'
# curl -X POST "http://127.0.0.1:8020/query" -H "Content-Type: application/json" -d '{"query": "your query here", "mode": "hybrid", "response_type": "Multiple Paragraphs", "top_k": 60}'
# Returns: {"status": "success", "data": "generated response here", "message": "message"}

# 2. Query the context retrieved only (without generating the response)
# curl -X POST "http://127.0.0.1:8020/query" -H "Content-Type: application/json" -d '{"query": "your query here", "only_need_context": true}'
# Returns: {"status": "success", "data": "context here", "message": "message"}

# 3. Start indexing (X-Github-Token required for private repos, can be ignored for public repos)
# curl -X POST "http://127.0.0.1:8020/index" -H "Content-Type: application/json, X-Github-Token: your_github_token" -d '{"repo": "owner/repo", "branch": "main"}' 
# Returns: {"status": "accepted", "job_id": "index_1", "message": "Indexing job started"}

# 4. Check indexing status
# curl -X GET "http://127.0.0.1:8020/index/status/index_1"
# Returns: {"status": "pending", "job_id": "index_1", "message": "Files processed, inserting into RAG..."}

# 5. Health check:
# curl -X GET "http://127.0.0.1:8020/health"