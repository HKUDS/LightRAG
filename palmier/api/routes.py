from fastapi import FastAPI, HTTPException, BackgroundTasks, Header
# from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi_versioning import VersionedFastAPI, version
from palmier.api.types import *
from palmier.config import load_config
from palmier.chunking.code_chunker import CodeChunker
from palmier.chunking.repo import get_github_repo
from lightrag import LightRAG, QueryParam
from lightrag.llm import gpt_4o_mini_complete, openai_embedding

import asyncio
import nest_asyncio
import shutil
import os

# TODO: Add API key as Bearer token
# security = HTTPBearer()

# Apply nest_asyncio to solve event loop issues
nest_asyncio.apply()

app = FastAPI(title="Palmier API", version="1.0.0")

# Global variables
config = load_config()

class LightRAGIndex():
    """a lightrag instance for a repository"""

    instance: LightRAG
    status: Status
    message: Optional[str] = None

    def __init__(self, instance: LightRAG, status: Status, message: Optional[str] = None):
        self.instance = instance
        self.status = status
        self.message = message

lightrag_list: dict[str, LightRAGIndex] = {}

# Configure working directory
WORKING_DIR = config['working_dir']
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)
print(f"WORKING_DIR: {WORKING_DIR}")

# Helper functions
async def process_index_request(request: IndexRequest, working_dir: str, github_token: str | None):
    try:
        repo = request.repo
        if repo not in lightrag_list:
            raise HTTPException(status_code=404, detail="Repository not found")
        instance = lightrag_list[repo]

        # Download the Github repo to the working directory
        root_dir = get_github_repo(request.repo, request.branch, working_dir, github_token=github_token)
        instance.message = "Repository downloaded, processing files..."

        # Chunk the repo and write the chunks into .txt files
        max_tokens = config['chunker']['max_tokens']
        chunker = CodeChunker(root_dir, working_dir, max_tokens)
        chunker.process_files()
        instance.message = "Files chunked, inserting into RAG..."

        # Get all the files in the input folder
        input_folder = os.path.join(working_dir, "input")
        texts_to_insert = []
        for filename in os.listdir(input_folder):
            if filename.endswith(".txt"):
                file_path = os.path.join(input_folder, filename)
                with open(file_path, "r") as f:
                    texts_to_insert.append(f.read())

        rag = lightrag_list[repo].instance

        # Insert the chunks into lightrag
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: rag.insert(texts_to_insert))
        
        instance.status = Status.COMPLETED
        instance.message = "Indexing completed successfully"
    except Exception as e:
        instance.status = Status.FAILED
        instance.message = f"Indexing failed: {str(e)}"
        print(f"Indexing failed: {str(e)}")

        if os.path.exists(os.path.join(WORKING_DIR, request.repo)):
            shutil.rmtree(os.path.join(WORKING_DIR, request.repo))

# Routes

@app.post("/query", response_model=Response)
async def query_endpoint(request: QueryRequest):
    try:
        repo = request.repo
        if repo not in lightrag_list:
            raise HTTPException(status_code=404, detail="Repository not found")
        instance = lightrag_list[repo]

        match instance.status:
            case Status.ACCEPTED | Status.PENDING:
                raise HTTPException(status_code=400, detail=f"Requested repository is still in the process of indexing, please check the status at /index/status/{repo}")
            case Status.COMPLETED:
                pass
            case Status.FAILED:
                raise HTTPException(status_code=500, detail=f"Indexing failed for repository {repo}, please retry indexing before querying")

        rag = instance.instance

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

@app.post("/index", response_model=Response)
async def index_endpoint(
    request: IndexRequest, 
    background_tasks: BackgroundTasks,
    x_github_token: str | None = Header(None, alias="X-Github-Token")
):
    try:
        repo = request.repo
        working_dir = os.path.join(WORKING_DIR, repo)
        if not os.path.exists(working_dir):
            os.makedirs(working_dir, exist_ok=True)
        
        # Repo has not been indexed yet
        if repo not in lightrag_list:
            rag = LightRAG(
                working_dir=working_dir,
                llm_model_func=gpt_4o_mini_complete,
                embedding_func=openai_embedding
            )
            lightrag_list[repo] = LightRAGIndex(instance=rag, status=Status.ACCEPTED, message="Indexing started")
        # Repo is already being indexed
        else:
            if lightrag_list[repo].status == Status.PENDING:
                raise HTTPException(status_code=400, detail=f"Requested repository is still in the process of indexing, please check the status at /index/status/{repo}")

        lightrag_list[repo].status = Status.PENDING
        lightrag_list[repo].message = "Indexing started"
        
        # Move the indexing logic to a background task
        background_tasks.add_task(
            process_index_request,
            request,
            working_dir,
            x_github_token
        )
        
        return Response(
            status=Status.ACCEPTED,
            data=None,
            message=f"Indexing job started for repository {repo}"
        )
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/index/status/{owner}/{repo}", response_model=Response)
async def get_index_status(owner: str, repo: str):
    repo = f"{owner}/{repo}"
    if repo not in lightrag_list:
        raise HTTPException(status_code=404, detail="Repository not found")
    
    instance = lightrag_list[repo]
    return Response(
        status=instance.status,
        data=None,
        message=instance.message
    )

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

api_app = VersionedFastAPI(app,
    version_format='{major}',
    prefix_format='/v{major}')

# Example requests:
# 1. Query:
# curl -X POST "http://127.0.0.1:8020/v1/query" -H "Content-Type: application/json" -d '{"repo": "owner/repo", "query": "your query here"}'
# curl -X POST "http://127.0.0.1:8020/v1/query" -H "Content-Type: application/json" -d '{"repo": "owner/repo", "query": "your query here", "mode": "hybrid", "response_type": "Multiple Paragraphs", "top_k": 60}'
# Returns: {"status": "success", "data": "generated response here", "message": "message"}

# 2. Query the context retrieved only (without generating the response)
# curl -X POST "http://127.0.0.1:8020/v1/query" -H "Content-Type: application/json" -d '{"repo": "owner/repo", "query": "your query here", "only_need_context": true}'
# Returns: {"status": "success", "data": "context here", "message": "message"}

# 3. Start indexing (X-Github-Token required for private repos, can be ignored for public repos)
# curl -X POST "http://127.0.0.1:8020/v1/index" -H "Content-Type: application/json, X-Github-Token: your_github_token" -d '{"repo": "owner/repo", "branch": "main"}' 
# Returns: {"status": "accepted", "job_id": "index_1", "message": "Indexing job started"}

# 4. Check indexing status
# curl -X GET "http://127.0.0.1:8020/v1/index/status/index_1"
# Returns: {"status": "pending", "job_id": "index_1", "message": "Files processed, inserting into RAG..."}

# 5. Health check:
# curl -X GET "http://127.0.0.1:8020/v1/health"