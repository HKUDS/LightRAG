from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import asyncio
import os
import logging
import argparse
from lightrag import LightRAG, QueryParam
from lightrag.llm import ollama_model_complete, ollama_embedding
from lightrag.utils import EmbeddingFunc
from typing import Optional, List
from enum import Enum
import io

def parse_args():
    parser = argparse.ArgumentParser(
        description="""
LightRAG FastAPI Server
======================

A REST API server for text querying using LightRAG. Supports multiple search modes,
streaming responses, and document management.

Features:
- Multiple search modes (naive, local, global, hybrid)
- Streaming and non-streaming responses
- Document insertion and management
- Configurable model parameters
- REST API with automatic documentation
""",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Server configuration
    parser.add_argument('--host', default='0.0.0.0', help='Server host (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8000, help='Server port (default: 8000)')
    
    # Model configuration
    parser.add_argument('--model', default='gemma2:2b', help='LLM model name (default: gemma2:2b)')
    parser.add_argument('--embedding-model', default='nomic-embed-text', help='Embedding model name (default: nomic-embed-text)')
    parser.add_argument('--ollama-host', default='http://localhost:11434', help='Ollama host URL (default: http://localhost:11434)')
    
    # RAG configuration
    parser.add_argument('--working-dir', default='./dickens', help='Working directory for RAG (default: ./dickens)')
    parser.add_argument('--max-async', type=int, default=4, help='Maximum async operations (default: 4)')
    parser.add_argument('--max-tokens', type=int, default=32768, help='Maximum token size (default: 32768)')
    parser.add_argument('--embedding-dim', type=int, default=768, help='Embedding dimensions (default: 768)')
    parser.add_argument('--max-embed-tokens', type=int, default=8192, help='Maximum embedding token size (default: 8192)')
    
    # Input configuration
    parser.add_argument('--input-file', default='./book.txt', help='Initial input file to process (default: ./book.txt)')
    
    # Logging configuration
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      help='Logging level (default: INFO)')
    
    return parser.parse_args()

# Pydantic models
class SearchMode(str, Enum):
    naive = "naive"
    local = "local"
    global_ = "global"
    hybrid = "hybrid"

class QueryRequest(BaseModel):
    query: str
    mode: SearchMode = SearchMode.hybrid
    stream: bool = False

class QueryResponse(BaseModel):
    response: str

class InsertTextRequest(BaseModel):
    text: str
    description: Optional[str] = None

class InsertResponse(BaseModel):
    status: str
    message: str
    document_count: int

def create_app(args):
    # Setup logging
    logging.basicConfig(format="%(levelname)s:%(message)s", level=getattr(logging, args.log_level))

    # Initialize FastAPI app
    app = FastAPI(
        title="LightRAG API",
        description="""
        API for querying text using LightRAG.
        
        Configuration:
        - Model: {model}
        - Embedding Model: {embed_model}
        - Working Directory: {work_dir}
        - Max Tokens: {max_tokens}
        """.format(
            model=args.model,
            embed_model=args.embedding_model,
            work_dir=args.working_dir,
            max_tokens=args.max_tokens
        )
    )

    # Create working directory if it doesn't exist
    if not os.path.exists(args.working_dir):
        os.makedirs(args.working_dir)

    # Initialize RAG
    rag = LightRAG(
        working_dir=args.working_dir,
        llm_model_func=ollama_model_complete,
        llm_model_name=args.model,
        llm_model_max_async=args.max_async,
        llm_model_max_token_size=args.max_tokens,
        llm_model_kwargs={"host": args.ollama_host, "options": {"num_ctx": args.max_tokens}},
        embedding_func=EmbeddingFunc(
            embedding_dim=args.embedding_dim,
            max_token_size=args.max_embed_tokens,
            func=lambda texts: ollama_embedding(
                texts, embed_model=args.embedding_model, host=args.ollama_host
            ),
        ),
    )

    @app.on_event("startup")
    async def startup_event():
        try:
            with open(args.input_file, "r", encoding="utf-8") as f:
                rag.insert(f.read())
        except FileNotFoundError:
            logging.warning(f"Input file {args.input_file} not found. Please ensure the file exists before querying.")

    @app.post("/query", response_model=QueryResponse)
    async def query_text(request: QueryRequest):
        try:
            response = rag.query(
                request.query,
                param=QueryParam(mode=request.mode, stream=request.stream)
            )
            
            if request.stream:
                result = ""
                async for chunk in response:
                    result += chunk
                return QueryResponse(response=result)
            else:
                return QueryResponse(response=response)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/query/stream")
    async def query_text_stream(request: QueryRequest):
        try:
            response = rag.query(
                request.query,
                param=QueryParam(mode=request.mode, stream=True)
            )
            
            async def stream_generator():
                async for chunk in response:
                    yield chunk
            
            return stream_generator()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/documents/text", response_model=InsertResponse)
    async def insert_text(request: InsertTextRequest):
        try:
            rag.insert(request.text)
            return InsertResponse(
                status="success",
                message="Text successfully inserted",
                document_count=len(rag)
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/documents/file", response_model=InsertResponse)
    async def insert_file(
        file: UploadFile = File(...),
        description: str = Form(None)
    ):
        try:
            content = await file.read()
            
            if file.filename.endswith(('.txt', '.md')):
                text = content.decode('utf-8')
                rag.insert(text)
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Unsupported file type. Only .txt and .md files are supported"
                )
            
            return InsertResponse(
                status="success",
                message=f"File '{file.filename}' successfully inserted",
                document_count=len(rag)
            )
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="File encoding not supported")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/documents/batch", response_model=InsertResponse)
    async def insert_batch(files: List[UploadFile] = File(...)):
        try:
            inserted_count = 0
            failed_files = []
            
            for file in files:
                try:
                    content = await file.read()
                    if file.filename.endswith(('.txt', '.md')):
                        text = content.decode('utf-8')
                        rag.insert(text)
                        inserted_count += 1
                    else:
                        failed_files.append(f"{file.filename} (unsupported type)")
                except Exception as e:
                    failed_files.append(f"{file.filename} ({str(e)})")
            
            status_message = f"Successfully inserted {inserted_count} documents"
            if failed_files:
                status_message += f". Failed files: {', '.join(failed_files)}"
            
            return InsertResponse(
                status="success" if inserted_count > 0 else "partial_success",
                message=status_message,
                document_count=len(rag)
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/documents", response_model=InsertResponse)
    async def clear_documents():
        try:
            rag.text_chunks = []
            rag.entities_vdb = None
            rag.relationships_vdb = None
            return InsertResponse(
                status="success",
                message="All documents cleared successfully",
                document_count=0
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    async def health_check():
        return {
            "status": "healthy",
            "configuration": {
                "model": args.model,
                "embedding_model": args.embedding_model,
                "working_dir": args.working_dir,
                "max_tokens": args.max_tokens,
                "ollama_host": args.ollama_host
            }
        }

    return app

if __name__ == "__main__":
    args = parse_args()
    import uvicorn
    app = create_app(args)
    uvicorn.run(app, host=args.host, port=args.port)
