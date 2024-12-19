from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from pydantic import BaseModel
import logging
import argparse
from lightrag import LightRAG, QueryParam
from lightrag.llm import ollama_model_complete, ollama_embedding
from lightrag.utils import EmbeddingFunc
from typing import Optional, List
from enum import Enum
from pathlib import Path
import shutil
import aiofiles
from ascii_colors import trace_exception


def parse_args():
    parser = argparse.ArgumentParser(
        description="LightRAG FastAPI Server with separate working and input directories"
    )

    # Server configuration
    parser.add_argument(
        "--host", default="0.0.0.0", help="Server host (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=9621, help="Server port (default: 9621)"
    )

    # Directory configuration
    parser.add_argument(
        "--working-dir",
        default="./rag_storage",
        help="Working directory for RAG storage (default: ./rag_storage)",
    )
    parser.add_argument(
        "--input-dir",
        default="./inputs",
        help="Directory containing input documents (default: ./inputs)",
    )

    # Model configuration
    parser.add_argument(
        "--model",
        default="mistral-nemo:latest",
        help="LLM model name (default: mistral-nemo:latest)",
    )
    parser.add_argument(
        "--embedding-model",
        default="bge-m3:latest",
        help="Embedding model name (default: bge-m3:latest)",
    )
    parser.add_argument(
        "--ollama-host",
        default="http://localhost:11434",
        help="Ollama host URL (default: http://localhost:11434)",
    )

    # RAG configuration
    parser.add_argument(
        "--max-async", type=int, default=4, help="Maximum async operations (default: 4)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32768,
        help="Maximum token size (default: 32768)",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=1024,
        help="Embedding dimensions (default: 1024)",
    )
    parser.add_argument(
        "--max-embed-tokens",
        type=int,
        default=8192,
        help="Maximum embedding token size (default: 8192)",
    )

    # Logging configuration
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)",
    )

    return parser.parse_args()


class DocumentManager:
    """Handles document operations and tracking"""

    def __init__(self, input_dir: str, supported_extensions: tuple = (".txt", ".md")):
        self.input_dir = Path(input_dir)
        self.supported_extensions = supported_extensions
        self.indexed_files = set()

        # Create input directory if it doesn't exist
        self.input_dir.mkdir(parents=True, exist_ok=True)

    def scan_directory(self) -> List[Path]:
        """Scan input directory for new files"""
        new_files = []
        for ext in self.supported_extensions:
            for file_path in self.input_dir.rglob(f"*{ext}"):
                if file_path not in self.indexed_files:
                    new_files.append(file_path)
        return new_files

    def mark_as_indexed(self, file_path: Path):
        """Mark a file as indexed"""
        self.indexed_files.add(file_path)

    def is_supported_file(self, filename: str) -> bool:
        """Check if file type is supported"""
        return any(filename.lower().endswith(ext) for ext in self.supported_extensions)


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
    logging.basicConfig(
        format="%(levelname)s:%(message)s", level=getattr(logging, args.log_level)
    )

    # Initialize FastAPI app
    app = FastAPI(
        title="LightRAG API",
        description="API for querying text using LightRAG with separate storage and input directories",
    )

    # Create working directory if it doesn't exist
    Path(args.working_dir).mkdir(parents=True, exist_ok=True)

    # Initialize document manager
    doc_manager = DocumentManager(args.input_dir)

    # Initialize RAG
    rag = LightRAG(
        working_dir=args.working_dir,
        llm_model_func=ollama_model_complete,
        llm_model_name=args.model,
        llm_model_max_async=args.max_async,
        llm_model_max_token_size=args.max_tokens,
        llm_model_kwargs={
            "host": args.ollama_host,
            "options": {"num_ctx": args.max_tokens},
        },
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
        """Index all files in input directory during startup"""
        try:
            new_files = doc_manager.scan_directory()
            for file_path in new_files:
                try:
                    # Use async file reading
                    async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                        content = await f.read()
                        # Use the async version of insert directly
                        await rag.ainsert(content)
                        doc_manager.mark_as_indexed(file_path)
                        logging.info(f"Indexed file: {file_path}")
                except Exception as e:
                    trace_exception(e)
                    logging.error(f"Error indexing file {file_path}: {str(e)}")

            logging.info(f"Indexed {len(new_files)} documents from {args.input_dir}")

        except Exception as e:
            logging.error(f"Error during startup indexing: {str(e)}")

    @app.post("/documents/scan")
    async def scan_for_new_documents():
        """Manually trigger scanning for new documents"""
        try:
            new_files = doc_manager.scan_directory()
            indexed_count = 0

            for file_path in new_files:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        rag.insert(content)
                        doc_manager.mark_as_indexed(file_path)
                        indexed_count += 1
                except Exception as e:
                    logging.error(f"Error indexing file {file_path}: {str(e)}")

            return {
                "status": "success",
                "indexed_count": indexed_count,
                "total_documents": len(doc_manager.indexed_files),
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/documents/upload")
    async def upload_to_input_dir(file: UploadFile = File(...)):
        """Upload a file to the input directory"""
        try:
            if not doc_manager.is_supported_file(file.filename):
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type. Supported types: {doc_manager.supported_extensions}",
                )

            file_path = doc_manager.input_dir / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # Immediately index the uploaded file
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                rag.insert(content)
                doc_manager.mark_as_indexed(file_path)

            return {
                "status": "success",
                "message": f"File uploaded and indexed: {file.filename}",
                "total_documents": len(doc_manager.indexed_files),
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/query", response_model=QueryResponse)
    async def query_text(request: QueryRequest):
        try:
            response = await rag.aquery(
                request.query,
                param=QueryParam(mode=request.mode, stream=request.stream),
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
                request.query, param=QueryParam(mode=request.mode, stream=True)
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
                document_count=len(rag),
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/documents/file", response_model=InsertResponse)
    async def insert_file(file: UploadFile = File(...), description: str = Form(None)):
        try:
            content = await file.read()

            if file.filename.endswith((".txt", ".md")):
                text = content.decode("utf-8")
                rag.insert(text)
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Unsupported file type. Only .txt and .md files are supported",
                )

            return InsertResponse(
                status="success",
                message=f"File '{file.filename}' successfully inserted",
                document_count=len(rag),
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
                    if file.filename.endswith((".txt", ".md")):
                        text = content.decode("utf-8")
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
                document_count=len(rag),
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
                document_count=0,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    async def get_status():
        """Get current system status"""
        return {
            "status": "healthy",
            "working_directory": str(args.working_dir),
            "input_directory": str(args.input_dir),
            "indexed_files": len(doc_manager.indexed_files),
            "configuration": {
                "model": args.model,
                "embedding_model": args.embedding_model,
                "max_tokens": args.max_tokens,
                "ollama_host": args.ollama_host,
            },
        }

    return app


if __name__ == "__main__":
    args = parse_args()
    import uvicorn

    app = create_app(args)
    uvicorn.run(app, host=args.host, port=args.port)
