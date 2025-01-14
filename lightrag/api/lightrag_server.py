from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from pydantic import BaseModel
import logging
import argparse
from lightrag import LightRAG, QueryParam
from lightrag.llm import lollms_model_complete, lollms_embed
from lightrag.llm import ollama_model_complete, ollama_embed
from lightrag.llm import openai_complete_if_cache, openai_embedding
from lightrag.llm import azure_openai_complete_if_cache, azure_openai_embedding

from lightrag.utils import EmbeddingFunc
from typing import Optional, List, Union
from enum import Enum
from pathlib import Path
import shutil
import aiofiles
from ascii_colors import trace_exception
import os

from fastapi import Depends, Security
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware

from starlette.status import HTTP_403_FORBIDDEN
import pipmaster as pm


def get_default_host(binding_type: str) -> str:
    default_hosts = {
        "ollama": "http://localhost:11434",
        "lollms": "http://localhost:9600",
        "azure_openai": "https://api.openai.com/v1",
        "openai": "https://api.openai.com/v1",
    }
    return default_hosts.get(
        binding_type, "http://localhost:11434"
    )  # fallback to ollama if unknown


def parse_args():
    parser = argparse.ArgumentParser(
        description="LightRAG FastAPI Server with separate working and input directories"
    )

    # Start by the bindings
    parser.add_argument(
        "--llm-binding",
        default="ollama",
        help="LLM binding to be used. Supported: lollms, ollama, openai (default: ollama)",
    )
    parser.add_argument(
        "--embedding-binding",
        default="ollama",
        help="Embedding binding to be used. Supported: lollms, ollama, openai (default: ollama)",
    )

    # Parse just these arguments first
    temp_args, _ = parser.parse_known_args()

    # Add remaining arguments with dynamic defaults for hosts
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

    # LLM Model configuration
    default_llm_host = get_default_host(temp_args.llm_binding)
    parser.add_argument(
        "--llm-binding-host",
        default=default_llm_host,
        help=f"llm server host URL (default: {default_llm_host})",
    )

    parser.add_argument(
        "--llm-model",
        default="mistral-nemo:latest",
        help="LLM model name (default: mistral-nemo:latest)",
    )

    # Embedding model configuration
    default_embedding_host = get_default_host(temp_args.embedding_binding)
    parser.add_argument(
        "--embedding-binding-host",
        default=default_embedding_host,
        help=f"embedding server host URL (default: {default_embedding_host})",
    )

    parser.add_argument(
        "--embedding-model",
        default="bge-m3:latest",
        help="Embedding model name (default: bge-m3:latest)",
    )

    def timeout_type(value):
        if value is None or value == "None":
            return None
        return int(value)

    parser.add_argument(
        "--timeout",
        default=None,
        type=timeout_type,
        help="Timeout in seconds (useful when using slow AI). Use None for infinite timeout",
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

    parser.add_argument(
        "--key",
        type=str,
        help="API key for authentication. This protects lightrag server against unauthorized access",
        default=None,
    )

    # Optional https parameters
    parser.add_argument(
        "--ssl", action="store_true", help="Enable HTTPS (default: False)"
    )
    parser.add_argument(
        "--ssl-certfile",
        default=None,
        help="Path to SSL certificate file (required if --ssl is enabled)",
    )
    parser.add_argument(
        "--ssl-keyfile",
        default=None,
        help="Path to SSL private key file (required if --ssl is enabled)",
    )
    return parser.parse_args()


class DocumentManager:
    """Handles document operations and tracking"""

    def __init__(
        self,
        input_dir: str,
        supported_extensions: tuple = (".txt", ".md", ".pdf", ".docx", ".pptx"),
    ):
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
    only_need_context: bool = False


class QueryResponse(BaseModel):
    response: str


class InsertTextRequest(BaseModel):
    text: str
    description: Optional[str] = None


class InsertResponse(BaseModel):
    status: str
    message: str
    document_count: int


def get_api_key_dependency(api_key: Optional[str]):
    if not api_key:
        # If no API key is configured, return a dummy dependency that always succeeds
        async def no_auth():
            return None

        return no_auth

    # If API key is configured, use proper authentication
    api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

    async def api_key_auth(api_key_header_value: str | None = Security(api_key_header)):
        if not api_key_header_value:
            raise HTTPException(
                status_code=HTTP_403_FORBIDDEN, detail="API Key required"
            )
        if api_key_header_value != api_key:
            raise HTTPException(
                status_code=HTTP_403_FORBIDDEN, detail="Invalid API Key"
            )
        return api_key_header_value

    return api_key_auth


def create_app(args):
    # Verify that bindings arer correctly setup
    if args.llm_binding not in ["lollms", "ollama", "openai"]:
        raise Exception("llm binding not supported")

    if args.embedding_binding not in ["lollms", "ollama", "openai"]:
        raise Exception("embedding binding not supported")

    # Add SSL validation
    if args.ssl:
        if not args.ssl_certfile or not args.ssl_keyfile:
            raise Exception(
                "SSL certificate and key files must be provided when SSL is enabled"
            )
        if not os.path.exists(args.ssl_certfile):
            raise Exception(f"SSL certificate file not found: {args.ssl_certfile}")
        if not os.path.exists(args.ssl_keyfile):
            raise Exception(f"SSL key file not found: {args.ssl_keyfile}")

    # Setup logging
    logging.basicConfig(
        format="%(levelname)s:%(message)s", level=getattr(logging, args.log_level)
    )

    # Check if API key is provided either through env var or args
    api_key = os.getenv("LIGHTRAG_API_KEY") or args.key

    # Initialize FastAPI
    app = FastAPI(
        title="LightRAG API",
        description="API for querying text using LightRAG with separate storage and input directories"
        + "(With authentication)"
        if api_key
        else "",
        version="1.0.2",
        openapi_tags=[{"name": "api"}],
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Create the optional API key dependency
    optional_api_key = get_api_key_dependency(api_key)

    # Create working directory if it doesn't exist
    Path(args.working_dir).mkdir(parents=True, exist_ok=True)

    # Initialize document manager
    doc_manager = DocumentManager(args.input_dir)

    # Initialize RAG
    rag = LightRAG(
        working_dir=args.working_dir,
        llm_model_func=lollms_model_complete
        if args.llm_binding == "lollms"
        else ollama_model_complete
        if args.llm_binding == "ollama"
        else azure_openai_complete_if_cache
        if args.llm_binding == "azure_openai"
        else openai_complete_if_cache,
        llm_model_name=args.llm_model,
        llm_model_max_async=args.max_async,
        llm_model_max_token_size=args.max_tokens,
        llm_model_kwargs={
            "host": args.llm_binding_host,
            "timeout": args.timeout,
            "options": {"num_ctx": args.max_tokens},
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=args.embedding_dim,
            max_token_size=args.max_embed_tokens,
            func=lambda texts: lollms_embed(
                texts,
                embed_model=args.embedding_model,
                host=args.embedding_binding_host,
            )
            if args.llm_binding == "lollms"
            else ollama_embed(
                texts,
                embed_model=args.embedding_model,
                host=args.embedding_binding_host,
            )
            if args.llm_binding == "ollama"
            else azure_openai_embedding(
                texts,
                model=args.embedding_model,  # no host is used for openai
            )
            if args.llm_binding == "azure_openai"
            else openai_embedding(
                texts,
                model=args.embedding_model,  # no host is used for openai
            ),
        ),
    )

    async def index_file(file_path: Union[str, Path]) -> None:
        """Index all files inside the folder with support for multiple file formats

        Args:
            file_path: Path to the file to be indexed (str or Path object)

        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file doesn't exist
        """
        if not pm.is_installed("aiofiles"):
            pm.install("aiofiles")

        # Convert to Path object if string
        file_path = Path(file_path)

        # Check if file exists
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        content = ""
        # Get file extension in lowercase
        ext = file_path.suffix.lower()

        match ext:
            case ".txt" | ".md":
                # Text files handling
                async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                    content = await f.read()

            case ".pdf":
                if not pm.is_installed("pypdf2"):
                    pm.install("pypdf2")
                from pypdf2 import PdfReader

                # PDF handling
                reader = PdfReader(str(file_path))
                content = ""
                for page in reader.pages:
                    content += page.extract_text() + "\n"

            case ".docx":
                if not pm.is_installed("docx"):
                    pm.install("docx")
                from docx import Document

                # Word document handling
                doc = Document(file_path)
                content = "\n".join([paragraph.text for paragraph in doc.paragraphs])

            case ".pptx":
                if not pm.is_installed("pptx"):
                    pm.install("pptx")
                from pptx import Presentation

                # PowerPoint handling
                prs = Presentation(file_path)
                content = ""
                for slide in prs.slides:
                    for shape in slide.shapes:
                        if hasattr(shape, "text"):
                            content += shape.text + "\n"

            case _:
                raise ValueError(f"Unsupported file format: {ext}")

        # Insert content into RAG system
        if content:
            await rag.ainsert(content)
            doc_manager.mark_as_indexed(file_path)
            logging.info(f"Successfully indexed file: {file_path}")
        else:
            logging.warning(f"No content extracted from file: {file_path}")

    @app.on_event("startup")
    async def startup_event():
        """Index all files in input directory during startup"""
        try:
            new_files = doc_manager.scan_directory()
            for file_path in new_files:
                try:
                    await index_file(file_path)
                except Exception as e:
                    trace_exception(e)
                    logging.error(f"Error indexing file {file_path}: {str(e)}")

            logging.info(f"Indexed {len(new_files)} documents from {args.input_dir}")

        except Exception as e:
            logging.error(f"Error during startup indexing: {str(e)}")

    @app.post("/documents/scan", dependencies=[Depends(optional_api_key)])
    async def scan_for_new_documents():
        """Manually trigger scanning for new documents"""
        try:
            new_files = doc_manager.scan_directory()
            indexed_count = 0

            for file_path in new_files:
                try:
                    await index_file(file_path)
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

    @app.post("/documents/upload", dependencies=[Depends(optional_api_key)])
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
            await index_file(file_path)

            return {
                "status": "success",
                "message": f"File uploaded and indexed: {file.filename}",
                "total_documents": len(doc_manager.indexed_files),
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(
        "/query", response_model=QueryResponse, dependencies=[Depends(optional_api_key)]
    )
    async def query_text(request: QueryRequest):
        try:
            response = await rag.aquery(
                request.query,
                param=QueryParam(
                    mode=request.mode,
                    stream=request.stream,
                    only_need_context=request.only_need_context,
                ),
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

    @app.post("/query/stream", dependencies=[Depends(optional_api_key)])
    async def query_text_stream(request: QueryRequest):
        try:
            response = rag.query(
                request.query,
                param=QueryParam(
                    mode=request.mode,
                    stream=True,
                    only_need_context=request.only_need_context,
                ),
            )

            async def stream_generator():
                async for chunk in response:
                    yield chunk

            return stream_generator()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(
        "/documents/text",
        response_model=InsertResponse,
        dependencies=[Depends(optional_api_key)],
    )
    async def insert_text(request: InsertTextRequest):
        try:
            await rag.ainsert(request.text)
            return InsertResponse(
                status="success",
                message="Text successfully inserted",
                document_count=1,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(
        "/documents/file",
        response_model=InsertResponse,
        dependencies=[Depends(optional_api_key)],
    )
    async def insert_file(file: UploadFile = File(...), description: str = Form(None)):
        """Insert a file directly into the RAG system

        Args:
            file: Uploaded file
            description: Optional description of the file

        Returns:
            InsertResponse: Status of the insertion operation

        Raises:
            HTTPException: For unsupported file types or processing errors
        """
        try:
            content = ""
            # Get file extension in lowercase
            ext = Path(file.filename).suffix.lower()

            match ext:
                case ".txt" | ".md":
                    # Text files handling
                    text_content = await file.read()
                    content = text_content.decode("utf-8")

                case ".pdf":
                    if not pm.is_installed("pypdf2"):
                        pm.install("pypdf2")
                    from pypdf2 import PdfReader
                    from io import BytesIO

                    # Read PDF from memory
                    pdf_content = await file.read()
                    pdf_file = BytesIO(pdf_content)
                    reader = PdfReader(pdf_file)
                    content = ""
                    for page in reader.pages:
                        content += page.extract_text() + "\n"

                case ".docx":
                    if not pm.is_installed("docx"):
                        pm.install("docx")
                    from docx import Document
                    from io import BytesIO

                    # Read DOCX from memory
                    docx_content = await file.read()
                    docx_file = BytesIO(docx_content)
                    doc = Document(docx_file)
                    content = "\n".join(
                        [paragraph.text for paragraph in doc.paragraphs]
                    )

                case ".pptx":
                    if not pm.is_installed("pptx"):
                        pm.install("pptx")
                    from pptx import Presentation
                    from io import BytesIO

                    # Read PPTX from memory
                    pptx_content = await file.read()
                    pptx_file = BytesIO(pptx_content)
                    prs = Presentation(pptx_file)
                    content = ""
                    for slide in prs.slides:
                        for shape in slide.shapes:
                            if hasattr(shape, "text"):
                                content += shape.text + "\n"

                case _:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unsupported file type. Supported types: {doc_manager.supported_extensions}",
                    )

            # Insert content into RAG system
            if content:
                # Add description if provided
                if description:
                    content = f"{description}\n\n{content}"

                await rag.ainsert(content)
                logging.info(f"Successfully indexed file: {file.filename}")

                return InsertResponse(
                    status="success",
                    message=f"File '{file.filename}' successfully inserted",
                    document_count=1,
                )
            else:
                raise HTTPException(
                    status_code=400,
                    detail="No content could be extracted from the file",
                )

        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="File encoding not supported")
        except Exception as e:
            logging.error(f"Error processing file {file.filename}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post(
        "/documents/batch",
        response_model=InsertResponse,
        dependencies=[Depends(optional_api_key)],
    )
    async def insert_batch(files: List[UploadFile] = File(...)):
        """Process multiple files in batch mode

        Args:
            files: List of files to process

        Returns:
            InsertResponse: Status of the batch insertion operation

        Raises:
            HTTPException: For processing errors
        """
        try:
            inserted_count = 0
            failed_files = []

            for file in files:
                try:
                    content = ""
                    ext = Path(file.filename).suffix.lower()

                    match ext:
                        case ".txt" | ".md":
                            text_content = await file.read()
                            content = text_content.decode("utf-8")

                        case ".pdf":
                            if not pm.is_installed("pypdf2"):
                                pm.install("pypdf2")
                            from pypdf2 import PdfReader
                            from io import BytesIO

                            pdf_content = await file.read()
                            pdf_file = BytesIO(pdf_content)
                            reader = PdfReader(pdf_file)
                            for page in reader.pages:
                                content += page.extract_text() + "\n"

                        case ".docx":
                            if not pm.is_installed("docx"):
                                pm.install("docx")
                            from docx import Document
                            from io import BytesIO

                            docx_content = await file.read()
                            docx_file = BytesIO(docx_content)
                            doc = Document(docx_file)
                            content = "\n".join(
                                [paragraph.text for paragraph in doc.paragraphs]
                            )

                        case ".pptx":
                            if not pm.is_installed("pptx"):
                                pm.install("pptx")
                            from pptx import Presentation
                            from io import BytesIO

                            pptx_content = await file.read()
                            pptx_file = BytesIO(pptx_content)
                            prs = Presentation(pptx_file)
                            for slide in prs.slides:
                                for shape in slide.shapes:
                                    if hasattr(shape, "text"):
                                        content += shape.text + "\n"

                        case _:
                            failed_files.append(f"{file.filename} (unsupported type)")
                            continue

                    if content:
                        await rag.ainsert(content)
                        inserted_count += 1
                        logging.info(f"Successfully indexed file: {file.filename}")
                    else:
                        failed_files.append(f"{file.filename} (no content extracted)")

                except UnicodeDecodeError:
                    failed_files.append(f"{file.filename} (encoding error)")
                except Exception as e:
                    failed_files.append(f"{file.filename} ({str(e)})")
                    logging.error(f"Error processing file {file.filename}: {str(e)}")

            # Prepare status message
            if inserted_count == len(files):
                status = "success"
                status_message = f"Successfully inserted all {inserted_count} documents"
            elif inserted_count > 0:
                status = "partial_success"
                status_message = f"Successfully inserted {inserted_count} out of {len(files)} documents"
                if failed_files:
                    status_message += f". Failed files: {', '.join(failed_files)}"
            else:
                status = "failure"
                status_message = "No documents were successfully inserted"
                if failed_files:
                    status_message += f". Failed files: {', '.join(failed_files)}"

            return InsertResponse(
                status=status,
                message=status_message,
                document_count=inserted_count,
            )

        except Exception as e:
            logging.error(f"Batch processing error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete(
        "/documents",
        response_model=InsertResponse,
        dependencies=[Depends(optional_api_key)],
    )
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

    @app.get("/health", dependencies=[Depends(optional_api_key)])
    async def get_status():
        """Get current system status"""
        return {
            "status": "healthy",
            "working_directory": str(args.working_dir),
            "input_directory": str(args.input_dir),
            "indexed_files": len(doc_manager.indexed_files),
            "configuration": {
                # LLM configuration binding/host address (if applicable)/model (if applicable)
                "llm_binding": args.llm_binding,
                "llm_binding_host": args.llm_binding_host,
                "llm_model": args.llm_model,
                # embedding model configuration binding/host address (if applicable)/model (if applicable)
                "embedding_binding": args.embedding_binding,
                "embedding_binding_host": args.embedding_binding_host,
                "embedding_model": args.embedding_model,
                "max_tokens": args.max_tokens,
            },
        }

    return app


def main():
    args = parse_args()
    import uvicorn

    app = create_app(args)
    uvicorn_config = {
        "app": app,
        "host": args.host,
        "port": args.port,
    }
    if args.ssl:
        uvicorn_config.update(
            {
                "ssl_certfile": args.ssl_certfile,
                "ssl_keyfile": args.ssl_keyfile,
            }
        )
    uvicorn.run(**uvicorn_config)


if __name__ == "__main__":
    main()
