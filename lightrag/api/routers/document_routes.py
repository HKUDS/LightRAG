"""
This module contains all document-related routes for the LightRAG API.
"""

import asyncio
from lightrag.utils import logger
import aiofiles
import shutil
import traceback
import pipmaster as pm
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field, field_validator

from lightrag import LightRAG
from lightrag.base import DocProcessingStatus, DocStatus
from lightrag.api.utils_api import (
    get_api_key_dependency,
    global_args,
    get_auth_dependency,
)

router = APIRouter(
    prefix="/documents",
    tags=["documents"],
    dependencies=[Depends(get_auth_dependency())],
)

# Temporary file prefix
temp_prefix = "__tmp__"


class InsertTextRequest(BaseModel):
    text: str = Field(
        min_length=1,
        description="The text to insert",
    )

    @field_validator("text", mode="after")
    @classmethod
    def strip_after(cls, text: str) -> str:
        return text.strip()


class InsertTextsRequest(BaseModel):
    texts: list[str] = Field(
        min_length=1,
        description="The texts to insert",
    )

    @field_validator("texts", mode="after")
    @classmethod
    def strip_after(cls, texts: list[str]) -> list[str]:
        return [text.strip() for text in texts]


class InsertResponse(BaseModel):
    status: str = Field(description="Status of the operation")
    message: str = Field(description="Message describing the operation result")


class DocStatusResponse(BaseModel):
    @staticmethod
    def format_datetime(dt: Any) -> Optional[str]:
        if dt is None:
            return None
        if isinstance(dt, str):
            return dt
        return dt.isoformat()

    """Response model for document status

    Attributes:
        id: Document identifier
        content_summary: Summary of document content
        content_length: Length of document content
        status: Current processing status
        created_at: Creation timestamp (ISO format string)
        updated_at: Last update timestamp (ISO format string)
        chunks_count: Number of chunks (optional)
        error: Error message if any (optional)
        metadata: Additional metadata (optional)
    """

    id: str
    content_summary: str
    content_length: int
    status: DocStatus
    created_at: str
    updated_at: str
    chunks_count: Optional[int] = None
    error: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


class DocsStatusesResponse(BaseModel):
    statuses: Dict[DocStatus, List[DocStatusResponse]] = {}


class PipelineStatusResponse(BaseModel):
    """Response model for pipeline status

    Attributes:
        autoscanned: Whether auto-scan has started
        busy: Whether the pipeline is currently busy
        job_name: Current job name (e.g., indexing files/indexing texts)
        job_start: Job start time as ISO format string (optional)
        docs: Total number of documents to be indexed
        batchs: Number of batches for processing documents
        cur_batch: Current processing batch
        request_pending: Flag for pending request for processing
        latest_message: Latest message from pipeline processing
        history_messages: List of history messages
    """

    autoscanned: bool = False
    busy: bool = False
    job_name: str = "Default Job"
    job_start: Optional[str] = None
    docs: int = 0
    batchs: int = 0
    cur_batch: int = 0
    request_pending: bool = False
    latest_message: str = ""
    history_messages: Optional[List[str]] = None

    class Config:
        extra = "allow"  # Allow additional fields from the pipeline status


class DocumentManager:
    def __init__(
        self,
        input_dir: str,
        supported_extensions: tuple = (
            ".txt",
            ".md",
            ".pdf",
            ".docx",
            ".pptx",
            ".xlsx",
            ".rtf",  # Rich Text Format
            ".odt",  # OpenDocument Text
            ".tex",  # LaTeX
            ".epub",  # Electronic Publication
            ".html",  # HyperText Markup Language
            ".htm",  # HyperText Markup Language
            ".csv",  # Comma-Separated Values
            ".json",  # JavaScript Object Notation
            ".xml",  # eXtensible Markup Language
            ".yaml",  # YAML Ain't Markup Language
            ".yml",  # YAML
            ".log",  # Log files
            ".conf",  # Configuration files
            ".ini",  # Initialization files
            ".properties",  # Java properties files
            ".sql",  # SQL scripts
            ".bat",  # Batch files
            ".sh",  # Shell scripts
            ".c",  # C source code
            ".cpp",  # C++ source code
            ".py",  # Python source code
            ".java",  # Java source code
            ".js",  # JavaScript source code
            ".ts",  # TypeScript source code
            ".swift",  # Swift source code
            ".go",  # Go source code
            ".rb",  # Ruby source code
            ".php",  # PHP source code
            ".css",  # Cascading Style Sheets
            ".scss",  # Sassy CSS
            ".less",  # LESS CSS
        ),
    ):
        self.input_dir = Path(input_dir)
        self.supported_extensions = supported_extensions
        self.indexed_files = set()

        # Create input directory if it doesn't exist
        self.input_dir.mkdir(parents=True, exist_ok=True)

    def scan_directory_for_new_files(self) -> List[Path]:
        """Scan input directory for new files"""
        new_files = []
        for ext in self.supported_extensions:
            logger.debug(f"Scanning for {ext} files in {self.input_dir}")
            for file_path in self.input_dir.rglob(f"*{ext}"):
                if file_path not in self.indexed_files:
                    new_files.append(file_path)
        return new_files

    def mark_as_indexed(self, file_path: Path):
        self.indexed_files.add(file_path)

    def is_supported_file(self, filename: str) -> bool:
        return any(filename.lower().endswith(ext) for ext in self.supported_extensions)


async def pipeline_enqueue_file(rag: LightRAG, file_path: Path) -> bool:
    """Add a file to the queue for processing

    Args:
        rag: LightRAG instance
        file_path: Path to the saved file
    Returns:
        bool: True if the file was successfully enqueued, False otherwise
    """

    try:
        content = ""
        ext = file_path.suffix.lower()

        file = None
        async with aiofiles.open(file_path, "rb") as f:
            file = await f.read()

        # Process based on file type
        match ext:
            case (
                ".txt"
                | ".md"
                | ".html"
                | ".htm"
                | ".tex"
                | ".json"
                | ".xml"
                | ".yaml"
                | ".yml"
                | ".rtf"
                | ".odt"
                | ".epub"
                | ".csv"
                | ".log"
                | ".conf"
                | ".ini"
                | ".properties"
                | ".sql"
                | ".bat"
                | ".sh"
                | ".c"
                | ".cpp"
                | ".py"
                | ".java"
                | ".js"
                | ".ts"
                | ".swift"
                | ".go"
                | ".rb"
                | ".php"
                | ".css"
                | ".scss"
                | ".less"
            ):
                try:
                    # Try to decode as UTF-8
                    content = file.decode("utf-8")

                    # Validate content
                    if not content or len(content.strip()) == 0:
                        logger.error(f"Empty content in file: {file_path.name}")
                        return False

                    # Check if content looks like binary data string representation
                    if content.startswith("b'") or content.startswith('b"'):
                        logger.error(
                            f"File {file_path.name} appears to contain binary data representation instead of text"
                        )
                        return False

                except UnicodeDecodeError:
                    logger.error(
                        f"File {file_path.name} is not valid UTF-8 encoded text. Please convert it to UTF-8 before processing."
                    )
                    return False
            case ".pdf":
                if global_args["main_args"].document_loading_engine == "DOCLING":
                    if not pm.is_installed("docling"):  # type: ignore
                        pm.install("docling")
                    from docling.document_converter import DocumentConverter  # type: ignore

                    converter = DocumentConverter()
                    result = converter.convert(file_path)
                    content = result.document.export_to_markdown()
                else:
                    if not pm.is_installed("pypdf2"):  # type: ignore
                        pm.install("pypdf2")
                    from PyPDF2 import PdfReader  # type: ignore
                    from io import BytesIO

                    pdf_file = BytesIO(file)
                    reader = PdfReader(pdf_file)
                    for page in reader.pages:
                        content += page.extract_text() + "\n"
            case ".docx":
                if global_args["main_args"].document_loading_engine == "DOCLING":
                    if not pm.is_installed("docling"):  # type: ignore
                        pm.install("docling")
                    from docling.document_converter import DocumentConverter  # type: ignore

                    converter = DocumentConverter()
                    result = converter.convert(file_path)
                    content = result.document.export_to_markdown()
                else:
                    if not pm.is_installed("python-docx"):  # type: ignore
                        pm.install("docx")
                    from docx import Document  # type: ignore
                    from io import BytesIO

                    docx_file = BytesIO(file)
                    doc = Document(docx_file)
                    content = "\n".join(
                        [paragraph.text for paragraph in doc.paragraphs]
                    )
            case ".pptx":
                if global_args["main_args"].document_loading_engine == "DOCLING":
                    if not pm.is_installed("docling"):  # type: ignore
                        pm.install("docling")
                    from docling.document_converter import DocumentConverter  # type: ignore

                    converter = DocumentConverter()
                    result = converter.convert(file_path)
                    content = result.document.export_to_markdown()
                else:
                    if not pm.is_installed("python-pptx"):  # type: ignore
                        pm.install("pptx")
                    from pptx import Presentation  # type: ignore
                    from io import BytesIO

                    pptx_file = BytesIO(file)
                    prs = Presentation(pptx_file)
                    for slide in prs.slides:
                        for shape in slide.shapes:
                            if hasattr(shape, "text"):
                                content += shape.text + "\n"
            case ".xlsx":
                if global_args["main_args"].document_loading_engine == "DOCLING":
                    if not pm.is_installed("docling"):  # type: ignore
                        pm.install("docling")
                    from docling.document_converter import DocumentConverter  # type: ignore

                    converter = DocumentConverter()
                    result = converter.convert(file_path)
                    content = result.document.export_to_markdown()
                else:
                    if not pm.is_installed("openpyxl"):  # type: ignore
                        pm.install("openpyxl")
                    from openpyxl import load_workbook  # type: ignore
                    from io import BytesIO

                    xlsx_file = BytesIO(file)
                    wb = load_workbook(xlsx_file)
                    for sheet in wb:
                        content += f"Sheet: {sheet.title}\n"
                        for row in sheet.iter_rows(values_only=True):
                            content += (
                                "\t".join(
                                    str(cell) if cell is not None else ""
                                    for cell in row
                                )
                                + "\n"
                            )
                        content += "\n"
            case _:
                logger.error(
                    f"Unsupported file type: {file_path.name} (extension {ext})"
                )
                return False

        # Insert into the RAG queue
        if content:
            await rag.apipeline_enqueue_documents(content)
            logger.info(f"Successfully fetched and enqueued file: {file_path.name}")
            return True
        else:
            logger.error(f"No content could be extracted from file: {file_path.name}")

    except Exception as e:
        logger.error(f"Error processing or enqueueing file {file_path.name}: {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        if file_path.name.startswith(temp_prefix):
            try:
                file_path.unlink()
            except Exception as e:
                logger.error(f"Error deleting file {file_path}: {str(e)}")
    return False


async def pipeline_index_file(rag: LightRAG, file_path: Path):
    """Index a file

    Args:
        rag: LightRAG instance
        file_path: Path to the saved file
    """
    try:
        if await pipeline_enqueue_file(rag, file_path):
            await rag.apipeline_process_enqueue_documents()

    except Exception as e:
        logger.error(f"Error indexing file {file_path.name}: {str(e)}")
        logger.error(traceback.format_exc())


async def pipeline_index_files(rag: LightRAG, file_paths: List[Path]):
    """Index multiple files concurrently

    Args:
        rag: LightRAG instance
        file_paths: Paths to the files to index
    """
    if not file_paths:
        return
    try:
        enqueued = False

        if len(file_paths) == 1:
            enqueued = await pipeline_enqueue_file(rag, file_paths[0])
        else:
            tasks = [pipeline_enqueue_file(rag, path) for path in file_paths]
            enqueued = any(await asyncio.gather(*tasks))

        if enqueued:
            await rag.apipeline_process_enqueue_documents()
    except Exception as e:
        logger.error(f"Error indexing files: {str(e)}")
        logger.error(traceback.format_exc())


async def pipeline_index_texts(rag: LightRAG, texts: List[str]):
    """Index a list of texts

    Args:
        rag: LightRAG instance
        texts: The texts to index
    """
    if not texts:
        return
    await rag.apipeline_enqueue_documents(texts)
    await rag.apipeline_process_enqueue_documents()


async def save_temp_file(input_dir: Path, file: UploadFile = File(...)) -> Path:
    """Save the uploaded file to a temporary location

    Args:
        file: The uploaded file

    Returns:
        Path: The path to the saved file
    """
    # Generate unique filename to avoid conflicts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_filename = f"{temp_prefix}{timestamp}_{file.filename}"

    # Create a temporary file to save the uploaded content
    temp_path = input_dir / "temp" / unique_filename
    temp_path.parent.mkdir(exist_ok=True)

    # Save the file
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return temp_path


async def run_scanning_process(rag: LightRAG, doc_manager: DocumentManager):
    """Background task to scan and index documents"""
    try:
        new_files = doc_manager.scan_directory_for_new_files()
        total_files = len(new_files)
        logger.info(f"Found {total_files} new files to index.")

        for idx, file_path in enumerate(new_files):
            try:
                await pipeline_index_file(rag, file_path)
            except Exception as e:
                logger.error(f"Error indexing file {file_path}: {str(e)}")

    except Exception as e:
        logger.error(f"Error during scanning process: {str(e)}")


def create_document_routes(
    rag: LightRAG, doc_manager: DocumentManager, api_key: Optional[str] = None
):
    optional_api_key = get_api_key_dependency(api_key)

    @router.post("/scan", dependencies=[Depends(optional_api_key)])
    async def scan_for_new_documents(background_tasks: BackgroundTasks):
        """
        Trigger the scanning process for new documents.

        This endpoint initiates a background task that scans the input directory for new documents
        and processes them. If a scanning process is already running, it returns a status indicating
        that fact.

        Returns:
            dict: A dictionary containing the scanning status
        """
        # Start the scanning process in the background
        background_tasks.add_task(run_scanning_process, rag, doc_manager)
        return {"status": "scanning_started"}

    @router.post("/upload", dependencies=[Depends(optional_api_key)])
    async def upload_to_input_dir(
        background_tasks: BackgroundTasks, file: UploadFile = File(...)
    ):
        """
        Upload a file to the input directory and index it.

        This API endpoint accepts a file through an HTTP POST request, checks if the
        uploaded file is of a supported type, saves it in the specified input directory,
        indexes it for retrieval, and returns a success status with relevant details.

        Args:
            background_tasks: FastAPI BackgroundTasks for async processing
            file (UploadFile): The file to be uploaded. It must have an allowed extension.

        Returns:
            InsertResponse: A response object containing the upload status and a message.

        Raises:
            HTTPException: If the file type is not supported (400) or other errors occur (500).
        """
        try:
            if not doc_manager.is_supported_file(file.filename):
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type. Supported types: {doc_manager.supported_extensions}",
                )

            file_path = doc_manager.input_dir / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # Add to background tasks
            background_tasks.add_task(pipeline_index_file, rag, file_path)

            return InsertResponse(
                status="success",
                message=f"File '{file.filename}' uploaded successfully. Processing will continue in background.",
            )
        except Exception as e:
            logger.error(f"Error /documents/upload: {file.filename}: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))

    @router.post(
        "/text", response_model=InsertResponse, dependencies=[Depends(optional_api_key)]
    )
    async def insert_text(
        request: InsertTextRequest, background_tasks: BackgroundTasks
    ):
        """
        Insert text into the RAG system.

        This endpoint allows you to insert text data into the RAG system for later retrieval
        and use in generating responses.

        Args:
            request (InsertTextRequest): The request body containing the text to be inserted.
            background_tasks: FastAPI BackgroundTasks for async processing

        Returns:
            InsertResponse: A response object containing the status of the operation.

        Raises:
            HTTPException: If an error occurs during text processing (500).
        """
        try:
            background_tasks.add_task(pipeline_index_texts, rag, [request.text])
            return InsertResponse(
                status="success",
                message="Text successfully received. Processing will continue in background.",
            )
        except Exception as e:
            logger.error(f"Error /documents/text: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))

    @router.post(
        "/texts",
        response_model=InsertResponse,
        dependencies=[Depends(optional_api_key)],
    )
    async def insert_texts(
        request: InsertTextsRequest, background_tasks: BackgroundTasks
    ):
        """
        Insert multiple texts into the RAG system.

        This endpoint allows you to insert multiple text entries into the RAG system
        in a single request.

        Args:
            request (InsertTextsRequest): The request body containing the list of texts.
            background_tasks: FastAPI BackgroundTasks for async processing

        Returns:
            InsertResponse: A response object containing the status of the operation.

        Raises:
            HTTPException: If an error occurs during text processing (500).
        """
        try:
            background_tasks.add_task(pipeline_index_texts, rag, request.texts)
            return InsertResponse(
                status="success",
                message="Text successfully received. Processing will continue in background.",
            )
        except Exception as e:
            logger.error(f"Error /documents/text: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))

    @router.post(
        "/file", response_model=InsertResponse, dependencies=[Depends(optional_api_key)]
    )
    async def insert_file(
        background_tasks: BackgroundTasks, file: UploadFile = File(...)
    ):
        """
        Insert a file directly into the RAG system.

        This endpoint accepts a file upload and processes it for inclusion in the RAG system.
        The file is saved temporarily and processed in the background.

        Args:
            background_tasks: FastAPI BackgroundTasks for async processing
            file (UploadFile): The file to be processed

        Returns:
            InsertResponse: A response object containing the status of the operation.

        Raises:
            HTTPException: If the file type is not supported (400) or other errors occur (500).
        """
        try:
            if not doc_manager.is_supported_file(file.filename):
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type. Supported types: {doc_manager.supported_extensions}",
                )

            temp_path = await save_temp_file(doc_manager.input_dir, file)

            # Add to background tasks
            background_tasks.add_task(pipeline_index_file, rag, temp_path)

            return InsertResponse(
                status="success",
                message=f"File '{file.filename}' saved successfully. Processing will continue in background.",
            )
        except Exception as e:
            logger.error(f"Error /documents/file: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))

    @router.post(
        "/file_batch",
        response_model=InsertResponse,
        dependencies=[Depends(optional_api_key)],
    )
    async def insert_batch(
        background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)
    ):
        """
        Process multiple files in batch mode.

        This endpoint allows uploading and processing multiple files simultaneously.
        It handles partial successes and provides detailed feedback about failed files.

        Args:
            background_tasks: FastAPI BackgroundTasks for async processing
            files (List[UploadFile]): List of files to process

        Returns:
            InsertResponse: A response object containing:
                - status: "success", "partial_success", or "failure"
                - message: Detailed information about the operation results

        Raises:
            HTTPException: If an error occurs during processing (500).
        """
        try:
            inserted_count = 0
            failed_files = []
            temp_files = []

            for file in files:
                if doc_manager.is_supported_file(file.filename):
                    # Create a temporary file to save the uploaded content
                    temp_files.append(await save_temp_file(doc_manager.input_dir, file))
                    inserted_count += 1
                else:
                    failed_files.append(f"{file.filename} (unsupported type)")

            if temp_files:
                background_tasks.add_task(pipeline_index_files, rag, temp_files)

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

            return InsertResponse(status=status, message=status_message)
        except Exception as e:
            logger.error(f"Error /documents/batch: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))

    @router.delete(
        "", response_model=InsertResponse, dependencies=[Depends(optional_api_key)]
    )
    async def clear_documents():
        """
        Clear all documents from the RAG system.

        This endpoint deletes all text chunks, entities vector database, and relationships
        vector database, effectively clearing all documents from the RAG system.

        Returns:
            InsertResponse: A response object containing the status and message.

        Raises:
            HTTPException: If an error occurs during the clearing process (500).
        """
        try:
            rag.text_chunks = []
            rag.entities_vdb = None
            rag.relationships_vdb = None
            return InsertResponse(
                status="success", message="All documents cleared successfully"
            )
        except Exception as e:
            logger.error(f"Error DELETE /documents: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))

    @router.get(
        "/pipeline_status",
        dependencies=[Depends(optional_api_key)],
        response_model=PipelineStatusResponse,
    )
    async def get_pipeline_status() -> PipelineStatusResponse:
        """
        Get the current status of the document indexing pipeline.

        This endpoint returns information about the current state of the document processing pipeline,
        including the processing status, progress information, and history messages.

        Returns:
            PipelineStatusResponse: A response object containing:
                - autoscanned (bool): Whether auto-scan has started
                - busy (bool): Whether the pipeline is currently busy
                - job_name (str): Current job name (e.g., indexing files/indexing texts)
                - job_start (str, optional): Job start time as ISO format string
                - docs (int): Total number of documents to be indexed
                - batchs (int): Number of batches for processing documents
                - cur_batch (int): Current processing batch
                - request_pending (bool): Flag for pending request for processing
                - latest_message (str): Latest message from pipeline processing
                - history_messages (List[str], optional): List of history messages

        Raises:
            HTTPException: If an error occurs while retrieving pipeline status (500)
        """
        try:
            from lightrag.kg.shared_storage import get_namespace_data

            pipeline_status = await get_namespace_data("pipeline_status")

            # Convert to regular dict if it's a Manager.dict
            status_dict = dict(pipeline_status)

            # Convert history_messages to a regular list if it's a Manager.list
            if "history_messages" in status_dict:
                status_dict["history_messages"] = list(status_dict["history_messages"])

            # Format the job_start time if it exists
            if status_dict.get("job_start"):
                status_dict["job_start"] = str(status_dict["job_start"])

            return PipelineStatusResponse(**status_dict)
        except Exception as e:
            logger.error(f"Error getting pipeline status: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("", dependencies=[Depends(optional_api_key)])
    async def documents() -> DocsStatusesResponse:
        """
        Get the status of all documents in the system.

        This endpoint retrieves the current status of all documents, grouped by their
        processing status (PENDING, PROCESSING, PROCESSED, FAILED).

        Returns:
            DocsStatusesResponse: A response object containing a dictionary where keys are
                                DocStatus values and values are lists of DocStatusResponse
                                objects representing documents in each status category.

        Raises:
            HTTPException: If an error occurs while retrieving document statuses (500).
        """
        try:
            statuses = (
                DocStatus.PENDING,
                DocStatus.PROCESSING,
                DocStatus.PROCESSED,
                DocStatus.FAILED,
            )

            tasks = [rag.get_docs_by_status(status) for status in statuses]
            results: List[Dict[str, DocProcessingStatus]] = await asyncio.gather(*tasks)

            response = DocsStatusesResponse()

            for idx, result in enumerate(results):
                status = statuses[idx]
                for doc_id, doc_status in result.items():
                    if status not in response.statuses:
                        response.statuses[status] = []
                    response.statuses[status].append(
                        DocStatusResponse(
                            id=doc_id,
                            content_summary=doc_status.content_summary,
                            content_length=doc_status.content_length,
                            status=doc_status.status,
                            created_at=DocStatusResponse.format_datetime(
                                doc_status.created_at
                            ),
                            updated_at=DocStatusResponse.format_datetime(
                                doc_status.updated_at
                            ),
                            chunks_count=doc_status.chunks_count,
                            error=doc_status.error,
                            metadata=doc_status.metadata,
                        )
                    )
            return response
        except Exception as e:
            logger.error(f"Error GET /documents: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))

    return router
