
import asyncio
import logging
from pathlib import Path
from typing import Callable, Dict, List, Any, Optional
from fastapi import APIRouter,BackgroundTasks, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel
from lightrag.api.new.new_routers.workspace_routes import DataResponse
from lightrag.api.routers.document_routes import DocStatusResponse, DocsStatusesResponse, DocumentManager, InsertResponse, InsertTextRequest, pipeline_index_file, pipeline_index_files, pipeline_index_texts, save_temp_file
from lightrag.base import DocProcessingStatus, DocStatus

router = APIRouter(prefix="/new/documents", tags=["documents"])

def create_new_document_routes(
    args,
    doc_manager: DocumentManager,
    api_key: Optional[str] = None,
    get_api_key_dependency: Optional[Callable] = None,
    get_working_dir_dependency: Optional[Callable] = None,
):
    # Setup logging
    logging.basicConfig(
        format="%(levelname)s:%(message)s", level=getattr(logging, args.log_level)
    )

    optional_api_key = get_api_key_dependency(api_key)
    optional_working_dir = get_working_dir_dependency(args)
    @router.post(
        "/text",
        response_model=InsertResponse,
        dependencies=[Depends(optional_api_key)],
    )
    async def insert_text(
        background_tasks: BackgroundTasks,
        request: InsertTextRequest, rag=Depends(optional_working_dir)
    ):
        """
        Insert text into the Retrieval-Augmented Generation (RAG) system.

        This endpoint allows you to insert text data into the RAG system for later retrieval and use in generating responses.

        Args:
            request (InsertTextRequest): The request body containing the text to be inserted.

        Returns:
            InsertResponse: A response object containing the status of the operation, a message, and the number of documents inserted.
        """
        try:
            background_tasks.add_task(pipeline_index_texts, rag, [request.text])
            return InsertResponse(
                status="success",
                message="Text successfully inserted"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.post(
        "/file",
        response_model=InsertResponse,
        dependencies=[Depends(optional_api_key)],
    )
    async def insert_file(
        background_tasks: BackgroundTasks,
        file: UploadFile,
        rag=Depends(optional_working_dir),
    ):
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
        except UnicodeDecodeError:
            raise HTTPException(status_code=400, detail="File encoding not supported")
        except Exception as e:
            logging.error(f"Error processing file {file.filename}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.post(
        "/file_batch",
        response_model=InsertResponse,
        dependencies=[Depends(optional_api_key)],
    )
    async def insert_batch(
        background_tasks: BackgroundTasks,
        files: List[UploadFile] = File(...),
        rag=Depends(optional_working_dir),
    ):
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
            logging.error(f"Batch processing error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.delete(
        "",
        response_model=InsertResponse,
        dependencies=[Depends(optional_api_key)],
    )
    async def clear_documents(rag=Depends(optional_working_dir)):
        """
        Clear all documents from the LightRAG system.

        This endpoint deletes all text chunks, entities vector database, and relationships vector database,
        effectively clearing all documents from the LightRAG system.

        Returns:
            InsertResponse: A response object containing the status, message, and the new document count (0 in this case).
        """
        try:
            rag.text_chunks = []
            rag.entities_vdb = None
            rag.relationships_vdb = None
            return InsertResponse(
                status="success",
                message="All documents cleared successfully"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    # 删除单个文档及其建立的知识图谱
    @router.delete(
        "/{document_id}",
        response_model=DataResponse,
        dependencies=[Depends(optional_api_key)],
    )
    async def delete_document(document_id: str, rag=Depends(optional_working_dir)):
        try:
            await rag.adelete_by_doc_id(document_id)
            return DataResponse(
                code=0,
                message=f"Document {document_id} cleared successfully",
                data="ok",
            )
        except Exception as e:
            print(e)
            raise HTTPException(status_code=500, detail=str(e))
    # 清空所有文档
    @router.delete(
        "/all",
        response_model=DataResponse,
        dependencies=[Depends(optional_api_key)],
    )
    async def clear_all_documents(rag=Depends(optional_working_dir)):
        try:
            # 删除相关数据
            await rag.llm_response_cache.drop()
            await rag.full_docs.drop()
            await rag.text_chunks.drop()
            await rag.doc_status.drop()
            await rag.chunk_entity_relation_graph.delete_all()
            await rag.entities_vdb.delete_all()
            await rag.relationships_vdb.delete_all()
            await rag.chunks_vdb.delete_all()

            await rag.llm_response_cache.index_done_callback()
            await rag.full_docs.index_done_callback()
            await rag.text_chunks.index_done_callback()
            await rag.doc_status.index_done_callback()
            await rag.chunk_entity_relation_graph.index_done_callback()
            await rag.entities_vdb.index_done_callback()
            await rag.relationships_vdb.index_done_callback()
            await rag.chunks_vdb.index_done_callback()

            return DataResponse(
                code=0, message="All documents cleared successfully", data="ok"
            )
        except Exception as e:
            print(e)
            raise HTTPException(status_code=500, detail=str(e))
    # 知识图谱-文档列表-查询
    @router.get(
        "",
        response_model=DocsStatusesResponse,
        dependencies=[Depends(optional_api_key)],
    )
    async def documents(rag=Depends(optional_working_dir)):
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
            raise HTTPException(status_code=500, detail=str(e))

    # 知识图谱-文档-查询
    @router.get(
        "/{document_id}",
        response_model=DataResponse,
        dependencies=[Depends(optional_api_key)],
    )
    async def get_graph_document_detail(document_id, rag=Depends(optional_working_dir)):
        try:
            # 查询文档状态
            doc_status_document = await rag.doc_status.get_by_id(document_id)
            document = await rag.full_docs.get_by_id(document_id)
            if not doc_status_document:
                return DataResponse(
                    code=0,
                    message=f"Document {document_id} get successfully",
                    data=None,
                )

            # 查询文档的chunk
            chunks = await rag.text_chunks.get_by_keys({"full_doc_id": document_id})
            # 将chunks转换为list,chunk[0]为id，chunk[1]为其他字段
            chunk_List = []
            for chunk in chunks:
                chunk_List.append(
                    {
                        "id": chunk[0],
                        # 解构chunk[1]为其他字段
                        **chunk[1],
                    }
                )
            # 返回知识图谱数据
            return DataResponse(
                code=0,
                message="ok",
                data={
                    "doc_status_document": doc_status_document,
                    "document": document,
                    "chunks": chunk_List,
                    # "details": details,
                },
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return router