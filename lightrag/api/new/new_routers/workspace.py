import ast
import base64
from enum import Enum
import json
import logging
import os
from pathlib import Path
from pdb import pm
import shutil
from typing import Callable, List, Any, Optional
from fastapi import APIRouter,BackgroundTasks, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel
from lightrag.api.routers.document_routes import DocumentManager, pipeline_index_file, pipeline_index_files, pipeline_index_texts, save_temp_file
from lightrag.base import QueryParam
from ascii_colors import trace_exception
from starlette.status import HTTP_403_FORBIDDEN

router = APIRouter(prefix="/new", tags=["workspace"])


# LightRAG query mode
class SearchMode(str, Enum):
    naive = "naive"
    local = "local"
    global_ = "global"
    hybrid = "hybrid"
    mix = "mix"


class QueryRequest(BaseModel):
    query: str
    mode: SearchMode = SearchMode.hybrid
    stream: bool = False
    only_need_context: bool = False
    only_need_prompt: bool = False
    response_type: str = "Multiple Paragraphs"
    stream: bool = False
    top_k: int = 60
    max_token_for_text_unit: int = 4000
    max_token_for_global_context: int = 4000
    max_token_for_local_context: int = 4000


class DataResponse(BaseModel):
    code: int
    message: str
    data: Any


class QueryResponse(BaseModel):
    response: str


class InsertTextRequest(BaseModel):
    text: str


class CreateWorkspaceRequest(BaseModel):
    workspace: str


class UpdateWorkspaceRequest(BaseModel):
    workspace: str


class InsertResponse(BaseModel):
    status: str
    message: str


def create_workspace_routes(
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
        "/query", response_model=QueryResponse, dependencies=[Depends(optional_api_key)]
    )
    async def query_text(request: QueryRequest, rag=Depends(optional_working_dir)):
        """
        Handle a POST request at the /query endpoint to process user queries using RAG capabilities.

        Parameters:
            request (QueryRequest): A Pydantic model containing the following fields:
                - query (str): The text of the user's query.
                - mode (ModeEnum): Optional. Specifies the mode of retrieval augmentation.
                - stream (bool): Optional. Determines if the response should be streamed.
                - only_need_context (bool): Optional. If true, returns only the context without further processing.

        Returns:
            QueryResponse: A Pydantic model containing the result of the query processing.
                           If a string is returned (e.g., cache hit), it's directly returned.
                           Otherwise, an async generator may be used to build the response.

        Raises:
            HTTPException: Raised when an error occurs during the request handling process,
                           with status code 500 and detail containing the exception message.
        """
        try:
            response = await rag.aquery(
                request.query,
                param=QueryParam(
                    mode=request.mode,
                    stream=bool(request.stream),
                    only_need_context=request.only_need_context,
                    only_need_prompt=request.only_need_prompt,
                    response_type=request.response_type,
                    top_k=int(request.top_k),
                    max_token_for_text_unit=int(request.max_token_for_text_unit),
                    # Number of tokens for the relationship descriptions
                    max_token_for_global_context=int(
                        request.max_token_for_global_context
                    ),
                    # Number of tokens for the entity descriptions
                    max_token_for_local_context=int(
                        request.max_token_for_local_context
                    ),
                ),
            )

            # If response is a string (e.g. cache hit), return directly
            if isinstance(response, str):
                return QueryResponse(response=response)

            if isinstance(response, dict):
                result = json.dumps(response, indent=2)
                return QueryResponse(response=result)
            else:
                return QueryResponse(response=str(response))

        except Exception as e:
            trace_exception(e)
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/query/stream", dependencies=[Depends(optional_api_key)])
    async def query_text_stream(
        request: QueryRequest, rag=Depends(optional_working_dir)
    ):
        """
        This endpoint performs a retrieval-augmented generation (RAG) query and streams the response.

        Args:
            request (QueryRequest): The request object containing the query parameters.
            optional_api_key (Optional[str], optional): An optional API key for authentication. Defaults to None.

        Returns:
            StreamingResponse: A streaming response containing the RAG query results.
        """
        try:
            response = await rag.aquery(  # Use aquery instead of query, and add await
                request.query,
                param=QueryParam(
                    mode=request.mode,
                    stream=True,
                    only_need_context=request.only_need_context,
                    only_need_prompt=request.only_need_prompt,
                    response_type=request.response_type,
                    top_k=int(request.top_k),
                    max_token_for_text_unit=int(request.max_token_for_text_unit),
                    # Number of tokens for the relationship descriptions
                    max_token_for_global_context=int(
                        request.max_token_for_global_context
                    ),
                    # Number of tokens for the entity descriptions
                    max_token_for_local_context=int(
                        request.max_token_for_local_context
                    ),
                ),
            )

            from fastapi.responses import StreamingResponse

            async def stream_generator():
                if isinstance(response, str):
                    # If it's a string, send it all at once
                    yield f"{json.dumps({'response': response})}\n"
                else:
                    # If it's an async generator, send chunks one by one
                    if hasattr(response, "__aiter__"):
                        try:
                            async for chunk in response:
                                if chunk:  # Only send non-empty content
                                    yield f"{json.dumps({'response': chunk})}\n"
                        except Exception as e:
                            logging.error(f"Streaming error: {str(e)}")
                            yield f"{json.dumps({'error': str(e)})}\n"
                    else:
                        # If it's not an async generator, treat it as a single response
                        yield f"{response}\n"

            return StreamingResponse(
                stream_generator(),
                media_type="application/x-ndjson",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Content-Type": "application/x-ndjson",
                    "X-Accel-Buffering": "no",  # Ensure proper handling of streaming response when proxied by Nginx
                },
            )
        except Exception as e:
            trace_exception(e)
            raise HTTPException(status_code=500, detail=str(e))

    @router.post(
        "/documents/text",
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
        "/documents/file",
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
        "/documents/batch",
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
        "/documents",
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

    # 获取所有工作空间
    @router.get(
        "/workspaces/all",
        response_model=DataResponse,
        dependencies=[Depends(optional_api_key)],
    )
    async def get_workspaces():
        try:
            working_dir = args.working_dir
            workspaces = []
            for item_name in os.listdir(working_dir):
                item_path = os.path.join(working_dir, item_name)
                if os.path.isdir(item_path):
                    # 获取文件夹相关信息
                    dir_info = os.stat(item_path)
                    workspaces.append(
                        {
                            "name": base64.urlsafe_b64decode(
                                item_name.encode("utf-8")
                            ).decode("utf-8"),
                            "mtime": dir_info.st_mtime,
                            "birthtime": getattr(
                                dir_info, "st_birthtime", dir_info.st_mtime
                            ),
                        }
                    )
            return DataResponse(
                code=0,
                message="ok",
                data=workspaces,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # 获取工作空间详情
    @router.get(
        "/workspaces/{workspace}",
        response_model=DataResponse,
        dependencies=[Depends(optional_api_key)],
    )
    async def get_workspace_detail(workspace: str):
        try:
            workspace_path = Path(
                args.working_dir,
                base64.urlsafe_b64encode(workspace.encode("utf-8")).decode("utf-8"),
            )
            # 如果文件夹已存在，抛出异常
            if not os.path.exists(workspace_path):
                raise FileExistsError(f"Workspace not found: {workspace}")
            # 获取文件夹相关信息
            dir_info = os.stat(workspace_path)
            data = {
                "name": workspace,
                "mtime": dir_info.st_mtime,
                "birthtime": getattr(dir_info, "st_birthtime", dir_info.st_mtime),
            }
            return DataResponse(code=0, message="ok", data=data)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # 创建新的工作空间
    @router.post(
        "/workspaces",
        response_model=DataResponse,
        dependencies=[Depends(optional_api_key)],
    )
    async def create_workspace(request: CreateWorkspaceRequest):
        try:
            workspace = request.workspace
            # 如果为空
            if not workspace:
                raise ValueError("Workspace name is required")

            workspace_path = Path(
                args.working_dir,
                base64.urlsafe_b64encode(workspace.encode("utf-8")).decode("utf-8"),
            )
            # 如果文件夹已存在，抛出异常
            if os.path.exists(workspace_path):
                raise FileExistsError(f"Workspace already exists: {workspace}")
            Path(workspace_path).mkdir(parents=False, exist_ok=True)
            data = {
                "name": workspace,
            }
            return DataResponse(code=0, message="ok", data=data)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # 修改工作空间
    @router.put(
        "/workspaces/{workspace}",
        response_model=DataResponse,
        dependencies=[Depends(optional_api_key)],
    )
    async def update_workspace(workspace: str, request: UpdateWorkspaceRequest):
        try:
            new_workspace = request.workspace
            # 如果为空
            if not workspace:
                raise ValueError("Workspace name is required")
            if not new_workspace:
                raise ValueError("New workspace name is required")
            # 如果工作区名称与新工作区名称相同，抛出异常
            if workspace == new_workspace:
                raise ValueError("New workspace name is the same as the original one")
            print("Renaming workspace...")
            print(f"Original workspace: {workspace}")
            print(f"New workspace: {new_workspace}")
            origin_workspace_path = Path(
                args.working_dir,
                base64.urlsafe_b64encode(workspace.encode("utf-8")).decode("utf-8"),
            )

            # 如果文件夹不存在，抛出异常
            if not Path(origin_workspace_path).exists():
                raise FileExistsError(f"Workspace not exists: {workspace}")
            new_workspace_path = Path(
                args.working_dir,
                base64.urlsafe_b64encode(new_workspace.encode("utf-8")).decode("utf-8"),
            )
            # 如果文件夹已存在，抛出异常
            if Path(new_workspace_path).exists():
                raise FileExistsError(f"Workspace '{new_workspace}' already exists")
            # 修改工作空间
            Path(origin_workspace_path).rename(new_workspace_path)
            data = {
                "name": new_workspace,
            }
            return DataResponse(code=0, message="ok", data=data)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # 删除工作空间
    @router.delete("/workspaces/{workspace}", response_model=DataResponse)
    async def delete_workspace(workspace: str):
        try:
            # 如果为空
            if not workspace:
                raise ValueError("Workspace name is required")
            workspace_path = Path(
                args.working_dir,
                base64.urlsafe_b64encode(workspace.encode("utf-8")).decode("utf-8"),
            )
            # 如果存在 workspace，则删除它
            if Path(workspace_path).exists():
                shutil.rmtree(workspace_path)
            return DataResponse(code=0, message="ok", data=None)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # 清空所有文档
    @router.delete(
        "/documents/all",
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

    # 删除缓存
    @router.post("/resetcache", dependencies=[Depends(optional_api_key)])
    async def reset_cache(rag=Depends(optional_working_dir)):
        """Manually reset cache"""
        try:
            cachefile = rag.working_dir + "/kv_store_llm_response_cache.json"
            if os.path.exists(cachefile):
                with open(cachefile, "w") as f:
                    f.write("{}")
            return DataResponse(
                code=0,
                message=f"Manually reset cache successfully",
                data="ok",
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # 删除单个文档及其建立的知识图谱
    @router.delete(
        "/documents/{document_id}",
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

    # 知识图谱-实体修改
    @router.put(
        "/graph/entity/{entity_name}",
        response_model=DataResponse,
        dependencies=[Depends(optional_api_key)],
    )
    async def update_entity(
        entity_name: str, entity_data: dict, rag=Depends(optional_working_dir)
    ):
        try:
            entity_type = entity_data.get("entity_type", None)
            description = entity_data.get("description", None)
            source_id = entity_data.get("source_id", None)
            # Prepare node data
            node_data = {
                "entity_type": entity_type,
                "description": description,
                "source_id": source_id,
            }
            # Insert node data into the knowledge graph
            await rag.chunk_entity_relation_graph.upsert_node(
                entity_name, node_data=node_data
            )
            await rag.chunk_entity_relation_graph.index_done_callback()
            data = {"id": entity_name, "label": entity_name, **node_data}
            return DataResponse(code=0, message="ok", data=data)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # 知识图谱-实体删除
    @router.delete(
        "/graph/entity/{entity_name}",
        response_model=DataResponse,
        dependencies=[Depends(optional_api_key)],
    )
    async def delete_entity(entity_name: str, rag=Depends(optional_working_dir)):
        print(f"Deleting entity {entity_name}")
        try:
            await rag.adelete_by_entity(entity_name)
            return DataResponse(code=0, message="ok", data=None)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # 知识图谱-关系修改-通过开始节点和结束节点查询关系
    @router.put(
        "/graph/relation/by_nodes",
        response_model=DataResponse,
        dependencies=[Depends(optional_api_key)],
    )
    async def update_relation_by_nodes(
        src_entity_name: str,
        tgt_entity_name: str,
        relation_data: dict,
        rag=Depends(optional_working_dir),
    ):
        try:
            src_id = src_entity_name
            tgt_id = tgt_entity_name
            weight = relation_data.get("weight", None)
            keywords = relation_data.get("keywords", None)
            description = relation_data.get("description", None)
            source_id = relation_data.get("source_id", None)

            edge_data = dict(
                weight=weight,
                description=description,
                keywords=keywords,
                source_id=source_id,
            )

            print("Edge Data:", edge_data)  # Debugging print
            # Insert node data into the knowledge graph
            await rag.chunk_entity_relation_graph.upsert_edge(
                src_id,
                tgt_id,
                edge_data=edge_data,
            )
            await rag.chunk_entity_relation_graph.index_done_callback()
            data = {
                "id": src_id + "_" + tgt_id,
                "source": src_id,
                "target": tgt_id,
                **edge_data,
            }
            return DataResponse(code=0, message="ok", data=data)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # 知识图谱-关系删除-通过开始节点和结束节点查询关系
    @router.delete(
        "/graph/relation/by_nodes",
        response_model=DataResponse,
        dependencies=[Depends(optional_api_key)],
    )
    async def delete_relation_by_nodes(
        src_entity_name: str, tgt_entity_name: str, rag=Depends(optional_working_dir)
    ):
        try:
            await rag.relationships_vdb.delete_entity_relation_by_nodes(
                src_entity_name, tgt_entity_name
            )
            relationships_to_delete = set()
            relationships_to_delete.add((src_entity_name, tgt_entity_name))
            rag.chunk_entity_relation_graph.remove_edges(list(relationships_to_delete))
            await rag.relationships_vdb.index_done_callback()
            await rag.chunk_entity_relation_graph.index_done_callback()
            return DataResponse(code=0, message="ok", data="")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # 知识图谱-实体查询
    @router.get(
        "/graph/entity/{entity_name}",
        response_model=DataResponse,
        dependencies=[Depends(optional_api_key)],
    )
    async def get_node(entity_name: str, rag=Depends(optional_working_dir)):
        try:
            node = await rag.chunk_entity_relation_graph.get_node(entity_name)
            return DataResponse(code=0, message="ok", data=node)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # 知识图谱-关系查询-通过开始节点和结束节点查询关系
    @router.get(
        "/graph/relation/by_nodes",
        response_model=DataResponse,
        dependencies=[Depends(optional_api_key)],
    )
    async def get_relation_by_nodes(
        src_entity_name: str, tgt_entity_name: str, rag=Depends(optional_working_dir)
    ):
        try:
            relation = await rag.chunk_entity_relation_graph.get_edge(
                src_entity_name, tgt_entity_name
            )
            return DataResponse(code=0, message="ok", data=relation)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # 知识图谱-关系查询-通过节点ID查询关系
    @router.get(
        "/graph/relation/node/{node_id}",
        response_model=DataResponse,
        dependencies=[Depends(optional_api_key)],
    )
    async def get_relation_by_node(node_id: str, rag=Depends(optional_working_dir)):
        try:
            relations = await rag.chunk_entity_relation_graph.get_node_edges(node_id)
            return DataResponse(code=0, message="ok", data=relations)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # 知识图谱-实体列表-查询
    @router.get(
        "/graph/entity",
        response_model=DataResponse,
        dependencies=[Depends(optional_api_key)],
    )
    async def get_graph_entity_list(rag=Depends(optional_working_dir)):
        try:
            # 提取所有实体和关系
            entities = await rag.chunk_entity_relation_graph.query_all()

            # 返回知识图谱数据
            return DataResponse(
                code=0,
                message="ok",
                data=entities,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # 知识图谱-文档列表-查询
    @router.get(
        "/graph/document",
        response_model=DataResponse,
        dependencies=[Depends(optional_api_key)],
    )
    async def get_graph_document_list(rag=Depends(optional_working_dir)):
        try:
            # 提取所有文档
            documents = await rag.doc_status.get_all_docs()

            # 返回知识图谱数据
            return DataResponse(
                code=0,
                message="ok",
                data=documents,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # 知识图谱-文档-查询
    @router.get(
        "/graph/document/{document_id}",
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

    # 知识图谱-数据查询
    @router.get(
        "/graph/data",
        response_model=DataResponse,
        dependencies=[Depends(optional_api_key)],
    )
    async def get_graph_data(rag=Depends(optional_working_dir)):
        try:
            # 提取所有实体和关系
            entities = await rag.chunk_entity_relation_graph.query_all()
            # 返回知识图谱数据
            return DataResponse(code=0, message="ok", data=entities)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return router
