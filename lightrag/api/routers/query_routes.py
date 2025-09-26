"""
This module contains all query-related routes for the LightRAG API.
"""

import json
import logging
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from lightrag.base import QueryParam
from lightrag.api.utils_api import get_combined_auth_dependency
from pydantic import BaseModel, Field, field_validator

from ascii_colors import trace_exception

router = APIRouter(tags=["query"])


class QueryRequest(BaseModel):
    query: str = Field(
        min_length=3,
        description="The query text",
    )

    mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"] = Field(
        default="mix",
        description="Query mode",
    )

    only_need_context: Optional[bool] = Field(
        default=None,
        description="If True, only returns the retrieved context without generating a response.",
    )

    only_need_prompt: Optional[bool] = Field(
        default=None,
        description="If True, only returns the generated prompt without producing a response.",
    )

    response_type: Optional[str] = Field(
        min_length=1,
        default=None,
        description="Defines the response format. Examples: 'Multiple Paragraphs', 'Single Paragraph', 'Bullet Points'.",
    )

    top_k: Optional[int] = Field(
        ge=1,
        default=None,
        description="Number of top items to retrieve. Represents entities in 'local' mode and relationships in 'global' mode.",
    )

    chunk_top_k: Optional[int] = Field(
        ge=1,
        default=None,
        description="Number of text chunks to retrieve initially from vector search and keep after reranking.",
    )

    max_entity_tokens: Optional[int] = Field(
        default=None,
        description="Maximum number of tokens allocated for entity context in unified token control system.",
        ge=1,
    )

    max_relation_tokens: Optional[int] = Field(
        default=None,
        description="Maximum number of tokens allocated for relationship context in unified token control system.",
        ge=1,
    )

    max_total_tokens: Optional[int] = Field(
        default=None,
        description="Maximum total tokens budget for the entire query context (entities + relations + chunks + system prompt).",
        ge=1,
    )

    conversation_history: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Stores past conversation history to maintain context. Format: [{'role': 'user/assistant', 'content': 'message'}].",
    )

    user_prompt: Optional[str] = Field(
        default=None,
        description="User-provided prompt for the query. If provided, this will be used instead of the default value from prompt template.",
    )

    enable_rerank: Optional[bool] = Field(
        default=None,
        description="Enable reranking for retrieved text chunks. If True but no rerank model is configured, a warning will be issued. Default is True.",
    )

    include_references: Optional[bool] = Field(
        default=True,
        description="If True, includes reference list in responses. Affects /query and /query/stream endpoints. /query/data always includes references.",
    )

    @field_validator("query", mode="after")
    @classmethod
    def query_strip_after(cls, query: str) -> str:
        return query.strip()

    @field_validator("conversation_history", mode="after")
    @classmethod
    def conversation_history_role_check(
        cls, conversation_history: List[Dict[str, Any]] | None
    ) -> List[Dict[str, Any]] | None:
        if conversation_history is None:
            return None
        for msg in conversation_history:
            if "role" not in msg or msg["role"] not in {"user", "assistant"}:
                raise ValueError(
                    "Each message must have a 'role' key with value 'user' or 'assistant'."
                )
        return conversation_history

    def to_query_params(self, is_stream: bool) -> "QueryParam":
        """Converts a QueryRequest instance into a QueryParam instance."""
        # Use Pydantic's `.model_dump(exclude_none=True)` to remove None values automatically
        request_data = self.model_dump(exclude_none=True, exclude={"query"})

        # Ensure `mode` and `stream` are set explicitly
        param = QueryParam(**request_data)
        param.stream = is_stream
        return param


class QueryResponse(BaseModel):
    response: str = Field(
        description="The generated response",
    )
    references: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Reference list (only included when include_references=True, /query/data always includes references.)",
    )


class QueryDataResponse(BaseModel):
    status: str = Field(description="Query execution status")
    message: str = Field(description="Status message")
    data: Dict[str, Any] = Field(
        description="Query result data containing entities, relationships, chunks, and references"
    )
    metadata: Dict[str, Any] = Field(
        description="Query metadata including mode, keywords, and processing information"
    )


def create_query_routes(rag, api_key: Optional[str] = None, top_k: int = 60):
    combined_auth = get_combined_auth_dependency(api_key)

    @router.post(
        "/query", response_model=QueryResponse, dependencies=[Depends(combined_auth)]
    )
    async def query_text(request: QueryRequest):
        """
        This endpoint performs a RAG query with non-streaming response.

        Parameters:
            request (QueryRequest): The request object containing the query parameters.
        Returns:
            QueryResponse: A Pydantic model containing the result of the query processing.
                       If include_references=True, also includes reference list.

        Raises:
            HTTPException: Raised when an error occurs during the request handling process,
                       with status code 500 and detail containing the exception message.
        """
        try:
            param = request.to_query_params(
                False
            )  # Ensure stream=False for non-streaming endpoint
            # Force stream=False for /query endpoint regardless of include_references setting
            param.stream = False

            # Unified approach: always use aquery_llm for both cases
            result = await rag.aquery_llm(request.query, param=param)

            # Extract LLM response and references from unified result
            llm_response = result.get("llm_response", {})
            references = result.get("data", {}).get("references", [])

            # Get the non-streaming response content
            response_content = llm_response.get("content", "")
            if not response_content:
                response_content = "No relevant context found for the query."

            # Return response with or without references based on request
            if request.include_references:
                return QueryResponse(response=response_content, references=references)
            else:
                return QueryResponse(response=response_content, references=None)
        except Exception as e:
            trace_exception(e)
            raise HTTPException(status_code=500, detail=str(e))

    @router.post("/query/stream", dependencies=[Depends(combined_auth)])
    async def query_text_stream(request: QueryRequest):
        """
        This endpoint performs RAG query with streaming response.
        Streaming can be turn off by setting stream=False in QueryRequest.

        The streaming response includes:
        1. Reference list (sent first as a single message, if include_references=True)
        2. LLM response content (streamed as multiple chunks)

        Args:
            request (QueryRequest): The request object containing the query parameters.

        Returns:
            StreamingResponse: A streaming response containing:
                - First message: {"references": [...]} - Complete reference list (if requested)
                - Subsequent messages: {"response": "..."} - LLM response chunks
                - Error messages: {"error": "..."} - If any errors occur
        """
        try:
            param = request.to_query_params(
                True
            )  # Ensure stream=True for streaming endpoint

            from fastapi.responses import StreamingResponse

            # Unified approach: always use aquery_llm for all cases
            result = await rag.aquery_llm(request.query, param=param)

            async def stream_generator():

                # Extract references and LLM response from unified result
                references = result.get("data", {}).get("references", [])
                llm_response = result.get("llm_response", {})

                # Send reference list first if requested
                if request.include_references:
                    yield f"{json.dumps({'references': references})}\n"

                # Then stream the LLM response content
                if llm_response.get("is_streaming"):
                    response_stream = llm_response.get("response_iterator")
                    if response_stream:
                        try:
                            async for chunk in response_stream:
                                if chunk:  # Only send non-empty content
                                    yield f"{json.dumps({'response': chunk})}\n"
                        except Exception as e:
                            logging.error(f"Streaming error: {str(e)}")
                            yield f"{json.dumps({'error': str(e)})}\n"
                else:
                    # Non-streaming response (fallback)
                    response_content = llm_response.get("content", "")
                    if response_content:
                        yield f"{json.dumps({'response': response_content})}\n"
                    else:
                        yield f"{json.dumps({'response': 'No relevant context found for the query.'})}\n"

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
        "/query/data",
        response_model=QueryDataResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def query_data(request: QueryRequest):
        """
        Retrieve structured data without LLM generation.

        This endpoint returns raw retrieval results including entities, relationships,
        and text chunks that would be used for RAG, but without generating a final response.
        All parameters are compatible with the regular /query endpoint.

        Parameters:
            request (QueryRequest): The request object containing the query parameters.

        Returns:
            QueryDataResponse: A Pydantic model containing structured data with status,
                             message, data (entities, relationships, chunks, references),
                             and metadata.

        Raises:
            HTTPException: Raised when an error occurs during the request handling process,
                         with status code 500 and detail containing the exception message.
        """
        try:
            param = request.to_query_params(False)  # No streaming for data endpoint
            response = await rag.aquery_data(request.query, param=param)

            # aquery_data returns the new format with status, message, data, and metadata
            if isinstance(response, dict):
                return QueryDataResponse(**response)
            else:
                # Handle unexpected response format
                return QueryDataResponse(
                    status="failure",
                    message="Invalid response type",
                    data={},
                )
        except Exception as e:
            trace_exception(e)
            raise HTTPException(status_code=500, detail=str(e))

    return router
