from enum import Enum
import json
import logging
from typing import Callable, Any, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from lightrag.base import  QueryParam
from ascii_colors import trace_exception

router = APIRouter(prefix="/new", tags=["query"])


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


def create_new_query_routes(
    args,
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

    return router
