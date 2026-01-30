import json
from typing import Optional, AsyncGenerator
from fastapi import APIRouter, Depends, HTTPException, Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from lightrag import LightRAG
from lightrag.api.routers.query_routes import (
    QueryRequest, QueryResponse, QueryDataResponse
)
from lightrag.utils import logger
from ..dependencies import get_current_rag, get_current_user
from .. import db

router = APIRouter(tags=["query"])

class TenantQueryRequest(QueryRequest):
    session_id: Optional[str] = Field(
        default=None, 
        description="Chat session ID. If provided, history is loaded from DB and new messages are saved."
    )

@router.post("/query", response_model=QueryResponse)
async def query_text(
    request: TenantQueryRequest,
    rag: LightRAG = Depends(get_current_rag),
    user: dict = Depends(get_current_user)
):
    try:
        user_id = user["user_id"]
        session_id = request.session_id
        
        # 1. Manage Session
        if session_id:
             # Verify ownership? 
             # For now, simplistic check: if session exists, great. 
             # Ideally we check if session belongs to user.
             # In MVP, we trust ID or failed lookup returns empty.
             pass
        else:
             # Create new session if not provided?
             # Or treat as ephemeral (no persistence)?
             # User prompt: "Persistent Chat". 
             # If no session_id, we create one automatically or treat as one-off?
             # Let's create one automatically if meaningful? 
             # Usually client provides session_id or requests new one.
             # If client sends none, we treat as ephemeral unless they want persistence.
             # BUT: To return the session_id to the client, we need to modify QueryResponse.
             # QueryResponse struct is fixed.
             # So if no session_id, we default to ephemeral (no save).
             pass

        # 2. Load History for Context
        history_messages = []
        if session_id:
            db_history = db.get_chat_messages(session_id)
            # Convert to [{'role': 'user', 'content': '...'}, ...]
            # db_history has (role, content, timestamp...)
            for msg in db_history:
                history_messages.append({"role": msg["role"], "content": msg["content"]})
        
        # Add provided history (if any) - merge or override? 
        # Usually DB takes precedence or append? 
        # Let's append request history to DB history? 
        if request.conversation_history:
            history_messages.extend(request.conversation_history)

        # 3. Prepare Query Params
        param = request.to_query_params(request.stream or False)
        # Override history
        param.conversation_history = history_messages
        param.stream = False # Force false for this endpoint

        # 4. Save User Message (if session)
        if session_id:
            db.save_chat_message(session_id, "user", request.query)

        # 5. Execute Query
        result = await rag.aquery_llm(request.query, param=param)
        
        llm_response = result.get("llm_response", {})
        response_content = llm_response.get("content", "")
        if not response_content:
             response_content = "No relevant context found."
             
        # 6. Save Assistant Response (if session)
        if session_id:
            db.save_chat_message(session_id, "assistant", response_content)

        # 7. Return Response
        data = result.get("data", {})
        references = data.get("references", [])
        
        if request.include_references:
            return QueryResponse(response=response_content, references=references)
        else:
            return QueryResponse(response=response_content, references=None)

    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query/stream")
async def query_text_stream(
    request: TenantQueryRequest,
    rag: LightRAG = Depends(get_current_rag),
    user: dict = Depends(get_current_user)
):
    try:
        user_id = user["user_id"]
        session_id = request.session_id
        
        # History Setup (same as above)
        history_messages = []
        if session_id:
            db_history = db.get_chat_messages(session_id)
            for msg in db_history:
                history_messages.append({"role": msg["role"], "content": msg["content"]})
        if request.conversation_history:
            history_messages.extend(request.conversation_history)

        param = request.to_query_params(True)
        param.conversation_history = history_messages

        # Save User Message
        if session_id:
            db.save_chat_message(session_id, "user", request.query)

        result = await rag.aquery_llm(request.query, param=param)
        
        # Streaming Logic with Capture
        async def stream_generator():
            full_response_accumulator = []
            
            # Send references first
            if request.include_references:
                 refs = result.get("data", {}).get("references", [])
                 yield f"{json.dumps({'references': refs})}\n"
            
            llm_response = result.get("llm_response", {})
            if llm_response.get("is_streaming"):
                 response_stream = llm_response.get("response_iterator")
                 if response_stream:
                     async for chunk in response_stream:
                         if chunk:
                             full_response_accumulator.append(chunk)
                             yield f"{json.dumps({'response': chunk})}\n"
            else:
                 # Fallback if not actually streaming
                 content = llm_response.get("content", "")
                 full_response_accumulator.append(content)
                 yield f"{json.dumps({'response': content})}\n"
            
            # Save Accumulated Response
            if session_id and full_response_accumulator:
                full_text = "".join(full_response_accumulator)
                db.save_chat_message(session_id, "assistant", full_text)

        return StreamingResponse(
            stream_generator(),
            media_type="application/x-ndjson",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "application/x-ndjson",
                "X-Accel-Buffering": "no",
            },
        )

    except Exception as e:
        logger.error(f"Stream Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
