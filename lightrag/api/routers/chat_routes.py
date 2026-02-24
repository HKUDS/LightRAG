from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from ..dependencies import get_current_user
from .. import db

router = APIRouter(tags=["chats"])

class ChatSessionResponse(BaseModel):
    id: str
    title: Optional[str] = None
    created_at: str
    updated_at: str

class CreateChatRequest(BaseModel):
    title: Optional[str] = None

class ChatMessageResponse(BaseModel):
    role: str
    content: str
    created_at: str

@router.get("/chats", response_model=List[ChatSessionResponse])
async def list_chats(user: dict = Depends(get_current_user)):
    user_id = user["user_id"]
    sessions = db.get_user_chat_sessions(user_id)
    # Convert DB rows to pydantic
    return [
        ChatSessionResponse(
            id=s["id"], 
            title=s.get("name"), # DB has 'name', API uses 'title'
            created_at=s["created_at"],
            updated_at=s["updated_at"]
        ) for s in sessions
    ]

@router.post("/chats", response_model=ChatSessionResponse)
async def create_chat(
    request: CreateChatRequest, 
    user: dict = Depends(get_current_user)
):
    user_id = user["user_id"]
    try:
        session = db.create_chat_session(user_id, request.title or "New Chat")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create chat: {str(e)}")
    
    return ChatSessionResponse(
        id=session["id"],
        title=session.get("name"),
        created_at=session["created_at"],
        updated_at=session["updated_at"]
    )
    
@router.delete("/chats/{session_id}")
async def delete_chat(
    session_id: str,
    user: dict = Depends(get_current_user)
):
    # Verify ownership
    user_id = user["user_id"]
    # We need to check if session belongs to user.
    # db.py doesn't have `get_session`.
    # simplistic: list all user sessions and check if id in list.
    user_sessions = db.get_user_chat_sessions(user_id)
    if not any(s["id"] == session_id for s in user_sessions):
        raise HTTPException(status_code=404, detail="Session not found")
        
    db.delete_chat_session(session_id)
    return {"status": "success"}

@router.get("/chats/{session_id}/messages", response_model=List[ChatMessageResponse])
async def get_chat_messages(
    session_id: str,
    user: dict = Depends(get_current_user)
):
    # Verify ownership
    user_id = user["user_id"]
    user_sessions = db.get_user_chat_sessions(user_id)
    if not any(s["id"] == session_id for s in user_sessions):
        raise HTTPException(status_code=404, detail="Session not found")
        
    messages = db.get_chat_messages(session_id)
    return [
        ChatMessageResponse(
            role=m["role"],
            content=m["content"],
            created_at=m["created_at"]
        ) for m in messages
    ]
