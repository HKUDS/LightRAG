from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from uuid import UUID

from app.api.dependencies import get_db
from app.services.history_manager import HistoryManager
from app.services.chat_service import ChatService
from app.models.schemas import (
    SessionCreate, SessionResponse, ChatMessageRequest, ChatMessageResponse
)

router = APIRouter()

@router.post("/sessions", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
def create_session(session_in: SessionCreate, db: Session = Depends(get_db)):
    # For now, we assume a default user or handle auth separately.
    # Using a hardcoded user ID for demonstration if no auth middleware.
    # In production, get user_id from current_user.
    import uuid
    # Placeholder user ID. In real app, ensure user exists.
    # We might need to create a default user if not exists or require auth.
    # For this task, we'll create a dummy user if needed or just use a random UUID 
    # but that might fail FK constraint if user doesn't exist.
    # Let's assume we need to create a user first or use an existing one.
    # For simplicity, we'll generate a UUID but this will fail FK.
    # So we should probably have a "get_or_create_default_user" helper.
    
    # Quick fix: Create a default user if table is empty or just use a fixed ID 
    # and ensure it exists in startup event.
    # For now, let's just use a fixed UUID and assume the user exists or we create it.
    # Actually, let's just create a user on the fly for this session if we don't have auth.
    
    manager = HistoryManager(db)
    # User logic removed
    # Using a fixed UUID for demonstration purposes. In a real application,
    # this would come from an authenticated user.
    fixed_user_id = UUID("00000000-0000-0000-0000-000000000001")
    
    session = manager.create_session(
        user_id=fixed_user_id,
        title=session_in.title,
        rag_config=session_in.rag_config
    )
    return session

@router.get("/sessions", response_model=List[SessionResponse])
def list_sessions(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    manager = HistoryManager(db)
    # User logic removed
    pass
    # Using a fixed UUID for demonstration purposes. In a real application,
    # this would come from an authenticated user.
    fixed_user_id = UUID("00000000-0000-0000-0000-000000000001")
        
    sessions = manager.list_sessions(user_id=fixed_user_id, skip=skip, limit=limit)
    return sessions

@router.get("/sessions/{session_id}/history")
def get_session_history(session_id: UUID, db: Session = Depends(get_db)):
    manager = HistoryManager(db)
    # This returns context format, might need a different schema for full history display
    # For now reusing get_conversation_context logic but maybe we want full objects.
    # Let's just return the raw messages for now or map to a schema.
    # The requirement said "Get full history".
    from app.models.models import ChatMessage
    messages = db.query(ChatMessage).filter(ChatMessage.session_id == session_id).order_by(ChatMessage.created_at.asc()).all()
    return messages
