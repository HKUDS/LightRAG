"""
Session History Routes for LightRAG API

This module provides REST API endpoints for managing chat sessions
and conversation history.
"""

from fastapi import APIRouter, Depends, HTTPException, Header, status
from sqlalchemy.orm import Session
from typing import List, Optional
from uuid import UUID
import time

from lightrag.api.session_database import get_db
from lightrag.api.session_manager import SessionHistoryManager
from lightrag.api.session_schemas import (
    SessionResponse,
    SessionCreate,
    ChatMessageResponse,
    ChatMessageRequest,
)
from lightrag.utils import logger

router = APIRouter(prefix="/history", tags=["Session History"])


async def get_current_user_id(
    x_user_id: Optional[str] = Header(None, alias="X-User-ID")
) -> str:
    """
    Extract user ID from request header.
    
    Args:
        x_user_id: User ID from X-User-ID header.
        
    Returns:
        User ID string, defaults to 'default_user' if not provided.
    """
    return x_user_id or "default_user"


@router.get("/sessions", response_model=List[SessionResponse])
async def list_sessions(
    skip: int = 0,
    limit: int = 20,
    db: Session = Depends(get_db),
    current_user_id: str = Depends(get_current_user_id),
):
    """
    List all chat sessions for the current user.
    
    Args:
        skip: Number of sessions to skip (for pagination).
        limit: Maximum number of sessions to return.
        db: Database session.
        current_user_id: Current user identifier.
        
    Returns:
        List of session response objects.
    """
    try:
        manager = SessionHistoryManager(db)
        sessions = manager.list_sessions(user_id=current_user_id, skip=skip, limit=limit)
        return sessions
    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sessions", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def create_session(
    session_in: SessionCreate,
    db: Session = Depends(get_db),
    current_user_id: str = Depends(get_current_user_id),
):
    """
    Create a new chat session.
    
    Args:
        session_in: Session creation request.
        db: Database session.
        current_user_id: Current user identifier.
        
    Returns:
        Created session response.
    """
    try:
        manager = SessionHistoryManager(db)
        session = manager.create_session(
            user_id=current_user_id,
            title=session_in.title,
            rag_config=session_in.rag_config,
        )
        return session
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}/history", response_model=List[ChatMessageResponse])
async def get_session_history(
    session_id: UUID,
    db: Session = Depends(get_db),
):
    """
    Get all messages for a specific session.
    
    Args:
        session_id: Session UUID.
        db: Database session.
        
    Returns:
        List of chat message responses with citations.
    """
    try:
        manager = SessionHistoryManager(db)
        messages = manager.get_session_history(session_id)
        return messages
    except Exception as e:
        logger.error(f"Error getting session history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: UUID,
    db: Session = Depends(get_db),
    current_user_id: str = Depends(get_current_user_id),
):
    """
    Delete a chat session and all its messages.
    
    Args:
        session_id: Session UUID.
        db: Database session.
        current_user_id: Current user identifier.
    """
    try:
        manager = SessionHistoryManager(db)
        
        # Verify session belongs to user
        session = manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        if session.user_id != current_user_id:
            raise HTTPException(status_code=403, detail="Not authorized to delete this session")
        
        manager.delete_session(session_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session: {e}")
        raise HTTPException(status_code=500, detail=str(e))
