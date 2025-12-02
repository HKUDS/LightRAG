from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy.orm import Session
from typing import List, Optional
from uuid import UUID
import sys
import os

# Ensure service module is in path (similar to query_routes.py)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
service_dir = os.path.join(project_root, "service")
if service_dir not in sys.path:
    sys.path.append(service_dir)

try:
    from app.core.database import get_db
    from app.services.history_manager import HistoryManager
    from app.models.schemas import SessionResponse, SessionCreate, ChatMessageResponse
except ImportError:
    # Fallback if service not found (shouldn't happen if setup is correct)
    get_db = None
    HistoryManager = None
    SessionResponse = None
    SessionCreate = None
    ChatMessageResponse = None

router = APIRouter()

def check_dependencies():
    if not HistoryManager:
        raise HTTPException(status_code=503, detail="History service not available")

async def get_current_user_id(
    x_user_id: Optional[str] = Header(None, alias="X-User-ID")
) -> str:
    # Prefer X-User-ID, default to default_user
    uid = x_user_id
    if not uid:
        # Fallback to default user if no header provided (for backward compatibility or dev)
        # Or raise error if strict
        return "default_user"
    return uid

@router.get("/sessions", response_model=List[SessionResponse], tags=["History"])
def list_sessions(
    skip: int = 0, 
    limit: int = 20, 
    db: Session = Depends(get_db),
    current_user_id: str = Depends(get_current_user_id)
):
    check_dependencies()
    manager = HistoryManager(db)
    sessions = manager.list_sessions(user_id=current_user_id, skip=skip, limit=limit)
    return sessions

@router.post("/sessions", response_model=SessionResponse, tags=["History"])
def create_session(
    session_in: SessionCreate, 
    db: Session = Depends(get_db),
    current_user_id: str = Depends(get_current_user_id)
):
    check_dependencies()
    manager = HistoryManager(db)
    return manager.create_session(user_id=current_user_id, title=session_in.title)

@router.get("/sessions/{session_id}/history", response_model=List[ChatMessageResponse], tags=["History"])
def get_session_history(
    session_id: str, 
    db: Session = Depends(get_db)
):
    check_dependencies()
    manager = HistoryManager(db)
    return manager.get_session_history(session_id)
