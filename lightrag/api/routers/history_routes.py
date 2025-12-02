from fastapi import APIRouter, Depends, HTTPException
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
    from app.models.models import User
except ImportError:
    # Fallback if service not found (shouldn't happen if setup is correct)
    get_db = None
    HistoryManager = None
    SessionResponse = None
    SessionCreate = None
    ChatMessageResponse = None
    User = None

router = APIRouter()

def check_dependencies():
    if not HistoryManager:
        raise HTTPException(status_code=503, detail="History service not available")

@router.get("/sessions", response_model=List[SessionResponse], tags=["History"])
def list_sessions(
    skip: int = 0, 
    limit: int = 20, 
    db: Session = Depends(get_db)
):
    check_dependencies()
    manager = HistoryManager(db)
    # For now, get default user or create one
    user = db.query(User).filter(User.username == "default_user").first()
    if not user:
        user = User(username="default_user", email="default@example.com")
        db.add(user)
        db.commit()
        db.refresh(user)
        
    sessions = manager.list_sessions(user_id=user.id, skip=skip, limit=limit)
    return sessions

@router.post("/sessions", response_model=SessionResponse, tags=["History"])
def create_session(
    session_in: SessionCreate, 
    db: Session = Depends(get_db)
):
    check_dependencies()
    manager = HistoryManager(db)
    user = db.query(User).filter(User.username == "default_user").first()
    if not user:
        user = User(username="default_user", email="default@example.com")
        db.add(user)
        db.commit()
    
    return manager.create_session(user_id=user.id, title=session_in.title)

@router.get("/sessions/{session_id}/history", response_model=List[ChatMessageResponse], tags=["History"])
def get_session_history(
    session_id: str, 
    db: Session = Depends(get_db)
):
    check_dependencies()
    manager = HistoryManager(db)
    return manager.get_session_history(session_id)
