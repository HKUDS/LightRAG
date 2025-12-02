from fastapi import APIRouter, Depends, HTTPException, Security, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from typing import List, Optional
from uuid import UUID
import sys
import os
from lightrag.api.auth import auth_handler

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

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login", auto_error=False)

def check_dependencies():
    if not HistoryManager:
        raise HTTPException(status_code=503, detail="History service not available")

async def get_current_user(
    token: str = Security(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    check_dependencies()
    
    if not token:
        # If no token provided, try to use default user if configured or allowed
        # For now, we'll return the default user for backward compatibility if needed,
        # but ideally we should require auth.
        # Let's check if we have a default user
        user = db.query(User).filter(User.username == "default_user").first()
        if not user:
             user = User(username="default_user", email="default@example.com")
             db.add(user)
             db.commit()
             db.refresh(user)
        return user

    try:
        user_data = auth_handler.validate_token(token)
        username = user_data["username"]
        
        user = db.query(User).filter(User.username == username).first()
        if not user:
            # Create user if not exists (auto-registration on first login)
            # In a real app you might want to fetch email from token metadata or require explicit registration
            user = User(username=username, email=f"{username}@example.com") 
            db.add(user)
            db.commit()
            db.refresh(user)
            
        return user
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

@router.get("/sessions", response_model=List[SessionResponse], tags=["History"])
def list_sessions(
    skip: int = 0, 
    limit: int = 20, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    check_dependencies()
    manager = HistoryManager(db)
    sessions = manager.list_sessions(user_id=current_user.id, skip=skip, limit=limit)
    return sessions

@router.post("/sessions", response_model=SessionResponse, tags=["History"])
def create_session(
    session_in: SessionCreate, 
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    check_dependencies()
    manager = HistoryManager(db)
    return manager.create_session(user_id=current_user.id, title=session_in.title)

@router.get("/sessions/{session_id}/history", response_model=List[ChatMessageResponse], tags=["History"])
def get_session_history(
    session_id: str, 
    db: Session = Depends(get_db)
):
    check_dependencies()
    manager = HistoryManager(db)
    return manager.get_session_history(session_id)
