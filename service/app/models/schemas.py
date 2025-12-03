from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime

class SessionCreate(BaseModel):
    title: Optional[str] = None
    rag_config: Optional[Dict[str, Any]] = {}

class SessionResponse(BaseModel):
    id: UUID
    title: Optional[str]
    created_at: datetime
    last_message_at: Optional[datetime]

    class Config:
        from_attributes = True

class ChatMessageRequest(BaseModel):
    session_id: UUID
    content: str
    mode: Optional[str] = "hybrid"
    stream: Optional[bool] = False

class Citation(BaseModel):
    source_doc_id: str
    file_path: str
    chunk_content: Optional[str]
    relevance_score: Optional[float]

    class Config:
        from_attributes = True

class ChatMessageResponse(BaseModel):
    id: UUID
    content: str
    role: str
    created_at: datetime
    citations: List[Citation] = []

    class Config:
        from_attributes = True
