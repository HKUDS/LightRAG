"""
Session History Pydantic Schemas for LightRAG API

This module provides Pydantic schemas for request/response validation
of session history endpoints.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime


class SessionCreate(BaseModel):
    """Schema for creating a new chat session."""
    
    title: Optional[str] = Field(None, description="Optional title for the session")
    rag_config: Optional[Dict[str, Any]] = Field(default_factory=dict, description="RAG configuration for this session")


class SessionResponse(BaseModel):
    """Schema for chat session response."""
    
    id: UUID
    title: Optional[str]
    created_at: datetime
    last_message_at: Optional[datetime]

    class Config:
        from_attributes = True


class ChatMessageRequest(BaseModel):
    """Schema for sending a chat message."""
    
    session_id: UUID = Field(..., description="Session ID to add message to")
    content: str = Field(..., description="Message content")
    mode: Optional[str] = Field("hybrid", description="Query mode: local, global, hybrid, naive, mix")
    stream: Optional[bool] = Field(False, description="Enable streaming response")


class Citation(BaseModel):
    """Schema for message citation."""
    
    source_doc_id: str
    file_path: str
    chunk_content: Optional[str] = None
    relevance_score: Optional[float] = None

    class Config:
        from_attributes = True


class ChatMessageResponse(BaseModel):
    """Schema for chat message response."""
    
    id: UUID
    content: str
    role: str
    created_at: datetime
    citations: List[Citation] = Field(default_factory=list)

    class Config:
        from_attributes = True

