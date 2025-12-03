"""
Session History Manager for LightRAG API

This module provides business logic for managing chat sessions,
messages, and citations.
"""

from sqlalchemy.orm import Session
from lightrag.api.session_models import ChatMessage, ChatSession, MessageCitation
from typing import List, Dict, Optional
import uuid


class SessionHistoryManager:
    """Manager for chat session history operations."""
    
    def __init__(self, db: Session):
        """
        Initialize session history manager.
        
        Args:
            db: SQLAlchemy database session.
        """
        self.db = db

    def get_conversation_context(
        self, 
        session_id: uuid.UUID, 
        max_tokens: int = 4000
    ) -> List[Dict[str, str]]:
        """
        Retrieve conversation history formatted for LLM context.
        
        Args:
            session_id: Session UUID to retrieve messages from.
            max_tokens: Maximum number of tokens to include.
            
        Returns:
            List of message dictionaries with 'role' and 'content' keys.
        """
        # Get latest messages first
        raw_messages = (
            self.db.query(ChatMessage)
            .filter(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.desc())
            .limit(20)  # Safe buffer
            .all()
        )
        
        context = []
        current_tokens = 0
        
        for msg in raw_messages:
            # Simple token estimation (approx 4 chars per token)
            msg_tokens = msg.token_count or len(msg.content) // 4
            if current_tokens + msg_tokens > max_tokens:
                break
            
            context.append({"role": msg.role, "content": msg.content})
            current_tokens += msg_tokens
            
        return list(reversed(context))

    def create_session(
        self, 
        user_id: str, 
        title: str = None, 
        rag_config: dict = None
    ) -> ChatSession:
        """
        Create a new chat session.
        
        Args:
            user_id: User identifier.
            title: Optional session title.
            rag_config: Optional RAG configuration dictionary.
            
        Returns:
            Created ChatSession instance.
        """
        session = ChatSession(
            user_id=user_id,
            title=title,
            rag_config=rag_config or {}
        )
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)
        return session

    def get_session(self, session_id: uuid.UUID) -> Optional[ChatSession]:
        """
        Get a session by ID.
        
        Args:
            session_id: Session UUID.
            
        Returns:
            ChatSession instance or None if not found.
        """
        return self.db.query(ChatSession).filter(ChatSession.id == session_id).first()

    def list_sessions(
        self, 
        user_id: str, 
        skip: int = 0, 
        limit: int = 100
    ) -> List[ChatSession]:
        """
        List sessions for a user.
        
        Args:
            user_id: User identifier.
            skip: Number of sessions to skip.
            limit: Maximum number of sessions to return.
            
        Returns:
            List of ChatSession instances.
        """
        return (
            self.db.query(ChatSession)
            .filter(ChatSession.user_id == user_id)
            .order_by(ChatSession.last_message_at.desc())
            .offset(skip)
            .limit(limit)
            .all()
        )

    def save_message(
        self,
        session_id: uuid.UUID,
        role: str,
        content: str,
        token_count: int = None,
        processing_time: float = None
    ) -> ChatMessage:
        """
        Save a message to a session.
        
        Args:
            session_id: Session UUID.
            role: Message role (user, assistant, system).
            content: Message content.
            token_count: Optional token count.
            processing_time: Optional processing time in seconds.
            
        Returns:
            Created ChatMessage instance.
        """
        message = ChatMessage(
            session_id=session_id,
            role=role,
            content=content,
            token_count=token_count,
            processing_time=processing_time
        )
        self.db.add(message)
        self.db.commit()
        self.db.refresh(message)
        
        # Update session last_message_at
        session = self.get_session(session_id)
        if session:
            session.last_message_at = message.created_at
            self.db.commit()
            
        return message

    def save_citations(self, message_id: uuid.UUID, citations: List[Dict]):
        """
        Save citations for a message.
        
        Args:
            message_id: Message UUID.
            citations: List of citation dictionaries.
        """
        for cit in citations:
            # Handle both list and string content
            content = cit.get("content", "")
            if isinstance(content, list):
                content = "\n".join(content)
            
            citation = MessageCitation(
                message_id=message_id,
                source_doc_id=cit.get("reference_id", cit.get("source_doc_id", "unknown")),
                file_path=cit.get("file_path", "unknown"),
                chunk_content=content,
                relevance_score=cit.get("relevance_score")
            )
            self.db.add(citation)
        self.db.commit()

    def get_session_history(self, session_id: uuid.UUID) -> List[ChatMessage]:
        """
        Get all messages for a session.
        
        Args:
            session_id: Session UUID.
            
        Returns:
            List of ChatMessage instances ordered by creation time.
        """
        return (
            self.db.query(ChatMessage)
            .filter(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.asc())
            .all()
        )
    
    def delete_session(self, session_id: uuid.UUID) -> bool:
        """
        Delete a session and all its messages.
        
        Args:
            session_id: Session UUID.
            
        Returns:
            True if session was deleted, False if not found.
        """
        session = self.get_session(session_id)
        if session:
            self.db.delete(session)
            self.db.commit()
            return True
        return False

