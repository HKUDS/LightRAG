from sqlalchemy.orm import Session
from app.models.models import ChatMessage, ChatSession, MessageCitation
from typing import List, Dict, Optional
import uuid

class HistoryManager:
    def __init__(self, db: Session):
        self.db = db

    def get_conversation_context(self, session_id: uuid.UUID, max_tokens: int = 4000) -> List[Dict]:
        """
        Retrieves conversation history formatted for LLM context, truncated to fit max_tokens.
        """
        # Get latest messages first
        raw_messages = (
            self.db.query(ChatMessage)
            .filter(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.desc())
            .limit(20) # Safe buffer
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

    def create_session(self, user_id: str, title: str = None, rag_config: dict = None) -> ChatSession:
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
        return self.db.query(ChatSession).filter(ChatSession.id == session_id).first()

    def list_sessions(self, user_id: str, skip: int = 0, limit: int = 100) -> List[ChatSession]:
        return (
            self.db.query(ChatSession)
            .filter(ChatSession.user_id == user_id)
            .order_by(ChatSession.last_message_at.desc())
            .offset(skip)
            .limit(limit)
            .all()
        )

    def save_message(self, session_id: uuid.UUID, role: str, content: str, token_count: int = None, processing_time: float = None) -> ChatMessage:
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
        for cit in citations:
            content = "\n".join(cit.get("content", []))
            citation = MessageCitation(
                message_id=message_id,
                source_doc_id=cit.get("reference_id", "unknown"),
                file_path=cit.get("file_path", "unknown"),
                chunk_content=content,
                relevance_score=cit.get("relevance_score")
            )
            self.db.add(citation)
        self.db.commit()

    def get_session_history(self, session_id: str) -> List[ChatMessage]:
        return (
            self.db.query(ChatMessage)
            .filter(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.asc())
            .all()
        )
