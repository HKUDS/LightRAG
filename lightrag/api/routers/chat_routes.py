# chat_routes.py
"""
Chat routes for the LightRAG API.
Parity with Node/Express:
- POST   /chat/               -> sendMessage
- GET    /chat/session        -> getMessagesBySession (query param: session_id)
- GET    /chat/rag/health     -> getRagHealth
- GET    /chat/rag/documents  -> getRagDocs
"""

from __future__ import annotations

import os
import json
import logging
from datetime import datetime
from typing import Any, Optional, List
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field, ConfigDict
from sqlalchemy import select
from sqlalchemy.orm import Session

from ascii_colors import trace_exception

from ..utils_api import get_combined_auth_dependency
from ..database import get_db
from ..models import ChatMessage, ChatSession, ChatRole, User, Project

# LightRAG in-process access (same pattern as query/session/workspace)
from lightrag import LightRAG
from lightrag.base import QueryParam

# ---------- LLM + prompts (embedded helpers) ----------
from dotenv import load_dotenv
load_dotenv()

import tiktoken
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage
from langchain_core.messages import SystemMessage

from ..prompts import (
    INTENT_DETECTION_PROMPT,
    QUERY_INTERPRETATION_PROMPT,
    BASIC_CHAT_STATELESS_PROMPT,
    QUESTION_GENERATION_PROMPT,
)

# LLM instances (env-driven)
llm = ChatOpenAI(
    temperature=0.7,
    model=os.getenv("OPENAI_CHAT_MODEL"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    streaming=False,
)
generation_llm = ChatOpenAI(
    model=os.getenv("OPENAI_GENERATION_MODEL"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    streaming=False,
)
query_interpretation_llm = ChatOpenAI(
    temperature=0.7,
    model=os.getenv("OPENAI_QUERY_INTERPRETATION_MODEL"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    streaming=False,
)

tiktoken_model = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(tiktoken_model.encode(text))

def load_memory(serialized_json: Optional[str]) -> ConversationBufferMemory:
    """Rebuild ConversationBufferMemory from stored JSON (if any)."""
    memory = ConversationBufferMemory(return_messages=True)
    if serialized_json:
        try:
            for m in json.loads(serialized_json):
                if m.get("type") == "human":
                    memory.chat_memory.add_user_message(HumanMessage(content=m["content"]))
                elif m.get("type") == "ai":
                    memory.chat_memory.add_ai_message(AIMessage(content=m["content"]))
                elif m.get("type") == "system":
                    memory.chat_memory.messages.append(SystemMessage(content=m["content"]))
        except Exception as e:
            logging.warning(f"Failed to load memory: {e}")
    return memory

def stringify_memory(memory: ConversationBufferMemory) -> str:
    """Plain-text transcript: Human: ..., AI: ..., System: ..."""
    lines: List[str] = []
    for m in memory.chat_memory.messages:
        if m.type == "human":
            lines.append(f"Human: {m.content}")
        elif m.type == "ai":
            lines.append(f"AI: {m.content}")
        elif m.type == "system":
            lines.append(f"System: {m.content}")
    return "\n".join(lines)

def dump_memory(memory: ConversationBufferMemory) -> str:
    return json.dumps([m.dict() for m in memory.chat_memory.messages])

# Intent detection
async def get_user_intent_controller(payload, db: Session) -> str:
    try:
        session_id = payload.session_id
        user_message = payload.user_message
        session = db.get(ChatSession, session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found.")
        memory = load_memory(session.memory_state)
        chat_history = stringify_memory(memory)

        prompt = INTENT_DETECTION_PROMPT.format(user_message=user_message, history=chat_history)
        response = await llm.ainvoke(prompt)

        # Update memory with the user message (stateful like Node)
        memory.chat_memory.add_user_message(HumanMessage(content=user_message))
        session.memory_state = dump_memory(memory)
        db.commit()

        return response.content
    except Exception as e:
        logging.error(f"Error in get_user_intent_controller: {e!r}")
        raise

# Query interpretation -> structured fields
class _QIResponse(BaseModel):
    question_count: str
    question: str
    command: str
    difficulty_level: str

async def get_query_interpretation_controller(payload, db: Session) -> _QIResponse:
    try:
        session_id = payload.session_id
        user_message = payload.user_message

        session = db.get(ChatSession, session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found.")
        memory = load_memory(session.memory_state)
        chat_history = stringify_memory(memory)

        prompt = QUERY_INTERPRETATION_PROMPT.format(user_message=user_message, history=chat_history)
        structured_llm = query_interpretation_llm.with_structured_output(_QIResponse)
        response = await structured_llm.ainvoke(prompt)
        return response
    except Exception as e:
        logging.error(f"Error in get_query_interpretation_controller: {e!r}")
        raise

# Create questions (uses generation model)
class _Question(BaseModel):
    question: str
    options: List[str]
    correct_options: List[int]
    difficulty_level: str
    tags: List[str]
    source: str

class _GQResponse(BaseModel):
    questions: List[_Question]
    message: str

async def create_questions_controller(payload, db: Session) -> _GQResponse:
    try:
        session_id = payload.session_id
        session = db.get(ChatSession, session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found.")
        memory = load_memory(session.memory_state)
        chat_history = stringify_memory(memory)

        prompt = QUESTION_GENERATION_PROMPT.format(
            question_count=payload.question_count,
            history=chat_history,
            difficulty_level=payload.difficulty_level,
            search_result=payload.search_result,
            command=payload.command,
            instructions=payload.instructions,
        )
        structured_llm = generation_llm.with_structured_output(_GQResponse)
        response = await structured_llm.ainvoke(prompt)

        # Store assistant response summary in memory (optional)
        memory.chat_memory.add_ai_message(AIMessage(content=str(response)))
        session.memory_state = dump_memory(memory)
        db.commit()

        return response
    except Exception as e:
        logging.error(f"Error in create_questions_controller: {e!r}")
        raise

# Basic chat fallback
async def get_chat_controller(payload, db: Session) -> str:
    try:
        session_id = payload.session_id
        user_message = payload.user_message

        session = db.get(ChatSession, session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found.")
        memory = load_memory(session.memory_state)
        chat_history = stringify_memory(memory)

        prompt = BASIC_CHAT_STATELESS_PROMPT.format(user_message=user_message, history=chat_history)
        response = await llm.ainvoke(prompt)

        # Update memory with AI reply
        memory.chat_memory.add_ai_message(AIMessage(content=response.content))
        session.memory_state = dump_memory(memory)
        db.commit()

        return response.content
    except Exception as e:
        logging.error(f"Error in get_chat_controller: {e!r}")
        raise

# ---------- RAG integration helpers ----------
async def _get_rag(request: Request, user_id: str, workspace_id: str) -> LightRAG:
    """Obtain/create LightRAG instance via app.state.instance_manager."""
    try:
        manager = request.app.state.instance_manager
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Instance manager not available: {e}")
    try:
        rag, _ = await manager.get_instance(user_id, workspace_id)
        return rag
    except Exception as e:
        trace_exception(e)
        raise HTTPException(status_code=500, detail=f"Failed to initialize LightRAG instance: {e}")

def _serialize_output(system_output: Any) -> str:
    if system_output is None or system_output == "":
        return ""
    try:
        return json.dumps(system_output, ensure_ascii=False)
    except Exception:
        return str(system_output)

# Optional HTTP client for external RAG service health/docs
try:
    import httpx
except ImportError:
    httpx = None

router = APIRouter(prefix="/chat", tags=["chat"])

# ---------- API Schemas ----------
class SendMessageRequest(BaseModel):
    user_id: str
    session_id: str
    project_id: str
    role: str = Field(description="Expected 'user' for inbound messages")
    user_message: str

class SendMessageResponse(BaseModel):
    success: bool = True
    message: str
    user_message_id: str
    system_message_id: str
    ai_message: str
    ai_questions: Any
    user_created_at: datetime
    system_created_at: datetime

class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    session_id: str
    role: ChatRole
    content: str
    output: Optional[str] = None
    created_at: datetime

class MessagesResponse(BaseModel):
    success: bool = True
    messages: List[MessageOut]

class HealthOk(BaseModel):
    status: str = "ok"
    data: Any | None = None

class HealthErr(BaseModel):
    status: str = "error"
    message: str

# ---------- Router factory ----------
def create_chat_routes(api_key: Optional[str] = None) -> APIRouter:
    combined_auth = get_combined_auth_dependency(api_key)

    # GET /chat/session?session_id=...
    @router.get(
        "/session",
        response_model=MessagesResponse,
        dependencies=[Depends(combined_auth)],
        summary="Get all messages for a session (ascending by created_at)",
    )
    async def get_messages_by_session(
        session_id: str = Query(..., description="Chat session id"),
        db: Session = Depends(get_db),
    ):
        stmt = (
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.asc())
        )
        rows = db.execute(stmt).scalars().all()
        return MessagesResponse(
            success=True,
            messages=[MessageOut.model_validate(r) for r in rows],
        )

    # POST /chat/
    @router.post(
        "/",
        response_model=SendMessageResponse,
        dependencies=[Depends(combined_auth)],
        summary="Send a user message; run intent pipeline; persist assistant reply",
    )
    async def send_message(
        request: Request,
        payload: SendMessageRequest,
        db: Session = Depends(get_db),
    ):
        # Validate references
        if db.get(User, payload.user_id) is None:
            raise HTTPException(status_code=400, detail="Invalid user_id: user not found.")
        sess = db.get(ChatSession, payload.session_id)
        if sess is None:
            raise HTTPException(status_code=400, detail="Invalid session_id: session not found.")
        if db.get(Project, payload.project_id) is None:
            raise HTTPException(status_code=400, detail="Invalid project_id: project not found.")

        # Insert the user's message
        user_message_id = str(uuid4())
        user_created_at = datetime.utcnow()
        try:
            db.add(
                ChatMessage(
                    id=user_message_id,
                    session_id=payload.session_id,
                    role=ChatRole.user,  # normalize to enum
                    content=payload.user_message,
                    created_at=user_created_at,
                )
            )
            db.commit()
        except Exception as e:
            db.rollback()
            trace_exception(e)
            raise HTTPException(status_code=500, detail="Failed to insert user message.")

        # 1) Intent detection
        try:
            uid_res_json = await get_user_intent_controller(
                type("UIDPayload", (), {"user_message": payload.user_message, "session_id": payload.session_id}),
                db,
            )
            try:
                uid = json.loads(uid_res_json)
            except Exception:
                uid = {"intent": "basic_chat", "reasoning": str(uid_res_json)}
            intent = (uid.get("intent") or "").strip()
            reasoning = uid.get("reasoning")
            logging.info(f"[INTENT] intent={intent} reasoning={reasoning}")
        except Exception as e:
            trace_exception(e)
            raise HTTPException(status_code=500, detail=f"Intent detection failed: {e}")

        system_message: str = ""
        system_output: Any = ""

        # 2) Branch on intent
        try:
            if intent == "search_graph":
                qi = await get_query_interpretation_controller(
                    type("QIPayload", (), {"user_message": payload.user_message, "session_id": payload.session_id}),
                    db,
                )
                question = qi.question

                # RAG global search
                rag = await _get_rag(request, user_id=payload.user_id, workspace_id=payload.project_id)
                qparam = QueryParam(
                    mode="global",
                    top_k=10,
                    max_token_for_text_unit=4000,
                    max_token_for_global_context=4000,
                    max_token_for_local_context=4000,
                    stream=False,
                )
                rag_res = await rag.aquery(question, param=qparam)
                if isinstance(rag_res, str):
                    system_message = rag_res
                elif isinstance(rag_res, dict):
                    system_message = json.dumps(rag_res, indent=2, ensure_ascii=False)
                else:
                    system_message = str(rag_res)

            elif intent == "generate_questions":
                qi = await get_query_interpretation_controller(
                    type("QIPayload", (), {"user_message": payload.user_message, "session_id": payload.session_id}),
                    db,
                )
                question = qi.question
                question_count = qi.question_count
                command = qi.command
                difficulty_level = qi.difficulty_level

                # RAG context-only retrieval
                rag = await _get_rag(request, user_id=payload.user_id, workspace_id=payload.project_id)
                qparam = QueryParam(
                    mode="mix",
                    only_need_context=True,
                    top_k=10,
                    max_token_for_text_unit=4000,
                    max_token_for_global_context=4000,
                    max_token_for_local_context=4000,
                    stream=False,
                )
                rag_res = await rag.aquery(question, param=qparam)

                cq_payload = type(
                    "CQPayload",
                    (),
                    {
                        "session_id": payload.session_id,
                        "question_count": str(question_count),
                        "difficulty_level": str(difficulty_level),
                        "search_result": rag_res if isinstance(rag_res, str) else json.dumps(rag_res, ensure_ascii=False),
                        "command": str(command),
                        "instructions": "",
                    },
                )
                gq = await create_questions_controller(cq_payload, db)
                system_message = gq.message
                system_output = gq.questions

            else:
                # Fallback to basic chat
                bc_payload = type("ChatPayload", (), {"session_id": payload.session_id, "user_message": payload.user_message})
                system_message = await get_chat_controller(bc_payload, db)

        except HTTPException:
            raise
        except Exception as e:
            trace_exception(e)
            raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

        # 3) Persist assistant message
        system_message_id = str(uuid4())
        system_created_at = datetime.utcnow()
        try:
            db.add(
                ChatMessage(
                    id=system_message_id,
                    session_id=payload.session_id,
                    role=ChatRole.assistant,
                    content=system_message,
                    output=_serialize_output(system_output),
                    created_at=system_created_at,
                )
            )
            db.commit()
        except Exception as e:
            db.rollback()
            trace_exception(e)
            raise HTTPException(status_code=500, detail="Failed to insert assistant message.")

        return SendMessageResponse(
            success=True,
            message="Functionality Ran Successfully.",
            user_message_id=user_message_id,
            system_message_id=system_message_id,
            ai_message=system_message,
            ai_questions=system_output or "",
            user_created_at=user_created_at,
            system_created_at=system_created_at,
        )

    return router