# session_routes.py
"""
Chat Session routes for the LightRAG API.
Converted from the Node/Express session routes and aligned with workspace_routes.py style.
Service helpers (title + memory summarization) are embedded directly in this file.
"""

from __future__ import annotations

import os
import json
import logging
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, Body, Depends, HTTPException, Path
from pydantic import BaseModel, Field, ConfigDict
from sqlalchemy import select, update, delete
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ascii_colors import trace_exception

from ..utils_api import get_combined_auth_dependency
from ..database import get_db
from ..models import User, Project, ChatSession

from dotenv import load_dotenv
load_dotenv()

import tiktoken
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage
from langchain_core.messages import SystemMessage

from ..prompts import (
    MEMORY_SUMMARIZATION_PROMPT,
    SESSION_TOPIC_GENERATION_PROMPT
)

# LLM instances (use your env vars)
summarizer_llm = ChatOpenAI(
    temperature=0.0,
    model=os.getenv("OPENAI_TOPIC_SUMMARY_MODEL"),
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


async def summarize_memory_controller_new(
    memory: ConversationBufferMemory, max_tokens: int = 2000
) -> bool:
    """Token-budget summarizer: compress older messages into a single SystemMessage."""
    human_ai = [m for m in memory.chat_memory.messages if m.type in ("human", "ai")]
    system_msgs = [m for m in memory.chat_memory.messages if m.type == "system"]

    total = sum(count_tokens(m.content) for m in human_ai)
    if total <= max_tokens:
        return False

    # Keep ~last 500 tokens of conversation intact
    latest_msgs: List[Any] = []
    cur = 0
    for m in reversed(human_ai):
        t = count_tokens(m.content)
        if cur + t > 500:
            break
        latest_msgs.insert(0, m)
        cur += t

    to_summarize = human_ai[: -len(latest_msgs)] if latest_msgs else human_ai

    prior_summary = "\n".join(m.content for m in system_msgs)
    message_block = "\n".join(f"{m.type.title()}: {m.content}" for m in to_summarize)

    prompt = MEMORY_SUMMARIZATION_PROMPT.format(
        prior_summary=prior_summary.strip(),
        message_block=message_block.strip(),
    )
    resp = await summarizer_llm.ainvoke(prompt)

    updated_summary = SystemMessage(content=resp.content)
    memory.chat_memory.messages = [updated_summary] + latest_msgs
    return True


async def generate_title_controller(session_id: str, db: Session) -> str:
    """Generate a concise session title from its conversation memory."""
    session = db.get(ChatSession, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")

    memory = load_memory(session.memory_state)
    chat_history = stringify_memory(memory)

    prompt = SESSION_TOPIC_GENERATION_PROMPT.format(chats=chat_history)
    resp = await summarizer_llm.ainvoke(prompt)
    return resp.content


router = APIRouter(
    prefix="/session",
    tags=["session"]
)


class SessionCreateRequest(BaseModel):
    user_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)


class SessionUpdateRequest(BaseModel):
    topic: Optional[str] = Field(default=None, description="New topic for the session.")


class SessionIdBody(BaseModel):
    session_id: str = Field(min_length=1)


class SessionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    topic: Optional[str] = None
    user_id: str
    project_id: str
    memory_state: Optional[Dict[str, Any]] = None
    created_at: Any
    last_active_at: Any


class ApiSuccess(BaseModel):
    success: bool = True
    message: Optional[str] = None


class SessionCreateResponse(ApiSuccess):
    session_id: str


class SessionSingleResponse(BaseModel):
    success: bool = True
    session: SessionOut


class SessionListResponse(BaseModel):
    success: bool = True
    sessions: List[SessionOut]


class TopicResponse(BaseModel):
    success: bool = True
    topic: str
    session_id: str


class SummaryResponse(BaseModel):
    success: bool = True
    message: str
    updated: bool
    session_id: str


def create_session_routes(api_key: Optional[str] = None) -> APIRouter:
    combined_auth = get_combined_auth_dependency(api_key)

    # POST "/"
    @router.post(
        "/",
        response_model=SessionCreateResponse,
        dependencies=[Depends(combined_auth)],
        summary="Create a new chat session",
    )
    async def create_new(payload: SessionCreateRequest, db: Session = Depends(get_db)):
        # Validate refs (like Node did implicitly)
        if db.get(User, payload.user_id) is None:
            raise HTTPException(status_code=400, detail="Invalid user_id: user not found.")
        if db.get(Project, payload.project_id) is None:
            raise HTTPException(status_code=400, detail="Invalid project_id: project not found.")

        sid = str(uuid4())
        session = ChatSession(
            id=sid,
            topic="Untitled",
            user_id=payload.user_id,
            project_id=payload.project_id,
        )
        try:
            db.add(session)
            db.commit()
        except Exception as e:
            db.rollback()
            trace_exception(e)
            raise HTTPException(status_code=500, detail="Failed to create new chat session.")

        return SessionCreateResponse(
            success=True,
            message="New Chat Session is created successfully.",
            session_id=sid,
        )

    # GET "/"
    @router.get(
        "/",
        response_model=SessionListResponse,
        dependencies=[Depends(combined_auth)],
        summary="Get all sessions (descending by created_at)",
    )
    async def get_all_sessions(db: Session = Depends(get_db)):
        stmt = select(ChatSession).order_by(ChatSession.created_at.desc())
        rows = db.execute(stmt).scalars().all()
        return SessionListResponse(success=True, sessions=[SessionOut.model_validate(r) for r in rows])

    # GET "/user/{user_id}/project/{project_id}"
    @router.get(
        "/user/{user_id}/project/{project_id}",
        response_model=SessionListResponse,
        dependencies=[Depends(combined_auth)],
        summary="Get sessions by user and project (limit 30)",
    )
    async def get_sessions_by_project_id(
        user_id: str = Path(...),
        project_id: str = Path(...),
        db: Session = Depends(get_db),
    ):
        stmt = (
            select(ChatSession)
            .where(ChatSession.user_id == user_id, ChatSession.project_id == project_id)
            .order_by(ChatSession.created_at.desc())
            .limit(30)
        )
        rows = db.execute(stmt).scalars().all()
        return SessionListResponse(success=True, sessions=[SessionOut.model_validate(r) for r in rows])

    # GET "/{id}"
    @router.get(
        "/{id}",
        response_model=SessionSingleResponse,
        dependencies=[Depends(combined_auth)],
        summary="Get a session by id",
    )
    async def get_session_by_id(id: str = Path(...), db: Session = Depends(get_db)):
        row = db.get(ChatSession, id)
        if row is None:
            raise HTTPException(status_code=404, detail="Session not found.")
        return SessionSingleResponse(success=True, session=SessionOut.model_validate(row))

    # PUT "/{id}"
    @router.put(
        "/{id}",
        response_model=ApiSuccess,
        dependencies=[Depends(combined_auth)],
        summary="Update a session (topic only)",
    )
    async def update_session(
        id: str = Path(...),
        payload: SessionUpdateRequest = Body(...),
        db: Session = Depends(get_db),
    ):
        if payload.topic is None:
            raise HTTPException(status_code=400, detail="Provide `topic` to update.")

        stmt = update(ChatSession).where(ChatSession.id == id).values(topic=payload.topic)
        try:
            res = db.execute(stmt)
            db.commit()
            if res.rowcount == 0:
                raise HTTPException(status_code=404, detail="Session not found.")
        except HTTPException:
            raise
        except Exception as e:
            db.rollback()
            trace_exception(e)
            raise HTTPException(status_code=500, detail="Failed to update session.")
        return ApiSuccess(message="Session updated successfully.")

    # DELETE "/{id}"
    @router.delete(
        "/{id}",
        response_model=ApiSuccess,
        dependencies=[Depends(combined_auth)],
        summary="Delete a session",
    )
    async def delete_session(id: str = Path(...), db: Session = Depends(get_db)):
        try:
            stmt = delete(ChatSession).where(ChatSession.id == id)
            res = db.execute(stmt)
            db.commit()
            if res.rowcount == 0:
                raise HTTPException(status_code=404, detail="Session not found.")
        except HTTPException:
            raise
        except IntegrityError as e:
            db.rollback()
            trace_exception(e)
            raise HTTPException(status_code=409, detail="Cannot delete session due to existing references.")
        except Exception as e:
            db.rollback()
            trace_exception(e)
            raise HTTPException(status_code=500, detail="Failed to delete session.")
        return ApiSuccess(message="Session deleted successfully.")

    # POST "/title"
    @router.post(
        "/title",
        response_model=TopicResponse,
        dependencies=[Depends(combined_auth)],
        summary="Generate a concise session title and persist it",
    )
    async def generate_session_topic(payload: SessionIdBody, db: Session = Depends(get_db)):
        # Validate
        sess = db.get(ChatSession, payload.session_id)
        if sess is None:
            raise HTTPException(status_code=404, detail="Session not found.")

        try:
            topic = (await generate_title_controller(payload.session_id, db)).strip()
            if not topic:
                raise RuntimeError("Empty title produced by title generator.")
            stmt = update(ChatSession).where(ChatSession.id == payload.session_id).values(topic=topic)
            res = db.execute(stmt)
            db.commit()
            if res.rowcount == 0:
                raise HTTPException(status_code=404, detail="Session not found during update.")
            return TopicResponse(success=True, topic=topic, session_id=payload.session_id)
        except Exception as e:
            db.rollback()
            trace_exception(e)
            raise HTTPException(status_code=500, detail=f"Failed to generate topic. {e}")

    # POST "/summarize"
    @router.post(
        "/summarize",
        response_model=SummaryResponse,
        dependencies=[Depends(combined_auth)],
        summary="Summarize/refresh session memory (compress long histories)",
    )
    async def summarize_memory(payload: SessionIdBody, db: Session = Depends(get_db)):
        sess = db.get(ChatSession, payload.session_id)
        if sess is None:
            raise HTTPException(status_code=404, detail="Session not found.")

        try:
            memory = load_memory(sess.memory_state)
            updated = await summarize_memory_controller_new(memory, max_tokens=2000)
            # Persist regardless (safe to rewrite same state)
            sess.memory_state = dump_memory(memory)
            db.commit()
            message = "Memory summarized." if updated else "No summarization needed."
            return SummaryResponse(success=True, message=message, updated=updated, session_id=payload.session_id)
        except Exception as e:
            db.rollback()
            trace_exception(e)
            raise HTTPException(status_code=500, detail=f"Failed to summarize memory. {e}")

    return router