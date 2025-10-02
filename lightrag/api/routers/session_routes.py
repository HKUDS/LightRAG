from __future__ import annotations

import os
import json
import logging
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from fastapi import APIRouter, Body, Depends, HTTPException, Path, Query
from pydantic import BaseModel, Field, ConfigDict, field_validator
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

# ---------- LLM + token counting ----------
summarizer_llm = ChatOpenAI(
    temperature=0.0,
    model=os.getenv("OPENAI_TOPIC_SUMMARY_MODEL"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    streaming=False,
)

tiktoken_model = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(tiktoken_model.encode(text))

# ---------- Memory helpers ----------
def load_memory(serialized: Optional[Union[str, List[dict[str, Any]]]]) -> ConversationBufferMemory:
    mem = ConversationBufferMemory(return_messages=True)
    if not serialized:
        return mem
    try:
        items = serialized if isinstance(serialized, list) else json.loads(serialized)
        for m in items:
            if m.get("type") == "human":
                mem.chat_memory.add_user_message(HumanMessage(content=m["content"]))
            elif m.get("type") == "ai":
                mem.chat_memory.add_ai_message(AIMessage(content=m["content"]))
            elif m.get("type") == "system":
                mem.chat_memory.messages.append(SystemMessage(content=m["content"]))
    except Exception as e:
        logging.warning(f"Failed to load memory: {e}")
    return mem

def stringify_memory(memory: ConversationBufferMemory) -> str:
    # Plain-text transcript: Human: ..., AI: ..., System: ...
    lines: List[str] = []
    for m in memory.chat_memory.messages:
        if m.type == "human":
            lines.append(f"Human: {m.content}")
        elif m.type == "ai":
            lines.append(f"AI: {m.content}")
        elif m.type == "system":
            lines.append(f"System: {m.content}")
    return "\n".join(lines)

def dump_memory(memory: ConversationBufferMemory) -> List[dict]:
    # Return a Python list; SQLAlchemy JSON column can store directly
    return [m.dict() for m in memory.chat_memory.messages]

async def summarize_memory_controller_new(
    memory: ConversationBufferMemory, max_tokens: int = 2000
) -> bool:
    """Token-budget summarizer: compress older messages into one SystemMessage."""
    human_ai = [m for m in memory.chat_memory.messages if m.type in ("human", "ai")]
    system_msgs = [m for m in memory.chat_memory.messages if m.type == "system"]

    total = sum(count_tokens(m.content) for m in human_ai)
    if total <= max_tokens:
        return False

    # Keep ~last 500 tokens intact
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

# ---------- Router ----------
router = APIRouter(
    prefix="/sessions",
    tags=["sessions"],
)

# ---------- Schemas ----------
class SessionCreateRequest(BaseModel):
    user_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    topic: Optional[str] = Field(default="Untitled", description="Optional initial topic/title.")

class SessionPatchRequest(BaseModel):
    # Partial field updates
    topic: Optional[str] = Field(default=None, description="New topic/title for the session.")

    # Server-computed actions (flags)
    generate_title: Optional[bool] = Field(default=False, description="If true, generate & set a concise title (stored in `topic`).")
    refresh_memory: Optional[bool] = Field(default=False, description="If true, summarize long history and persist.")

class SessionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    topic: Optional[str] = None
    user_id: str
    project_id: str
    memory_state: Optional[List[dict[str, Any]]] = None
    created_at: Any
    last_active_at: Any

    @field_validator("memory_state", mode="before")
    @classmethod
    def _coerce_memory(cls, v):
        if v is None or isinstance(v, list):
            return v
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                return parsed if isinstance(parsed, list) else []
            except Exception:
                return []
        if isinstance(v, dict):
            return [v]
        return v

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

# ---------- Routes ----------

def create_session_routes(api_key: Optional[str] = None) -> APIRouter:
    combined_auth = get_combined_auth_dependency(api_key)

    # POST /sessions
    @router.post(
        "",
        response_model=SessionCreateResponse,
        dependencies=[Depends(combined_auth)],
        summary="Create a new chat session",
    )
    async def create_new(payload: SessionCreateRequest, db: Session = Depends(get_db)):
        # Validate refs
        if db.get(User, payload.user_id) is None:
            raise HTTPException(status_code=400, detail="Invalid user_id: user not found.")
        if db.get(Project, payload.project_id) is None:
            raise HTTPException(status_code=400, detail="Invalid project_id: project not found.")

        sid = str(uuid4())
        session = ChatSession(
            id=sid,
            topic=payload.topic or "Untitled",
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
            message="New chat session created.",
            session_id=sid,
        )

    # GET /sessions  (filters + paging + sort)
    @router.get(
        "",
        response_model=SessionListResponse,
        dependencies=[Depends(combined_auth)],
        summary="List sessions (filter by user/project; sort; paging)",
    )
    async def list_sessions(
        user_id: Optional[str] = Query(None, description="Filter by user_id"),
        project_id: Optional[str] = Query(None, description="Filter by project_id"),
        limit: int = Query(30, ge=1, le=100),
        offset: int = Query(0, ge=0),
        sort: str = Query("-created_at", description="Sort by 'created_at' or '-created_at' (default)."),
        db: Session = Depends(get_db),
    ):
        # Validate sort
        sort_col = ChatSession.created_at
        if sort not in ("created_at", "-created_at"):
            raise HTTPException(status_code=400, detail="Unsupported sort. Use 'created_at' or '-created_at'.")
        order_by = sort_col.asc() if sort == "created_at" else sort_col.desc()

        stmt = select(ChatSession)
        if user_id:
            stmt = stmt.where(ChatSession.user_id == user_id)
        if project_id:
            stmt = stmt.where(ChatSession.project_id == project_id)

        stmt = stmt.order_by(order_by).offset(offset).limit(limit)
        rows = db.execute(stmt).scalars().all()
        return SessionListResponse(success=True, sessions=[SessionOut.model_validate(r) for r in rows])

    # GET /sessions/{id}
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

    # PATCH /sessions/{id}
    @router.patch(
        "/{id}",
        response_model=SessionSingleResponse,
        dependencies=[Depends(combined_auth)],
        summary="Partially update a session (topic or flags: generate_title, refresh_memory)",
    )
    async def patch_session(
        id: str = Path(...),
        payload: SessionPatchRequest = Body(...),
        db: Session = Depends(get_db),
    ):
        sess = db.get(ChatSession, id)
        if sess is None:
            raise HTTPException(status_code=404, detail="Session not found.")

        try:
            # 1) Field update: topic
            if payload.topic is not None:
                sess.topic = payload.topic

            # 2) Action: generate title (stored in `topic`)
            if payload.generate_title:
                generated = (await generate_title_controller(id, db)).strip()
                if not generated:
                    raise RuntimeError("Empty title produced by generator.")
                sess.topic = generated  # using `topic` as the canonical title field

            # 3) Action: refresh/summarize memory
            if payload.refresh_memory:
                memory = load_memory(sess.memory_state)
                updated = await summarize_memory_controller_new(memory, max_tokens=2000)
                # Persist regardless (safe overwrite)
                sess.memory_state = dump_memory(memory)

            db.commit()
            db.refresh(sess)
            return SessionSingleResponse(success=True, session=SessionOut.model_validate(sess))

        except Exception as e:
            db.rollback()
            trace_exception(e)
            raise HTTPException(status_code=500, detail=f"Failed to update session. {e}")

    # DELETE /sessions/{id}
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

    return router
