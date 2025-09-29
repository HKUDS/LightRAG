from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import uuid4
from sqlalchemy import select, update
from sqlalchemy.orm import Session
from ..models import Question, Project

import os
import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Body
from pydantic import BaseModel, Field, ConfigDict
from sqlalchemy.orm import Session

from ascii_colors import trace_exception

from ..utils_api import get_combined_auth_dependency
from ..database import get_db
from ..models import Question as QuestionModel, ChatMessage, ChatRole


# -------- createQuestions ----------
def create_questions(
    db: Session,
    *,
    user_id: str,
    session_id: str,
    project_id: Optional[str],
    questions: List[Dict[str, Any]],   # expects keys like: question, options, correct_options, difficulty_level, tags, source
) -> List[str]:
    ids: List[str] = []
    for q in questions:
        qid = str(uuid4())
        obj = Question(
            id=qid,
            user_id=user_id,
            session_id=session_id,
            project_id=project_id,
            question_text=q.get("question", ""),
            options=q.get("options", []) or [],
            correct_answers=q.get("correct_options", []) or [],
            difficulty_level=q.get("difficulty_level") or None,
            tags=q.get("tags", []) or [],
            source=q.get("source") or "",
            type=q.get("type") or "",
        )
        db.add(obj)
        ids.append(qid)
    db.commit()
    return ids

# -------- getQuestionById ----------
def get_question_by_id(db: Session, *, id: str) -> Optional[Question]:
    return db.get(Question, id)

# -------- getQuestionsByUserId ----------
# Matches your JOIN to include project name; returns list of dicts for convenience.
def get_questions_by_user_id(db: Session, *, user_id: str) -> List[Dict[str, Any]]:
    stmt = (
        select(Question, Project.name.label("project_name"))
        .join(Project, Project.id == Question.project_id, isouter=True)
        .where(Question.user_id == user_id, Question.isArchived == False)  # noqa: E712
        .order_by(Question.updated_at.desc())
    )
    rows = db.execute(stmt).all()
    out: List[Dict[str, Any]] = []
    for q, project_name in rows:
        out.append({
            "id": q.id,
            "user_id": q.user_id,
            "session_id": q.session_id,
            "project_id": q.project_id,
            "project_name": project_name,
            "question_text": q.question_text,
            "options": q.options,
            "correct_answers": q.correct_answers,
            "difficulty_level": q.difficulty_level,
            "tags": q.tags,
            "source": q.source,
            "isApproved": q.isApproved,
            "isArchived": q.isArchived,
            "created_at": q.created_at,
            "updated_at": q.updated_at,
        })
    return out

# -------- getQuestionsBySessionId ----------
def get_questions_by_session_id(db: Session, *, session_id: str) -> List[Question]:
    stmt = (
        select(Question)
        .where(Question.session_id == session_id, Question.isArchived == False)  # noqa: E712
        .order_by(Question.updated_at.desc())
    )
    return list(db.execute(stmt).scalars())

# -------- getQuestionsByGraphId (project_id) ----------
def get_questions_by_graph_id(db: Session, *, project_id: str) -> List[Question]:
    stmt = (
        select(Question)
        .where(Question.project_id == project_id, Question.isArchived == False)  # noqa: E712
        .order_by(Question.updated_at.desc())
    )
    return list(db.execute(stmt).scalars())

# -------- updateQuestion ----------
def update_question(db: Session, *, id: str, updates: Dict[str, Any]) -> int:
    """
    Allowed updatable fields mirror the JS:
      question_text, options, correct_answers, difficulty_level, tags, isApproved
    Automatically bumps updated_at via column onupdate.
    Returns number of rows updated (0 or 1).
    """
    allowed = {
        "question_text": "question_text",
        "options": "options",
        "correct_answers": "correct_answers",
        "difficulty_level": "difficulty_level",
        "tags": "tags",
        "isApproved": "isApproved",
    }
    values: Dict[str, Any] = {}
    for key, col in allowed.items():
        if key in updates:
            values[col] = updates[key]

    if not values:
        raise ValueError("No valid fields to update.")

    stmt = (
        update(Question)
        .where(Question.id == id, Question.isArchived == False)  # noqa: E712
        .values(**values)
    )
    res = db.execute(stmt)
    db.commit()
    return res.rowcount or 0

# -------- archiveQuestion ----------
def archive_question(db: Session, *, id: str) -> int:
    stmt = (
        update(Question)
        .where(Question.id == id)
        .values(isArchived=True)
    )
    res = db.execute(stmt)
    db.commit()
    return res.rowcount or 0

# -------- approveQuestion ----------
def approve_question(db: Session, *, id: str) -> int:
    stmt = (
        update(Question)
        .where(Question.id == id, Question.isArchived == False)  # noqa: E712
        .values(isApproved=True)
    )
    res = db.execute(stmt)
    db.commit()
    return res.rowcount or 0

# -------- undoApprovalQuestion ----------
def undo_approval_question(db: Session, *, id: str) -> int:
    stmt = (
        update(Question)
        .where(Question.id == id, Question.isArchived == False)  # noqa: E712
        .values(isApproved=False)
    )
    res = db.execute(stmt)
    db.commit()
    return res.rowcount or 0

router = APIRouter(
    prefix="/question",
    tags=["questions"]
)

class QuestionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    project_id: Optional[str] = None
    question_text: str
    options: List[str] = []
    correct_answers: List[int] = []
    difficulty_level: Optional[str] = None
    tags: List[str] = []
    source: Optional[str] = None
    isApproved: bool = False
    isArchived: bool = False
    created_at: datetime
    updated_at: datetime

class CreateQuestionsIn(BaseModel):
    user_id: str
    session_id: str
    project_id: Optional[str] = Field(default=None, description="Preferred key")
    graph_id: Optional[str] = Field(default=None, description="Alias of project_id (for parity)")
    questions: List[Dict[str, Any]]  # items with keys: question, options, correct_options, difficulty_level, tags, source

class CreateQuestionsOut(BaseModel):
    success: bool = True
    message: str = "Questions created."
    ids: List[str]

class SuccessOut(BaseModel):
    success: bool = True
    message: str

class ItemOut(BaseModel):
    success: bool = True
    question: QuestionOut

class ListOut(BaseModel):
    success: bool = True
    questions: List[QuestionOut]

class UpdateQuestionIn(BaseModel):
    # Allowed fields
    question_text: Optional[str] = None
    options: Optional[List[str]] = None
    correct_answers: Optional[List[int]] = None
    difficulty_level: Optional[str] = None
    tags: Optional[List[str]] = None
    isApproved: Optional[bool] = None

class TweakQuestionIn(BaseModel):
    session_id: str
    user_message: str

class TweakQuestionOut(BaseModel):
    success: bool = True
    message: str = "Functionality Ran Successfully."
    user_message_id: str
    system_message_id: str
    ai_message: str
    ai_questions: Dict[str, Any]
    user_created_at: datetime
    system_created_at: datetime

class ExportQuestionItem(BaseModel):
    # This mirrors your DB shape; we’ll remap before calling ML service
    question_text: str
    options: List[str]
    correct_answers: List[int]

class ExportQuestionsIn(BaseModel):
    filename: str
    fileformat: str
    questions: List[ExportQuestionItem]

class ExportQuestionsOut(BaseModel):
    success: bool = True
    message: str = "Question Exported successfully."
    export: Any

def _serialize_output(system_output: Any) -> str:
    if system_output is None or system_output == "":
        return ""
    try:
        return json.dumps(system_output, ensure_ascii=False)
    except Exception:
        return str(system_output)


def _to_question_out(q: QuestionModel) -> QuestionOut:
    # Ensure correct list types and naming parity
    return QuestionOut.model_validate(q)

def create_question_routes(api_key: Optional[str] = None) -> APIRouter:
    """
    Factory to create the /questions router with the auth dependency injected.
    """
    combined_auth = get_combined_auth_dependency(api_key)

        # POST /questions  -> createQuestions
    @router.post(
        "/",
        response_model=CreateQuestionsOut,
        dependencies=[Depends(combined_auth)],
        summary="Create questions (bulk).",
    )
    def create_questions(
        payload: CreateQuestionsIn,
        db: Session = Depends(get_db),
    ):
        try:
            project_id = payload.project_id or payload.graph_id
            ids = create_questions(
                db,
                user_id=payload.user_id,
                session_id=payload.session_id,
                project_id=project_id,
                questions=payload.questions,
            )
            return CreateQuestionsOut(success=True, message="Questions created.", ids=ids)
        except Exception as e:
            trace_exception(e)
            raise HTTPException(status_code=500, detail=f"Failed to create questions: {e}")
        
    # GET /questions/user/{user_id} -> getQuestionsByUserId
    @router.get(
        "/user/{user_id}",
        response_model=ListOut,
        dependencies=[Depends(combined_auth)],
        summary="Get questions by user id",
    )
    def get_by_user(
        user_id: str = Path(...),
        db: Session = Depends(get_db),
    ):
        try:
            rows = get_questions_by_user_id(db, user_id=user_id)
            # rows already dicts (service returns joined data); coerce to QuestionOut
            items = [QuestionOut(**row) for row in rows]
            return ListOut(success=True, questions=items)
        except Exception as e:
            trace_exception(e)
            raise HTTPException(status_code=500, detail=f"Failed to fetch questions: {e}")
        
    # GET /questions/session/{session_id} -> getQuestionsBySessionId
    @router.get(
        "/session/{session_id}",
        response_model=ListOut,
        dependencies=[Depends(combined_auth)],
        summary="Get questions by session id",
    )
    def get_by_session(
        session_id: str = Path(...),
        db: Session = Depends(get_db),
    ):
        try:
            rows = get_questions_by_session_id(db, session_id=session_id)
            items = [_to_question_out(q) for q in rows]
            return ListOut(success=True, questions=items)
        except Exception as e:
            trace_exception(e)
            raise HTTPException(status_code=500, detail=f"Failed to fetch questions: {e}")

    # GET /questions/project/{project_id} -> getQuestionsByProjectId
    @router.get(
        "/project/{project_id}",
        response_model=ListOut,
        dependencies=[Depends(combined_auth)],
        summary="Get questions by project (graph) id",
    )
    def get_by_project(
        project_id: str = Path(...),
        db: Session = Depends(get_db),
    ):
        try:
            rows = get_questions_by_graph_id(db, project_id=project_id)
            items = [_to_question_out(q) for q in rows]
            return ListOut(success=True, questions=items)
        except Exception as e:
            trace_exception(e)
            raise HTTPException(status_code=500, detail=f"Failed to fetch questions: {e}")

    # IMPORTANT: define specific routes before the generic "/{id}"

    # PUT /questions/{id} -> updateQuestion
    @router.put(
        "/{id}",
        response_model=SuccessOut,
        dependencies=[Depends(combined_auth)],
        summary="Update a question",
    )
    def update_question(
        id: str,
        payload: UpdateQuestionIn,
        db: Session = Depends(get_db),
    ):
        try:
            count = update_question(db, id=id, updates=payload.model_dump(exclude_none=True))
            if count == 0:
                raise HTTPException(status_code=404, detail="Question not found or no changes.")
            return SuccessOut(success=True, message="Question updated.")
        except HTTPException:
            raise
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            trace_exception(e)
            raise HTTPException(status_code=500, detail=f"Failed to update question: {e}")

    # POST /questions/archive/{id} -> archiveQuestion
    @router.post(
        "/archive/{id}",
        response_model=SuccessOut,
        dependencies=[Depends(combined_auth)],
        summary="Archive a question",
    )
    def archive(
        id: str,
        db: Session = Depends(get_db),
    ):
        try:
            count = archive_question(db, id=id)
            if count == 0:
                raise HTTPException(status_code=404, detail="Question not found.")
            return SuccessOut(success=True, message="Question archived.")
        except HTTPException:
            raise
        except Exception as e:
            trace_exception(e)
            raise HTTPException(status_code=500, detail=f"Failed to archive question: {e}")

    # POST /questions/approve/{id} -> approveQuestion
    @router.post(
        "/approve/{id}",
        response_model=SuccessOut,
        dependencies=[Depends(combined_auth)],
        summary="Approve a question",
    )
    def approve(
        id: str,
        payload: Dict[str, Any] = Body(default_factory=dict),  # parity with Node (session_id in body, but not used here)
        db: Session = Depends(get_db),
    ):
        try:
            count = approve_question(db, id=id)
            if count == 0:
                raise HTTPException(status_code=404, detail="Question not found.")
            return SuccessOut(success=True, message="Question approved successfully.")
        except HTTPException:
            raise
        except Exception as e:
            trace_exception(e)
            raise HTTPException(status_code=500, detail=f"Failed to approve question: {e}")

    # POST /questions/undoApproval/{id} -> undoApprovalQuestion
    @router.post(
        "/undoApproval/{id}",
        response_model=SuccessOut,
        dependencies=[Depends(combined_auth)],
        summary="Undo approval on a question",
    )
    def undo_approval(
        id: str,
        db: Session = Depends(get_db),
    ):
        try:
            count = undo_approval_question(db, id=id)
            if count == 0:
                raise HTTPException(status_code=404, detail="Question not found.")
            return SuccessOut(success=True, message="Approval reverted successfully.")
        except HTTPException:
            raise
        except Exception as e:
            trace_exception(e)
            raise HTTPException(status_code=500, detail=f"Failed to revert approval: {e}")

    @router.get(
        "/{id}",
        response_model=ItemOut,
        dependencies=[Depends(combined_auth)],
        summary="Get question by id",
    )
    def get_by_id(
        id: str,
        db: Session = Depends(get_db),
    ):
        try:
            q = get_question_by_id(db, id=id)
            if not q:
                raise HTTPException(status_code=404, detail="Question not found.")
            return ItemOut(success=True, question=_to_question_out(q))
        except HTTPException:
            raise
        except Exception as e:
            trace_exception(e)
            raise HTTPException(status_code=500, detail=f"Failed to fetch question: {e}")

    return router