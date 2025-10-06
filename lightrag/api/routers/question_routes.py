from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
from uuid import uuid4
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Body, Path
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select, update, func
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import Question as QuestionModel, QuestionOptionVariant as VariantModel
from ..utils_api import get_combined_auth_dependency
from ascii_colors import trace_exception

def create_questions(
    db: Session,
    *,
    user_id: str,
    session_id: str,
    project_id: Optional[str],
    questions: List[Dict[str, Any]],
) -> List[str]:
    ids: List[str] = []
    for q in questions:
        qid = str(uuid4())
        obj = QuestionModel(
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

# ---------------------- Schemas ----------------------

class QuestionVariantOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    question_id: str
    difficulty_level: Optional[str] = None
    options: List[str] = []
    correct_answers: List[int] = []
    rationale: str
    created_at: datetime
    updated_at: datetime

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
    type: Optional[str] = None
    isApproved: bool = False
    isArchived: bool = False
    created_at: datetime
    updated_at: datetime
    variants: List[QuestionVariantOut] = []

class CreateQuestionItem(BaseModel):
    user_id: str
    session_id: str
    project_id: Optional[str] = None
    question: str = Field(alias="question_text")
    options: List[str] = []
    correct_options: List[int] = Field(default_factory=list, alias="correct_answers")
    difficulty_level: Optional[str] = None
    tags: List[str] = []
    source: Optional[str] = None
    type: Optional[str] = None

    # Accept both old and new keys seamlessly
    def to_model_kwargs(self) -> Dict[str, Any]:
        return {
            "id": str(uuid4()),
            "user_id": self.user_id,
            "session_id": self.session_id,
            "project_id": self.project_id,
            "question_text": self.question,
            "options": self.options or [],
            "correct_answers": self.correct_options or [],
            "difficulty_level": self.difficulty_level,
            "tags": self.tags or [],
            "source": self.source or "",
            "type": self.type or "",
        }

class CreateQuestionsOut(BaseModel):
    success: bool = True
    message: str = "Questions created."
    ids: List[str]

class ListOut(BaseModel):
    success: bool = True
    total: int
    page: int
    pageSize: int
    questions: List[QuestionOut]

class ItemOut(BaseModel):
    success: bool = True
    question: QuestionOut

class SuccessOut(BaseModel):
    success: bool = True
    message: str

class PatchQuestionIn(BaseModel):
    # Partial update fields (state + content)
    question_text: Optional[str] = None
    options: Optional[List[str]] = None
    correct_answers: Optional[List[int]] = None
    difficulty_level: Optional[str] = None
    tags: Optional[List[str]] = None
    source: Optional[str] = None
    type: Optional[str] = None
    isApproved: Optional[bool] = None
    isArchived: Optional[bool] = None
    project_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None  # if you want to allow reassignment

class BulkPatchIn(BaseModel):
    ids: List[str]
    patch: PatchQuestionIn

# ---------------------- Router ----------------------

def create_question_routes(api_key: Optional[str] = None) -> APIRouter:
    router = APIRouter(prefix="/questions", tags=["questions"])
    auth = get_combined_auth_dependency(api_key)

    # --- POST /v1/questions (single or bulk create)
    @router.post("/", response_model=CreateQuestionsOut, dependencies=[Depends(auth)])
    def create_questions(
        payload: Union[CreateQuestionItem, List[CreateQuestionItem]] = Body(...),
        db: Session = Depends(get_db),
    ):
        try:
            items = payload if isinstance(payload, list) else [payload]
            ids: List[str] = []
            for item in items:
                kwargs = item.to_model_kwargs()
                obj = QuestionModel(**kwargs)
                db.add(obj)
                ids.append(kwargs["id"])
            db.commit()
            return CreateQuestionsOut(ids=ids)
        except Exception as e:
            trace_exception(e)
            raise HTTPException(status_code=500, detail=f"Failed to create questions: {e}")

    # --- GET /v1/questions (list + filters)
    @router.get("/", response_model=ListOut, dependencies=[Depends(auth)])
    def list_questions(
        db: Session = Depends(get_db),
        # Filters
        user_id: Optional[str] = Query(None),
        session_id: Optional[str] = Query(None),
        project_id: Optional[str] = Query(None),
        type: Optional[str] = Query(None, description="Filter by question type ('mcq' or 'multiple_response')"),
        hasVariants: Optional[bool] = Query(None, description="Filter questions that have variants"),
        isApproved: Optional[bool] = Query(None),
        isArchived: Optional[bool] = Query(False),
        q: Optional[str] = Query(None, description="Full-text search on question_text (simple ILIKE)"),
        # Pagination & sorting
        page: int = Query(1, ge=1),
        pageSize: int = Query(50, ge=1, le=200),
        sort: str = Query("updated_at"),
        order: str = Query("desc", regex="^(asc|desc)$"),
    ):
        try:
            stmt = select(QuestionModel)
            # Filters
            where = []
            if user_id:
                where.append(QuestionModel.user_id == user_id)
            if session_id:
                where.append(QuestionModel.session_id == session_id)
            if project_id:
                where.append(QuestionModel.project_id == project_id)
            if type:
                normalised_type = type.strip().lower()
                if normalised_type not in {"mcq", "multiple_response"}:
                    raise HTTPException(status_code=400, detail="Unsupported type filter. Use 'mcq' or 'multiple_response'.")
                where.append(QuestionModel.type == normalised_type)
            if hasVariants is not None:
                variant_exists = select(VariantModel.id).where(VariantModel.question_id == QuestionModel.id).limit(1).exists()
                if hasVariants:
                    where.append(variant_exists)
                else:
                    where.append(~variant_exists)
            if isApproved is not None:
                where.append(QuestionModel.isApproved == isApproved)
            if isArchived is not None:
                where.append(QuestionModel.isArchived == isArchived)
            if q:
                where.append(QuestionModel.question_text.ilike(f"%{q}%"))

            if where:
                stmt = stmt.where(*where)

            # Count
            count_stmt = select(func.count()).select_from(stmt.subquery())
            total = db.execute(count_stmt).scalar_one()

            # Sort
            sort_col = getattr(QuestionModel, sort, QuestionModel.updated_at)
            if order.lower() == "desc":
                sort_col = sort_col.desc()
            stmt = stmt.order_by(sort_col)

            # Pagination
            offset = (page - 1) * pageSize
            stmt = stmt.offset(offset).limit(pageSize)

            # Base questions
            questions = list(db.execute(stmt).scalars())

            # Fetch variants in one shot
            id_list = [q.id for q in questions]
            variants_by_qid: dict[str, list[VariantModel]] = {}
            if id_list:
                v_stmt = select(VariantModel).where(VariantModel.question_id.in_(id_list))
                variant_rows = list(db.execute(v_stmt).scalars())
                for v in variant_rows:
                    variants_by_qid.setdefault(v.question_id, []).append(v)
            
            # Build payloads with variants
            out_items: List[QuestionOut] = []
            for qobj in questions:
                variants = [
                    QuestionVariantOut.model_validate(v)
                    for v in variants_by_qid.get(qobj.id, [])
                ]
                item = QuestionOut.model_validate(qobj)
                item.variants = variants
                out_items.append(item)

            return ListOut(
                total=total,
                page=page,
                pageSize=pageSize,
                questions=out_items,
            )
        except Exception as e:
            trace_exception(e)
            raise HTTPException(status_code=500, detail=f"Failed to list questions: {e}")

    # --- GET /v1/questions/{id} (read)
    @router.get("/{id}", response_model=ItemOut, dependencies=[Depends(auth)])
    def get_question(id: str = Path(...), db: Session = Depends(get_db)):
        try:
            obj = db.get(QuestionModel, id)
            if not obj:
                raise HTTPException(status_code=404, detail="Question not found.")

            # Fetch all variants for this question
            v_stmt = select(VariantModel).where(VariantModel.question_id == id)
            variant_rows = list(db.execute(v_stmt).scalars())
            variants = [QuestionVariantOut.model_validate(v) for v in variant_rows]

            out = QuestionOut.model_validate(obj)
            out.variants = variants
            return ItemOut(question=out)
        except HTTPException:
            raise
        except Exception as e:
            trace_exception(e)
            raise HTTPException(status_code=500, detail=f"Failed to fetch question: {e}")

    # --- PATCH /v1/questions/{id} (partial update incl. approve/archive/unapprove/unarchive)
    @router.patch("/{id}", response_model=SuccessOut, dependencies=[Depends(auth)])
    def patch_question(
        id: str,
        payload: PatchQuestionIn,
        db: Session = Depends(get_db),
    ):
        try:
            updates = {k: v for k, v in payload.model_dump(exclude_none=True).items()}
            if not updates:
                raise HTTPException(status_code=400, detail="No fields to update.")
            stmt = (
                update(QuestionModel)
                .where(QuestionModel.id == id)
                .values(**updates)
            )
            res = db.execute(stmt)
            db.commit()
            if (res.rowcount or 0) == 0:
                raise HTTPException(status_code=404, detail="Question not found or no changes.")
            return SuccessOut(message="Question updated.")
        except HTTPException:
            raise
        except Exception as e:
            trace_exception(e)
            raise HTTPException(status_code=500, detail=f"Failed to update question: {e}")

    # --- DELETE /v1/questions/{id} (soft delete → isArchived=true)
    @router.delete("/{id}", response_model=SuccessOut, dependencies=[Depends(auth)])
    def delete_question(id: str, db: Session = Depends(get_db)):
        try:
            stmt = (
                update(QuestionModel)
                .where(QuestionModel.id == id)
                .values(isArchived=True)
            )
            res = db.execute(stmt)
            db.commit()
            if (res.rowcount or 0) == 0:
                raise HTTPException(status_code=404, detail="Question not found.")
            return SuccessOut(message="Question archived.")
        except HTTPException:
            raise
        except Exception as e:
            trace_exception(e)
            raise HTTPException(status_code=500, detail=f"Failed to archive question: {e}")

    # --- PATCH /v1/questions (bulk update via { ids, patch })
    @router.patch("/", response_model=SuccessOut, dependencies=[Depends(auth)])
    def bulk_patch(
        body: BulkPatchIn,
        db: Session = Depends(get_db),
    ):
        try:
            if not body.ids:
                raise HTTPException(status_code=400, detail="ids cannot be empty.")
            patch = body.patch.model_dump(exclude_none=True)
            if not patch:
                raise HTTPException(status_code=400, detail="patch cannot be empty.")
            stmt = (
                update(QuestionModel)
                .where(QuestionModel.id.in_(body.ids))
                .values(**patch)
            )
            res = db.execute(stmt)
            db.commit()
            if (res.rowcount or 0) == 0:
                raise HTTPException(status_code=404, detail="No matching questions updated.")
            return SuccessOut(message=f"Updated {res.rowcount} question(s).")
        except HTTPException:
            raise
        except Exception as e:
            trace_exception(e)
            raise HTTPException(status_code=500, detail=f"Bulk update failed: {e}")

    return router
