"""
Prompt template management routes (admin-only for writes).

These endpoints expose git-backed user templates managed by the LightRAG framework.
"""

from __future__ import annotations

from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from lightrag.base import PromptOrigin, PromptTemplate, PromptType
from lightrag.prompt_template.errors import (
    PromptTemplateNotFoundError,
    PromptTemplateValidationError,
)
from lightrag.utils import logger

from ..utils_api import get_combined_auth_dependency, require_admin_user


router = APIRouter(tags=["prompts"])


class PromptTemplateInfo(BaseModel):
    type: PromptType
    name: str
    origin: PromptOrigin
    file_path: str

    @classmethod
    def from_template(cls, t: PromptTemplate) -> "PromptTemplateInfo":
        return cls(type=t.type, name=t.name, origin=t.origin, file_path=t.file_path)


class PromptTemplateDetail(PromptTemplateInfo):
    content: str

    @classmethod
    def from_template(cls, t: PromptTemplate) -> "PromptTemplateDetail":
        return cls(
            type=t.type,
            name=t.name,
            origin=t.origin,
            file_path=t.file_path,
            content=t.content,
        )


class PromptUpdateRequest(BaseModel):
    content: str = Field(..., min_length=1, description="Full template content")
    commit_message: Optional[str] = Field(
        default=None, description="Optional git commit message"
    )


class PromptUpdateResponse(BaseModel):
    template: PromptTemplateDetail
    user_repo_head: Optional[str] = None


def create_prompt_routes(rag, api_key: Optional[str] = None):
    combined_auth = get_combined_auth_dependency(api_key)

    @router.get(
        "/prompts",
        response_model=List[PromptTemplateInfo],
        dependencies=[Depends(combined_auth)],
    )
    async def list_prompts(
        type: Optional[PromptType] = Query(default=None, description="Filter by type"),
        origin: Optional[PromptOrigin] = Query(
            default=None, description="Filter by origin (system/user)"
        ),
        resolved: bool = Query(
            default=True,
            description="If true, returns the resolved overlay view (user overrides system).",
        ),
    ):
        try:
            templates = rag.prompt_manager.list_templates(
                prompt_type=type, origin=origin, resolved=resolved
            )
            return [PromptTemplateInfo.from_template(t) for t in templates]
        except Exception as e:
            logger.error(f"Error listing prompts: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.get(
        "/prompts/{prompt_type}/{template_name}",
        response_model=PromptTemplateDetail,
        dependencies=[Depends(combined_auth)],
    )
    async def get_prompt(prompt_type: PromptType, template_name: str):
        try:
            tmpl = await rag.prompt_manager.get_template(prompt_type, template_name)
            if tmpl is None:
                raise HTTPException(status_code=404, detail="Template not found")
            return PromptTemplateDetail.from_template(tmpl)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting prompt {prompt_type}-{template_name}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @router.put(
        "/prompts/{prompt_type}/{template_name}",
        response_model=PromptUpdateResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def upsert_prompt(
        prompt_type: PromptType,
        template_name: str,
        req: PromptUpdateRequest,
        admin_info: dict = Depends(require_admin_user),
    ):
        username = admin_info.get("username", "unknown")
        try:
            tmpl = await rag.prompt_manager.upsert_user_template(
                prompt_type,
                template_name,
                content=req.content,
                author=username,
                commit_message=req.commit_message,
            )
            return PromptUpdateResponse(
                template=PromptTemplateDetail.from_template(tmpl),
                user_repo_head=rag.prompt_manager.get_user_repo_head(),
            )
        except PromptTemplateNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except PromptTemplateValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(
                f"Error updating prompt {prompt_type}-{template_name} by {username}: {e}"
            )
            raise HTTPException(status_code=500, detail=str(e))

    return router
