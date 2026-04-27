from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class LittleBullArea(BaseModel):
    id: str
    label: str
    slug: str
    description: str
    privacy: str
    document_count: int = 0
    ready_count: int = 0
    processing_count: int = 0
    accent: str = "#2563EB"
    emoji: str = "📁"


class LittleBullDocument(BaseModel):
    id: str
    file_path: str
    title: str
    status: str
    content_summary: str = ""
    content_length: int = 0
    updated_at: str | None = None
    created_at: str | None = None
    track_id: str | None = None
    chunks_count: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class LittleBullDocumentsResponse(BaseModel):
    documents: list[LittleBullDocument]
    total_count: int
    status_counts: dict[str, int] = Field(default_factory=dict)


class LittleBullQueryRequest(BaseModel):
    workspace_id: str
    query: str = Field(min_length=3)
    mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"] = "mix"
    response_type: str = "Multiple Paragraphs"
    top_k: int | None = None
    include_references: bool = True
    include_chunk_content: bool = False
    conversation_history: list[dict[str, Any]] = Field(default_factory=list)
    confidentiality: Literal["normal", "sensivel", "privado"] = "normal"
    model_profile: str = "equilibrado"


class LittleBullQueryResponse(BaseModel):
    response: str
    references: list[dict[str, Any]] = Field(default_factory=list)
    workspace_id: str
    model_profile: str


class LittleBullUploadResponse(BaseModel):
    status: str
    message: str
    track_id: str | None = None
    workspace_id: str


class LittleBullActivityItem(BaseModel):
    id: str
    action: str
    result: str
    created_at: str
    actor_user_id: str
    workspace_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class LittleBullAssistant(BaseModel):
    id: str
    name: str
    description: str
    enabled: bool = True
    response_rules: list[str] = Field(default_factory=list)

