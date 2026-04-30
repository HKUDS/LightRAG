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
    data_plane_attached: bool = False
    chat_model_id: str | None = None
    embedding_model_id: str | None = None
    embedding_reindex_required: bool = False


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
    agent_id: str | None = None


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


class LittleBullReindexArchivedResponse(BaseModel):
    status: str
    message: str
    track_id: str | None = None
    workspace_id: str
    recovered_count: int = 0
    skipped_count: int = 0
    files: list[str] = Field(default_factory=list)


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


class LittleBullModelSetting(BaseModel):
    model_setting_id: str | None = None
    tenant_id: str | None = None
    workspace_id: str | None = None
    usage: Literal["chat", "embedding", "rerank", "agent"] = "chat"
    provider: str = "openrouter"
    binding: str = "openai"
    binding_host: str = ""
    model_id: str
    display_name: str
    enabled: bool = True
    is_default: bool = False
    config: dict[str, Any] = Field(default_factory=dict)
    created_by: str | None = None
    updated_by: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class LittleBullEmbeddingCatalogItem(BaseModel):
    model_id: str
    display_name: str
    provider: str = "openrouter"
    binding: str = "openai"
    binding_host: str
    context_length: int
    prompt_cost_per_million_tokens: float
    prompt_cost_per_token: float
    estimated_cost_100k_tokens: float
    estimated_cost_200k_tokens: float
    quality_tier: str
    recommended_chunk_tokens: int
    notes: str


class LittleBullKnowledgeBase(BaseModel):
    workspace_id: str
    tenant_id: str | None = None
    name: str
    slug: str
    description: str = ""
    privacy: str = "team"
    data_plane_attached: bool
    document_count: int = 0
    ready_count: int = 0
    processing_count: int = 0
    chat_model: LittleBullModelSetting | None = None
    embedding_model: LittleBullModelSetting | None = None
    embedding_reindex_required: bool = False
    embedding_estimated_tokens: int = 0
    embedding_estimated_cost_usd: float = 0


class LittleBullKnowledgeBaseUpsertRequest(BaseModel):
    workspace_id: str | None = None
    name: str = Field(min_length=1)
    slug: str | None = None
    description: str = ""
    privacy: str = "team"
    embedding_model_id: str | None = None
    estimated_tokens: int | None = Field(default=None, ge=0)


class LittleBullKnowledgeBaseAttachResponse(BaseModel):
    status: str
    message: str
    workspace_id: str
    data_plane_attached: bool
    input_dir: str | None = None
    working_dir: str | None = None


class LittleBullKnowledgeBaseReindexRequest(BaseModel):
    approval_id: str | None = None
    include_archived: bool = True
    include_input_root: bool = True
    destructive_rebuild: bool = False


class LittleBullKnowledgeBaseReindexResponse(BaseModel):
    status: str
    message: str
    workspace_id: str
    track_id: str | None = None
    approval: dict[str, Any] | None = None
    destructive_rebuild: bool = False
    snapshot_id: str | None = None
    snapshot_path: str | None = None
    rollback_available: bool = False
    queued_count: int = 0
    skipped_count: int = 0
    files: list[str] = Field(default_factory=list)


class LittleBullKnowledgeBaseRollbackRequest(BaseModel):
    snapshot_id: str = Field(min_length=1)


class LittleBullKnowledgeBaseRollbackResponse(BaseModel):
    status: str
    message: str
    workspace_id: str
    snapshot_id: str
    restored_path: str | None = None
    preserved_current_snapshot_id: str | None = None
    preserved_current_snapshot_path: str | None = None


class LittleBullEmbeddingCostEstimateRequest(BaseModel):
    workspace_id: str
    model_id: str
    estimated_tokens: int | None = Field(default=None, ge=0)
    page_count: int | None = Field(default=None, ge=0)
    words_per_page: int = Field(default=400, ge=1, le=5000)


class LittleBullEmbeddingCostEstimateResponse(BaseModel):
    workspace_id: str
    model_id: str
    display_name: str
    estimated_tokens: int
    estimated_cost_usd: float
    prompt_cost_per_million_tokens: float
    context_length: int
    recommended_chunk_tokens: int
    reindex_required: bool = True
    notes: list[str] = Field(default_factory=list)


class LittleBullAgentConfig(BaseModel):
    agent_id: str | None = None
    tenant_id: str | None = None
    workspace_id: str | None = None
    name: str
    description: str = ""
    enabled: bool = True
    model_setting_id: str | None = None
    system_prompt: str = ""
    response_rules: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    config: dict[str, Any] = Field(default_factory=dict)
    created_by: str | None = None
    updated_by: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class LittleBullAgentStudioIssue(BaseModel):
    severity: Literal["error", "warning"]
    field: str
    message: str


class LittleBullAgentStudioPreviewRequest(BaseModel):
    workspace_id: str
    agent: LittleBullAgentConfig
    test_input: str = ""


class LittleBullAgentStudioPreviewResponse(BaseModel):
    agent: LittleBullAgentConfig
    issues: list[LittleBullAgentStudioIssue] = Field(default_factory=list)
    readiness_score: int = 0
    ready_to_publish: bool = False
    compiled_prompt: str
    test_input: str = ""
    test_summary: str = ""


class LittleBullConversationMessage(BaseModel):
    message_id: str | None = None
    id: str | None = None
    role: Literal["user", "assistant", "system"]
    content: str
    references: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None


class LittleBullConversationSaveRequest(BaseModel):
    conversation_id: str | None = None
    workspace_id: str
    title: str = ""
    agent_id: str | None = None
    model_profile: str = "equilibrado"
    confidentiality: Literal["normal", "sensivel", "privado"] = "normal"
    messages: list[LittleBullConversationMessage] = Field(default_factory=list)


class LittleBullConversation(BaseModel):
    conversation_id: str
    tenant_id: str | None = None
    workspace_id: str
    user_id: str
    title: str
    agent_id: str | None = None
    model_profile: str
    confidentiality: str
    message_count: int = 0
    messages: list[LittleBullConversationMessage] = Field(default_factory=list)
    created_at: str | None = None
    updated_at: str | None = None


class LittleBullCorrelationSuggestionRequest(BaseModel):
    workspace_id: str
    source_label: str = Field(min_length=1)
    target_label: str = Field(min_length=1)
    reason: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class LittleBullCorrelationSuggestion(BaseModel):
    suggestion_id: str
    tenant_id: str | None = None
    workspace_id: str
    user_id: str
    source_label: str
    target_label: str
    reason: str
    status: Literal["pending", "approved", "rejected"]
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None
    decided_at: str | None = None
    decided_by: str | None = None
