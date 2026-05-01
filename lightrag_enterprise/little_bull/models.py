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
    group_id: str | None = None
    subgroup_id: str | None = None
    registry_document_id: str | None = None
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
    top_k: int | None = Field(default=None, ge=1)
    group_id: str | None = None
    subgroup_id: str | None = None
    document_ids: list[str] = Field(default_factory=list)
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


class LittleBullContextEstimateRequest(BaseModel):
    workspace_id: str
    query: str = Field(min_length=3)
    conversation_history: list[dict[str, Any]] = Field(default_factory=list)
    agent_id: str | None = None
    group_id: str | None = None
    subgroup_id: str | None = None
    document_ids: list[str] = Field(default_factory=list)
    model_profile: str = "equilibrado"
    mode: Literal["local", "global", "hybrid", "naive", "mix", "bypass"] = "mix"
    top_k: int | None = Field(default=None, ge=1)
    reserved_response_tokens: int | None = Field(default=None, ge=0)


class LittleBullContextEstimateResponse(BaseModel):
    workspace_id: str
    agent_id: str | None = None
    group_id: str | None = None
    subgroup_id: str | None = None
    model_setting_id: str | None = None
    model_id: str | None = None
    context_window_tokens: int
    query_tokens: int
    history_tokens: int
    agent_prompt_tokens: int
    document_tokens: int
    chunk_tokens: int
    reserved_response_tokens: int
    total_estimated_tokens: int
    available_context_tokens: int
    overflow: bool
    overflow_tokens: int = 0
    document_count: int = 0
    chunk_count: int = 0
    retrieval_chunk_limit: int = 0
    notes: list[str] = Field(default_factory=list)


class LittleBullUploadResponse(BaseModel):
    status: str
    message: str
    track_id: str | None = None
    workspace_id: str
    group_id: str | None = None
    subgroup_id: str | None = None
    registry_document_id: str | None = None


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
    usage: Literal["chat", "embedding", "rerank", "agent", "agent_builder"] = "chat"
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


class LittleBullKnowledgeGroupRequest(BaseModel):
    group_id: str | None = None
    name: str = Field(min_length=1)
    slug: str | None = None
    description: str = ""
    privacy: str = "team"
    color: str = "#2563EB"
    metadata: dict[str, Any] = Field(default_factory=dict)


class LittleBullKnowledgeSubgroupRequest(BaseModel):
    subgroup_id: str | None = None
    group_id: str
    name: str = Field(min_length=1)
    slug: str | None = None
    description: str = ""
    privacy: str = "team"
    metadata: dict[str, Any] = Field(default_factory=dict)


class LittleBullMarkdownNoteRequest(BaseModel):
    note_id: str | None = None
    title: str = Field(min_length=1)
    slug: str | None = None
    group_id: str
    subgroup_id: str
    markdown: str = Field(min_length=1)
    privacy: str = "team"
    source_document_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class LittleBullBacklinkRequest(BaseModel):
    source_kind: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    target_kind: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    link_text: str = ""
    origin_type: str = "manual"
    graph_edge_origin_id: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class LittleBullSourceProvenanceRequest(BaseModel):
    source_kind: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    document_id: str | None = None
    note_id: str | None = None
    chunk_id: str = ""
    model_id: str = ""
    agent_id: str | None = None
    usage_ledger_id: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    locator: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class LittleBullCanvasBoardRequest(BaseModel):
    canvas_board_id: str | None = None
    title: str = Field(min_length=1)
    slug: str | None = None
    group_id: str
    subgroup_id: str
    layout: dict[str, Any] = Field(default_factory=dict)
    status: str = "active"


class LittleBullCanvasNodeRequest(BaseModel):
    canvas_node_id: str | None = None
    node_kind: str = Field(min_length=1)
    ref_kind: str = ""
    ref_id: str = ""
    x: float = 0
    y: float = 0
    width: float = Field(default=280, gt=0)
    height: float = Field(default=160, gt=0)
    content: dict[str, Any] = Field(default_factory=dict)


class LittleBullCanvasEdgeRequest(BaseModel):
    canvas_edge_id: str | None = None
    source_node_id: str
    target_node_id: str
    edge_kind: str = "manual"
    label: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class LittleBullContentMapRequest(BaseModel):
    content_map_id: str | None = None
    title: str = Field(min_length=1)
    slug: str | None = None
    group_id: str
    subgroup_id: str
    root_note_id: str | None = None
    description: str = ""
    map_body: dict[str, Any] = Field(default_factory=dict)
    status: str = "draft"


class LittleBullKnowledgeTrailRequest(BaseModel):
    knowledge_trail_id: str | None = None
    title: str = Field(min_length=1)
    slug: str | None = None
    group_id: str
    subgroup_id: str
    trail_type: str = "study"
    description: str = ""
    status: str = "draft"
    metadata: dict[str, Any] = Field(default_factory=dict)


class LittleBullKnowledgeTrailStepRequest(BaseModel):
    knowledge_trail_step_id: str | None = None
    step_order: int = Field(ge=0)
    title: str = Field(min_length=1)
    step_kind: str = "note"
    note_id: str | None = None
    document_id: str | None = None
    canvas_board_id: str | None = None
    instructions: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class LittleBullInboxItemRequest(BaseModel):
    inbox_item_id: str | None = None
    group_id: str | None = None
    subgroup_id: str | None = None
    item_kind: str = Field(min_length=1)
    title: str = Field(min_length=1)
    body: str = ""
    source_kind: str = ""
    source_id: str = ""
    status: str = "open"
    priority: str = "normal"
    metadata: dict[str, Any] = Field(default_factory=dict)


class LittleBullInboxItemStatusRequest(BaseModel):
    status: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class LittleBullCuratorSuggestionRequest(BaseModel):
    workspace_id: str
    suggestion_kind: Literal["backlink", "content_map", "subgroup", "conversation_note", "canvas_dossier"]
    title: str = ""
    body: str = ""
    group_id: str | None = None
    subgroup_id: str | None = None
    source_kind: str = ""
    source_id: str = ""
    target_kind: str = ""
    target_id: str = ""
    priority: str = "normal"
    metadata: dict[str, Any] = Field(default_factory=dict)


class LittleBullCuratorSuggestionResponse(BaseModel):
    inbox_item: dict[str, Any]
    requires_approval: bool = True
    allowed_actions: list[str] = Field(default_factory=lambda: ["review", "approve", "reject"])


class LegalMatterExtractionPayload(BaseModel):
    processos: list[dict[str, Any]] = Field(default_factory=list)
    partes: list[dict[str, Any]] = Field(default_factory=list)
    advogados: list[dict[str, Any]] = Field(default_factory=list)
    juizo: dict[str, Any] = Field(default_factory=dict)
    tribunal: dict[str, Any] = Field(default_factory=dict)
    magistrados: list[dict[str, Any]] = Field(default_factory=list)
    testemunhas: list[dict[str, Any]] = Field(default_factory=list)
    causa_de_pedir: list[dict[str, Any]] = Field(default_factory=list)
    pedidos: list[dict[str, Any]] = Field(default_factory=list)
    valores: list[dict[str, Any]] = Field(default_factory=list)
    decisoes: list[dict[str, Any]] = Field(default_factory=list)
    sentencas: list[dict[str, Any]] = Field(default_factory=list)
    acordaos: list[dict[str, Any]] = Field(default_factory=list)
    liquidacoes: list[dict[str, Any]] = Field(default_factory=list)
    prazos: list[dict[str, Any]] = Field(default_factory=list)
    jurimetria: dict[str, Any] = Field(default_factory=dict)


class LittleBullLegalMatterExtractionRequest(BaseModel):
    workspace_id: str
    group_id: str
    subgroup_id: str
    document_id: str
    matter_reference: str = ""
    extraction_model_id: str = ""
    schema_version: str = "legal-matter/v1"
    extracted_payload: LegalMatterExtractionPayload = Field(default_factory=LegalMatterExtractionPayload)
    source_refs: list[dict[str, Any]] = Field(min_length=1)
    confidence: float | None = Field(default=None, ge=0, le=1)


class LittleBullLegalMatterReviewRequest(BaseModel):
    review_status: Literal["approved", "rejected", "needs_changes"]
    error_message: str = ""


class LittleBullLegalMatterExtractionResponse(BaseModel):
    run: dict[str, Any]
    requires_human_review: bool = True
    schema_contract: dict[str, Any] = Field(default_factory=dict)


class LittleBullDossierExportRequest(BaseModel):
    format: Literal["txt", "md", "docx", "xlsx"] = "md"
    destination: Literal["internal", "external"] = "internal"
    approval_id: str | None = None
    include_audit: bool = True


class LittleBullDailyNoteRequest(BaseModel):
    daily_note_id: str | None = None
    note_date: str | None = None
    group_id: str
    subgroup_id: str
    summary: str = ""
    decisions: list[dict[str, Any]] = Field(default_factory=list)
    pending_items: list[dict[str, Any]] = Field(default_factory=list)
    cost_snapshot: dict[str, Any] = Field(default_factory=dict)


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


class LittleBullAgentBuilderSessionRequest(BaseModel):
    agent_builder_session_id: str | None = None
    model_setting_id: str | None = None
    current_step: str = "intake"
    user_message: str = Field(min_length=1)
    generated_config: dict[str, Any] = Field(default_factory=dict)


class LittleBullAgentBuilderPublishRequest(BaseModel):
    approved: bool = False
    enabled: bool = False


class LittleBullAgentContextBudgetRequest(BaseModel):
    agent_context_budget_id: str | None = None
    agent_id: str
    model_setting_id: str | None = None
    max_context_tokens: int = Field(default=0, ge=0)
    reserved_response_tokens: int = Field(default=0, ge=0)
    max_prompt_tokens: int = Field(default=0, ge=0)
    daily_cost_limit_usd: float | None = Field(default=None, ge=0)
    monthly_cost_limit_usd: float | None = Field(default=None, ge=0)
    policy: dict[str, Any] = Field(default_factory=dict)


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
    scope_snapshot: dict[str, Any] = Field(default_factory=dict)
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
    scope_snapshot: dict[str, Any] = Field(default_factory=dict)
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


class LittleBullOperationalChatRequest(LittleBullQueryRequest):
    conversation_id: str | None = None
    title: str = ""
    save_conversation: bool = True
    transform_to: Literal["none", "note", "suggestion"] = "none"
    note_title: str | None = None
    note_slug: str | None = None
    suggestion_target_label: str | None = None


class LittleBullOperationalChatResponse(BaseModel):
    response: str
    sources: list[dict[str, Any]] = Field(default_factory=list)
    workspace_id: str
    model_profile: str
    context: dict[str, Any] = Field(default_factory=dict)
    cost_estimate: dict[str, Any] = Field(default_factory=dict)
    conversation: LittleBullConversation | None = None
    note: dict[str, Any] | None = None
    suggestion: LittleBullCorrelationSuggestion | None = None


class ScopedContract(BaseModel):
    tenant_id: str
    workspace_id: str
    created_by: str
    updated_by: str
    created_at: str | None = None
    updated_at: str | None = None


class ProviderCredential(BaseModel):
    provider_credential_id: str | None = None
    tenant_id: str
    workspace_id: str | None = None
    provider: str = "openrouter"
    label: str
    credential_kind: str = "api_key"
    secret_ref: str = Field(min_length=1)
    secret_fingerprint: str = ""
    status: str = "active"
    scopes: list[str] = Field(default_factory=list)
    config_public: dict[str, Any] = Field(default_factory=dict)
    last_validated_at: str | None = None
    expires_at: str | None = None
    created_by: str
    updated_by: str
    created_at: str | None = None
    updated_at: str | None = None


class ModelCatalogSnapshot(BaseModel):
    model_catalog_snapshot_id: str | None = None
    tenant_id: str
    workspace_id: str | None = None
    provider_credential_id: str | None = None
    provider: str = "openrouter"
    source: str
    catalog_hash: str
    model_count: int = Field(default=0, ge=0)
    catalog: list[dict[str, Any]] = Field(default_factory=list)
    privacy_metadata: dict[str, Any] = Field(default_factory=dict)
    synced_at: str | None = None
    created_by: str
    updated_by: str
    created_at: str | None = None
    updated_at: str | None = None


class KnowledgeGroup(ScopedContract):
    group_id: str | None = None
    slug: str
    name: str
    description: str = ""
    privacy: str = "team"
    color: str = "#2563EB"
    metadata: dict[str, Any] = Field(default_factory=dict)


class KnowledgeSubgroup(ScopedContract):
    subgroup_id: str | None = None
    group_id: str
    slug: str
    name: str
    description: str = ""
    privacy: str = "team"
    metadata: dict[str, Any] = Field(default_factory=dict)


class EmbeddingIndexVersion(ScopedContract):
    embedding_version_id: str | None = None
    group_id: str | None = None
    subgroup_id: str | None = None
    model_setting_id: str | None = None
    provider: str
    model_id: str
    dimensions: int | None = Field(default=None, ge=1)
    chunking_policy: dict[str, Any] = Field(default_factory=dict)
    embedding_config_hash: str
    status: str = "draft"
    is_active: bool = False
    reindex_required: bool = True


class DocumentRegistry(ScopedContract):
    document_id: str | None = None
    group_id: str | None = None
    subgroup_id: str | None = None
    embedding_version_id: str | None = None
    title: str
    source_uri: str = ""
    source_kind: str = "upload"
    mime_type: str = ""
    content_hash: str = ""
    confidentiality: str = "normal"
    status: str = "registered"
    chunk_count: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class NoteRegistry(ScopedContract):
    note_id: str | None = None
    group_id: str | None = None
    subgroup_id: str | None = None
    title: str
    slug: str
    note_type: str = "markdown"
    privacy: str = "team"
    status: str = "active"
    metadata: dict[str, Any] = Field(default_factory=dict)


class IndexingJob(ScopedContract):
    indexing_job_id: str | None = None
    group_id: str | None = None
    subgroup_id: str | None = None
    document_id: str | None = None
    note_id: str | None = None
    embedding_version_id: str | None = None
    job_type: str = "index"
    status: str = "queued"
    progress: dict[str, Any] = Field(default_factory=dict)
    error_message: str = ""
    started_at: str | None = None
    completed_at: str | None = None


class LlmUsageLedger(ScopedContract):
    usage_ledger_id: str | None = None
    group_id: str | None = None
    subgroup_id: str | None = None
    user_id: str | None = None
    agent_id: str | None = None
    conversation_id: str | None = None
    model_setting_id: str | None = None
    provider: str
    model_id: str
    operation: str
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    estimated_cost_usd: float = Field(default=0, ge=0)
    actual_cost_usd: float | None = Field(default=None, ge=0)
    currency: str = "USD"
    request_hash: str
    response_hash: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    previous_ledger_hash: str = ""
    ledger_hash: str


class LittleBullCostPeriodSummary(BaseModel):
    name: str
    since: str | None = None
    request_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0
    actual_cost_usd: float = 0
    cost_usd: float = 0


class LittleBullCostBreakdownItem(BaseModel):
    key: str
    label: str = ""
    request_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0
    actual_cost_usd: float = 0
    cost_usd: float = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class LittleBullCostSummaryResponse(BaseModel):
    workspace_id: str
    currency: str = "USD"
    generated_at: str
    filters: dict[str, Any] = Field(default_factory=dict)
    periods: dict[str, LittleBullCostPeriodSummary] = Field(default_factory=dict)
    by_user: list[LittleBullCostBreakdownItem] = Field(default_factory=list)
    by_agent: list[LittleBullCostBreakdownItem] = Field(default_factory=list)
    by_model: list[LittleBullCostBreakdownItem] = Field(default_factory=list)
    by_group_subgroup: list[LittleBullCostBreakdownItem] = Field(default_factory=list)
    by_operation: list[LittleBullCostBreakdownItem] = Field(default_factory=list)


class GraphEdgeOrigin(ScopedContract):
    graph_edge_origin_id: str | None = None
    group_id: str | None = None
    subgroup_id: str | None = None
    source_node_id: str
    target_node_id: str
    edge_type: str
    origin_type: str
    origin_ref_id: str = ""
    confidence: float | None = Field(default=None, ge=0, le=1)
    provenance: dict[str, Any] = Field(default_factory=dict)
    status: str = "active"


class GraphCluster(ScopedContract):
    graph_cluster_id: str | None = None
    group_id: str | None = None
    subgroup_id: str | None = None
    label: str
    algorithm: str = ""
    node_count: int = Field(default=0, ge=0)
    edge_count: int = Field(default=0, ge=0)
    summary: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class KnowledgeTrail(ScopedContract):
    knowledge_trail_id: str | None = None
    group_id: str | None = None
    subgroup_id: str | None = None
    title: str
    slug: str
    trail_type: str = "study"
    description: str = ""
    status: str = "draft"
    metadata: dict[str, Any] = Field(default_factory=dict)


class KnowledgeTrailStep(ScopedContract):
    knowledge_trail_step_id: str | None = None
    knowledge_trail_id: str
    step_order: int = Field(ge=0)
    title: str
    step_kind: str = "note"
    note_id: str | None = None
    document_id: str | None = None
    canvas_board_id: str | None = None
    instructions: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class Backlink(ScopedContract):
    backlink_id: str | None = None
    source_kind: str
    source_id: str
    target_kind: str
    target_id: str
    link_text: str = ""
    origin_type: str = "manual"
    graph_edge_origin_id: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphChatSession(ScopedContract):
    graph_chat_session_id: str | None = None
    conversation_id: str | None = None
    focus_node_id: str = ""
    graph_scope: str = "workspace"
    context_snapshot: dict[str, Any] = Field(default_factory=dict)
    cost_estimate: dict[str, Any] = Field(default_factory=dict)
    status: str = "active"


class LittleBullGraphNode(BaseModel):
    node_id: str
    kind: str
    ref_id: str
    label: str
    group_id: str | None = None
    subgroup_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class LittleBullGraphEdge(BaseModel):
    edge_id: str
    source_node_id: str
    target_node_id: str
    origin_type: str
    edge_type: str = "relates"
    confidence: float | None = Field(default=None, ge=0, le=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class LittleBullGraphClusterSummary(BaseModel):
    cluster_id: str
    node_ids: list[str] = Field(default_factory=list)
    node_count: int = 0
    edge_count: int = 0
    label: str = ""


class LittleBullObsidianGraphResponse(BaseModel):
    workspace_id: str
    scope: Literal["global", "workspace", "group", "subgroup"] = "workspace"
    central_node_id: str | None = None
    filters: dict[str, Any] = Field(default_factory=dict)
    nodes: list[LittleBullGraphNode] = Field(default_factory=list)
    edges: list[LittleBullGraphEdge] = Field(default_factory=list)
    clusters: list[LittleBullGraphClusterSummary] = Field(default_factory=list)
    trails: list[dict[str, Any]] = Field(default_factory=list)
    chat_context: dict[str, Any] = Field(default_factory=dict)


class AgentBuilderSession(ScopedContract):
    agent_builder_session_id: str | None = None
    user_id: str
    agent_id: str | None = None
    model_setting_id: str | None = None
    status: str = "draft"
    current_step: str = "intake"
    builder_transcript: list[dict[str, Any]] = Field(default_factory=list)
    generated_config: dict[str, Any] = Field(default_factory=dict)
    readiness_score: int = Field(default=0, ge=0, le=100)
    requires_review: bool = True


class AgentContextBudget(ScopedContract):
    agent_context_budget_id: str | None = None
    agent_id: str
    model_setting_id: str | None = None
    max_context_tokens: int = Field(default=0, ge=0)
    reserved_response_tokens: int = Field(default=0, ge=0)
    max_prompt_tokens: int = Field(default=0, ge=0)
    daily_cost_limit_usd: float | None = Field(default=None, ge=0)
    monthly_cost_limit_usd: float | None = Field(default=None, ge=0)
    policy: dict[str, Any] = Field(default_factory=dict)


class MarkdownNote(ScopedContract):
    markdown_note_id: str | None = None
    note_id: str
    version_number: int = Field(default=1, ge=1)
    markdown: str
    rendered_summary: str = ""
    content_hash: str
    status: str = "current"
    source_document_id: str | None = None


class WikiLink(ScopedContract):
    wiki_link_id: str | None = None
    source_note_id: str
    target_note_id: str | None = None
    target_label: str
    link_text: str = ""
    link_status: str = "unresolved"
    metadata: dict[str, Any] = Field(default_factory=dict)


class TagRegistry(ScopedContract):
    tag_id: str | None = None
    tag: str
    label: str
    description: str = ""
    color: str = "#64748B"
    metadata: dict[str, Any] = Field(default_factory=dict)


class LittleBullMarkdownNoteResponse(BaseModel):
    registry: NoteRegistry
    note: MarkdownNote
    wiki_links: list[WikiLink] = Field(default_factory=list)
    tags: list[TagRegistry] = Field(default_factory=list)


class LittleBullProvenancePanel(BaseModel):
    target_kind: str
    target_id: str
    mentioned_in: list[Backlink] = Field(default_factory=list)
    cited_by: list[Backlink] = Field(default_factory=list)
    used_in_responses: list[SourceProvenance] = Field(default_factory=list)


class LittleBullCanvasBoardDetail(BaseModel):
    board: CanvasBoard
    nodes: list[CanvasNode] = Field(default_factory=list)
    edges: list[CanvasEdge] = Field(default_factory=list)


class LittleBullCanvasAnalysis(BaseModel):
    canvas_board_id: str
    node_count: int = 0
    edge_count: int = 0
    node_kind_counts: dict[str, int] = Field(default_factory=dict)
    clusters: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class LittleBullKnowledgeTrailDetail(BaseModel):
    trail: KnowledgeTrail
    steps: list[KnowledgeTrailStep] = Field(default_factory=list)


class ContentMap(ScopedContract):
    content_map_id: str | None = None
    group_id: str | None = None
    subgroup_id: str | None = None
    title: str
    slug: str
    root_note_id: str | None = None
    description: str = ""
    map_body: dict[str, Any] = Field(default_factory=dict)
    status: str = "draft"


class CanvasBoard(ScopedContract):
    canvas_board_id: str | None = None
    group_id: str | None = None
    subgroup_id: str | None = None
    title: str
    slug: str
    layout: dict[str, Any] = Field(default_factory=dict)
    status: str = "active"


class CanvasNode(ScopedContract):
    canvas_node_id: str | None = None
    canvas_board_id: str
    node_kind: str
    ref_kind: str = ""
    ref_id: str = ""
    x: float = 0
    y: float = 0
    width: float = 280
    height: float = 160
    content: dict[str, Any] = Field(default_factory=dict)


class CanvasEdge(ScopedContract):
    canvas_edge_id: str | None = None
    canvas_board_id: str
    source_node_id: str
    target_node_id: str
    edge_kind: str = "manual"
    label: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class KnowledgeInboxItem(ScopedContract):
    inbox_item_id: str | None = None
    group_id: str | None = None
    subgroup_id: str | None = None
    item_kind: str
    title: str
    body: str = ""
    source_kind: str = ""
    source_id: str = ""
    status: str = "open"
    priority: str = "normal"
    metadata: dict[str, Any] = Field(default_factory=dict)


class DailyNote(ScopedContract):
    daily_note_id: str | None = None
    note_id: str
    note_date: str
    summary: str = ""
    decisions: list[dict[str, Any]] = Field(default_factory=list)
    pending_items: list[dict[str, Any]] = Field(default_factory=list)
    cost_snapshot: dict[str, Any] = Field(default_factory=dict)


class NoteTemplate(ScopedContract):
    note_template_id: str | None = None
    title: str
    slug: str
    template_kind: str = "note"
    markdown_template: str
    variables_schema: dict[str, Any] = Field(default_factory=dict)
    status: str = "active"


class CommandPaletteAction(BaseModel):
    command_palette_action_id: str | None = None
    tenant_id: str
    workspace_id: str | None = None
    command_id: str
    title: str
    category: str = "workspace"
    handler_key: str
    required_permission: str = ""
    hotkey: str = ""
    enabled: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_by: str
    updated_by: str
    created_at: str | None = None
    updated_at: str | None = None


class SourceProvenance(ScopedContract):
    source_provenance_id: str | None = None
    source_kind: str
    source_id: str
    document_id: str | None = None
    note_id: str | None = None
    chunk_id: str = ""
    model_id: str = ""
    agent_id: str | None = None
    usage_ledger_id: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    locator: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class KnowledgeDossier(ScopedContract):
    knowledge_dossier_id: str | None = None
    group_id: str | None = None
    subgroup_id: str | None = None
    title: str
    slug: str
    dossier_kind: str = "knowledge"
    status: str = "draft"
    content_refs: list[dict[str, Any]] = Field(default_factory=list)
    export_policy: dict[str, Any] = Field(default_factory=dict)
    approval_id: str | None = None


class LegalMatterExtractionRun(ScopedContract):
    legal_matter_extraction_run_id: str | None = None
    group_id: str | None = None
    subgroup_id: str | None = None
    document_id: str | None = None
    matter_reference: str = ""
    extraction_model_id: str = ""
    schema_version: str
    run_status: str = "queued"
    extracted_payload: dict[str, Any] = Field(default_factory=dict)
    source_refs: list[dict[str, Any]] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0, le=1)
    review_status: str = "pending"
    requires_human_review: bool = True
    approved_by: str | None = None
    approved_at: str | None = None
    error_message: str = ""
