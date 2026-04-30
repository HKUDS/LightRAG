from __future__ import annotations


LITTLE_BULL_ADMIN_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS little_bull_model_settings (
    model_setting_id TEXT PRIMARY KEY,
    tenant_id TEXT REFERENCES system_tenants(tenant_id) ON DELETE CASCADE,
    workspace_id TEXT REFERENCES system_workspaces(workspace_id) ON DELETE CASCADE,
    usage TEXT NOT NULL,
    provider TEXT NOT NULL,
    binding TEXT NOT NULL,
    binding_host TEXT NOT NULL DEFAULT '',
    model_id TEXT NOT NULL,
    display_name TEXT NOT NULL,
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    is_default BOOLEAN NOT NULL DEFAULT FALSE,
    config JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_by TEXT NOT NULL,
    updated_by TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_little_bull_model_settings_scope
    ON little_bull_model_settings(tenant_id, workspace_id, usage, enabled);

CREATE TABLE IF NOT EXISTS little_bull_agent_configs (
    agent_id TEXT PRIMARY KEY,
    tenant_id TEXT REFERENCES system_tenants(tenant_id) ON DELETE CASCADE,
    workspace_id TEXT REFERENCES system_workspaces(workspace_id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    model_setting_id TEXT REFERENCES little_bull_model_settings(model_setting_id) ON DELETE SET NULL,
    system_prompt TEXT NOT NULL DEFAULT '',
    response_rules TEXT[] NOT NULL DEFAULT '{}',
    tools TEXT[] NOT NULL DEFAULT '{}',
    config JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_by TEXT NOT NULL,
    updated_by TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_little_bull_agent_configs_scope
    ON little_bull_agent_configs(tenant_id, workspace_id, enabled);

CREATE TABLE IF NOT EXISTS little_bull_conversations (
    conversation_id TEXT PRIMARY KEY,
    tenant_id TEXT REFERENCES system_tenants(tenant_id) ON DELETE CASCADE,
    workspace_id TEXT NOT NULL REFERENCES system_workspaces(workspace_id) ON DELETE CASCADE,
    user_id TEXT NOT NULL REFERENCES system_users(user_id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    agent_id TEXT,
    model_profile TEXT NOT NULL DEFAULT 'equilibrado',
    confidentiality TEXT NOT NULL DEFAULT 'normal',
    scope_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE little_bull_conversations
    ADD COLUMN IF NOT EXISTS scope_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb;

CREATE INDEX IF NOT EXISTS idx_little_bull_conversations_scope
    ON little_bull_conversations(tenant_id, workspace_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS little_bull_conversation_messages (
    message_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES little_bull_conversations(conversation_id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    message_references JSONB NOT NULL DEFAULT '[]'::jsonb,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_little_bull_conversation_messages_conversation
    ON little_bull_conversation_messages(conversation_id, created_at);

CREATE TABLE IF NOT EXISTS little_bull_correlation_suggestions (
    suggestion_id TEXT PRIMARY KEY,
    tenant_id TEXT REFERENCES system_tenants(tenant_id) ON DELETE CASCADE,
    workspace_id TEXT NOT NULL REFERENCES system_workspaces(workspace_id) ON DELETE CASCADE,
    user_id TEXT NOT NULL REFERENCES system_users(user_id) ON DELETE CASCADE,
    source_label TEXT NOT NULL,
    target_label TEXT NOT NULL,
    reason TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    decided_at TIMESTAMPTZ,
    decided_by TEXT REFERENCES system_users(user_id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_little_bull_correlation_suggestions_scope
    ON little_bull_correlation_suggestions(tenant_id, workspace_id, status, created_at DESC);

CREATE TABLE IF NOT EXISTS little_bull_provider_credentials (
    provider_credential_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL REFERENCES system_tenants(tenant_id) ON DELETE CASCADE,
    workspace_id TEXT REFERENCES system_workspaces(workspace_id) ON DELETE CASCADE,
    provider TEXT NOT NULL,
    label TEXT NOT NULL,
    credential_kind TEXT NOT NULL DEFAULT 'api_key',
    secret_ref TEXT NOT NULL,
    secret_fingerprint TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'active',
    scopes TEXT[] NOT NULL DEFAULT '{}',
    config_public JSONB NOT NULL DEFAULT '{}'::jsonb,
    last_validated_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ,
    created_by TEXT NOT NULL REFERENCES system_users(user_id) ON DELETE RESTRICT,
    updated_by TEXT NOT NULL REFERENCES system_users(user_id) ON DELETE RESTRICT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (secret_ref <> ''),
    CHECK (secret_fingerprint = '' OR secret_fingerprint <> secret_ref)
);

CREATE INDEX IF NOT EXISTS idx_little_bull_provider_credentials_scope
    ON little_bull_provider_credentials(tenant_id, workspace_id, provider, status);

CREATE UNIQUE INDEX IF NOT EXISTS uq_little_bull_provider_credentials_workspace
    ON little_bull_provider_credentials(tenant_id, workspace_id, provider, label)
    WHERE workspace_id IS NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS uq_little_bull_provider_credentials_tenant
    ON little_bull_provider_credentials(tenant_id, provider, label)
    WHERE workspace_id IS NULL;

CREATE TABLE IF NOT EXISTS little_bull_model_catalog_snapshots (
    model_catalog_snapshot_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL REFERENCES system_tenants(tenant_id) ON DELETE CASCADE,
    workspace_id TEXT REFERENCES system_workspaces(workspace_id) ON DELETE CASCADE,
    provider_credential_id TEXT REFERENCES little_bull_provider_credentials(provider_credential_id) ON DELETE SET NULL,
    provider TEXT NOT NULL,
    source TEXT NOT NULL,
    catalog_hash TEXT NOT NULL,
    model_count INTEGER NOT NULL DEFAULT 0,
    catalog JSONB NOT NULL DEFAULT '[]'::jsonb,
    privacy_metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    synced_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_by TEXT NOT NULL REFERENCES system_users(user_id) ON DELETE RESTRICT,
    updated_by TEXT NOT NULL REFERENCES system_users(user_id) ON DELETE RESTRICT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_little_bull_model_catalog_snapshots_scope
    ON little_bull_model_catalog_snapshots(tenant_id, workspace_id, provider, synced_at DESC);

CREATE TABLE IF NOT EXISTS little_bull_knowledge_groups (
    group_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL REFERENCES system_tenants(tenant_id) ON DELETE CASCADE,
    workspace_id TEXT NOT NULL REFERENCES system_workspaces(workspace_id) ON DELETE CASCADE,
    slug TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    privacy TEXT NOT NULL DEFAULT 'team',
    color TEXT NOT NULL DEFAULT '#2563EB',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_by TEXT NOT NULL REFERENCES system_users(user_id) ON DELETE RESTRICT,
    updated_by TEXT NOT NULL REFERENCES system_users(user_id) ON DELETE RESTRICT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (workspace_id, slug)
);

CREATE INDEX IF NOT EXISTS idx_little_bull_knowledge_groups_scope
    ON little_bull_knowledge_groups(tenant_id, workspace_id, privacy);

CREATE TABLE IF NOT EXISTS little_bull_knowledge_subgroups (
    subgroup_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL REFERENCES system_tenants(tenant_id) ON DELETE CASCADE,
    workspace_id TEXT NOT NULL REFERENCES system_workspaces(workspace_id) ON DELETE CASCADE,
    group_id TEXT NOT NULL REFERENCES little_bull_knowledge_groups(group_id) ON DELETE CASCADE,
    slug TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    privacy TEXT NOT NULL DEFAULT 'team',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_by TEXT NOT NULL REFERENCES system_users(user_id) ON DELETE RESTRICT,
    updated_by TEXT NOT NULL REFERENCES system_users(user_id) ON DELETE RESTRICT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (group_id, slug)
);

CREATE INDEX IF NOT EXISTS idx_little_bull_knowledge_subgroups_scope
    ON little_bull_knowledge_subgroups(tenant_id, workspace_id, group_id, privacy);

CREATE TABLE IF NOT EXISTS little_bull_embedding_index_versions (
    embedding_version_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL REFERENCES system_tenants(tenant_id) ON DELETE CASCADE,
    workspace_id TEXT NOT NULL REFERENCES system_workspaces(workspace_id) ON DELETE CASCADE,
    group_id TEXT REFERENCES little_bull_knowledge_groups(group_id) ON DELETE SET NULL,
    subgroup_id TEXT REFERENCES little_bull_knowledge_subgroups(subgroup_id) ON DELETE SET NULL,
    model_setting_id TEXT REFERENCES little_bull_model_settings(model_setting_id) ON DELETE SET NULL,
    provider TEXT NOT NULL,
    model_id TEXT NOT NULL,
    dimensions INTEGER,
    chunking_policy JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding_config_hash TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'draft',
    is_active BOOLEAN NOT NULL DEFAULT FALSE,
    reindex_required BOOLEAN NOT NULL DEFAULT TRUE,
    created_by TEXT NOT NULL REFERENCES system_users(user_id) ON DELETE RESTRICT,
    updated_by TEXT NOT NULL REFERENCES system_users(user_id) ON DELETE RESTRICT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_little_bull_embedding_index_versions_scope
    ON little_bull_embedding_index_versions(tenant_id, workspace_id, group_id, subgroup_id, status);

CREATE TABLE IF NOT EXISTS little_bull_document_registry (
    document_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL REFERENCES system_tenants(tenant_id) ON DELETE CASCADE,
    workspace_id TEXT NOT NULL REFERENCES system_workspaces(workspace_id) ON DELETE CASCADE,
    group_id TEXT REFERENCES little_bull_knowledge_groups(group_id) ON DELETE SET NULL,
    subgroup_id TEXT REFERENCES little_bull_knowledge_subgroups(subgroup_id) ON DELETE SET NULL,
    embedding_version_id TEXT REFERENCES little_bull_embedding_index_versions(embedding_version_id) ON DELETE SET NULL,
    title TEXT NOT NULL,
    source_uri TEXT NOT NULL DEFAULT '',
    source_kind TEXT NOT NULL DEFAULT 'upload',
    mime_type TEXT NOT NULL DEFAULT '',
    content_hash TEXT NOT NULL DEFAULT '',
    confidentiality TEXT NOT NULL DEFAULT 'normal',
    status TEXT NOT NULL DEFAULT 'registered',
    chunk_count INTEGER NOT NULL DEFAULT 0,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_by TEXT NOT NULL REFERENCES system_users(user_id) ON DELETE RESTRICT,
    updated_by TEXT NOT NULL REFERENCES system_users(user_id) ON DELETE RESTRICT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_little_bull_document_registry_scope
    ON little_bull_document_registry(tenant_id, workspace_id, group_id, subgroup_id, status);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'chk_little_bull_document_registry_upload_classified'
    ) THEN
        ALTER TABLE little_bull_document_registry
            ADD CONSTRAINT chk_little_bull_document_registry_upload_classified
            CHECK (source_kind <> 'upload' OR (group_id IS NOT NULL AND subgroup_id IS NOT NULL))
            NOT VALID;
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS little_bull_note_registry (
    note_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL REFERENCES system_tenants(tenant_id) ON DELETE CASCADE,
    workspace_id TEXT NOT NULL REFERENCES system_workspaces(workspace_id) ON DELETE CASCADE,
    group_id TEXT REFERENCES little_bull_knowledge_groups(group_id) ON DELETE SET NULL,
    subgroup_id TEXT REFERENCES little_bull_knowledge_subgroups(subgroup_id) ON DELETE SET NULL,
    title TEXT NOT NULL,
    slug TEXT NOT NULL,
    note_type TEXT NOT NULL DEFAULT 'markdown',
    privacy TEXT NOT NULL DEFAULT 'team',
    status TEXT NOT NULL DEFAULT 'active',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_by TEXT NOT NULL REFERENCES system_users(user_id) ON DELETE RESTRICT,
    updated_by TEXT NOT NULL REFERENCES system_users(user_id) ON DELETE RESTRICT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (workspace_id, slug)
);

CREATE INDEX IF NOT EXISTS idx_little_bull_note_registry_scope
    ON little_bull_note_registry(tenant_id, workspace_id, group_id, subgroup_id, status);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'chk_little_bull_note_registry_markdown_classified'
    ) THEN
        ALTER TABLE little_bull_note_registry
            ADD CONSTRAINT chk_little_bull_note_registry_markdown_classified
            CHECK (note_type <> 'markdown' OR (group_id IS NOT NULL AND subgroup_id IS NOT NULL))
            NOT VALID;
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS little_bull_indexing_jobs (
    indexing_job_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL REFERENCES system_tenants(tenant_id) ON DELETE CASCADE,
    workspace_id TEXT NOT NULL REFERENCES system_workspaces(workspace_id) ON DELETE CASCADE,
    group_id TEXT REFERENCES little_bull_knowledge_groups(group_id) ON DELETE SET NULL,
    subgroup_id TEXT REFERENCES little_bull_knowledge_subgroups(subgroup_id) ON DELETE SET NULL,
    document_id TEXT REFERENCES little_bull_document_registry(document_id) ON DELETE SET NULL,
    note_id TEXT REFERENCES little_bull_note_registry(note_id) ON DELETE SET NULL,
    embedding_version_id TEXT REFERENCES little_bull_embedding_index_versions(embedding_version_id) ON DELETE SET NULL,
    job_type TEXT NOT NULL DEFAULT 'index',
    status TEXT NOT NULL DEFAULT 'queued',
    progress JSONB NOT NULL DEFAULT '{}'::jsonb,
    error_message TEXT NOT NULL DEFAULT '',
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_by TEXT NOT NULL REFERENCES system_users(user_id) ON DELETE RESTRICT,
    updated_by TEXT NOT NULL REFERENCES system_users(user_id) ON DELETE RESTRICT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_little_bull_indexing_jobs_scope
    ON little_bull_indexing_jobs(tenant_id, workspace_id, status, created_at DESC);

CREATE TABLE IF NOT EXISTS little_bull_llm_usage_ledger (
    usage_ledger_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL REFERENCES system_tenants(tenant_id) ON DELETE CASCADE,
    workspace_id TEXT NOT NULL REFERENCES system_workspaces(workspace_id) ON DELETE CASCADE,
    group_id TEXT REFERENCES little_bull_knowledge_groups(group_id) ON DELETE SET NULL,
    subgroup_id TEXT REFERENCES little_bull_knowledge_subgroups(subgroup_id) ON DELETE SET NULL,
    user_id TEXT REFERENCES system_users(user_id) ON DELETE SET NULL,
    agent_id TEXT REFERENCES little_bull_agent_configs(agent_id) ON DELETE SET NULL,
    conversation_id TEXT REFERENCES little_bull_conversations(conversation_id) ON DELETE SET NULL,
    model_setting_id TEXT REFERENCES little_bull_model_settings(model_setting_id) ON DELETE SET NULL,
    provider TEXT NOT NULL,
    model_id TEXT NOT NULL,
    operation TEXT NOT NULL,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    estimated_cost_usd NUMERIC(18,8) NOT NULL DEFAULT 0,
    actual_cost_usd NUMERIC(18,8),
    currency TEXT NOT NULL DEFAULT 'USD',
    request_hash TEXT NOT NULL,
    response_hash TEXT NOT NULL DEFAULT '',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    previous_ledger_hash TEXT NOT NULL DEFAULT '',
    ledger_hash TEXT NOT NULL,
    created_by TEXT NOT NULL REFERENCES system_users(user_id) ON DELETE RESTRICT,
    updated_by TEXT NOT NULL REFERENCES system_users(user_id) ON DELETE RESTRICT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (updated_at = created_at),
    CHECK (updated_by = created_by)
);

ALTER TABLE little_bull_llm_usage_ledger
    ADD COLUMN IF NOT EXISTS group_id TEXT REFERENCES little_bull_knowledge_groups(group_id) ON DELETE SET NULL;

ALTER TABLE little_bull_llm_usage_ledger
    ADD COLUMN IF NOT EXISTS subgroup_id TEXT REFERENCES little_bull_knowledge_subgroups(subgroup_id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_little_bull_llm_usage_ledger_scope
    ON little_bull_llm_usage_ledger(tenant_id, workspace_id, provider, model_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_little_bull_llm_usage_ledger_group_scope
    ON little_bull_llm_usage_ledger(tenant_id, workspace_id, group_id, subgroup_id, operation, created_at DESC);

CREATE OR REPLACE FUNCTION little_bull_prevent_usage_ledger_update()
RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'little_bull_llm_usage_ledger is append-only';
END;
$$ LANGUAGE plpgsql;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_trigger
        WHERE tgname = 'trg_little_bull_usage_ledger_append_only'
    ) THEN
        CREATE TRIGGER trg_little_bull_usage_ledger_append_only
        BEFORE UPDATE OR DELETE ON little_bull_llm_usage_ledger
        FOR EACH ROW EXECUTE FUNCTION little_bull_prevent_usage_ledger_update();
    END IF;
END;
$$;

CREATE TABLE IF NOT EXISTS little_bull_graph_edge_origins (
    graph_edge_origin_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL REFERENCES system_tenants(tenant_id) ON DELETE CASCADE,
    workspace_id TEXT NOT NULL REFERENCES system_workspaces(workspace_id) ON DELETE CASCADE,
    group_id TEXT REFERENCES little_bull_knowledge_groups(group_id) ON DELETE SET NULL,
    subgroup_id TEXT REFERENCES little_bull_knowledge_subgroups(subgroup_id) ON DELETE SET NULL,
    source_node_id TEXT NOT NULL,
    target_node_id TEXT NOT NULL,
    edge_type TEXT NOT NULL,
    origin_type TEXT NOT NULL,
    origin_ref_id TEXT NOT NULL DEFAULT '',
    confidence NUMERIC(5,4),
    provenance JSONB NOT NULL DEFAULT '{}'::jsonb,
    status TEXT NOT NULL DEFAULT 'active',
    created_by TEXT NOT NULL REFERENCES system_users(user_id) ON DELETE RESTRICT,
    updated_by TEXT NOT NULL REFERENCES system_users(user_id) ON DELETE RESTRICT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_little_bull_graph_edge_origins_scope
    ON little_bull_graph_edge_origins(tenant_id, workspace_id, group_id, subgroup_id, edge_type);

CREATE TABLE IF NOT EXISTS little_bull_graph_clusters (
    graph_cluster_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL REFERENCES system_tenants(tenant_id) ON DELETE CASCADE,
    workspace_id TEXT NOT NULL REFERENCES system_workspaces(workspace_id) ON DELETE CASCADE,
    group_id TEXT REFERENCES little_bull_knowledge_groups(group_id) ON DELETE SET NULL,
    subgroup_id TEXT REFERENCES little_bull_knowledge_subgroups(subgroup_id) ON DELETE SET NULL,
    label TEXT NOT NULL,
    algorithm TEXT NOT NULL DEFAULT '',
    node_count INTEGER NOT NULL DEFAULT 0,
    edge_count INTEGER NOT NULL DEFAULT 0,
    summary TEXT NOT NULL DEFAULT '',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_by TEXT NOT NULL REFERENCES system_users(user_id) ON DELETE RESTRICT,
    updated_by TEXT NOT NULL REFERENCES system_users(user_id) ON DELETE RESTRICT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_little_bull_graph_clusters_scope
    ON little_bull_graph_clusters(tenant_id, workspace_id, group_id, subgroup_id);

CREATE TABLE IF NOT EXISTS little_bull_knowledge_trails (
    knowledge_trail_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL REFERENCES system_tenants(tenant_id) ON DELETE CASCADE,
    workspace_id TEXT NOT NULL REFERENCES system_workspaces(workspace_id) ON DELETE CASCADE,
    group_id TEXT REFERENCES little_bull_knowledge_groups(group_id) ON DELETE SET NULL,
    subgroup_id TEXT REFERENCES little_bull_knowledge_subgroups(subgroup_id) ON DELETE SET NULL,
    title TEXT NOT NULL,
    slug TEXT NOT NULL,
    trail_type TEXT NOT NULL DEFAULT 'study',
    description TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'draft',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_by TEXT NOT NULL REFERENCES system_users(user_id) ON DELETE RESTRICT,
    updated_by TEXT NOT NULL REFERENCES system_users(user_id) ON DELETE RESTRICT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (workspace_id, slug)
);

CREATE INDEX IF NOT EXISTS idx_little_bull_knowledge_trails_scope
    ON little_bull_knowledge_trails(tenant_id, workspace_id, group_id, subgroup_id, status);

CREATE TABLE IF NOT EXISTS little_bull_backlinks (
    backlink_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL REFERENCES system_tenants(tenant_id) ON DELETE CASCADE,
    workspace_id TEXT NOT NULL REFERENCES system_workspaces(workspace_id) ON DELETE CASCADE,
    source_kind TEXT NOT NULL,
    source_id TEXT NOT NULL,
    target_kind TEXT NOT NULL,
    target_id TEXT NOT NULL,
    link_text TEXT NOT NULL DEFAULT '',
    origin_type TEXT NOT NULL DEFAULT 'manual',
    graph_edge_origin_id TEXT REFERENCES little_bull_graph_edge_origins(graph_edge_origin_id) ON DELETE SET NULL,
    confidence NUMERIC(5,4),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_by TEXT NOT NULL REFERENCES system_users(user_id) ON DELETE RESTRICT,
    updated_by TEXT NOT NULL REFERENCES system_users(user_id) ON DELETE RESTRICT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (workspace_id, source_kind, source_id, target_kind, target_id, origin_type)
);

CREATE INDEX IF NOT EXISTS idx_little_bull_backlinks_target
    ON little_bull_backlinks(tenant_id, workspace_id, target_kind, target_id);

CREATE INDEX IF NOT EXISTS idx_little_bull_backlinks_source
    ON little_bull_backlinks(tenant_id, workspace_id, source_kind, source_id);

CREATE TABLE IF NOT EXISTS little_bull_graph_chat_sessions (
    graph_chat_session_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL REFERENCES system_tenants(tenant_id) ON DELETE CASCADE,
    workspace_id TEXT NOT NULL REFERENCES system_workspaces(workspace_id) ON DELETE CASCADE,
    conversation_id TEXT REFERENCES little_bull_conversations(conversation_id) ON DELETE SET NULL,
    focus_node_id TEXT NOT NULL DEFAULT '',
    graph_scope TEXT NOT NULL DEFAULT 'workspace',
    context_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb,
    cost_estimate JSONB NOT NULL DEFAULT '{}'::jsonb,
    status TEXT NOT NULL DEFAULT 'active',
    created_by TEXT NOT NULL REFERENCES system_users(user_id) ON DELETE RESTRICT,
    updated_by TEXT NOT NULL REFERENCES system_users(user_id) ON DELETE RESTRICT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_little_bull_graph_chat_sessions_scope
    ON little_bull_graph_chat_sessions(tenant_id, workspace_id, graph_scope, updated_at DESC);

CREATE TABLE IF NOT EXISTS little_bull_agent_builder_sessions (
    agent_builder_session_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL REFERENCES system_tenants(tenant_id) ON DELETE CASCADE,
    workspace_id TEXT NOT NULL REFERENCES system_workspaces(workspace_id) ON DELETE CASCADE,
    user_id TEXT NOT NULL REFERENCES system_users(user_id) ON DELETE CASCADE,
    agent_id TEXT REFERENCES little_bull_agent_configs(agent_id) ON DELETE SET NULL,
    model_setting_id TEXT REFERENCES little_bull_model_settings(model_setting_id) ON DELETE SET NULL,
    status TEXT NOT NULL DEFAULT 'draft',
    current_step TEXT NOT NULL DEFAULT 'intake',
    builder_transcript JSONB NOT NULL DEFAULT '[]'::jsonb,
    generated_config JSONB NOT NULL DEFAULT '{}'::jsonb,
    readiness_score INTEGER NOT NULL DEFAULT 0,
    requires_review BOOLEAN NOT NULL DEFAULT TRUE,
    created_by TEXT NOT NULL REFERENCES system_users(user_id) ON DELETE RESTRICT,
    updated_by TEXT NOT NULL REFERENCES system_users(user_id) ON DELETE RESTRICT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_little_bull_agent_builder_sessions_scope
    ON little_bull_agent_builder_sessions(tenant_id, workspace_id, user_id, status, updated_at DESC);

CREATE TABLE IF NOT EXISTS little_bull_agent_context_budgets (
    agent_context_budget_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL REFERENCES system_tenants(tenant_id) ON DELETE CASCADE,
    workspace_id TEXT NOT NULL REFERENCES system_workspaces(workspace_id) ON DELETE CASCADE,
    agent_id TEXT NOT NULL REFERENCES little_bull_agent_configs(agent_id) ON DELETE CASCADE,
    model_setting_id TEXT REFERENCES little_bull_model_settings(model_setting_id) ON DELETE SET NULL,
    max_context_tokens INTEGER NOT NULL DEFAULT 0,
    reserved_response_tokens INTEGER NOT NULL DEFAULT 0,
    max_prompt_tokens INTEGER NOT NULL DEFAULT 0,
    daily_cost_limit_usd NUMERIC(18,8),
    monthly_cost_limit_usd NUMERIC(18,8),
    policy JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_by TEXT NOT NULL REFERENCES system_users(user_id) ON DELETE RESTRICT,
    updated_by TEXT NOT NULL REFERENCES system_users(user_id) ON DELETE RESTRICT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (max_context_tokens >= 0),
    CHECK (reserved_response_tokens >= 0),
    CHECK (max_prompt_tokens >= 0)
);

CREATE INDEX IF NOT EXISTS idx_little_bull_agent_context_budgets_scope
    ON little_bull_agent_context_budgets(tenant_id, workspace_id, agent_id);

CREATE UNIQUE INDEX IF NOT EXISTS uq_little_bull_agent_context_budgets_model
    ON little_bull_agent_context_budgets(agent_id, model_setting_id)
    WHERE model_setting_id IS NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS uq_little_bull_agent_context_budgets_default
    ON little_bull_agent_context_budgets(agent_id)
    WHERE model_setting_id IS NULL;

CREATE TABLE IF NOT EXISTS little_bull_markdown_notes (
    markdown_note_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL REFERENCES system_tenants(tenant_id) ON DELETE CASCADE,
    workspace_id TEXT NOT NULL REFERENCES system_workspaces(workspace_id) ON DELETE CASCADE,
    note_id TEXT NOT NULL REFERENCES little_bull_note_registry(note_id) ON DELETE CASCADE,
    version_number INTEGER NOT NULL DEFAULT 1,
    markdown TEXT NOT NULL,
    rendered_summary TEXT NOT NULL DEFAULT '',
    content_hash TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'current',
    source_document_id TEXT REFERENCES little_bull_document_registry(document_id) ON DELETE SET NULL,
    created_by TEXT NOT NULL REFERENCES system_users(user_id) ON DELETE RESTRICT,
    updated_by TEXT NOT NULL REFERENCES system_users(user_id) ON DELETE RESTRICT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (note_id, version_number)
);

CREATE INDEX IF NOT EXISTS idx_little_bull_markdown_notes_note
    ON little_bull_markdown_notes(note_id, status, version_number DESC);

CREATE TABLE IF NOT EXISTS little_bull_wiki_links (
    wiki_link_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL REFERENCES system_tenants(tenant_id) ON DELETE CASCADE,
    workspace_id TEXT NOT NULL REFERENCES system_workspaces(workspace_id) ON DELETE CASCADE,
    source_note_id TEXT NOT NULL REFERENCES little_bull_note_registry(note_id) ON DELETE CASCADE,
    target_note_id TEXT REFERENCES little_bull_note_registry(note_id) ON DELETE SET NULL,
    target_label TEXT NOT NULL,
    link_text TEXT NOT NULL DEFAULT '',
    link_status TEXT NOT NULL DEFAULT 'unresolved',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_by TEXT NOT NULL REFERENCES system_users(user_id) ON DELETE RESTRICT,
    updated_by TEXT NOT NULL REFERENCES system_users(user_id) ON DELETE RESTRICT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_little_bull_wiki_links_source
    ON little_bull_wiki_links(tenant_id, workspace_id, source_note_id);

CREATE TABLE IF NOT EXISTS little_bull_tag_registry (
    tag_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL REFERENCES system_tenants(tenant_id) ON DELETE CASCADE,
    workspace_id TEXT NOT NULL REFERENCES system_workspaces(workspace_id) ON DELETE CASCADE,
    tag TEXT NOT NULL,
    label TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    color TEXT NOT NULL DEFAULT '#64748B',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_by TEXT NOT NULL REFERENCES system_users(user_id) ON DELETE RESTRICT,
    updated_by TEXT NOT NULL REFERENCES system_users(user_id) ON DELETE RESTRICT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (workspace_id, tag)
);

CREATE INDEX IF NOT EXISTS idx_little_bull_tag_registry_scope
    ON little_bull_tag_registry(tenant_id, workspace_id, tag);

CREATE TABLE IF NOT EXISTS little_bull_content_maps (
    content_map_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL REFERENCES system_tenants(tenant_id) ON DELETE CASCADE,
    workspace_id TEXT NOT NULL REFERENCES system_workspaces(workspace_id) ON DELETE CASCADE,
    group_id TEXT REFERENCES little_bull_knowledge_groups(group_id) ON DELETE SET NULL,
    subgroup_id TEXT REFERENCES little_bull_knowledge_subgroups(subgroup_id) ON DELETE SET NULL,
    title TEXT NOT NULL,
    slug TEXT NOT NULL,
    root_note_id TEXT REFERENCES little_bull_note_registry(note_id) ON DELETE SET NULL,
    description TEXT NOT NULL DEFAULT '',
    map_body JSONB NOT NULL DEFAULT '{}'::jsonb,
    status TEXT NOT NULL DEFAULT 'draft',
    created_by TEXT NOT NULL REFERENCES system_users(user_id) ON DELETE RESTRICT,
    updated_by TEXT NOT NULL REFERENCES system_users(user_id) ON DELETE RESTRICT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (workspace_id, slug)
);

CREATE INDEX IF NOT EXISTS idx_little_bull_content_maps_scope
    ON little_bull_content_maps(tenant_id, workspace_id, group_id, subgroup_id, status);

CREATE INDEX IF NOT EXISTS idx_little_bull_content_maps_root_note
    ON little_bull_content_maps(tenant_id, workspace_id, root_note_id)
    WHERE root_note_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS little_bull_canvas_boards (
    canvas_board_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL REFERENCES system_tenants(tenant_id) ON DELETE CASCADE,
    workspace_id TEXT NOT NULL REFERENCES system_workspaces(workspace_id) ON DELETE CASCADE,
    group_id TEXT REFERENCES little_bull_knowledge_groups(group_id) ON DELETE SET NULL,
    subgroup_id TEXT REFERENCES little_bull_knowledge_subgroups(subgroup_id) ON DELETE SET NULL,
    title TEXT NOT NULL,
    slug TEXT NOT NULL,
    layout JSONB NOT NULL DEFAULT '{}'::jsonb,
    status TEXT NOT NULL DEFAULT 'active',
    created_by TEXT NOT NULL REFERENCES system_users(user_id) ON DELETE RESTRICT,
    updated_by TEXT NOT NULL REFERENCES system_users(user_id) ON DELETE RESTRICT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (workspace_id, slug)
);

CREATE INDEX IF NOT EXISTS idx_little_bull_canvas_boards_scope
    ON little_bull_canvas_boards(tenant_id, workspace_id, group_id, subgroup_id, status);

CREATE TABLE IF NOT EXISTS little_bull_knowledge_trail_steps (
    knowledge_trail_step_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL REFERENCES system_tenants(tenant_id) ON DELETE CASCADE,
    workspace_id TEXT NOT NULL REFERENCES system_workspaces(workspace_id) ON DELETE CASCADE,
    knowledge_trail_id TEXT NOT NULL REFERENCES little_bull_knowledge_trails(knowledge_trail_id) ON DELETE CASCADE,
    step_order INTEGER NOT NULL,
    title TEXT NOT NULL,
    step_kind TEXT NOT NULL DEFAULT 'note',
    note_id TEXT REFERENCES little_bull_note_registry(note_id) ON DELETE SET NULL,
    document_id TEXT REFERENCES little_bull_document_registry(document_id) ON DELETE SET NULL,
    canvas_board_id TEXT REFERENCES little_bull_canvas_boards(canvas_board_id) ON DELETE SET NULL,
    instructions TEXT NOT NULL DEFAULT '',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_by TEXT NOT NULL REFERENCES system_users(user_id) ON DELETE RESTRICT,
    updated_by TEXT NOT NULL REFERENCES system_users(user_id) ON DELETE RESTRICT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (knowledge_trail_id, step_order)
);

CREATE INDEX IF NOT EXISTS idx_little_bull_knowledge_trail_steps_trail
    ON little_bull_knowledge_trail_steps(knowledge_trail_id, step_order);

CREATE INDEX IF NOT EXISTS idx_little_bull_knowledge_trail_steps_note
    ON little_bull_knowledge_trail_steps(tenant_id, workspace_id, note_id)
    WHERE note_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_little_bull_knowledge_trail_steps_document
    ON little_bull_knowledge_trail_steps(tenant_id, workspace_id, document_id)
    WHERE document_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_little_bull_knowledge_trail_steps_canvas
    ON little_bull_knowledge_trail_steps(tenant_id, workspace_id, canvas_board_id)
    WHERE canvas_board_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS little_bull_canvas_nodes (
    canvas_node_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL REFERENCES system_tenants(tenant_id) ON DELETE CASCADE,
    workspace_id TEXT NOT NULL REFERENCES system_workspaces(workspace_id) ON DELETE CASCADE,
    canvas_board_id TEXT NOT NULL REFERENCES little_bull_canvas_boards(canvas_board_id) ON DELETE CASCADE,
    node_kind TEXT NOT NULL,
    ref_kind TEXT NOT NULL DEFAULT '',
    ref_id TEXT NOT NULL DEFAULT '',
    x NUMERIC(18,6) NOT NULL DEFAULT 0,
    y NUMERIC(18,6) NOT NULL DEFAULT 0,
    width NUMERIC(18,6) NOT NULL DEFAULT 280,
    height NUMERIC(18,6) NOT NULL DEFAULT 160,
    content JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_by TEXT NOT NULL REFERENCES system_users(user_id) ON DELETE RESTRICT,
    updated_by TEXT NOT NULL REFERENCES system_users(user_id) ON DELETE RESTRICT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_little_bull_canvas_nodes_board
    ON little_bull_canvas_nodes(canvas_board_id, node_kind);

CREATE INDEX IF NOT EXISTS idx_little_bull_canvas_nodes_ref
    ON little_bull_canvas_nodes(tenant_id, workspace_id, ref_kind, ref_id)
    WHERE ref_id <> '';

CREATE TABLE IF NOT EXISTS little_bull_canvas_edges (
    canvas_edge_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL REFERENCES system_tenants(tenant_id) ON DELETE CASCADE,
    workspace_id TEXT NOT NULL REFERENCES system_workspaces(workspace_id) ON DELETE CASCADE,
    canvas_board_id TEXT NOT NULL REFERENCES little_bull_canvas_boards(canvas_board_id) ON DELETE CASCADE,
    source_node_id TEXT NOT NULL REFERENCES little_bull_canvas_nodes(canvas_node_id) ON DELETE CASCADE,
    target_node_id TEXT NOT NULL REFERENCES little_bull_canvas_nodes(canvas_node_id) ON DELETE CASCADE,
    edge_kind TEXT NOT NULL DEFAULT 'manual',
    label TEXT NOT NULL DEFAULT '',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_by TEXT NOT NULL REFERENCES system_users(user_id) ON DELETE RESTRICT,
    updated_by TEXT NOT NULL REFERENCES system_users(user_id) ON DELETE RESTRICT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_little_bull_canvas_edges_board
    ON little_bull_canvas_edges(canvas_board_id, edge_kind);

CREATE TABLE IF NOT EXISTS little_bull_knowledge_inbox_items (
    inbox_item_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL REFERENCES system_tenants(tenant_id) ON DELETE CASCADE,
    workspace_id TEXT NOT NULL REFERENCES system_workspaces(workspace_id) ON DELETE CASCADE,
    group_id TEXT REFERENCES little_bull_knowledge_groups(group_id) ON DELETE SET NULL,
    subgroup_id TEXT REFERENCES little_bull_knowledge_subgroups(subgroup_id) ON DELETE SET NULL,
    item_kind TEXT NOT NULL,
    title TEXT NOT NULL,
    body TEXT NOT NULL DEFAULT '',
    source_kind TEXT NOT NULL DEFAULT '',
    source_id TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'open',
    priority TEXT NOT NULL DEFAULT 'normal',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_by TEXT NOT NULL REFERENCES system_users(user_id) ON DELETE RESTRICT,
    updated_by TEXT NOT NULL REFERENCES system_users(user_id) ON DELETE RESTRICT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_little_bull_knowledge_inbox_items_scope
    ON little_bull_knowledge_inbox_items(tenant_id, workspace_id, status, priority, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_little_bull_knowledge_inbox_items_source
    ON little_bull_knowledge_inbox_items(tenant_id, workspace_id, source_kind, source_id)
    WHERE source_id <> '';

CREATE TABLE IF NOT EXISTS little_bull_daily_notes (
    daily_note_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL REFERENCES system_tenants(tenant_id) ON DELETE CASCADE,
    workspace_id TEXT NOT NULL REFERENCES system_workspaces(workspace_id) ON DELETE CASCADE,
    note_id TEXT NOT NULL REFERENCES little_bull_note_registry(note_id) ON DELETE CASCADE,
    note_date DATE NOT NULL,
    summary TEXT NOT NULL DEFAULT '',
    decisions JSONB NOT NULL DEFAULT '[]'::jsonb,
    pending_items JSONB NOT NULL DEFAULT '[]'::jsonb,
    cost_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_by TEXT NOT NULL REFERENCES system_users(user_id) ON DELETE RESTRICT,
    updated_by TEXT NOT NULL REFERENCES system_users(user_id) ON DELETE RESTRICT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (workspace_id, note_date)
);

CREATE INDEX IF NOT EXISTS idx_little_bull_daily_notes_scope
    ON little_bull_daily_notes(tenant_id, workspace_id, note_date DESC);

CREATE TABLE IF NOT EXISTS little_bull_note_templates (
    note_template_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL REFERENCES system_tenants(tenant_id) ON DELETE CASCADE,
    workspace_id TEXT NOT NULL REFERENCES system_workspaces(workspace_id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    slug TEXT NOT NULL,
    template_kind TEXT NOT NULL DEFAULT 'note',
    markdown_template TEXT NOT NULL,
    variables_schema JSONB NOT NULL DEFAULT '{}'::jsonb,
    status TEXT NOT NULL DEFAULT 'active',
    created_by TEXT NOT NULL REFERENCES system_users(user_id) ON DELETE RESTRICT,
    updated_by TEXT NOT NULL REFERENCES system_users(user_id) ON DELETE RESTRICT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (workspace_id, slug)
);

CREATE INDEX IF NOT EXISTS idx_little_bull_note_templates_scope
    ON little_bull_note_templates(tenant_id, workspace_id, template_kind, status);

CREATE TABLE IF NOT EXISTS little_bull_command_palette_actions (
    command_palette_action_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL REFERENCES system_tenants(tenant_id) ON DELETE CASCADE,
    workspace_id TEXT REFERENCES system_workspaces(workspace_id) ON DELETE CASCADE,
    command_id TEXT NOT NULL,
    title TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT 'workspace',
    handler_key TEXT NOT NULL,
    required_permission TEXT NOT NULL DEFAULT '',
    hotkey TEXT NOT NULL DEFAULT '',
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_by TEXT NOT NULL REFERENCES system_users(user_id) ON DELETE RESTRICT,
    updated_by TEXT NOT NULL REFERENCES system_users(user_id) ON DELETE RESTRICT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_little_bull_command_palette_actions_scope
    ON little_bull_command_palette_actions(tenant_id, workspace_id, category, enabled);

CREATE UNIQUE INDEX IF NOT EXISTS uq_little_bull_command_palette_actions_workspace
    ON little_bull_command_palette_actions(tenant_id, workspace_id, command_id)
    WHERE workspace_id IS NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS uq_little_bull_command_palette_actions_tenant
    ON little_bull_command_palette_actions(tenant_id, command_id)
    WHERE workspace_id IS NULL;

CREATE TABLE IF NOT EXISTS little_bull_source_provenance (
    source_provenance_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL REFERENCES system_tenants(tenant_id) ON DELETE CASCADE,
    workspace_id TEXT NOT NULL REFERENCES system_workspaces(workspace_id) ON DELETE CASCADE,
    source_kind TEXT NOT NULL,
    source_id TEXT NOT NULL,
    document_id TEXT REFERENCES little_bull_document_registry(document_id) ON DELETE SET NULL,
    note_id TEXT REFERENCES little_bull_note_registry(note_id) ON DELETE SET NULL,
    chunk_id TEXT NOT NULL DEFAULT '',
    model_id TEXT NOT NULL DEFAULT '',
    agent_id TEXT REFERENCES little_bull_agent_configs(agent_id) ON DELETE SET NULL,
    usage_ledger_id TEXT REFERENCES little_bull_llm_usage_ledger(usage_ledger_id) ON DELETE SET NULL,
    confidence NUMERIC(5,4),
    locator JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_by TEXT NOT NULL REFERENCES system_users(user_id) ON DELETE RESTRICT,
    updated_by TEXT NOT NULL REFERENCES system_users(user_id) ON DELETE RESTRICT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_little_bull_source_provenance_scope
    ON little_bull_source_provenance(tenant_id, workspace_id, source_kind, source_id);

CREATE INDEX IF NOT EXISTS idx_little_bull_source_provenance_document
    ON little_bull_source_provenance(tenant_id, workspace_id, document_id)
    WHERE document_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_little_bull_source_provenance_note
    ON little_bull_source_provenance(tenant_id, workspace_id, note_id)
    WHERE note_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS little_bull_knowledge_dossiers (
    knowledge_dossier_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL REFERENCES system_tenants(tenant_id) ON DELETE CASCADE,
    workspace_id TEXT NOT NULL REFERENCES system_workspaces(workspace_id) ON DELETE CASCADE,
    group_id TEXT REFERENCES little_bull_knowledge_groups(group_id) ON DELETE SET NULL,
    subgroup_id TEXT REFERENCES little_bull_knowledge_subgroups(subgroup_id) ON DELETE SET NULL,
    title TEXT NOT NULL,
    slug TEXT NOT NULL,
    dossier_kind TEXT NOT NULL DEFAULT 'knowledge',
    status TEXT NOT NULL DEFAULT 'draft',
    content_refs JSONB NOT NULL DEFAULT '[]'::jsonb,
    export_policy JSONB NOT NULL DEFAULT '{}'::jsonb,
    approval_id TEXT REFERENCES system_approval_requests(approval_id) ON DELETE SET NULL,
    created_by TEXT NOT NULL REFERENCES system_users(user_id) ON DELETE RESTRICT,
    updated_by TEXT NOT NULL REFERENCES system_users(user_id) ON DELETE RESTRICT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (workspace_id, slug)
);

CREATE INDEX IF NOT EXISTS idx_little_bull_knowledge_dossiers_scope
    ON little_bull_knowledge_dossiers(tenant_id, workspace_id, dossier_kind, status);

CREATE TABLE IF NOT EXISTS little_bull_legal_matter_extraction_runs (
    legal_matter_extraction_run_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL REFERENCES system_tenants(tenant_id) ON DELETE CASCADE,
    workspace_id TEXT NOT NULL REFERENCES system_workspaces(workspace_id) ON DELETE CASCADE,
    group_id TEXT REFERENCES little_bull_knowledge_groups(group_id) ON DELETE SET NULL,
    subgroup_id TEXT REFERENCES little_bull_knowledge_subgroups(subgroup_id) ON DELETE SET NULL,
    document_id TEXT REFERENCES little_bull_document_registry(document_id) ON DELETE SET NULL,
    matter_reference TEXT NOT NULL DEFAULT '',
    extraction_model_id TEXT NOT NULL DEFAULT '',
    schema_version TEXT NOT NULL,
    run_status TEXT NOT NULL DEFAULT 'queued',
    extracted_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    source_refs JSONB NOT NULL DEFAULT '[]'::jsonb,
    confidence NUMERIC(5,4),
    review_status TEXT NOT NULL DEFAULT 'pending',
    requires_human_review BOOLEAN NOT NULL DEFAULT TRUE,
    approved_by TEXT REFERENCES system_users(user_id) ON DELETE SET NULL,
    approved_at TIMESTAMPTZ,
    error_message TEXT NOT NULL DEFAULT '',
    created_by TEXT NOT NULL REFERENCES system_users(user_id) ON DELETE RESTRICT,
    updated_by TEXT NOT NULL REFERENCES system_users(user_id) ON DELETE RESTRICT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_little_bull_legal_matter_extraction_runs_scope
    ON little_bull_legal_matter_extraction_runs(tenant_id, workspace_id, document_id, run_status, review_status);
"""
