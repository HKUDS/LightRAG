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
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

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
"""
