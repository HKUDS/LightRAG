from __future__ import annotations

import os

from .little_bull_admin_schema import LITTLE_BULL_ADMIN_SCHEMA_SQL


def get_database_url() -> str | None:
    return os.getenv("LIGHTRAG_SYSTEM_DATABASE_URL") or os.getenv("DATABASE_URL")


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS system_users (
    user_id TEXT PRIMARY KEY,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    display_name TEXT NOT NULL,
    is_master_global BOOLEAN NOT NULL DEFAULT FALSE,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    permission_version INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS system_tenants (
    tenant_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS system_workspaces (
    workspace_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL REFERENCES system_tenants(tenant_id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    slug TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    privacy TEXT NOT NULL DEFAULT 'team',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (tenant_id, slug)
);

CREATE TABLE IF NOT EXISTS system_memberships (
    membership_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES system_users(user_id) ON DELETE CASCADE,
    tenant_id TEXT NOT NULL REFERENCES system_tenants(tenant_id) ON DELETE CASCADE,
    workspace_id TEXT NOT NULL REFERENCES system_workspaces(workspace_id) ON DELETE CASCADE,
    roles TEXT[] NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (user_id, workspace_id)
);

CREATE TABLE IF NOT EXISTS system_roles (
    role_id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS system_permissions (
    permission_id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS system_role_permissions (
    role_id TEXT NOT NULL REFERENCES system_roles(role_id) ON DELETE CASCADE,
    permission_id TEXT NOT NULL REFERENCES system_permissions(permission_id) ON DELETE CASCADE,
    PRIMARY KEY (role_id, permission_id)
);

CREATE TABLE IF NOT EXISTS system_policies (
    policy_id TEXT PRIMARY KEY,
    tenant_id TEXT REFERENCES system_tenants(tenant_id) ON DELETE CASCADE,
    workspace_id TEXT REFERENCES system_workspaces(workspace_id) ON DELETE CASCADE,
    key TEXT NOT NULL,
    value JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS system_approval_requests (
    approval_id TEXT PRIMARY KEY,
    action TEXT NOT NULL,
    actor_user_id TEXT NOT NULL REFERENCES system_users(user_id) ON DELETE CASCADE,
    tenant_id TEXT REFERENCES system_tenants(tenant_id) ON DELETE CASCADE,
    workspace_id TEXT REFERENCES system_workspaces(workspace_id) ON DELETE CASCADE,
    payload_hash TEXT NOT NULL,
    reason TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    requested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    decided_at TIMESTAMPTZ,
    decided_by TEXT REFERENCES system_users(user_id) ON DELETE SET NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS system_audit_events (
    event_id TEXT PRIMARY KEY,
    actor_user_id TEXT NOT NULL,
    action TEXT NOT NULL,
    tenant_id TEXT,
    workspace_id TEXT,
    result TEXT NOT NULL,
    approval_id TEXT,
    model TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_system_memberships_user ON system_memberships(user_id);
CREATE INDEX IF NOT EXISTS idx_system_workspaces_tenant ON system_workspaces(tenant_id);
CREATE INDEX IF NOT EXISTS idx_system_approvals_scope ON system_approval_requests(tenant_id, workspace_id, status);
CREATE INDEX IF NOT EXISTS idx_system_audit_scope ON system_audit_events(tenant_id, workspace_id, created_at DESC);
""" + LITTLE_BULL_ADMIN_SCHEMA_SQL


async def run_schema(database_url: str | None = None) -> bool:
    url = database_url or get_database_url()
    if not url:
        return False

    import asyncpg

    conn = await asyncpg.connect(url)
    try:
        await conn.execute(SCHEMA_SQL)
    finally:
        await conn.close()
    return True
