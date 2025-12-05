-- ============================================================================
-- Migration: Add User-Tenant Membership System
-- Version: 001
-- Date: 2025-11-23
-- Description: Adds user-tenant membership table with RBAC support
-- ============================================================================

-- ============================================================================
-- User-Tenant Memberships Table
--
-- Stores user memberships in tenants with role-based access control
-- Roles: owner, admin, editor, viewer
-- ============================================================================

CREATE TABLE IF NOT EXISTS user_tenant_memberships (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    tenant_id VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL CHECK (role IN ('owner', 'admin', 'editor', 'viewer')),
    created_at TIMESTAMP DEFAULT NOW(),
    created_by VARCHAR(255),
    updated_at TIMESTAMP DEFAULT NOW(),

    FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    CONSTRAINT uk_user_tenant UNIQUE(user_id, tenant_id)
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_user_memberships ON user_tenant_memberships(user_id);
CREATE INDEX IF NOT EXISTS idx_tenant_members ON user_tenant_memberships(tenant_id);
CREATE INDEX IF NOT EXISTS idx_user_tenant_role ON user_tenant_memberships(user_id, tenant_id, role);

-- ============================================================================
-- Migrate Existing Tenant Creators to Owners
--
-- Add existing tenant creators as owners in the membership table
-- This ensures backward compatibility
-- ============================================================================

-- Add created_by column to tenants table if it doesn't exist
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'tenants' AND column_name = 'created_by'
    ) THEN
        ALTER TABLE tenants ADD COLUMN created_by VARCHAR(255);
    END IF;
END $$;

-- Migrate existing tenants: assume 'admin' created demo tenants
INSERT INTO user_tenant_memberships (user_id, tenant_id, role, created_by)
SELECT
    COALESCE(t.created_by, 'admin') as user_id,
    t.tenant_id,
    'owner' as role,
    'system' as created_by
FROM tenants t
WHERE NOT EXISTS (
    SELECT 1 FROM user_tenant_memberships m
    WHERE m.tenant_id = t.tenant_id
    AND m.user_id = COALESCE(t.created_by, 'admin')
)
ON CONFLICT (user_id, tenant_id) DO NOTHING;

-- ============================================================================
-- Helper Functions
-- ============================================================================

-- Function to check if user has required role for tenant
CREATE OR REPLACE FUNCTION has_tenant_access(
    p_user_id VARCHAR(255),
    p_tenant_id VARCHAR(255),
    p_required_role VARCHAR(50) DEFAULT 'viewer'
) RETURNS BOOLEAN AS $$
DECLARE
    v_user_role VARCHAR(50);
    v_role_hierarchy INTEGER;
    v_required_hierarchy INTEGER;
BEGIN
    -- Get user's role for the tenant
    SELECT role INTO v_user_role
    FROM user_tenant_memberships
    WHERE user_id = p_user_id AND tenant_id = p_tenant_id;

    -- If no membership found, check if tenant is public
    IF v_user_role IS NULL THEN
        RETURN EXISTS (
            SELECT 1 FROM tenants
            WHERE tenant_id = p_tenant_id
            AND (metadata->>'is_public')::boolean = true
        );
    END IF;

    -- Role hierarchy: owner(4) > admin(3) > editor(2) > viewer(1)
    v_role_hierarchy := CASE v_user_role
        WHEN 'owner' THEN 4
        WHEN 'admin' THEN 3
        WHEN 'editor' THEN 2
        WHEN 'viewer' THEN 1
        ELSE 0
    END;

    v_required_hierarchy := CASE p_required_role
        WHEN 'owner' THEN 4
        WHEN 'admin' THEN 3
        WHEN 'editor' THEN 2
        WHEN 'viewer' THEN 1
        ELSE 0
    END;

    RETURN v_role_hierarchy >= v_required_hierarchy;
END;
$$ LANGUAGE plpgsql;

-- Function to get user's role for a tenant
CREATE OR REPLACE FUNCTION get_user_tenant_role(
    p_user_id VARCHAR(255),
    p_tenant_id VARCHAR(255)
) RETURNS VARCHAR(50) AS $$
DECLARE
    v_role VARCHAR(50);
BEGIN
    SELECT role INTO v_role
    FROM user_tenant_memberships
    WHERE user_id = p_user_id AND tenant_id = p_tenant_id;

    RETURN v_role;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- Audit Log
-- ============================================================================

COMMENT ON TABLE user_tenant_memberships IS 'Stores user memberships in tenants with RBAC';
COMMENT ON COLUMN user_tenant_memberships.role IS 'User role: owner, admin, editor, or viewer';
COMMENT ON FUNCTION has_tenant_access IS 'Check if user has required role for tenant';
COMMENT ON FUNCTION get_user_tenant_role IS 'Get user role for a specific tenant';
