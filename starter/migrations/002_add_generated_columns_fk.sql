-- ============================================================================
-- LightRAG Multi-Tenant Schema Migration - Complete Implementation
-- Version: 2.0.0 (Consolidated)
-- Date: December 4, 2025
-- Purpose: Add generated columns, indexes, and auto-sync triggers to LIGHTRAG_* tables
-- 
-- This script implements the HYBRID APPROACH recommended by the audit:
-- - Keep LIGHTRAG_* tables as the authoritative storage
-- - Add generated columns for tenant_id and kb_id extraction from workspace
-- - Add indexes for query performance
-- - Add triggers to auto-register tenants/KBs
-- 
-- Usage:
--   PGPASSWORD=lightrag123 psql -h localhost -p 15432 -U lightrag -d lightrag_multitenant -f 002_add_generated_columns_fk.sql
-- ============================================================================

BEGIN;

-- ============================================================================
-- STEP 0: Create schema_migrations table if not exists
-- ============================================================================

CREATE TABLE IF NOT EXISTS schema_migrations (
    version VARCHAR(50) PRIMARY KEY,
    description TEXT,
    status VARCHAR(20) DEFAULT 'applied',
    applied_at TIMESTAMP DEFAULT NOW()
);

-- ============================================================================
-- STEP 1: Add generated columns to all LIGHTRAG_* tables
-- These columns auto-extract tenant_id and kb_id from workspace column
-- ============================================================================

-- LIGHTRAG_DOC_FULL
ALTER TABLE lightrag_doc_full 
ADD COLUMN IF NOT EXISTS tenant_id VARCHAR(255) GENERATED ALWAYS AS (
    CASE 
        WHEN workspace LIKE '%:%' THEN SPLIT_PART(workspace, ':', 1)
        ELSE workspace
    END
) STORED;

ALTER TABLE lightrag_doc_full 
ADD COLUMN IF NOT EXISTS kb_id VARCHAR(255) GENERATED ALWAYS AS (
    CASE 
        WHEN workspace LIKE '%:%' THEN SPLIT_PART(workspace, ':', 2)
        ELSE 'default'
    END
) STORED;

-- LIGHTRAG_DOC_CHUNKS
ALTER TABLE lightrag_doc_chunks 
ADD COLUMN IF NOT EXISTS tenant_id VARCHAR(255) GENERATED ALWAYS AS (
    CASE 
        WHEN workspace LIKE '%:%' THEN SPLIT_PART(workspace, ':', 1)
        ELSE workspace
    END
) STORED;

ALTER TABLE lightrag_doc_chunks 
ADD COLUMN IF NOT EXISTS kb_id VARCHAR(255) GENERATED ALWAYS AS (
    CASE 
        WHEN workspace LIKE '%:%' THEN SPLIT_PART(workspace, ':', 2)
        ELSE 'default'
    END
) STORED;

-- LIGHTRAG_DOC_STATUS
ALTER TABLE lightrag_doc_status 
ADD COLUMN IF NOT EXISTS tenant_id VARCHAR(255) GENERATED ALWAYS AS (
    CASE 
        WHEN workspace LIKE '%:%' THEN SPLIT_PART(workspace, ':', 1)
        ELSE workspace
    END
) STORED;

ALTER TABLE lightrag_doc_status 
ADD COLUMN IF NOT EXISTS kb_id VARCHAR(255) GENERATED ALWAYS AS (
    CASE 
        WHEN workspace LIKE '%:%' THEN SPLIT_PART(workspace, ':', 2)
        ELSE 'default'
    END
) STORED;

-- LIGHTRAG_VDB_CHUNKS
ALTER TABLE lightrag_vdb_chunks 
ADD COLUMN IF NOT EXISTS tenant_id VARCHAR(255) GENERATED ALWAYS AS (
    CASE 
        WHEN workspace LIKE '%:%' THEN SPLIT_PART(workspace, ':', 1)
        ELSE workspace
    END
) STORED;

ALTER TABLE lightrag_vdb_chunks 
ADD COLUMN IF NOT EXISTS kb_id VARCHAR(255) GENERATED ALWAYS AS (
    CASE 
        WHEN workspace LIKE '%:%' THEN SPLIT_PART(workspace, ':', 2)
        ELSE 'default'
    END
) STORED;

-- LIGHTRAG_VDB_ENTITY
ALTER TABLE lightrag_vdb_entity 
ADD COLUMN IF NOT EXISTS tenant_id VARCHAR(255) GENERATED ALWAYS AS (
    CASE 
        WHEN workspace LIKE '%:%' THEN SPLIT_PART(workspace, ':', 1)
        ELSE workspace
    END
) STORED;

ALTER TABLE lightrag_vdb_entity 
ADD COLUMN IF NOT EXISTS kb_id VARCHAR(255) GENERATED ALWAYS AS (
    CASE 
        WHEN workspace LIKE '%:%' THEN SPLIT_PART(workspace, ':', 2)
        ELSE 'default'
    END
) STORED;

-- LIGHTRAG_VDB_RELATION
ALTER TABLE lightrag_vdb_relation 
ADD COLUMN IF NOT EXISTS tenant_id VARCHAR(255) GENERATED ALWAYS AS (
    CASE 
        WHEN workspace LIKE '%:%' THEN SPLIT_PART(workspace, ':', 1)
        ELSE workspace
    END
) STORED;

ALTER TABLE lightrag_vdb_relation 
ADD COLUMN IF NOT EXISTS kb_id VARCHAR(255) GENERATED ALWAYS AS (
    CASE 
        WHEN workspace LIKE '%:%' THEN SPLIT_PART(workspace, ':', 2)
        ELSE 'default'
    END
) STORED;

-- LIGHTRAG_FULL_ENTITIES
ALTER TABLE lightrag_full_entities 
ADD COLUMN IF NOT EXISTS tenant_id VARCHAR(255) GENERATED ALWAYS AS (
    CASE 
        WHEN workspace LIKE '%:%' THEN SPLIT_PART(workspace, ':', 1)
        ELSE workspace
    END
) STORED;

ALTER TABLE lightrag_full_entities 
ADD COLUMN IF NOT EXISTS kb_id VARCHAR(255) GENERATED ALWAYS AS (
    CASE 
        WHEN workspace LIKE '%:%' THEN SPLIT_PART(workspace, ':', 2)
        ELSE 'default'
    END
) STORED;

-- LIGHTRAG_FULL_RELATIONS
ALTER TABLE lightrag_full_relations 
ADD COLUMN IF NOT EXISTS tenant_id VARCHAR(255) GENERATED ALWAYS AS (
    CASE 
        WHEN workspace LIKE '%:%' THEN SPLIT_PART(workspace, ':', 1)
        ELSE workspace
    END
) STORED;

ALTER TABLE lightrag_full_relations 
ADD COLUMN IF NOT EXISTS kb_id VARCHAR(255) GENERATED ALWAYS AS (
    CASE 
        WHEN workspace LIKE '%:%' THEN SPLIT_PART(workspace, ':', 2)
        ELSE 'default'
    END
) STORED;

-- LIGHTRAG_LLM_CACHE
ALTER TABLE lightrag_llm_cache 
ADD COLUMN IF NOT EXISTS tenant_id VARCHAR(255) GENERATED ALWAYS AS (
    CASE 
        WHEN workspace LIKE '%:%' THEN SPLIT_PART(workspace, ':', 1)
        ELSE workspace
    END
) STORED;

ALTER TABLE lightrag_llm_cache 
ADD COLUMN IF NOT EXISTS kb_id VARCHAR(255) GENERATED ALWAYS AS (
    CASE 
        WHEN workspace LIKE '%:%' THEN SPLIT_PART(workspace, ':', 2)
        ELSE 'default'
    END
) STORED;

-- ============================================================================
-- STEP 2: Add performance indexes on tenant_id and kb_id columns
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_lightrag_doc_full_tenant ON lightrag_doc_full(tenant_id);
CREATE INDEX IF NOT EXISTS idx_lightrag_doc_full_tenant_kb ON lightrag_doc_full(tenant_id, kb_id);

CREATE INDEX IF NOT EXISTS idx_lightrag_doc_chunks_tenant ON lightrag_doc_chunks(tenant_id);
CREATE INDEX IF NOT EXISTS idx_lightrag_doc_chunks_tenant_kb ON lightrag_doc_chunks(tenant_id, kb_id);

CREATE INDEX IF NOT EXISTS idx_lightrag_doc_status_tenant ON lightrag_doc_status(tenant_id);
CREATE INDEX IF NOT EXISTS idx_lightrag_doc_status_tenant_kb ON lightrag_doc_status(tenant_id, kb_id);

CREATE INDEX IF NOT EXISTS idx_lightrag_vdb_chunks_tenant ON lightrag_vdb_chunks(tenant_id);
CREATE INDEX IF NOT EXISTS idx_lightrag_vdb_chunks_tenant_kb ON lightrag_vdb_chunks(tenant_id, kb_id);

CREATE INDEX IF NOT EXISTS idx_lightrag_vdb_entity_tenant ON lightrag_vdb_entity(tenant_id);
CREATE INDEX IF NOT EXISTS idx_lightrag_vdb_entity_tenant_kb ON lightrag_vdb_entity(tenant_id, kb_id);

CREATE INDEX IF NOT EXISTS idx_lightrag_vdb_relation_tenant ON lightrag_vdb_relation(tenant_id);
CREATE INDEX IF NOT EXISTS idx_lightrag_vdb_relation_tenant_kb ON lightrag_vdb_relation(tenant_id, kb_id);

CREATE INDEX IF NOT EXISTS idx_lightrag_full_entities_tenant ON lightrag_full_entities(tenant_id);
CREATE INDEX IF NOT EXISTS idx_lightrag_full_entities_tenant_kb ON lightrag_full_entities(tenant_id, kb_id);

CREATE INDEX IF NOT EXISTS idx_lightrag_full_relations_tenant ON lightrag_full_relations(tenant_id);
CREATE INDEX IF NOT EXISTS idx_lightrag_full_relations_tenant_kb ON lightrag_full_relations(tenant_id, kb_id);

CREATE INDEX IF NOT EXISTS idx_lightrag_llm_cache_tenant ON lightrag_llm_cache(tenant_id);
CREATE INDEX IF NOT EXISTS idx_lightrag_llm_cache_tenant_kb ON lightrag_llm_cache(tenant_id, kb_id);

-- ============================================================================
-- STEP 3: Create auto-sync trigger function
-- This automatically registers tenants and knowledge bases when data is inserted
-- ============================================================================

CREATE OR REPLACE FUNCTION sync_tenant_from_workspace()
RETURNS TRIGGER AS $$
DECLARE
    v_tenant_id VARCHAR(255);
    v_kb_id VARCHAR(255);
BEGIN
    -- Extract tenant_id and kb_id from workspace
    v_tenant_id := SPLIT_PART(NEW.workspace, ':', 1);
    v_kb_id := SPLIT_PART(NEW.workspace, ':', 2);
    
    -- Skip if workspace doesn't contain colon (old format)
    IF v_kb_id = '' OR v_kb_id IS NULL THEN
        RETURN NEW;
    END IF;
    
    -- Insert tenant if not exists (use tenant_id column, not id)
    INSERT INTO tenants (tenant_id, name, created_at, updated_at)
    VALUES (v_tenant_id, v_tenant_id, NOW(), NOW())
    ON CONFLICT (tenant_id) DO NOTHING;
    
    -- Insert knowledge base if not exists
    INSERT INTO knowledge_bases (tenant_id, kb_id, name, created_at, updated_at)
    VALUES (v_tenant_id, v_kb_id, v_kb_id, NOW(), NOW())
    ON CONFLICT (tenant_id, kb_id) DO NOTHING;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- STEP 4: Add triggers to all LIGHTRAG_* tables
-- ============================================================================

DROP TRIGGER IF EXISTS trg_sync_tenant_doc_full ON lightrag_doc_full;
CREATE TRIGGER trg_sync_tenant_doc_full
    BEFORE INSERT OR UPDATE ON lightrag_doc_full
    FOR EACH ROW EXECUTE FUNCTION sync_tenant_from_workspace();

DROP TRIGGER IF EXISTS trg_sync_tenant_doc_chunks ON lightrag_doc_chunks;
CREATE TRIGGER trg_sync_tenant_doc_chunks
    BEFORE INSERT OR UPDATE ON lightrag_doc_chunks
    FOR EACH ROW EXECUTE FUNCTION sync_tenant_from_workspace();

DROP TRIGGER IF EXISTS trg_sync_tenant_doc_status ON lightrag_doc_status;
CREATE TRIGGER trg_sync_tenant_doc_status
    BEFORE INSERT OR UPDATE ON lightrag_doc_status
    FOR EACH ROW EXECUTE FUNCTION sync_tenant_from_workspace();

DROP TRIGGER IF EXISTS trg_sync_tenant_vdb_chunks ON lightrag_vdb_chunks;
CREATE TRIGGER trg_sync_tenant_vdb_chunks
    BEFORE INSERT OR UPDATE ON lightrag_vdb_chunks
    FOR EACH ROW EXECUTE FUNCTION sync_tenant_from_workspace();

DROP TRIGGER IF EXISTS trg_sync_tenant_vdb_entity ON lightrag_vdb_entity;
CREATE TRIGGER trg_sync_tenant_vdb_entity
    BEFORE INSERT OR UPDATE ON lightrag_vdb_entity
    FOR EACH ROW EXECUTE FUNCTION sync_tenant_from_workspace();

DROP TRIGGER IF EXISTS trg_sync_tenant_vdb_relation ON lightrag_vdb_relation;
CREATE TRIGGER trg_sync_tenant_vdb_relation
    BEFORE INSERT OR UPDATE ON lightrag_vdb_relation
    FOR EACH ROW EXECUTE FUNCTION sync_tenant_from_workspace();

DROP TRIGGER IF EXISTS trg_sync_tenant_full_entities ON lightrag_full_entities;
CREATE TRIGGER trg_sync_tenant_full_entities
    BEFORE INSERT OR UPDATE ON lightrag_full_entities
    FOR EACH ROW EXECUTE FUNCTION sync_tenant_from_workspace();

DROP TRIGGER IF EXISTS trg_sync_tenant_full_relations ON lightrag_full_relations;
CREATE TRIGGER trg_sync_tenant_full_relations
    BEFORE INSERT OR UPDATE ON lightrag_full_relations
    FOR EACH ROW EXECUTE FUNCTION sync_tenant_from_workspace();

DROP TRIGGER IF EXISTS trg_sync_tenant_llm_cache ON lightrag_llm_cache;
CREATE TRIGGER trg_sync_tenant_llm_cache
    BEFORE INSERT OR UPDATE ON lightrag_llm_cache
    FOR EACH ROW EXECUTE FUNCTION sync_tenant_from_workspace();

-- ============================================================================
-- STEP 5: Create user_tenant_memberships table for RBAC
-- ============================================================================

CREATE TABLE IF NOT EXISTS user_tenant_memberships (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    tenant_id VARCHAR(255) NOT NULL REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    role VARCHAR(50) NOT NULL DEFAULT 'viewer',
    permissions JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(user_id, tenant_id),
    CONSTRAINT valid_role CHECK (role IN ('owner', 'admin', 'editor', 'viewer'))
);

CREATE INDEX IF NOT EXISTS idx_user_memberships_user ON user_tenant_memberships(user_id);
CREATE INDEX IF NOT EXISTS idx_user_memberships_tenant ON user_tenant_memberships(tenant_id);

-- ============================================================================
-- STEP 6: Record migrations
-- ============================================================================

INSERT INTO schema_migrations (version, description, status, applied_at) VALUES
('1.0.0', 'Initial schema with LIGHTRAG_* tables', 'applied', '2025-12-03 00:00:00'),
('2.0.0', 'Add generated columns (tenant_id, kb_id) and indexes to LIGHTRAG_* tables', 'applied', NOW()),
('2.1.0', 'Add auto-sync trigger for tenant/KB registration', 'applied', NOW()),
('2.2.0', 'Create user_tenant_memberships table for RBAC', 'applied', NOW())
ON CONFLICT (version) DO UPDATE SET status = 'applied', applied_at = NOW();

COMMIT;

-- ============================================================================
-- Verification
-- ============================================================================

-- Show all migrations
SELECT version, description, status, applied_at FROM schema_migrations ORDER BY version;

-- Show table count
SELECT COUNT(*) as table_count FROM pg_tables WHERE schemaname = 'public';

-- Show sample data with generated columns
SELECT workspace, tenant_id, kb_id FROM lightrag_doc_full LIMIT 1;
