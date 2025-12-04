-- ============================================================================
-- LightRAG Multi-Tenant PostgreSQL Schema Initialization
-- Version: 3.0.0
-- Date: December 4, 2025
-- 
-- This script initializes the PostgreSQL database with multi-tenant support.
-- It is automatically executed when PostgreSQL container starts.
-- 
-- Architecture:
--   • LIGHTRAG_* tables: Core storage with workspace-based multi-tenancy
--   • tenants/knowledge_bases: Metadata registry for API layer
--   • Generated columns: Auto-extract tenant_id/kb_id from workspace
--   • Auto-sync triggers: Auto-register tenants/KBs on data insert
-- 
-- Features:
--   • pgvector support for embeddings
--   • Automatic indexes for performance
--   • RBAC with user_tenant_memberships
--   • Sample data for testing
-- ============================================================================

-- ============================================================================
-- Extensions
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";
CREATE EXTENSION IF NOT EXISTS "age";

-- ============================================================================
-- Schema Migrations Tracking Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS schema_migrations (
    version VARCHAR(50) PRIMARY KEY,
    description TEXT,
    status VARCHAR(20) DEFAULT 'applied',
    applied_at TIMESTAMP DEFAULT NOW()
);

-- ============================================================================
-- Tenants Table
-- 
-- Stores tenant information for multi-tenant system
-- Each tenant represents an organization, customer, or project
-- ============================================================================

CREATE TABLE IF NOT EXISTS tenants (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_tenants_id ON tenants(tenant_id);

-- ============================================================================
-- Knowledge Bases Table
-- 
-- Stores knowledge base metadata for each tenant
-- Each tenant can have multiple KBs (prod, dev, staging, etc.)
-- ============================================================================

CREATE TABLE IF NOT EXISTS knowledge_bases (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id VARCHAR(255) NOT NULL,
    kb_id VARCHAR(255) NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    CONSTRAINT uk_kb_tenant_kb UNIQUE(tenant_id, kb_id)
);

CREATE INDEX IF NOT EXISTS idx_kbs_tenant_kb ON knowledge_bases(tenant_id, kb_id);
CREATE INDEX IF NOT EXISTS idx_kbs_tenant ON knowledge_bases(tenant_id);

-- ============================================================================
-- LIGHTRAG Core Storage Tables
-- 
-- These are the PRODUCTION storage tables used by postgres_impl.py.
-- They use workspace-based multi-tenancy: workspace = "{tenant_id}:{kb_id}"
-- Generated columns auto-extract tenant_id and kb_id for queries.
-- ============================================================================

-- Document Status Table (tracks document processing)
CREATE TABLE IF NOT EXISTS LIGHTRAG_DOC_STATUS (
    workspace VARCHAR(255) NOT NULL,
    id VARCHAR(255) NOT NULL,
    content_summary VARCHAR(255) NULL,
    content_length INT NULL,
    chunks_count INT NULL,
    status VARCHAR(64) NULL,
    file_path TEXT NULL,
    chunks_list JSONB NULL DEFAULT '[]'::jsonb,
    track_id VARCHAR(255) NULL,
    metadata JSONB NULL DEFAULT '{}'::jsonb,
    error_msg TEXT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    -- Generated columns for multi-tenant queries
    tenant_id VARCHAR(255) GENERATED ALWAYS AS (
        CASE WHEN workspace LIKE '%:%' THEN SPLIT_PART(workspace, ':', 1) ELSE workspace END
    ) STORED,
    kb_id VARCHAR(255) GENERATED ALWAYS AS (
        CASE WHEN workspace LIKE '%:%' THEN SPLIT_PART(workspace, ':', 2) ELSE 'default' END
    ) STORED,
    CONSTRAINT LIGHTRAG_DOC_STATUS_PK PRIMARY KEY (workspace, id)
);

CREATE INDEX IF NOT EXISTS idx_lightrag_doc_status_workspace ON LIGHTRAG_DOC_STATUS(workspace);
CREATE INDEX IF NOT EXISTS idx_lightrag_doc_status_status ON LIGHTRAG_DOC_STATUS(workspace, status);
CREATE INDEX IF NOT EXISTS idx_lightrag_doc_status_tenant ON LIGHTRAG_DOC_STATUS(tenant_id);
CREATE INDEX IF NOT EXISTS idx_lightrag_doc_status_tenant_kb ON LIGHTRAG_DOC_STATUS(tenant_id, kb_id);

-- Full Documents Table (stores complete document content)
CREATE TABLE IF NOT EXISTS LIGHTRAG_DOC_FULL (
    id VARCHAR(255),
    workspace VARCHAR(255),
    doc_name VARCHAR(1024),
    content TEXT,
    meta JSONB,
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    -- Generated columns for multi-tenant queries
    tenant_id VARCHAR(255) GENERATED ALWAYS AS (
        CASE WHEN workspace LIKE '%:%' THEN SPLIT_PART(workspace, ':', 1) ELSE workspace END
    ) STORED,
    kb_id VARCHAR(255) GENERATED ALWAYS AS (
        CASE WHEN workspace LIKE '%:%' THEN SPLIT_PART(workspace, ':', 2) ELSE 'default' END
    ) STORED,
    CONSTRAINT LIGHTRAG_DOC_FULL_PK PRIMARY KEY (workspace, id)
);

CREATE INDEX IF NOT EXISTS idx_lightrag_doc_full_workspace ON LIGHTRAG_DOC_FULL(workspace);
CREATE INDEX IF NOT EXISTS idx_lightrag_doc_full_tenant ON LIGHTRAG_DOC_FULL(tenant_id);
CREATE INDEX IF NOT EXISTS idx_lightrag_doc_full_tenant_kb ON LIGHTRAG_DOC_FULL(tenant_id, kb_id);

-- Document Chunks Table (stores chunked document content)
CREATE TABLE IF NOT EXISTS LIGHTRAG_DOC_CHUNKS (
    id VARCHAR(255),
    workspace VARCHAR(255),
    full_doc_id VARCHAR(256),
    chunk_order_index INTEGER,
    tokens INTEGER,
    content TEXT,
    file_path TEXT NULL,
    llm_cache_list JSONB NULL DEFAULT '[]'::jsonb,
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    -- Generated columns for multi-tenant queries
    tenant_id VARCHAR(255) GENERATED ALWAYS AS (
        CASE WHEN workspace LIKE '%:%' THEN SPLIT_PART(workspace, ':', 1) ELSE workspace END
    ) STORED,
    kb_id VARCHAR(255) GENERATED ALWAYS AS (
        CASE WHEN workspace LIKE '%:%' THEN SPLIT_PART(workspace, ':', 2) ELSE 'default' END
    ) STORED,
    CONSTRAINT LIGHTRAG_DOC_CHUNKS_PK PRIMARY KEY (workspace, id)
);

CREATE INDEX IF NOT EXISTS idx_lightrag_doc_chunks_workspace ON LIGHTRAG_DOC_CHUNKS(workspace);
CREATE INDEX IF NOT EXISTS idx_lightrag_doc_chunks_full_doc_id ON LIGHTRAG_DOC_CHUNKS(workspace, full_doc_id);
CREATE INDEX IF NOT EXISTS idx_lightrag_doc_chunks_tenant ON LIGHTRAG_DOC_CHUNKS(tenant_id);
CREATE INDEX IF NOT EXISTS idx_lightrag_doc_chunks_tenant_kb ON LIGHTRAG_DOC_CHUNKS(tenant_id, kb_id);

-- Vector DB Chunks Table (stores vector embeddings for chunks)
CREATE TABLE IF NOT EXISTS LIGHTRAG_VDB_CHUNKS (
    id VARCHAR(255),
    workspace VARCHAR(255),
    full_doc_id VARCHAR(256),
    chunk_order_index INTEGER,
    tokens INTEGER,
    content TEXT,
    content_vector VECTOR(1536),
    file_path TEXT NULL,
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    -- Generated columns for multi-tenant queries
    tenant_id VARCHAR(255) GENERATED ALWAYS AS (
        CASE WHEN workspace LIKE '%:%' THEN SPLIT_PART(workspace, ':', 1) ELSE workspace END
    ) STORED,
    kb_id VARCHAR(255) GENERATED ALWAYS AS (
        CASE WHEN workspace LIKE '%:%' THEN SPLIT_PART(workspace, ':', 2) ELSE 'default' END
    ) STORED,
    CONSTRAINT LIGHTRAG_VDB_CHUNKS_PK PRIMARY KEY (workspace, id)
);

CREATE INDEX IF NOT EXISTS idx_lightrag_vdb_chunks_workspace ON LIGHTRAG_VDB_CHUNKS(workspace);
CREATE INDEX IF NOT EXISTS idx_lightrag_vdb_chunks_vector ON LIGHTRAG_VDB_CHUNKS USING ivfflat (content_vector vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_lightrag_vdb_chunks_tenant ON LIGHTRAG_VDB_CHUNKS(tenant_id);
CREATE INDEX IF NOT EXISTS idx_lightrag_vdb_chunks_tenant_kb ON LIGHTRAG_VDB_CHUNKS(tenant_id, kb_id);

-- Vector DB Entities Table (stores knowledge graph entities with embeddings)
CREATE TABLE IF NOT EXISTS LIGHTRAG_VDB_ENTITY (
    id VARCHAR(255),
    workspace VARCHAR(255),
    entity_name VARCHAR(512),
    content TEXT,
    content_vector VECTOR(1536),
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    chunk_ids VARCHAR(255)[] NULL,
    file_path TEXT NULL,
    -- Generated columns for multi-tenant queries
    tenant_id VARCHAR(255) GENERATED ALWAYS AS (
        CASE WHEN workspace LIKE '%:%' THEN SPLIT_PART(workspace, ':', 1) ELSE workspace END
    ) STORED,
    kb_id VARCHAR(255) GENERATED ALWAYS AS (
        CASE WHEN workspace LIKE '%:%' THEN SPLIT_PART(workspace, ':', 2) ELSE 'default' END
    ) STORED,
    CONSTRAINT LIGHTRAG_VDB_ENTITY_PK PRIMARY KEY (workspace, id)
);

CREATE INDEX IF NOT EXISTS idx_lightrag_vdb_entity_workspace ON LIGHTRAG_VDB_ENTITY(workspace);
CREATE INDEX IF NOT EXISTS idx_lightrag_vdb_entity_id ON LIGHTRAG_VDB_ENTITY(id);
CREATE INDEX IF NOT EXISTS idx_lightrag_vdb_entity_vector ON LIGHTRAG_VDB_ENTITY USING ivfflat (content_vector vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_lightrag_vdb_entity_hnsw_cosine ON LIGHTRAG_VDB_ENTITY USING hnsw (content_vector vector_cosine_ops) WITH (m = 16, ef_construction = 64);
CREATE INDEX IF NOT EXISTS idx_lightrag_vdb_entity_tenant ON LIGHTRAG_VDB_ENTITY(tenant_id);
CREATE INDEX IF NOT EXISTS idx_lightrag_vdb_entity_tenant_kb ON LIGHTRAG_VDB_ENTITY(tenant_id, kb_id);

-- Vector DB Relations Table (stores knowledge graph relationships with embeddings)
CREATE TABLE IF NOT EXISTS LIGHTRAG_VDB_RELATION (
    id VARCHAR(255),
    workspace VARCHAR(255),
    source_id VARCHAR(512),
    target_id VARCHAR(512),
    content TEXT,
    content_vector VECTOR(1536),
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    chunk_ids VARCHAR(255)[] NULL,
    file_path TEXT NULL,
    -- Generated columns for multi-tenant queries
    tenant_id VARCHAR(255) GENERATED ALWAYS AS (
        CASE WHEN workspace LIKE '%:%' THEN SPLIT_PART(workspace, ':', 1) ELSE workspace END
    ) STORED,
    kb_id VARCHAR(255) GENERATED ALWAYS AS (
        CASE WHEN workspace LIKE '%:%' THEN SPLIT_PART(workspace, ':', 2) ELSE 'default' END
    ) STORED,
    CONSTRAINT LIGHTRAG_VDB_RELATION_PK PRIMARY KEY (workspace, id)
);

CREATE INDEX IF NOT EXISTS idx_lightrag_vdb_relation_workspace ON LIGHTRAG_VDB_RELATION(workspace);
CREATE INDEX IF NOT EXISTS idx_lightrag_vdb_relation_vector ON LIGHTRAG_VDB_RELATION USING ivfflat (content_vector vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_lightrag_vdb_relation_tenant ON LIGHTRAG_VDB_RELATION(tenant_id);
CREATE INDEX IF NOT EXISTS idx_lightrag_vdb_relation_tenant_kb ON LIGHTRAG_VDB_RELATION(tenant_id, kb_id);

-- LLM Cache Table (caches LLM responses)
CREATE TABLE IF NOT EXISTS LIGHTRAG_LLM_CACHE (
    workspace VARCHAR(255) NOT NULL,
    id VARCHAR(255) NOT NULL,
    original_prompt TEXT,
    return_value TEXT,
    chunk_id VARCHAR(255) NULL,
    cache_type VARCHAR(32),
    queryparam JSONB NULL,
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    -- Generated columns for multi-tenant queries
    tenant_id VARCHAR(255) GENERATED ALWAYS AS (
        CASE WHEN workspace LIKE '%:%' THEN SPLIT_PART(workspace, ':', 1) ELSE workspace END
    ) STORED,
    kb_id VARCHAR(255) GENERATED ALWAYS AS (
        CASE WHEN workspace LIKE '%:%' THEN SPLIT_PART(workspace, ':', 2) ELSE 'default' END
    ) STORED,
    CONSTRAINT LIGHTRAG_LLM_CACHE_PK PRIMARY KEY (workspace, id)
);

CREATE INDEX IF NOT EXISTS idx_lightrag_llm_cache_workspace ON LIGHTRAG_LLM_CACHE(workspace);
CREATE INDEX IF NOT EXISTS idx_lightrag_llm_cache_chunk_id ON LIGHTRAG_LLM_CACHE(workspace, chunk_id);
CREATE INDEX IF NOT EXISTS idx_lightrag_llm_cache_tenant ON LIGHTRAG_LLM_CACHE(tenant_id);
CREATE INDEX IF NOT EXISTS idx_lightrag_llm_cache_tenant_kb ON LIGHTRAG_LLM_CACHE(tenant_id, kb_id);

-- Full Entities Table (aggregated entity data)
CREATE TABLE IF NOT EXISTS LIGHTRAG_FULL_ENTITIES (
    id VARCHAR(255),
    workspace VARCHAR(255),
    entity_names JSONB,
    count INTEGER,
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    -- Generated columns for multi-tenant queries
    tenant_id VARCHAR(255) GENERATED ALWAYS AS (
        CASE WHEN workspace LIKE '%:%' THEN SPLIT_PART(workspace, ':', 1) ELSE workspace END
    ) STORED,
    kb_id VARCHAR(255) GENERATED ALWAYS AS (
        CASE WHEN workspace LIKE '%:%' THEN SPLIT_PART(workspace, ':', 2) ELSE 'default' END
    ) STORED,
    CONSTRAINT LIGHTRAG_FULL_ENTITIES_PK PRIMARY KEY (workspace, id)
);

CREATE INDEX IF NOT EXISTS idx_lightrag_full_entities_workspace ON LIGHTRAG_FULL_ENTITIES(workspace);
CREATE INDEX IF NOT EXISTS idx_lightrag_full_entities_tenant ON LIGHTRAG_FULL_ENTITIES(tenant_id);
CREATE INDEX IF NOT EXISTS idx_lightrag_full_entities_tenant_kb ON LIGHTRAG_FULL_ENTITIES(tenant_id, kb_id);

-- Full Relations Table (aggregated relation data)
CREATE TABLE IF NOT EXISTS LIGHTRAG_FULL_RELATIONS (
    id VARCHAR(255),
    workspace VARCHAR(255),
    relation_pairs JSONB,
    count INTEGER,
    create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    update_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    -- Generated columns for multi-tenant queries
    tenant_id VARCHAR(255) GENERATED ALWAYS AS (
        CASE WHEN workspace LIKE '%:%' THEN SPLIT_PART(workspace, ':', 1) ELSE workspace END
    ) STORED,
    kb_id VARCHAR(255) GENERATED ALWAYS AS (
        CASE WHEN workspace LIKE '%:%' THEN SPLIT_PART(workspace, ':', 2) ELSE 'default' END
    ) STORED,
    CONSTRAINT LIGHTRAG_FULL_RELATIONS_PK PRIMARY KEY (workspace, id)
);

CREATE INDEX IF NOT EXISTS idx_lightrag_full_relations_workspace ON LIGHTRAG_FULL_RELATIONS(workspace);
CREATE INDEX IF NOT EXISTS idx_lightrag_full_relations_tenant ON LIGHTRAG_FULL_RELATIONS(tenant_id);
CREATE INDEX IF NOT EXISTS idx_lightrag_full_relations_tenant_kb ON LIGHTRAG_FULL_RELATIONS(tenant_id, kb_id);

-- ============================================================================
-- Auto-Sync Trigger Function
-- 
-- Automatically registers tenants and knowledge bases when data is inserted
-- into any LIGHTRAG_* table with a workspace column
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
    
    -- Insert tenant if not exists
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
-- Auto-Sync Triggers for all LIGHTRAG_* tables
-- ============================================================================

CREATE TRIGGER trg_sync_tenant_doc_full
    BEFORE INSERT OR UPDATE ON LIGHTRAG_DOC_FULL
    FOR EACH ROW EXECUTE FUNCTION sync_tenant_from_workspace();

CREATE TRIGGER trg_sync_tenant_doc_chunks
    BEFORE INSERT OR UPDATE ON LIGHTRAG_DOC_CHUNKS
    FOR EACH ROW EXECUTE FUNCTION sync_tenant_from_workspace();

CREATE TRIGGER trg_sync_tenant_doc_status
    BEFORE INSERT OR UPDATE ON LIGHTRAG_DOC_STATUS
    FOR EACH ROW EXECUTE FUNCTION sync_tenant_from_workspace();

CREATE TRIGGER trg_sync_tenant_vdb_chunks
    BEFORE INSERT OR UPDATE ON LIGHTRAG_VDB_CHUNKS
    FOR EACH ROW EXECUTE FUNCTION sync_tenant_from_workspace();

CREATE TRIGGER trg_sync_tenant_vdb_entity
    BEFORE INSERT OR UPDATE ON LIGHTRAG_VDB_ENTITY
    FOR EACH ROW EXECUTE FUNCTION sync_tenant_from_workspace();

CREATE TRIGGER trg_sync_tenant_vdb_relation
    BEFORE INSERT OR UPDATE ON LIGHTRAG_VDB_RELATION
    FOR EACH ROW EXECUTE FUNCTION sync_tenant_from_workspace();

CREATE TRIGGER trg_sync_tenant_full_entities
    BEFORE INSERT OR UPDATE ON LIGHTRAG_FULL_ENTITIES
    FOR EACH ROW EXECUTE FUNCTION sync_tenant_from_workspace();

CREATE TRIGGER trg_sync_tenant_full_relations
    BEFORE INSERT OR UPDATE ON LIGHTRAG_FULL_RELATIONS
    FOR EACH ROW EXECUTE FUNCTION sync_tenant_from_workspace();

CREATE TRIGGER trg_sync_tenant_llm_cache
    BEFORE INSERT OR UPDATE ON LIGHTRAG_LLM_CACHE
    FOR EACH ROW EXECUTE FUNCTION sync_tenant_from_workspace();

-- ============================================================================
-- User Tenant Memberships Table (RBAC)
-- ============================================================================

CREATE TABLE IF NOT EXISTS user_tenant_memberships (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    tenant_id VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL DEFAULT 'viewer',
    permissions JSONB DEFAULT '[]'::jsonb,
    created_by VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    CONSTRAINT uk_user_tenant UNIQUE(user_id, tenant_id),
    CONSTRAINT chk_role CHECK (role IN ('owner', 'admin', 'editor', 'viewer'))
);

CREATE INDEX IF NOT EXISTS idx_user_memberships_user ON user_tenant_memberships(user_id);
CREATE INDEX IF NOT EXISTS idx_user_memberships_tenant ON user_tenant_memberships(tenant_id);

-- ============================================================================
-- Tenant Access Helper Function
-- ============================================================================

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

-- ============================================================================
-- Sample Data for Testing Multi-Tenant Features
-- ============================================================================

-- Insert sample tenants
INSERT INTO tenants (tenant_id, name, description) VALUES
    ('acme-corp', 'Acme Corporation', 'Enterprise customer - production deployment'),
    ('techstart', 'TechStart Inc', 'Startup customer - evaluation environment')
ON CONFLICT (tenant_id) DO NOTHING;

-- Insert sample knowledge bases for Acme Corp
INSERT INTO knowledge_bases (tenant_id, kb_id, name, description) VALUES
    ('acme-corp', 'kb-prod', 'Production KB', 'Production knowledge base for Acme Corp'),
    ('acme-corp', 'kb-dev', 'Development KB', 'Development knowledge base for Acme Corp')
ON CONFLICT (tenant_id, kb_id) DO NOTHING;

-- Insert sample knowledge bases for TechStart
INSERT INTO knowledge_bases (tenant_id, kb_id, name, description) VALUES
    ('techstart', 'kb-main', 'Main KB', 'Main knowledge base for TechStart'),
    ('techstart', 'kb-backup', 'Backup KB', 'Backup knowledge base for TechStart')
ON CONFLICT (tenant_id, kb_id) DO NOTHING;

-- ============================================================================
-- Record Migrations
-- ============================================================================

INSERT INTO schema_migrations (version, description, status, applied_at) VALUES
    ('1.0.0', 'Initial schema with LIGHTRAG_* tables', 'applied', NOW()),
    ('2.0.0', 'Add generated columns (tenant_id, kb_id) and indexes', 'applied', NOW()),
    ('2.1.0', 'Add auto-sync trigger for tenant/KB registration', 'applied', NOW()),
    ('2.2.0', 'Create user_tenant_memberships table for RBAC', 'applied', NOW()),
    ('3.0.0', 'Remove unused tables, clean schema', 'applied', NOW())
ON CONFLICT (version) DO UPDATE SET status = 'applied', applied_at = NOW();

-- ============================================================================
-- Grant Permissions
-- ============================================================================

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO lightrag;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO lightrag;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO lightrag;
GRANT CREATE ON SCHEMA public TO lightrag;

-- ============================================================================
-- Schema Statistics
-- ============================================================================

ANALYZE tenants;
ANALYZE knowledge_bases;
ANALYZE user_tenant_memberships;

-- ============================================================================
-- Initialization Complete
-- ============================================================================
