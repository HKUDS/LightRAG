-- ============================================================================
-- LightRAG Multi-Tenant Schema Migration - ROLLBACK
-- Version: 2.0.0
-- Date: December 4, 2025
-- Purpose: Rollback the FK infrastructure migration
-- 
-- IMPORTANT: This will remove generated columns, triggers, and indexes.
--            The original workspace column and all data remain intact.
-- 
-- Usage:
--   PGPASSWORD=lightrag123 psql -h localhost -p 15432 -U lightrag -d lightrag_multitenant -f rollback_v2.0.0.sql
-- ============================================================================

BEGIN;

\echo '============================================'
\echo 'Starting Rollback of Migration 2.0.0'
\echo '============================================'

-- ============================================================================
-- STEP 1: Drop triggers
-- ============================================================================

DROP TRIGGER IF EXISTS trg_sync_tenant_doc_full ON lightrag_doc_full;
DROP TRIGGER IF EXISTS trg_sync_tenant_doc_chunks ON lightrag_doc_chunks;
DROP TRIGGER IF EXISTS trg_sync_tenant_doc_status ON lightrag_doc_status;
DROP TRIGGER IF EXISTS trg_sync_tenant_vdb_chunks ON lightrag_vdb_chunks;
DROP TRIGGER IF EXISTS trg_sync_tenant_vdb_entity ON lightrag_vdb_entity;
DROP TRIGGER IF EXISTS trg_sync_tenant_vdb_relation ON lightrag_vdb_relation;
DROP TRIGGER IF EXISTS trg_sync_tenant_full_entities ON lightrag_full_entities;
DROP TRIGGER IF EXISTS trg_sync_tenant_full_relations ON lightrag_full_relations;
DROP TRIGGER IF EXISTS trg_sync_tenant_llm_cache ON lightrag_llm_cache;

\echo 'Dropped all sync triggers'

-- ============================================================================
-- STEP 2: Drop trigger function
-- ============================================================================

DROP FUNCTION IF EXISTS sync_tenant_from_workspace();

\echo 'Dropped sync_tenant_from_workspace function'

-- ============================================================================
-- STEP 3: Drop indexes
-- ============================================================================

DROP INDEX IF EXISTS idx_lightrag_doc_full_tenant;
DROP INDEX IF EXISTS idx_lightrag_doc_full_tenant_kb;
DROP INDEX IF EXISTS idx_lightrag_doc_chunks_tenant;
DROP INDEX IF EXISTS idx_lightrag_doc_chunks_tenant_kb;
DROP INDEX IF EXISTS idx_lightrag_doc_status_tenant;
DROP INDEX IF EXISTS idx_lightrag_doc_status_tenant_kb;
DROP INDEX IF EXISTS idx_lightrag_vdb_chunks_tenant;
DROP INDEX IF EXISTS idx_lightrag_vdb_chunks_tenant_kb;
DROP INDEX IF EXISTS idx_lightrag_vdb_entity_tenant;
DROP INDEX IF EXISTS idx_lightrag_vdb_entity_tenant_kb;
DROP INDEX IF EXISTS idx_lightrag_vdb_relation_tenant;
DROP INDEX IF EXISTS idx_lightrag_vdb_relation_tenant_kb;
DROP INDEX IF EXISTS idx_lightrag_full_entities_tenant;
DROP INDEX IF EXISTS idx_lightrag_full_entities_tenant_kb;
DROP INDEX IF EXISTS idx_lightrag_full_relations_tenant;
DROP INDEX IF EXISTS idx_lightrag_full_relations_tenant_kb;
DROP INDEX IF EXISTS idx_lightrag_llm_cache_tenant;
DROP INDEX IF EXISTS idx_lightrag_llm_cache_tenant_kb;

\echo 'Dropped all tenant/kb indexes'

-- ============================================================================
-- STEP 4: Drop generated columns from all tables
-- ============================================================================

-- LIGHTRAG_DOC_FULL
ALTER TABLE lightrag_doc_full DROP COLUMN IF EXISTS tenant_id;
ALTER TABLE lightrag_doc_full DROP COLUMN IF EXISTS kb_id;

-- LIGHTRAG_DOC_CHUNKS
ALTER TABLE lightrag_doc_chunks DROP COLUMN IF EXISTS tenant_id;
ALTER TABLE lightrag_doc_chunks DROP COLUMN IF EXISTS kb_id;

-- LIGHTRAG_DOC_STATUS
ALTER TABLE lightrag_doc_status DROP COLUMN IF EXISTS tenant_id;
ALTER TABLE lightrag_doc_status DROP COLUMN IF EXISTS kb_id;

-- LIGHTRAG_VDB_CHUNKS
ALTER TABLE lightrag_vdb_chunks DROP COLUMN IF EXISTS tenant_id;
ALTER TABLE lightrag_vdb_chunks DROP COLUMN IF EXISTS kb_id;

-- LIGHTRAG_VDB_ENTITY
ALTER TABLE lightrag_vdb_entity DROP COLUMN IF EXISTS tenant_id;
ALTER TABLE lightrag_vdb_entity DROP COLUMN IF EXISTS kb_id;

-- LIGHTRAG_VDB_RELATION
ALTER TABLE lightrag_vdb_relation DROP COLUMN IF EXISTS tenant_id;
ALTER TABLE lightrag_vdb_relation DROP COLUMN IF EXISTS kb_id;

-- LIGHTRAG_FULL_ENTITIES
ALTER TABLE lightrag_full_entities DROP COLUMN IF EXISTS tenant_id;
ALTER TABLE lightrag_full_entities DROP COLUMN IF EXISTS kb_id;

-- LIGHTRAG_FULL_RELATIONS
ALTER TABLE lightrag_full_relations DROP COLUMN IF EXISTS tenant_id;
ALTER TABLE lightrag_full_relations DROP COLUMN IF EXISTS kb_id;

-- LIGHTRAG_LLM_CACHE
ALTER TABLE lightrag_llm_cache DROP COLUMN IF EXISTS tenant_id;
ALTER TABLE lightrag_llm_cache DROP COLUMN IF EXISTS kb_id;

\echo 'Dropped all generated columns'

-- ============================================================================
-- STEP 5: Update migration status
-- ============================================================================

UPDATE schema_migrations 
SET status = 'rolled_back', applied_at = NOW() 
WHERE version = '2.0.0';

\echo 'Updated migration status to rolled_back'

COMMIT;

-- ============================================================================
-- STEP 6: Analyze tables
-- ============================================================================

ANALYZE lightrag_doc_full;
ANALYZE lightrag_doc_chunks;
ANALYZE lightrag_doc_status;
ANALYZE lightrag_vdb_chunks;
ANALYZE lightrag_vdb_entity;
ANALYZE lightrag_vdb_relation;
ANALYZE lightrag_full_entities;
ANALYZE lightrag_full_relations;
ANALYZE lightrag_llm_cache;

\echo ''
\echo '============================================'
\echo 'Rollback of Migration 2.0.0 Complete!'
\echo '============================================'
\echo ''
\echo 'The following were removed:'
\echo '  - Generated columns (tenant_id, kb_id) from all LIGHTRAG_* tables'
\echo '  - Indexes on tenant_id and kb_id columns'
\echo '  - Auto-sync triggers and function'
\echo ''
\echo 'The following remain intact:'
\echo '  - All original data in LIGHTRAG_* tables'
\echo '  - workspace column (unchanged)'
\echo '  - tenants and knowledge_bases tables'
\echo ''
\echo 'NOTE: Tenants/KBs that were auto-created are NOT deleted.'
\echo '      Delete them manually if needed.'
