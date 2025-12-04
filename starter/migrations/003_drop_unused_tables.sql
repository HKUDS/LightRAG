-- ============================================================================
-- LightRAG Multi-Tenant Schema Migration - Drop Unused Tables
-- Version: 3.0.0
-- Date: December 4, 2025
-- Purpose: Remove empty "modern" tables that are not used by the storage engine
-- 
-- These tables were designed but never implemented in the storage engine.
-- All actual data storage uses LIGHTRAG_* tables with workspace column.
-- 
-- Usage:
--   PGPASSWORD=lightrag123 psql -h localhost -p 15432 -U lightrag -d lightrag_multitenant -f 003_drop_unused_tables.sql
-- ============================================================================

BEGIN;

-- Verify tables are empty (safety check)
DO $$
DECLARE
    cnt INTEGER;
BEGIN
    SELECT COUNT(*) INTO cnt FROM documents;
    IF cnt > 0 THEN RAISE EXCEPTION 'documents table is not empty (% rows)', cnt; END IF;
    
    SELECT COUNT(*) INTO cnt FROM document_status;
    IF cnt > 0 THEN RAISE EXCEPTION 'document_status table is not empty (% rows)', cnt; END IF;
    
    SELECT COUNT(*) INTO cnt FROM entities;
    IF cnt > 0 THEN RAISE EXCEPTION 'entities table is not empty (% rows)', cnt; END IF;
    
    SELECT COUNT(*) INTO cnt FROM relations;
    IF cnt > 0 THEN RAISE EXCEPTION 'relations table is not empty (% rows)', cnt; END IF;
    
    SELECT COUNT(*) INTO cnt FROM embeddings;
    IF cnt > 0 THEN RAISE EXCEPTION 'embeddings table is not empty (% rows)', cnt; END IF;
    
    SELECT COUNT(*) INTO cnt FROM kv_storage;
    IF cnt > 0 THEN RAISE EXCEPTION 'kv_storage table is not empty (% rows)', cnt; END IF;
    
    SELECT COUNT(*) INTO cnt FROM lightrag_tenants;
    IF cnt > 0 THEN RAISE EXCEPTION 'lightrag_tenants table is not empty (% rows)', cnt; END IF;
    
    SELECT COUNT(*) INTO cnt FROM lightrag_knowledge_bases;
    IF cnt > 0 THEN RAISE EXCEPTION 'lightrag_knowledge_bases table is not empty (% rows)', cnt; END IF;
    
    RAISE NOTICE 'All tables verified empty - safe to drop';
END $$;

-- Drop tables (in correct order due to FK dependencies)
DROP TABLE IF EXISTS embeddings CASCADE;
DROP TABLE IF EXISTS document_status CASCADE;
DROP TABLE IF EXISTS relations CASCADE;
DROP TABLE IF EXISTS entities CASCADE;
DROP TABLE IF EXISTS documents CASCADE;
DROP TABLE IF EXISTS kv_storage CASCADE;
DROP TABLE IF EXISTS lightrag_knowledge_bases CASCADE;
DROP TABLE IF EXISTS lightrag_tenants CASCADE;

-- Record migration
INSERT INTO schema_migrations (version, description, status, applied_at)
VALUES ('3.0.0', 'Drop unused modern tables', 'applied', NOW())
ON CONFLICT (version) DO UPDATE SET status = 'applied', applied_at = NOW();

COMMIT;
