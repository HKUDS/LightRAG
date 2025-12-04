-- Enable Row Level Security on all tables
ALTER TABLE LIGHTRAG_DOC_FULL ENABLE ROW LEVEL SECURITY;
ALTER TABLE LIGHTRAG_DOC_CHUNKS ENABLE ROW LEVEL SECURITY;
ALTER TABLE LIGHTRAG_VDB_CHUNKS ENABLE ROW LEVEL SECURITY;
ALTER TABLE LIGHTRAG_VDB_ENTITY ENABLE ROW LEVEL SECURITY;
ALTER TABLE LIGHTRAG_VDB_RELATION ENABLE ROW LEVEL SECURITY;
ALTER TABLE LIGHTRAG_LLM_CACHE ENABLE ROW LEVEL SECURITY;
ALTER TABLE LIGHTRAG_DOC_STATUS ENABLE ROW LEVEL SECURITY;
ALTER TABLE LIGHTRAG_FULL_ENTITIES ENABLE ROW LEVEL SECURITY;
ALTER TABLE LIGHTRAG_FULL_RELATIONS ENABLE ROW LEVEL SECURITY;

-- Create RLS Policies
-- Policy: Users can only see rows where tenant_id matches the current session setting

-- LIGHTRAG_DOC_FULL
CREATE POLICY tenant_isolation_policy ON LIGHTRAG_DOC_FULL
    USING (tenant_id = current_setting('app.current_tenant')::text);

-- LIGHTRAG_DOC_CHUNKS
CREATE POLICY tenant_isolation_policy ON LIGHTRAG_DOC_CHUNKS
    USING (tenant_id = current_setting('app.current_tenant')::text);

-- LIGHTRAG_VDB_CHUNKS
CREATE POLICY tenant_isolation_policy ON LIGHTRAG_VDB_CHUNKS
    USING (tenant_id = current_setting('app.current_tenant')::text);

-- LIGHTRAG_VDB_ENTITY
CREATE POLICY tenant_isolation_policy ON LIGHTRAG_VDB_ENTITY
    USING (tenant_id = current_setting('app.current_tenant')::text);

-- LIGHTRAG_VDB_RELATION
CREATE POLICY tenant_isolation_policy ON LIGHTRAG_VDB_RELATION
    USING (tenant_id = current_setting('app.current_tenant')::text);

-- LIGHTRAG_LLM_CACHE
CREATE POLICY tenant_isolation_policy ON LIGHTRAG_LLM_CACHE
    USING (tenant_id = current_setting('app.current_tenant')::text);

-- LIGHTRAG_DOC_STATUS
CREATE POLICY tenant_isolation_policy ON LIGHTRAG_DOC_STATUS
    USING (tenant_id = current_setting('app.current_tenant')::text);

-- LIGHTRAG_FULL_ENTITIES
CREATE POLICY tenant_isolation_policy ON LIGHTRAG_FULL_ENTITIES
    USING (tenant_id = current_setting('app.current_tenant')::text);

-- LIGHTRAG_FULL_RELATIONS
CREATE POLICY tenant_isolation_policy ON LIGHTRAG_FULL_RELATIONS
    USING (tenant_id = current_setting('app.current_tenant')::text);

-- Create a helper function to set the tenant (optional, but useful for testing)
CREATE OR REPLACE FUNCTION set_tenant_context(tenant_id text) RETURNS void AS $$
BEGIN
    PERFORM set_config('app.current_tenant', tenant_id, false);
END;
$$ LANGUAGE plpgsql;
