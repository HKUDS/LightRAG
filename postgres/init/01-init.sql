-- LightRAG Production Database Initialization
-- This script will be executed when the PostgreSQL container starts

-- Create extensions required by LightRAG
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
CREATE EXTENSION IF NOT EXISTS auto_explain;

-- Set auto_explain configuration
ALTER SYSTEM SET auto_explain.log_min_duration = '1s';
ALTER SYSTEM SET auto_explain.log_analyze = true;
ALTER SYSTEM SET auto_explain.log_buffers = true;

-- Reload configuration
SELECT pg_reload_conf();
