-- LightRAG Database Initialization
-- This script will be executed when the PostgreSQL container starts
-- It handles both default (rag/rag/rag) and custom database configurations

-- Create extensions required by LightRAG in the target database
-- This will work for both default 'rag' database and custom databases
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
CREATE EXTENSION IF NOT EXISTS auto_explain;
CREATE EXTENSION IF NOT EXISTS age;

-- Load AGE extension and set search path
LOAD 'age';
SET search_path = ag_catalog, "$user", public;

-- Ensure search path is set for all future sessions in this database
ALTER DATABASE CURRENT SET search_path = ag_catalog, "$user", public;

-- Set auto_explain configuration
ALTER SYSTEM SET auto_explain.log_min_duration = '1s';
ALTER SYSTEM SET auto_explain.log_analyze = true;
ALTER SYSTEM SET auto_explain.log_buffers = true;

-- Reload configuration
SELECT pg_reload_conf();
