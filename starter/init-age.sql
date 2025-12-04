-- AGE Extension Initialization
-- This script creates the AGE extension in the database

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";
CREATE EXTENSION IF NOT EXISTS age;

-- Load AGE in the search path
SET search_path = ag_catalog, "$user", public;

-- Verify AGE is installed
SELECT extname, extversion FROM pg_extension WHERE extname = 'age';
