-- Custom PostgreSQL Database Setup for LightRAG
-- This script handles custom database creation with proper extension support
-- It will be executed by docker-entrypoint-initdb.d when needed

-- Function to create database and user if they don't exist
DO $$
DECLARE
    target_user TEXT := coalesce(current_setting('custom.user', true), 'rag');
    target_db TEXT := coalesce(current_setting('custom.database', true), 'rag');
    target_password TEXT := coalesce(current_setting('custom.password', true), 'rag');
    user_exists BOOLEAN;
    db_exists BOOLEAN;
BEGIN
    -- Check if we're using custom credentials (different from defaults)
    IF target_user != 'rag' OR target_db != 'rag' OR target_password != 'rag' THEN
        RAISE NOTICE 'Setting up custom database: % for user: %', target_db, target_user;

        -- Check if user exists
        SELECT EXISTS(SELECT 1 FROM pg_roles WHERE rolname = target_user) INTO user_exists;

        -- Create user if doesn't exist
        IF NOT user_exists THEN
            EXECUTE format('CREATE USER %I WITH PASSWORD %L', target_user, target_password);
            RAISE NOTICE 'Created user: %', target_user;
        END IF;

        -- Check if database exists
        SELECT EXISTS(SELECT 1 FROM pg_database WHERE datname = target_db) INTO db_exists;

        -- Create database if doesn't exist
        IF NOT db_exists THEN
            EXECUTE format('CREATE DATABASE %I OWNER %I', target_db, target_user);
            RAISE NOTICE 'Created database: % owned by: %', target_db, target_user;
        END IF;

        -- Grant privileges
        EXECUTE format('GRANT ALL PRIVILEGES ON DATABASE %I TO %I', target_db, target_user);

    ELSE
        RAISE NOTICE 'Using default rag/rag/rag configuration';
    END IF;
END $$;
