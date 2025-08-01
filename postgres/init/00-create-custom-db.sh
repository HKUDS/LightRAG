#!/bin/bash
set -e

# Custom database setup for LightRAG with shangor/postgres-for-rag image
# This script creates custom user and database if they differ from defaults

CUSTOM_USER="${POSTGRES_USER:-rag}"
CUSTOM_PASSWORD="${POSTGRES_PASSWORD:-rag}"
CUSTOM_DATABASE="${POSTGRES_DATABASE:-rag}"

echo "PostgreSQL init: Setting up database '$CUSTOM_DATABASE' for user '$CUSTOM_USER'"

# Check if we need to create custom user/database (different from defaults)
if [ "$CUSTOM_USER" != "rag" ] || [ "$CUSTOM_DATABASE" != "rag" ]; then
    echo "Creating custom user and database..."

    # Create custom user if it doesn't exist
    su - postgres -c "psql -v ON_ERROR_STOP=1 --dbname postgres" <<-EOSQL
        DO \$\$
        BEGIN
            IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '$CUSTOM_USER') THEN
                CREATE USER $CUSTOM_USER WITH PASSWORD '$CUSTOM_PASSWORD';
                RAISE NOTICE 'User $CUSTOM_USER created';
            ELSE
                RAISE NOTICE 'User $CUSTOM_USER already exists';
            END IF;
        END
        \$\$;
EOSQL

    # Create custom database if it doesn't exist (direct command, not in function)
    su - postgres -c "psql -v ON_ERROR_STOP=1 --dbname postgres -c \"SELECT 1 FROM pg_database WHERE datname = '$CUSTOM_DATABASE'\"" | grep -q 1 || \
    su - postgres -c "psql -v ON_ERROR_STOP=1 --dbname postgres -c \"CREATE DATABASE $CUSTOM_DATABASE OWNER $CUSTOM_USER\""

    # Grant privileges
    su - postgres -c "psql -v ON_ERROR_STOP=1 --dbname postgres" <<-EOSQL
        GRANT ALL PRIVILEGES ON DATABASE $CUSTOM_DATABASE TO $CUSTOM_USER;
EOSQL

    # Install extensions in the custom database
    echo "Installing extensions in database '$CUSTOM_DATABASE'..."
    su - postgres -c "psql -v ON_ERROR_STOP=1 --dbname \"$CUSTOM_DATABASE\"" <<-EOSQL
        CREATE EXTENSION IF NOT EXISTS vector;
        CREATE EXTENSION IF NOT EXISTS pg_stat_statements;
        CREATE EXTENSION IF NOT EXISTS auto_explain;
        CREATE EXTENSION IF NOT EXISTS age;
        LOAD 'age';
        SET search_path = ag_catalog, "\$user", public;
        -- Make search_path persistent for this database
        ALTER DATABASE $CUSTOM_DATABASE SET search_path = ag_catalog, "\$user", public;
EOSQL

    echo "Custom database setup completed for '$CUSTOM_DATABASE'"
else
    echo "Using default rag/rag configuration - extensions should already be installed"
fi
