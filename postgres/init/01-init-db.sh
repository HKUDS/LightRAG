#!/bin/bash
set -e

# This script creates the desired user and database if they don't exist,
# and secures the default 'rag' user.

# It is safe to run this script multiple times.

export PGPASSWORD="${POSTGRES_PASSWORD}"

psql -v ON_ERROR_STOP=1 --username "postgres" --dbname "postgres" <<-EOSQL
    -- Create user if it does not exist
    DO
    $do$
    BEGIN
       IF NOT EXISTS (
          SELECT FROM pg_catalog.pg_roles
          WHERE  rolname = '${POSTGRES_USER}') THEN

          CREATE ROLE ${POSTGRES_USER} WITH LOGIN PASSWORD '${POSTGRES_PASSWORD}';
       END IF;
    END
    $do$;

    -- Update password just in case user already existed
    ALTER ROLE ${POSTGRES_USER} WITH PASSWORD '${POSTGRES_PASSWORD}';

    -- For security, also change the password of the default 'rag' user
    ALTER USER rag WITH PASSWORD '${POSTGRES_PASSWORD}';
EOSQL

# Check if the desired database exists
DB_EXISTS=$(psql -U postgres -lqt | cut -d \| -f 1 | grep -w ${POSTGRES_DATABASE} | wc -l)

if [ "$DB_EXISTS" -eq 0 ]; then
    echo "Database '${POSTGRES_DATABASE}' does not exist. Creating..."
    psql -v ON_ERROR_STOP=1 --username "postgres" <<-EOSQL
        CREATE DATABASE ${POSTGRES_DATABASE} OWNER ${POSTGRES_USER};
EOSQL
    echo "Database '${POSTGRES_DATABASE}' created."
else
    echo "Database '${POSTGRES_DATABASE}' already exists."
fi

# Grant all privileges on the new database to the new user
psql -v ON_ERROR_STOP=1 --username "postgres" --dbname "${POSTGRES_DATABASE}" <<-EOSQL
    GRANT ALL PRIVILEGES ON DATABASE ${POSTGRES_DATABASE} TO ${POSTGRES_USER};
    GRANT USAGE, CREATE ON SCHEMA public TO ${POSTGRES_USER};
EOSQL

unset PGPASSWORD


echo "****** Successfully configured user '${POSTGRES_USER}' and database '${POSTGRES_DATABASE}' ******"
