#!/bin/bash
# PostgreSQL initialization script for LightRAG with AGE support

set -e

echo "Installing PostgreSQL extensions..."

# Install build dependencies
apt-get update
apt-get install -y \
    build-essential \
    postgresql-server-dev-16 \
    git \
    ca-certificates

# Install uuid-ossp and vector extensions (built-in with PostgreSQL)
echo "Creating built-in extensions..."
psql -U lightrag -d lightrag_audit <<-EOSQL
    CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
    CREATE EXTENSION IF NOT EXISTS "vector";
EOSQL

# Install AGE extension
echo "Installing Apache AGE extension..."
cd /tmp
git clone https://github.com/apache/age.git
cd age
git checkout release/PG16
make
make install

# Create AGE extension in database
psql -U lightrag -d lightrag_audit <<-EOSQL
    CREATE EXTENSION IF NOT EXISTS age;
    SELECT load_labels('ag_catalog');
EOSQL

echo "Extensions installed successfully!"
