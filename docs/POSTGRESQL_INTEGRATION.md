# PostgreSQL Integration Documentation

## Overview

LightRAG provides comprehensive PostgreSQL integration that supports all four storage types required by the RAG system. This integration leverages the power of PostgreSQL's advanced features including vector similarity search, graph operations, and high-performance data storage.

## Features

- **Full Storage Backend Support**: KV, Vector, Graph, and Document Status storage
- **Vector Similarity Search**: Using pgvector extension for embedding operations
- **Graph Operations**: With Apache AGE extension for knowledge graph functionality
- **SSL/TLS Security**: Comprehensive SSL configuration options
- **Connection Pooling**: Optimized async connection management
- **Production Ready**: Includes monitoring, backup, and high availability features
- **Automated Setup**: Custom initialization scripts for seamless deployment

## Storage Implementations

### Available PostgreSQL Storage Backends

1. **PGKVStorage** - Key-Value storage for document chunks and LLM cache
2. **PGVectorStorage** - Vector storage for embeddings with similarity search
3. **PGGraphStorage** - Graph storage for entity relationships and knowledge graphs
4. **PGDocStatusStorage** - Document processing status tracking

## PostgreSQL Setup and Configuration

### Directory Structure

```
postgres/
├── config/
│   └── postgresql.conf          # Production-optimized PostgreSQL configuration
└── init/                        # Database initialization scripts
    ├── 00-create-custom-db.sh   # Custom database setup script
    ├── 00-custom-db-setup.sql   # SQL-based custom database creation
    ├── 01-init-db.sh           # User and database initialization
    └── 01-init.sql              # Extensions and performance setup
```

### Initialization Scripts

#### 1. `01-init.sql` - Extensions and Performance Setup

This script is the foundation of the PostgreSQL setup and automatically:

```sql
-- Create required extensions
CREATE EXTENSION IF NOT EXISTS vector;           -- Vector similarity search
CREATE EXTENSION IF NOT EXISTS pg_stat_statements; -- Query performance monitoring
CREATE EXTENSION IF NOT EXISTS auto_explain;     -- Automatic query plan explanation
CREATE EXTENSION IF NOT EXISTS age;              -- Apache AGE for graph operations

-- Load AGE and set search paths
LOAD 'age';
SET search_path = ag_catalog, "$user", public;
ALTER DATABASE CURRENT SET search_path = ag_catalog, "$user", public;

-- Configure performance monitoring
ALTER SYSTEM SET auto_explain.log_min_duration = '1s';
ALTER SYSTEM SET auto_explain.log_analyze = true;
ALTER SYSTEM SET auto_explain.log_buffers = true;
```

**Purpose**: Ensures all required extensions are available and configures optimal performance monitoring.

#### 2. `00-create-custom-db.sh` - Custom Database Setup Script

Flexible bash script that handles custom database creation:

```bash
#!/bin/bash
set -e

CUSTOM_USER="${POSTGRES_USER:-rag}"
CUSTOM_PASSWORD="${POSTGRES_PASSWORD:-rag}"
CUSTOM_DATABASE="${POSTGRES_DATABASE:-rag}"

# Creates custom user and database if they differ from defaults
# Installs all required extensions in the custom database
# Sets up proper search paths for AGE extension
```

**Key Features**:
- Environment variable driven configuration
- Safe to run multiple times (idempotent)
- Automatically installs extensions in custom databases
- Configures Apache AGE search paths

#### 3. `00-custom-db-setup.sql` - SQL-Based Custom Setup

Pure SQL approach for database initialization:

```sql
DO $$
DECLARE
    target_user TEXT := coalesce(current_setting('custom.user', true), 'rag');
    target_db TEXT := coalesce(current_setting('custom.database', true), 'rag');
    target_password TEXT := coalesce(current_setting('custom.password', true), 'rag');
BEGIN
    -- Safely creates users and databases only if they don't exist
    -- Grants appropriate privileges
    -- Handles custom vs default configurations
END $$;
```

**Purpose**: Provides SQL-only database setup for environments where shell scripts are restricted.

#### 4. `01-init-db.sh` - User and Database Initialization

Comprehensive user and database setup script:

```bash
#!/bin/bash
set -e

export PGPASSWORD="${POSTGRES_PASSWORD}"

# Creates user if it doesn't exist
# Updates passwords for security
# Creates database with proper ownership
# Grants all necessary privileges
```

**Security Features**:
- Updates default 'rag' user password for security
- Grants minimal required privileges
- Safe password handling with environment variables

### Configuration Files

#### `postgresql.conf` - Production-Optimized Configuration

Located at `postgres/config/postgresql.conf`, this file provides production-ready PostgreSQL settings:

```conf
# Connection and Authentication
listen_addresses = '*'
port = 5432
max_connections = 100
shared_preload_libraries = 'pg_stat_statements,auto_explain'

# Memory Optimization
shared_buffers = 256MB           # 25% of available RAM
effective_cache_size = 1GB       # 75% of available RAM
maintenance_work_mem = 64MB      # For index creation and maintenance
work_mem = 4MB                   # Per-query working memory

# Performance Tuning
random_page_cost = 1.1           # Optimized for SSD storage
effective_io_concurrency = 200   # Parallel I/O operations
checkpoint_completion_target = 0.9

# Write Ahead Log (WAL) Configuration
wal_level = replica              # Required for replication
max_wal_size = 1GB              # Maximum WAL size
min_wal_size = 80MB             # Minimum WAL size
checkpoint_timeout = 5min        # Automatic checkpoint interval

# Comprehensive Logging
log_min_duration_statement = 1000  # Log queries > 1 second
log_checkpoints = on             # Log checkpoint activity
log_connections = on             # Log new connections
log_disconnections = on          # Log connection terminations
log_lock_waits = on             # Log lock waits
log_temp_files = 0              # Log all temporary files

# Statistics Collection
track_activities = on            # Track running queries
track_counts = on               # Track table statistics
track_io_timing = on            # Track I/O timing
track_functions = all           # Track function calls

# Autovacuum Configuration
autovacuum = on                 # Enable automatic vacuum
autovacuum_max_workers = 3      # Number of autovacuum workers
autovacuum_naptime = 1min       # Time between autovacuum runs
autovacuum_vacuum_threshold = 50 # Minimum number of tuple updates
```

**Optimizations Include**:
- Memory settings optimized for typical LightRAG workloads
- SSD-optimized I/O settings
- Comprehensive logging for monitoring and debugging
- Autovacuum tuned for high-write workloads
- Performance monitoring enabled

## Environment Configuration

### Required Environment Variables

```bash
# Basic PostgreSQL Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=your_username
POSTGRES_PASSWORD='your_password'
POSTGRES_DATABASE=your_database
POSTGRES_MAX_CONNECTIONS=12

# Optional Workspace Configuration
# POSTGRES_WORKSPACE=forced_workspace_name
```

### SSL Configuration (Optional)

```bash
# SSL Modes: disable, allow, prefer, require, verify-ca, verify-full
POSTGRES_SSL_MODE=require

# Certificate Paths (for verify-ca and verify-full modes)
POSTGRES_SSL_CERT=/path/to/client-cert.pem
POSTGRES_SSL_KEY=/path/to/client-key.pem
POSTGRES_SSL_ROOT_CERT=/path/to/ca-cert.pem
POSTGRES_SSL_CRL=/path/to/crl.pem
```

### SSL Modes Explained

- `disable` - No SSL connection attempted
- `allow` - Try SSL connection, fallback to non-SSL if SSL fails
- `prefer` - Try SSL first, fallback to non-SSL if SSL unavailable
- `require` - Require SSL connection, fail if SSL unavailable
- `verify-ca` - Require SSL and verify CA certificate
- `verify-full` - Require SSL, verify CA certificate and hostname

## Docker Integration

### Development Environment

The project includes a complete Docker Compose setup using the specialized `shangor/postgres-for-rag:v1.0` image:

```yaml
services:
  postgres:
    container_name: lightrag_postgres_dev
    image: shangor/postgres-for-rag:v1.0
    restart: unless-stopped
    environment:
      POSTGRES_USER: ${POSTGRES_USER:-rag}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-rag}
      POSTGRES_DATABASE: ${POSTGRES_DATABASE:-rag}
    ports:
      - "${POSTGRES_PORT:-5432}:5432"
    volumes:
      - postgres_dev_data:/var/lib/postgresql/data
      - ./postgres/init:/docker-entrypoint-initdb.d:ro
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -h localhost -p 5432"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 30s
```

**Key Features**:
- Uses specialized Docker image optimized for RAG workloads
- Automatic initialization script execution
- Health checks for dependency management
- Persistent volume for data retention
- Read-only mount for initialization scripts

### Production Environment

```bash
# Start production stack with security hardening
docker compose -f docker-compose.production.yml up -d

# Monitor PostgreSQL logs
docker compose -f docker-compose.production.yml logs -f postgres

# Check health status
docker compose -f docker-compose.production.yml exec postgres pg_isready
```

## Code Implementation Details

### Connection Management

The PostgreSQL integration uses a sophisticated connection pool manager:

```python
class PostgreSQLDB:
    def __init__(self, config: dict[str, Any], **kwargs: Any):
        self.host = config["host"]
        self.port = config["port"]
        self.user = config["user"]
        self.password = config["password"]
        self.database = config["database"]
        self.workspace = config["workspace"]
        self.max = int(config["max_connections"])

        # SSL configuration support
        self.ssl_mode = config.get("ssl_mode")
        self.ssl_cert = config.get("ssl_cert")
        # ... additional SSL parameters
```

### SSL Context Creation

Robust SSL context creation with comprehensive certificate handling:

```python
def _create_ssl_context(self) -> ssl.SSLContext | None:
    """Create SSL context based on configuration parameters."""
    if ssl_mode in ["verify-ca", "verify-full"]:
        context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)

        # Configure certificate verification
        if ssl_mode == "verify-ca":
            context.check_hostname = False
        elif ssl_mode == "verify-full":
            context.check_hostname = True

        # Load certificates and handle errors gracefully
        # ... certificate loading logic
```

### Storage Backend Implementations

#### PGKVStorage - Key-Value Storage
- Handles document chunks and LLM response caching
- Workspace isolation support
- Optimized batch operations
- JSON data serialization

#### PGVectorStorage - Vector Operations
- pgvector integration for similarity search
- Configurable similarity metrics (cosine, L2, inner product)
- Batch vector upserts with error handling
- Automatic index creation and optimization

#### PGGraphStorage - Knowledge Graph
- Apache AGE integration for graph operations
- Entity and relationship management
- Graph traversal and pattern matching
- Cypher query support

#### PGDocStatusStorage - Document Status Tracking
- Processing status management
- Atomic status updates
- Batch status queries
- Workspace-aware document tracking

## Usage Examples

### Basic Setup

```python
from lightrag import LightRAG
from lightrag.kg import PGKVStorage, PGVectorStorage, PGGraphStorage, PGDocStatusStorage

# Initialize LightRAG with PostgreSQL storage
rag = LightRAG(
    working_dir="./rag_storage",
    kv_storage=PGKVStorage.from_dict({}),
    vector_db_storage=PGVectorStorage.from_dict({}),
    graph_storage=PGGraphStorage.from_dict({}),
    doc_status_storage=PGDocStatusStorage.from_dict({})
)

# Initialize storage connections
await rag.initialize_storages()

# Use LightRAG normally
await rag.ainsert("Your document content here")
result = await rag.aquery("Your question here", param="hybrid")

# Clean up connections
await rag.finalize_storages()
```

### Production Configuration with SSL

```python
import os
from lightrag import LightRAG

# Production environment with SSL
os.environ.update({
    "POSTGRES_HOST": "production-postgres-host",
    "POSTGRES_PORT": "5432",
    "POSTGRES_USER": "lightrag_prod_user",
    "POSTGRES_PASSWORD": "highly_secure_password",
    "POSTGRES_DATABASE": "lightrag_production",
    "POSTGRES_MAX_CONNECTIONS": "25",
    "POSTGRES_SSL_MODE": "verify-full",
    "POSTGRES_SSL_ROOT_CERT": "/certs/ca-cert.pem"
})

# Initialize with enhanced security
rag = LightRAG(
    working_dir="/app/production_storage",
    kv_storage="PGKVStorage",
    vector_db_storage="PGVectorStorage",
    graph_storage="PGGraphStorage",
    doc_status_storage="PGDocStatusStorage"
)
```

## Performance Optimization

### Database Indexing Strategy

The PostgreSQL implementation automatically creates optimized indexes:

```sql
-- Primary key indexes (automatic)
CREATE UNIQUE INDEX ON kv_storage (id, workspace);
CREATE UNIQUE INDEX ON vector_storage (id, workspace);

-- Vector similarity indexes (automatic via pgvector)
CREATE INDEX ON vector_storage USING ivfflat (vector vector_cosine_ops)
WITH (lists = 100);

-- Graph traversal indexes (automatic via AGE)
CREATE INDEX ON ag_vertex (properties);
CREATE INDEX ON ag_edge (start_id, end_id);
```

### Connection Pool Optimization

```python
# Optimized connection parameters
connection_params = {
    "min_size": 1,              # Minimum connections
    "max_size": self.max,       # Maximum connections (configurable)
    "command_timeout": 60,      # Command timeout
    "server_settings": {
        "jit": "off",           # Disable JIT for consistent performance
        "application_name": "lightrag"
    }
}
```

### Memory and Performance Settings

The included `postgresql.conf` optimizes for typical LightRAG workloads:

- **Memory**: Tuned for embedding storage and vector operations
- **I/O**: Optimized for SSD storage with parallel operations
- **Checkpointing**: Balanced for write performance and recovery time
- **Logging**: Comprehensive but performance-conscious

## Monitoring and Maintenance

### Health Monitoring

```bash
# Check database status
docker compose exec postgres pg_isready -h localhost -p 5432

# Monitor active connections
docker compose exec postgres psql -U lightrag -d lightrag -c \
  "SELECT count(*) as active_connections FROM pg_stat_activity WHERE state = 'active';"

# Check extension status
docker compose exec postgres psql -U lightrag -d lightrag -c "\dx"
```

### Performance Monitoring

```bash
# Top queries by execution time
docker compose exec postgres psql -U lightrag -d lightrag -c \
  "SELECT query, total_exec_time, calls, mean_exec_time
   FROM pg_stat_statements
   ORDER BY total_exec_time DESC LIMIT 10;"

# Vector storage performance
docker compose exec postgres psql -U lightrag -d lightrag -c \
  "SELECT schemaname, tablename, n_tup_ins, n_tup_upd, n_tup_del
   FROM pg_stat_user_tables
   WHERE tablename LIKE '%vector%';"

# Index usage statistics
docker compose exec postgres psql -U lightrag -d lightrag -c \
  "SELECT indexrelname, idx_tup_read, idx_tup_fetch
   FROM pg_stat_user_indexes
   ORDER BY idx_tup_read DESC;"
```

### Backup and Recovery

#### Automated Backup Script

```bash
#!/bin/bash
# Create timestamped backup
BACKUP_FILE="lightrag_backup_$(date +%Y%m%d_%H%M%S).sql"
docker compose exec postgres pg_dump -U lightrag lightrag > "$BACKUP_FILE"
echo "Backup created: $BACKUP_FILE"
```

#### Point-in-Time Recovery Setup

```bash
# Enable archive mode for point-in-time recovery
docker compose exec postgres psql -U postgres -c \
  "ALTER SYSTEM SET archive_mode = on;"
docker compose exec postgres psql -U postgres -c \
  "ALTER SYSTEM SET archive_command = 'cp %p /archive/%f';"
```

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. Connection Refused Errors

```bash
# Check if PostgreSQL is running
docker compose ps postgres

# Check container logs
docker compose logs postgres

# Test direct connection
docker compose exec postgres psql -U lightrag -d lightrag -c "SELECT version();"
```

#### 2. Extension Installation Issues

```bash
# Verify extensions are available
docker compose exec postgres psql -U lightrag -d lightrag -c "\dx"

# Check for extension errors in logs
docker compose logs postgres | grep -i "extension\|error"

# Manual extension installation
docker compose exec postgres psql -U postgres -d lightrag -c \
  "CREATE EXTENSION IF NOT EXISTS vector;"
```

#### 3. SSL Connection Problems

```bash
# Test SSL connectivity
docker compose exec postgres psql \
  "sslmode=require host=localhost user=lightrag dbname=lightrag"

# Check SSL configuration
docker compose exec postgres psql -U lightrag -d lightrag -c \
  "SHOW ssl;"

# Verify certificate files
docker compose exec postgres ls -la /path/to/certificates/
```

#### 4. Performance Issues

```bash
# Check connection pool usage
docker compose exec postgres psql -U lightrag -d lightrag -c \
  "SELECT count(*), state FROM pg_stat_activity GROUP BY state;"

# Monitor slow queries
docker compose exec postgres psql -U lightrag -d lightrag -c \
  "SELECT query, total_exec_time FROM pg_stat_statements
   WHERE total_exec_time > 1000 ORDER BY total_exec_time DESC;"

# Check table statistics
docker compose exec postgres psql -U lightrag -d lightrag -c \
  "SELECT relname, n_tup_ins, n_tup_upd, n_tup_del, n_dead_tup
   FROM pg_stat_user_tables;"
```

### Advanced Debugging

#### Enable Detailed Logging

```bash
# Temporarily enable detailed logging
docker compose exec postgres psql -U postgres -c \
  "ALTER SYSTEM SET log_min_messages = 'DEBUG1';"
docker compose exec postgres psql -U postgres -c \
  "SELECT pg_reload_conf();"
```

#### Connection Diagnostics

```python
import asyncio
import asyncpg

async def diagnose_connection():
    try:
        conn = await asyncpg.connect(
            host="localhost",
            port=5432,
            user="lightrag",
            password="your_password",
            database="lightrag"
        )

        version = await conn.fetchval("SELECT version()")
        print(f"PostgreSQL Version: {version}")

        extensions = await conn.fetch("SELECT extname FROM pg_extension")
        print(f"Extensions: {[ext['extname'] for ext in extensions]}")

        await conn.close()
        print("Connection successful!")

    except Exception as e:
        print(f"Connection failed: {e}")

# Run diagnostics
asyncio.run(diagnose_connection())
```

## Migration and Upgrade Guide

### From Other Storage Backends

1. **Backup Current Data**:
   ```bash
   tar -czf lightrag_backup_$(date +%Y%m%d).tar.gz ./rag_storage/
   ```

2. **Prepare PostgreSQL Environment**:
   ```bash
   cp env.example .env
   # Edit PostgreSQL configuration in .env
   docker compose up -d postgres
   ```

3. **Update Storage Configuration**:
   ```python
   # Update storage backend configuration
   rag = LightRAG(
       kv_storage="PGKVStorage",
       vector_db_storage="PGVectorStorage",
       graph_storage="PGGraphStorage",
       doc_status_storage="PGDocStatusStorage"
   )
   ```

4. **Re-index Documents**:
   ```python
   # Documents will be automatically reprocessed
   await rag.initialize_storages()
   # Previous document content will be reprocessed with new storage
   ```

### PostgreSQL Version Upgrades

```bash
# Create backup before upgrade
docker compose exec postgres pg_dumpall -U postgres > full_backup.sql

# Update Docker image version
# Edit docker-compose.yml to use newer PostgreSQL version

# Rebuild and start
docker compose down
docker compose up -d postgres

# Verify upgrade
docker compose exec postgres psql -U lightrag -d lightrag -c "SELECT version();"
```

## Security Best Practices

### Database Security

1. **Strong Authentication**:
   ```bash
   # Use complex passwords
   POSTGRES_PASSWORD='ComplexP@ssw0rd123!@#'

   # Enable SSL for all connections
   POSTGRES_SSL_MODE=require
   ```

2. **Network Security**:
   ```yaml
   # Restrict network access
   postgres:
     networks:
       - lightrag-internal
     # Don't expose ports in production
     # ports:
     #   - "5432:5432"
   ```

3. **Privilege Management**:
   ```sql
   -- Create application-specific role
   CREATE ROLE lightrag_app WITH LOGIN PASSWORD 'secure_password';
   GRANT CONNECT ON DATABASE lightrag TO lightrag_app;
   GRANT USAGE, CREATE ON SCHEMA public TO lightrag_app;
   ```

4. **Audit Logging**:
   ```conf
   # Enable comprehensive audit logging
   log_statement = 'all'
   log_connections = on
   log_disconnections = on
   log_hostname = on
   ```

### Container Security

```dockerfile
# Use non-root user in production
USER postgres
# Read-only root filesystem where possible
# Minimal base image
# Regular security updates
```

## Advanced Configuration

### Multi-Workspace Support

```python
# Workspace isolation
workspace_config = {
    "workspace": "project_alpha"  # Isolates data by project
}

rag = LightRAG(
    kv_storage=PGKVStorage.from_dict(workspace_config),
    vector_db_storage=PGVectorStorage.from_dict(workspace_config),
    graph_storage=PGGraphStorage.from_dict(workspace_config),
    doc_status_storage=PGDocStatusStorage.from_dict(workspace_config)
)
```

### Custom Connection Parameters

```python
# Advanced PostgreSQL configuration
advanced_config = {
    "host": "postgres-cluster.internal",
    "port": 5432,
    "user": "lightrag_service",
    "password": "service_password",
    "database": "lightrag_prod",
    "max_connections": 50,
    "ssl_mode": "verify-full",
    "ssl_root_cert": "/certs/ca-bundle.pem",
    "workspace": "production"
}

# Use advanced configuration
rag = LightRAG(
    kv_storage=PGKVStorage.from_dict(advanced_config),
    # ... other storage configurations
)
```

### High Availability Setup

```yaml
# PostgreSQL cluster with replication
services:
  postgres-primary:
    image: shangor/postgres-for-rag:v1.0
    environment:
      - POSTGRES_REPLICATION_MODE=master
      - POSTGRES_REPLICATION_USER=replicator
      - POSTGRES_REPLICATION_PASSWORD=replication_password

  postgres-replica:
    image: shangor/postgres-for-rag:v1.0
    environment:
      - POSTGRES_REPLICATION_MODE=slave
      - POSTGRES_MASTER_HOST=postgres-primary
      - POSTGRES_REPLICATION_USER=replicator
      - POSTGRES_REPLICATION_PASSWORD=replication_password
```

## Resources and References

- [PostgreSQL Official Documentation](https://www.postgresql.org/docs/)
- [pgvector Extension Guide](https://github.com/pgvector/pgvector)
- [Apache AGE Documentation](https://age.apache.org/)
- [asyncpg Driver Documentation](https://magicstack.github.io/asyncpg/)
- [Docker PostgreSQL Images](https://hub.docker.com/_/postgres)
- [LightRAG Production Deployment Guide](./PRODUCTION_DEPLOYMENT_GUIDE.md)
- [Security Hardening Guide](./SECURITY_HARDENING.md)

## Conclusion

The PostgreSQL integration in LightRAG provides a robust, scalable, and production-ready storage solution. With comprehensive initialization scripts, security features, performance optimizations, and monitoring capabilities, it's designed to handle enterprise-scale RAG applications with reliability and efficiency.

The automated setup scripts in the `postgres/` directory ensure consistent deployment across development and production environments, while the flexible configuration options allow customization for specific use cases and security requirements.
