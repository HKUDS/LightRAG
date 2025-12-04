# LightRAG Development Stack Scripts

Quick start, stop, and clean scripts for the LightRAG multi-tenant development environment.

## Quick Start

```bash
# Start all services
bash scripts/start-dev-stack.sh

# Stop all services (data persists)
bash scripts/stop-dev-stack.sh

# Complete cleanup (removes all data)
bash scripts/clean-dev-stack.sh
```

## Scripts Overview

### 🚀 `start-dev-stack.sh`

Starts the complete development stack with all services.

**What it does:**
1. Starts Docker containers (PostgreSQL + Redis)
2. Waits for PostgreSQL to be ready
3. Starts LightRAG API Server on port 9621
4. Starts React WebUI dev server on port 5173

**Usage:**
```bash
bash scripts/start-dev-stack.sh
```

**Output:**
```
✅ Stack Started Successfully!

Services Running:
  • PostgreSQL:  localhost:5433 (lightrag_audit)
  • Redis:       localhost:6380
  • API:         http://localhost:9621
  • WebUI:       http://localhost:5173

Useful Links:
  • API Docs:    http://localhost:9621/docs
  • OpenAPI:     http://localhost:9621/openapi.json
```

**Features:**
- Automatic PostgreSQL health check
- Automatic API health check
- Automatic WebUI readiness check
- Saves process IDs for later cleanup
- Creates log files in `/tmp/lightrag-*.log`

---

### 🛑 `stop-dev-stack.sh`

Gracefully stops all services while preserving data.

**What it does:**
1. Stops API Server
2. Stops WebUI Server
3. Stops Docker containers
4. Preserves all data in Docker volumes

**Usage:**
```bash
bash scripts/stop-dev-stack.sh
```

**Note:** All data is preserved in Docker volumes. To completely reset, use `clean-dev-stack.sh`.

---

### 🧹 `clean-dev-stack.sh`

Completely removes all containers, volumes, and data. **WARNING: Destructive operation!**

**What it does:**
1. Confirms action (requires "yes" response)
2. Kills all background processes
3. Stops Docker containers
4. Removes Docker volumes
5. Clears log files
6. Cleans local storage (rag_storage directory)

**Usage:**
```bash
bash scripts/clean-dev-stack.sh
```

**Warning:**
- 🚨 ALL DATABASE DATA WILL BE DELETED
- 🚨 ALL LOCAL RAG STORAGE WILL BE DELETED
- 🚨 Requires confirmation before proceeding

---

## Service Endpoints

Once started, the following services are available:

### API Server
- **Base URL:** `http://localhost:9621`
- **API Documentation:** `http://localhost:9621/docs`
- **OpenAPI Schema:** `http://localhost:9621/openapi.json`
- **Health Check:** `http://localhost:9621/health`
- **Login:** `POST http://localhost:9621/login`

### WebUI
- **URL:** `http://localhost:5173`
- **Framework:** React + Vite

### Databases
- **PostgreSQL:** `localhost:5433` (default database: `lightrag_audit`)
  - User: `lightrag`
  - Password: `lightrag123`
- **Redis:** `localhost:6380`

### Authentication
- **Default Admin User:** `admin`
- **Default Admin Password:** `admin123`

---

## Configuration

The scripts use these environment variables (all set automatically):

```bash
POSTGRES_HOST=localhost
POSTGRES_PORT=5433
POSTGRES_USER=lightrag
POSTGRES_PASSWORD=lightrag123
POSTGRES_DATABASE=lightrag_audit
REDIS_HOST=localhost
REDIS_PORT=6380
LIGHTRAG_MULTI_TENANT_STRICT=true
LIGHTRAG_REQUIRE_USER_AUTH=true
AUTH_USER=admin
AUTH_PASS=admin123
LLM_BINDING=ollama
LLM_BINDING_HOST=http://localhost:11434
EMBEDDING_BINDING=ollama
EMBEDDING_BINDING_HOST=http://localhost:11434
```

---

## Viewing Logs

After starting the stack, view logs with:

```bash
# API Server logs
tail -f /tmp/lightrag-api.log

# WebUI logs
tail -f /tmp/lightrag-webui.log

# Docker PostgreSQL logs
docker logs lightrag-audit-postgres

# Docker Redis logs
docker logs lightrag-audit-redis
```

---

## Troubleshooting

### API Server won't start
```bash
# Check if port 9621 is already in use
lsof -i :9621

# View full API logs
cat /tmp/lightrag-api.log

# Kill any orphaned processes
pkill -9 -f lightrag.api.lightrag_server
```

### PostgreSQL connection error
```bash
# Check PostgreSQL is running
docker ps | grep lightrag-audit-postgres

# Check PostgreSQL logs
docker logs lightrag-audit-postgres

# Restart PostgreSQL container
docker restart lightrag-audit-postgres
```

### WebUI not loading
```bash
# Check if port 5173 is already in use
lsof -i :5173

# View WebUI logs
cat /tmp/lightrag-webui.log

# Ensure Node.js dependencies are installed
cd lightrag_webui && npm install
```

### Docker volume cleanup issues
```bash
# List all volumes
docker volume ls

# Remove specific volume manually
docker volume rm lightrag_postgres_audit_data

# Remove all unused volumes
docker volume prune
```

---

## Process Management

The scripts save process IDs for clean shutdown:

- **API Server PID:** `/tmp/lightrag-api.pid`
- **WebUI PID:** `/tmp/lightrag-webui.pid`

These are automatically used by the stop script for graceful shutdown.

---

## Monitoring

Check running processes:
```bash
# List all LightRAG processes
ps aux | grep -E "lightrag|npm run dev" | grep -v grep

# Check specific ports
netstat -an | grep -E "9621|5173|5433|6380"
```

---

## Requirements

- Docker and Docker Compose
- Python 3.10+
- Node.js 18+
- npm or yarn
- macOS/Linux (for bash scripts)
