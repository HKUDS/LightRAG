# LightRAG Local Development Guide

This guide explains how to run LightRAG in **hybrid development mode**:
- **PostgreSQL + Redis** run in Docker containers
- **API Server + WebUI** run natively on your machine

This setup provides the best development experience with hot-reloading for the WebUI and easy debugging of the Python API.

## 🚀 Quick Start

```bash
# Start everything with one command
make dev

# Or use the script directly
./dev-start.sh
```

## 📋 Prerequisites

- **Docker** - For PostgreSQL and Redis
- **Python 3.10+** - For the API server
- **Bun or npm** - For the WebUI (Bun preferred for speed)
- **curl** - For health checks

## ⚙️ Configuration

All configuration is read from the `.env` file at the project root.

### Key Settings

```bash
# Database (Docker containers will use these)
POSTGRES_HOST=localhost
POSTGRES_PORT=15432
POSTGRES_USER=lightrag
POSTGRES_PASSWORD=lightrag123
POSTGRES_DATABASE=lightrag_multitenant

# Redis
REDIS_URI=redis://localhost:16379

# Authentication
AUTH_USER=admin
AUTH_PASS=admin123

# LLM Configuration (OpenAI example)
LLM_BINDING=openai
LLM_MODEL=gpt-4o-mini
LLM_BINDING_API_KEY=sk-your-key-here

# Embedding Configuration
EMBEDDING_BINDING=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIM=1536
EMBEDDING_BINDING_API_KEY=sk-your-key-here
```

## 📡 Service URLs

When the stack is running:

| Service | URL |
|---------|-----|
| **WebUI** | http://localhost:5173 |
| **API Server** | http://localhost:9621 |
| **API Documentation** | http://localhost:9621/docs |
| **Health Check** | http://localhost:9621/health |

## 🔐 Login Credentials

Default development credentials (configurable in `.env`):

- **Username:** `admin`
- **Password:** `admin123`

## 🛠️ Available Commands

### Full Stack (Recommended)

```bash
make dev              # Start everything
You can also auto-confirm killing any processes/containers occupying dev ports (dangerous in shared environments):

```bash
make dev CONFIRM_KILL=yes   # or
./dev-start.sh --yes
```

The script will then stop containers/processes using the dev ports before starting.

make dev-stop         # Stop everything
make dev-status       # Check what's running
make dev-logs         # View all logs
make dev-logs-api     # View API logs only
make dev-logs-webui   # View WebUI logs only
```

### Database Only

If you want to run only the databases (e.g., for custom API development):

```bash
make db-only          # Start PostgreSQL + Redis
make db-stop          # Stop databases
make db-shell         # Connect to PostgreSQL CLI
make db-logs          # View database logs
make clean-db         # Delete all data (⚠️ destructive!)
```

### Setup & Utilities

```bash
make install          # Install all dependencies
make test             # Run tests
make lint             # Run linters
```

## 📁 Files Created

| File | Purpose |
|------|---------|
| `dev-start.sh` | Main startup script |
| `dev-stop.sh` | Graceful shutdown script |
| `dev-status.sh` | Status check script |
| `docker-compose.dev-db.yml` | Docker Compose for databases only |
| `Makefile` | Convenient make commands |

## 🔄 How It Works

1. **Docker Compose** starts PostgreSQL (with AGE graph extension) and Redis
2. **Python** runs the LightRAG API server natively
3. **Bun/npm** runs the Vite dev server for the WebUI with hot-reload

This architecture means:
- Database data persists in Docker volumes
- API changes can be tested with a simple restart
- WebUI changes are reflected immediately (hot-reload)

## 🐛 Troubleshooting

### Port Already in Use

```bash
# Check what's using a port
lsof -i :9621
lsof -i :5173

# Kill the process
kill -9 <PID>

# Or let the scripts handle it
make dev-stop
make dev
```

### Database Connection Issues

```bash
# Check if PostgreSQL is running
docker ps | grep postgres

# View PostgreSQL logs
docker logs lightrag-dev-postgres

# Connect directly
docker exec -it lightrag-dev-postgres psql -U lightrag -d lightrag_multitenant
```

### WebUI Not Loading

```bash
# Check WebUI logs
tail -f /tmp/lightrag-dev-webui.log

# Reinstall dependencies
cd lightrag_webui && bun install
```

### API Server Crashes

```bash
# Check API logs
tail -f /tmp/lightrag-dev-api.log

# Common issues:
# - Missing LLM_BINDING_API_KEY in .env
# - Database not ready yet (wait a few seconds)
# - Python dependencies missing (run: pip install -e .)
```

## 🔧 Advanced: Manual Start

If you prefer to run services manually:

```bash
# 1. Start databases
docker compose -f docker-compose.dev-db.yml up -d

# 2. Start API server (in one terminal)
python -m lightrag.api.lightrag_server --host 0.0.0.0 --port 9621

# 3. Start WebUI (in another terminal)
cd lightrag_webui
VITE_API_BASE_URL=http://localhost:9621 bun run dev
```

## 📊 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Your Machine                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐        ┌─────────────────────────────────┐ │
│  │   WebUI (Vite)  │◄──────►│      LightRAG API Server       │ │
│  │   localhost:5173│        │      localhost:9621            │ │
│  │   (Native)      │        │      (Native Python)           │ │
│  └─────────────────┘        └──────────────┬──────────────────┘ │
│                                            │                     │
│  ┌─────────────────────────────────────────┴───────────────────┐ │
│  │                     Docker                                   │ │
│  │  ┌─────────────────────┐    ┌─────────────────────┐         │ │
│  │  │    PostgreSQL       │    │       Redis         │         │ │
│  │  │    localhost:15432  │    │    localhost:16379  │         │ │
│  │  │    + AGE Graph Ext  │    │                     │         │ │
│  │  └─────────────────────┘    └─────────────────────┘         │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 🆚 Comparison with Other Setups

| Feature | `make dev` (Hybrid) | `starter/make up` (Full Docker) | `scripts/start-dev-stack.sh` |
|---------|---------------------|----------------------------------|------------------------------|
| Databases | Docker | Docker | Docker |
| API Server | Native | Docker | Native |
| WebUI | Native | Docker | Native |
| Hot Reload | ✅ Yes | ❌ No | ✅ Yes |
| Uses root .env | ✅ Yes | ❌ Uses starter/.env | ❌ Hardcoded |
| Easy debugging | ✅ Yes | ❌ Harder | ✅ Yes |

## 🔀 Multi-Tenant Development & Testing

LightRAG supports multi-tenant operation with workspace isolation. Here's how to test the multi-tenant features:

### Running Multi-Tenant Tests

```bash
# Run the isolation test suite
./e2e/run_isolation_test.sh

# Run specific multi-tenant tests
python -m pytest tests/test_idempotency.py -v
python -m pytest e2e/test_multitenant_isolation.py -v
```

### Testing Tenant State Management

The WebUI includes state management features for multi-tenant UX:

1. **URL State Sync**: Navigate to documents with URL parameters:
   ```
   http://localhost:5173/documents#page=2&filter=processed
   ```

2. **Session Persistence**: Tenant selection persists in session storage
   - Switch between tenants without losing document page state
   - "Last selected" hint shows on tenant selection page

3. **Cross-Tab Sync**: Changes propagate across browser tabs via storage events

### Testing Idempotent Document Ingestion

```bash
# Test with curl - first insertion
curl -X POST "http://localhost:9621/documents/text" \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: test-tenant" \
  -H "X-KB-ID: kb-1" \
  -H "Authorization: Bearer your-token" \
  -d '{
    "text": "Document content here",
    "external_id": "doc-unique-id-123"
  }'

# Second insertion with same external_id returns existing document (no duplicate)
curl -X POST "http://localhost:9621/documents/text" \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: test-tenant" \
  -H "X-KB-ID: kb-1" \
  -H "Authorization: Bearer your-token" \
  -d '{
    "text": "Different content - will be ignored",
    "external_id": "doc-unique-id-123"
  }'
```

### Multi-Tenant Architecture

See [Multi-Tenant UX Documentation](archives/0004-multi-tenant-ux-state-management.md) for:
- State management architecture
- URL synchronization patterns
- Idempotency implementation details
- API examples

## 📝 Notes

- Log files are stored in `/tmp/lightrag-dev-*.log`
- PID files are stored in `/tmp/lightrag-dev-*.pid`
- Database data is persisted in Docker volumes (`lightrag_dev_postgres_data`, `lightrag_dev_redis_data`)
