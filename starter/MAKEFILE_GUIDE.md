# LightRAG Multi-Tenant Makefile - Complete Usage Guide

## Overview

This Makefile provides a comprehensive, user-friendly interface for managing the LightRAG multi-tenant stack with PostgreSQL backend. All commands are color-coded, well-documented, and designed for both developers and operations teams.

## Design Philosophy

✅ **User-Friendly**
- Clear, descriptive help messages with examples
- Organized by functional groups
- Visual feedback with colors and symbols
- No cryptic abbreviations

✅ **Safe Operations**
- Confirmation prompts for destructive operations
- Clear warnings for data loss
- Incremental operations (setup → up → init-db)
- Easy to understand state management

✅ **Comprehensive**
- Service control (start, stop, restart)
- Database operations (backup, restore, reset)
- Monitoring and health checks
- Testing and debugging utilities

✅ **Production-Ready**
- Resource limits enforced
- Health checks for all services
- Proper error handling
- Logging and monitoring

## Command Structure

### Color Coding
- 🔵 **BLUE** - Headers, sections, informational messages
- 🟢 **GREEN** - Success messages, completion status
- 🟡 **YELLOW** - Warnings, important notes, suggestions
- 🔴 **RED** - Errors, critical warnings, dangerous operations

### Symbol Usage
- 🔧 Setup commands
- 🚀 Start/Launch commands
- 🛑 Stop/Down commands
- 🔄 Restart commands
- 📋 View/List commands
- 🗄️ Database commands
- 🧪 Testing commands
- 🧹 Cleanup commands

## Command Categories

### 1. Initial Setup (Run Once)

```bash
make setup
```

**What it does:**
- Checks if `.env` file exists
- Creates `.env` from `env.template.example` if needed
- Provides instructions for configuration

**When to use:** First time only, before `make up`

**What to do next:**
```bash
# Edit .env with your settings
nano .env

# Then start services
make up
```

---

### 2. Service Control Commands

#### Start All Services
```bash
make up
```
**What it does:**
- Starts PostgreSQL
- Starts Redis
- Starts LightRAG API
- Starts WebUI
- Waits for health checks
- Shows endpoints and next steps

**Output includes:**
- Service endpoints (http://localhost:3000, etc.)
- Confirmation of successful startup
- Suggestions for next steps

**Typical flow:**
```bash
make up
make init-db    # Once services are ready
make status     # Verify everything is running
make api-health # Check API is responding
```

#### Stop All Services
```bash
make down
```
**What it does:**
- Stops all containers
- Preserves data and volumes
- Removes network

**Important:** Data is NOT deleted. Use `make reset` for full cleanup.

#### Restart Services
```bash
make restart
```
**Equivalent to:**
```bash
make down
sleep 2
make up
```

---

### 3. Logging Commands

#### View All Logs
```bash
make logs
```
**Shows:**
- Real-time logs from all services
- Press Ctrl+C to exit

**Useful for:**
- Monitoring startup
- Debugging issues
- Verifying operations

#### View Service-Specific Logs
```bash
make logs-api        # LightRAG API only
make logs-db         # PostgreSQL only
make logs-webui      # WebUI only
```

**Quick debugging:**
```bash
# See API errors
make logs-api | grep -i error

# See database messages
make logs-db | grep -i postgres

# Follow logs from multiple services
make logs | grep "lightrag-api\|postgres"
```

---

### 4. Database Management Commands

#### Initialize Database
```bash
make init-db
```
**Creates:**
- Database: `lightrag_multitenant`
- Schema tables (documents, entities, relations, etc.)
- Sample data (acme-corp, techstart tenants)
- Indexes for performance
- Sample knowledge bases

**Run after:** `make up` completes (wait 30 seconds)

**Idempotent:** Safe to run multiple times (uses `ON CONFLICT`)

#### Connect to Database Shell
```bash
make db-shell
```
**Useful SQL commands:**
```sql
-- List all tables
\dt

-- Show documents table structure
\d documents

-- Count documents by tenant
SELECT tenant_id, COUNT(*) FROM documents GROUP BY tenant_id;

-- Check all tenants
SELECT * FROM tenants;

-- Check knowledge bases
SELECT * FROM knowledge_bases;

-- Exit shell
\q
```

#### Backup Database
```bash
make db-backup
```
**Creates:**
- Directory: `./backups/` (if not exists)
- File: `lightrag_backup_YYYYMMDD_HHMMSS.sql`

**How to restore:**
```bash
make db-restore  # Restores latest backup

# Or manually restore specific backup
psql -U lightrag -d lightrag_multitenant < ./backups/lightrag_backup_20251120_143022.sql
```

#### Restore Database
```bash
make db-restore
```
**Restores:**
- Latest backup from `./backups/` directory
- Overwrites current database

**Warning:** Current data will be lost

#### Reset Database
```bash
make db-reset
```
**⚠️ WARNING - This deletes all data!**

**What it does:**
1. Asks for confirmation (requires typing 'yes')
2. Drops the database
3. Creates new database with fresh schema
4. Creates sample tenants and knowledge bases

**When to use:**
- Starting fresh
- Clearing test data
- Troubleshooting schema issues

---

### 5. Health & Status Commands

#### Check All Service Status
```bash
make status
```
**Shows:**
- Running containers
- Port mappings
- Service health status

**Example output:**
```
CONTAINER ID   IMAGE                    STATUS              PORTS
a1b2c3d4e5f6   postgres:15             Up 2 minutes        5432/tcp
f6e5d4c3b2a1   redis:7                 Up 2 minutes        6379/tcp
b2c3d4e5f6a1   lightrag-api            Up 2 minutes        9621/tcp
c3d4e5f6a1b2   lightrag-webui          Up 2 minutes        3000/tcp
```

#### Check API Health
```bash
make api-health
```
**Tests:**
- HTTP request to `/health` endpoint
- Parses JSON response
- Shows API status

**Expected response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-11-20T14:30:22.123Z",
  "services": {
    "database": "connected",
    "redis": "connected",
    "embedding": "ready"
  }
}
```

**Typical workflow:**
```bash
make up
sleep 30
make api-health  # Check if API is ready
```

---

### 6. Testing Commands

#### Run All Multi-Tenant Tests
```bash
make test
```
**Runs:**
- Multi-tenant backend tests
- Isolation verification
- Data integrity checks

#### Run Tenant Isolation Tests
```bash
make test-isolation
```
**Specifically tests:**
- Tenant data isolation
- Cross-tenant access prevention
- Tenant context enforcement

**Manual test example:**
```bash
# Insert for tenant A
curl -X POST http://localhost:9621/api/v1/insert \
  -H "X-Tenant-Id: acme-corp" \
  -H "X-KB-Id: kb-prod" \
  -d '{"document": "Acme data"}'

# Query as tenant A (should work)
curl "http://localhost:9621/api/v1/query" \
  -H "X-Tenant-Id: acme-corp" \
  -H "X-KB-Id: kb-prod"

# Query as tenant B (should NOT return acme data)
curl "http://localhost:9621/api/v1/query" \
  -H "X-Tenant-Id: techstart" \
  -H "X-KB-Id: kb-main"
```

---

### 7. Cleanup & Maintenance Commands

#### Clean Up Docker Resources
```bash
make clean
```
**Removes:**
- Stopped containers
- Dangling images
- Dangling volumes

**Safe:** Only cleans up unused resources, preserves running services

**Regular maintenance:**
```bash
# After stopping services
make down
make clean

# Then restart
make up
```

#### Full System Reset
```bash
make reset
```
**⚠️ WARNING - Complete data loss!**

**This command:**
1. Shows scary warning message
2. Asks for confirmation ('RESET' in all caps)
3. Stops all containers
4. Removes all containers
5. Deletes all volumes (including database)
6. Removes networks

**When to use:**
- Complete fresh start needed
- Migration to new environment
- Troubleshooting major issues

**Recover from accidental reset:**
```bash
# From backup
make db-restore

# Or manually
psql -U lightrag -d lightrag_multitenant < ./backups/lightrag_backup_*.sql
```

#### System Prune
```bash
make prune
```
**Removes:**
- Unused containers
- Unused images
- Unused volumes
- Unused networks

**More aggressive than `make clean`**

---

### 8. Utility Commands

#### Display Docker Compose Configuration
```bash
make view-compose
```
**Shows:**
- Full docker-compose.yml content
- Useful for understanding service setup
- Good for documentation

#### List Running Services
```bash
make ps
```
**Alias for `make status`** - shows docker-compose ps

---

## Workflow Examples

### Fresh Start Workflow
```bash
# 1. Clone and navigate
cd starter

# 2. Initial setup
make setup

# 3. Edit configuration
nano .env  # Add API keys, adjust settings

# 4. Start services
make up

# 5. Wait and initialize
sleep 10
make init-db

# 6. Verify health
make api-health
make status

# 7. Access application
# Open http://localhost:3000 in browser
```

### Development Workflow
```bash
# Morning: Start services
make up

# During day: Monitor logs
make logs

# Check specific issues
make logs-api

# Before making changes: Backup
make db-backup

# After changes: Verify
make api-health

# Evening: Clean shutdown
make down
```

### Troubleshooting Workflow
```bash
# 1. Check status
make status

# 2. View logs
make logs-api

# 3. Check health
make api-health

# 4. If API is stuck
docker compose -p lightrag-multitenant restart lightrag-api

# 5. If database is stuck
make db-shell
# SELECT 1;
# \q

# 6. Last resort: Full reset
make down
make reset
make setup
make up
make init-db
```

### Backup & Restore Workflow
```bash
# Before major change
make db-backup

# Make changes...

# If problems, restore
make db-restore

# Verify restored data
make db-shell
# SELECT count(*) FROM documents;
# \q
```

---

## Performance Tuning

### Database Optimization
```bash
# Connect to database
make db-shell

# Check query performance
EXPLAIN SELECT * FROM documents WHERE tenant_id='acme-corp' AND kb_id='kb-prod';

# Should show: Index Scan (not Seq Scan)

# Update statistics
ANALYZE;

# Check table sizes
SELECT schemaname, tablename, pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) 
FROM pg_tables 
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

### Memory Usage
```bash
# Check Docker memory usage
docker stats

# Adjust in docker-compose.yml if needed
# deploy:
#   resources:
#     limits:
#       memory: 8G  # Increase from 4G
```

---

## Environment Variable Reference

Key variables in `.env`:

```bash
# Server
PORT=9621
WEBUI_PORT=3000

# Database
POSTGRES_USER=lightrag
POSTGRES_PASSWORD=secure_password
POSTGRES_DATABASE=lightrag_multitenant

# LLM
LLM_BINDING=openai
LLM_MODEL=gpt-4o
LLM_BINDING_API_KEY=sk-...

# Embedding
EMBEDDING_BINDING=ollama
EMBEDDING_MODEL=bge-m3:latest
```

Change any variable and restart services:
```bash
nano .env
make restart
```

---

## Troubleshooting Matrix

| Issue | Check | Solution |
|-------|-------|----------|
| Services won't start | `make status` | `make logs` for error details |
| API not responding | `make api-health` | Wait longer, check `make logs-api` |
| Database errors | `make db-shell` | Run `make init-db` or `make db-reset` |
| Port already in use | `lsof -i :9621` | Change `PORT` in `.env`, restart |
| Memory issues | `docker stats` | Increase resource limits |
| Data loss | Have backup | `make db-restore` from backup |

---

## Color Legend

| Color | Meaning | Example |
|-------|---------|---------|
| 🔵 Blue | Headers, info | Section titles |
| 🟢 Green | Success | "✓ Services started" |
| 🟡 Yellow | Warning | "⚠️ This deletes data" |
| 🔴 Red | Error/Critical | "✗ Connection refused" |

---

## Exit Codes

```bash
# Command succeeded
make status
echo $?  # 0

# Command failed
make up
echo $?  # non-zero (1, 2, etc.)
```

---

## Tips & Tricks

### View logs with grep
```bash
# Show only errors
make logs | grep -i error

# Show only tenant-related logs
make logs | grep -i tenant

# Show last 50 lines
make logs | tail -50
```

### Fast database exports
```bash
# Export to CSV
make db-shell
\COPY (SELECT * FROM documents WHERE tenant_id='acme-corp') TO 'export.csv' CSV HEADER;
\q

# Import from CSV
make db-shell
\COPY documents FROM 'export.csv' CSV HEADER;
\q
```

### Monitor in real-time
```bash
# Split terminal or use tmux
make logs-api &
make logs-db &
make logs-webui &
```

---

## Advanced Usage

### Custom SQL Scripts
```bash
# Run SQL file against database
docker compose -p lightrag-multitenant exec -T postgres psql -U lightrag -d lightrag_multitenant -f script.sql

# Interactive SQL
make db-shell < script.sql
```

### Backup Automation
```bash
# Schedule daily backups with cron
0 2 * * * cd /path/to/starter && make db-backup

# Or with Docker
docker compose exec -T postgres pg_dump -U lightrag lightrag_multitenant > backup.sql
```

### Environment Switching
```bash
# Development environment
cp .env.development .env
make restart

# Production environment
cp .env.production .env
make restart
```

---

## Summary

This Makefile is designed to be:
- **Intuitive** - Clear command names matching their purpose
- **Safe** - Confirmations for destructive operations
- **Observable** - Color output and progress messages
- **Complete** - Covers all common operations
- **Documented** - Built-in help and examples

For additional help:
```bash
make help
```

---

**Last Updated**: November 20, 2025  
**Makefile Version**: 1.0  
**Status**: Production Ready
