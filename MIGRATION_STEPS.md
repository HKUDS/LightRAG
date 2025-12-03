# Migration Steps - Session History Integration

## Current Situation

You are on the `session_intergrate` branch which still uses the old `service/` folder approach. The new integration code I created uses `lightrag/api/session_*` modules.

## Quick Fix Applied

I've updated these files to use the new integrated modules:

### 1. `lightrag/api/routers/query_routes.py`
Changed imports from:
```python
from app.core.database import SessionLocal
from app.services.history_manager import HistoryManager
```

To:
```python
from lightrag.api.session_database import SessionLocal, get_db
from lightrag.api.session_manager import SessionHistoryManager
```

### 2. `lightrag/api/session_database.py`
Added SessionLocal alias for backward compatibility:
```python
SessionLocal = lambda: get_session_db_manager().get_session()
```

## Steps to Complete Migration

### 1. Install Dependencies
```bash
cd /d/work/LightRAG
pip install sqlalchemy psycopg2-binary httpx
```

### 2. Configure PostgreSQL
Ensure your `.env` file has PostgreSQL configured:
```bash
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
POSTGRES_DATABASE=lightrag_db
```

### 3. Start PostgreSQL
If using Docker:
```bash
docker run -d --name postgres \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=lightrag_db \
  -p 5432:5432 \
  postgres:16
```

Or use existing PostgreSQL instance.

### 4. Test Server
```bash
cd /d/work/LightRAG
lightrag-server
```

Check logs for:
```
INFO: Session history database initialized successfully
```

### 5. Test Session Endpoints
```bash
# Create a session
curl -X POST http://localhost:9621/history/sessions \
  -H "Content-Type: application/json" \
  -H "X-User-ID: test@example.com" \
  -d '{"title": "Test Session"}'

# List sessions
curl http://localhost:9621/history/sessions \
  -H "X-User-ID: test@example.com"
```

## Troubleshooting

### Error: "Failed to fetch sessions: 500"

**Cause**: PostgreSQL not configured or not running

**Fix**:
1. Check `.env` has `POSTGRES_*` variables
2. Start PostgreSQL
3. Check server logs for database connection errors

### Error: "ModuleNotFoundError: No module named 'httpx'"

**Fix**:
```bash
pip install httpx
```

### Error: "No module named 'sqlalchemy'"

**Fix**:
```bash
pip install sqlalchemy psycopg2-binary
```

### Database Connection Refused

**Fix**:
1. Check PostgreSQL is running:
   ```bash
   # Windows
   tasklist | findstr postgres
   
   # Linux/Mac
   ps aux | grep postgres
   ```

2. Test connection:
   ```bash
   psql -h localhost -U postgres -d lightrag_db
   ```

3. Check firewall not blocking port 5432

## Clean Migration (Recommended)

If you want to start fresh with the new integrated approach:

### 1. Backup Current Work
```bash
git stash save "backup before migration"
```

### 2. Create New Branch
```bash
git checkout -b session-integrated-clean
```

### 3. Apply New Files
Copy all the new files I created:
- `lightrag/api/session_models.py`
- `lightrag/api/session_schemas.py`
- `lightrag/api/session_database.py`
- `lightrag/api/session_manager.py`
- Updated `lightrag/api/routers/history_routes.py`
- Updated `lightrag/api/routers/query_routes.py`
- Updated `lightrag/api/lightrag_server.py`

### 4. Remove Old Service Folder
```bash
mv service service.backup
```

### 5. Test
```bash
lightrag-server
```

## Files Modified

- ✅ `lightrag/api/session_models.py` - NEW
- ✅ `lightrag/api/session_schemas.py` - NEW
- ✅ `lightrag/api/session_database.py` - NEW
- ✅ `lightrag/api/session_manager.py` - NEW
- ✅ `lightrag/api/routers/history_routes.py` - UPDATED
- ✅ `lightrag/api/routers/query_routes.py` - UPDATED
- ✅ `lightrag/api/lightrag_server.py` - UPDATED
- ✅ `docker-compose.yml` - SIMPLIFIED
- ✅ `env.example` - UPDATED
- ✅ `README.md` - UPDATED

## Next Steps

1. Test the integrated version
2. If working, commit the changes
3. Remove old `service/` folder
4. Update documentation
5. Deploy!

## Support

If issues persist:
1. Check all files are properly updated
2. Ensure PostgreSQL is accessible
3. Review server logs
4. Create GitHub issue with logs

