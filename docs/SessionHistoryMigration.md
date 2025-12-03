# Session History Migration Guide

## Overview

The session history functionality has been migrated from the standalone `service/` folder into the main LightRAG codebase as an integrated feature. This document explains the changes and migration steps.

## What Changed

### Before (Standalone Service)

- Session history was implemented as a separate service in the `service/` folder
- Required manual setup and configuration
- Separate database connections and initialization
- Required adding service path to sys.path

### After (Integrated Feature)

- Session history is now a built-in feature of LightRAG Server
- Automatically initialized when LightRAG Server starts
- Unified configuration through `.env` file
- Native integration with LightRAG API

## Migration Steps

### 1. Update Dependencies

The session history feature requires SQLAlchemy and PostgreSQL driver:

```bash
# Using uv (recommended)
uv pip install sqlalchemy psycopg2-binary

# Or using pip
pip install sqlalchemy psycopg2-binary
```

### 2. Update Configuration

Move your session database configuration to the main `.env` file:

```bash
# Enable session history feature
SESSION_HISTORY_ENABLED=true

# PostgreSQL configuration for session history
SESSION_POSTGRES_HOST=localhost
SESSION_POSTGRES_PORT=5433
SESSION_POSTGRES_USER=lightrag
SESSION_POSTGRES_PASSWORD=lightrag_password
SESSION_POSTGRES_DATABASE=lightrag_sessions
```

### 3. Update Docker Compose (if using Docker)

The new `docker-compose.yml` includes PostgreSQL service automatically:

```bash
# Stop existing services
docker compose down

# Pull/build new images
docker compose pull
docker compose build

# Start all services
docker compose up -d
```

### 4. API Endpoints

Session history endpoints are under the `/history` prefix:

```
POST   /history/sessions                 - Create session
GET    /history/sessions                 - List sessions
GET    /history/sessions/{id}/history    - Get messages
DELETE /history/sessions/{id}            - Delete session
```

### 5. Remove Old Service Folder

Once migration is complete and tested, you can safely remove the old `service/` folder:

```bash
# Backup first (optional)
mv service service.backup

# Or remove directly
rm -rf service
```

## New Features

The integrated session history includes several improvements:

1. **Automatic Initialization**: Session database is automatically initialized on server startup
2. **Graceful Degradation**: If session database is unavailable, server still starts (without history features)
3. **Better Error Handling**: Improved error messages and logging
4. **User Isolation**: Proper user ID handling via `X-User-ID` header
5. **Session Deletion**: New endpoint to delete sessions and messages

## Configuration Reference

### Configuration

Session history is **always enabled** and uses the same PostgreSQL as LightRAG:

- No environment variables needed
- Session tables created automatically in `POSTGRES_DATABASE`
- Works out of the box when PostgreSQL is configured

That's it - zero configuration!

## Troubleshooting

### Session history not available

**Symptom**: `/history/sessions` endpoints return 404

**Solution**: 
1. Check that `SESSION_HISTORY_ENABLED=true` in `.env`
2. Verify PostgreSQL is running and accessible
3. Check server logs for initialization errors

### Database connection errors

**Symptom**: Server starts but session endpoints fail with database errors

**Solution**:
1. Verify PostgreSQL credentials in `.env`
2. Ensure PostgreSQL is accessible from your network
3. Check PostgreSQL logs for connection issues
4. For Docker: ensure `session-db` container is running

### Migration from old service

**Symptom**: Want to preserve existing session data

**Solution**:
The database schema is compatible. Point `SESSION_DATABASE_URL` to your existing PostgreSQL database and the tables will be reused.

## API Examples

### Create a Session

```python
import requests

response = requests.post(
    "http://localhost:9621/history/sessions",
    json={"title": "Research Session"},
    headers={"X-User-ID": "user@example.com"}
)
print(response.json())
```

### List Sessions

```python
response = requests.get(
    "http://localhost:9621/history/sessions",
    headers={"X-User-ID": "user@example.com"}
)
print(response.json())
```

### Get Session History

```python
session_id = "..."  # UUID from create session
response = requests.get(
    f"http://localhost:9621/history/sessions/{session_id}/history"
)
print(response.json())
```

### Delete Session

```python
response = requests.delete(
    f"http://localhost:9621/history/sessions/{session_id}",
    headers={"X-User-ID": "user@example.com"}
)
print(response.status_code)  # 204 on success
```

## Support

For issues or questions:
- Check the main [README.md](../README.md)
- Review [LightRAG Server documentation](../lightrag/api/README.md)
- Open an issue on [GitHub](https://github.com/HKUDS/LightRAG/issues)

