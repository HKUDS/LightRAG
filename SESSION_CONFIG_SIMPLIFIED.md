# Session History Configuration - Simplified Approach

## Summary of Changes

Based on user feedback, the session history configuration has been **simplified** to avoid unnecessary complexity.

## What Changed

### Before (Over-complicated)
```bash
# Required separate PostgreSQL configuration
SESSION_POSTGRES_HOST=localhost
SESSION_POSTGRES_PORT=5433
SESSION_POSTGRES_USER=lightrag
SESSION_POSTGRES_PASSWORD=lightrag_password
SESSION_POSTGRES_DATABASE=lightrag_sessions
```
- ❌ Required users to configure separate database
- ❌ More environment variables to manage
- ❌ Confusion about when to use which settings

### After (Simplified)
```bash
# Just enable - uses existing PostgreSQL automatically
SESSION_HISTORY_ENABLED=true
```
- ✅ Uses existing `POSTGRES_*` configuration by default
- ✅ Minimal configuration needed
- ✅ Session tables created in same database as LightRAG
- ✅ Still allows separate database if needed (optional)

## Configuration Logic

The system now follows this priority order:

1. **`SESSION_DATABASE_URL`** (if set) - Full custom connection string
2. **`SESSION_POSTGRES_*`** (if set) - Override for separate database
3. **`POSTGRES_*`** (default) - Shared with LightRAG ✨ **RECOMMENDED**

## Use Cases

### 99% of Users (Recommended)
```bash
# In .env - just enable it!
SESSION_HISTORY_ENABLED=true

# Session tables will be created in POSTGRES_DATABASE automatically
# No additional configuration needed
```

**Result**: 
- Session tables: `lightrag_chat_sessions_history`, `lightrag_chat_messages_history`, `lightrag_message_citations_history`
- Created in the same PostgreSQL database as LightRAG storage
- Uses existing PostgreSQL connection settings

### Advanced Users (Separate Database)
```bash
SESSION_HISTORY_ENABLED=true

# Only if you REALLY need separate database
SESSION_POSTGRES_HOST=other-host
SESSION_POSTGRES_DATABASE=dedicated_sessions_db
```

## Docker Compose Changes

### Simplified (Default)
```yaml
services:
  lightrag:
    # ... existing config
    # No session-db dependency needed!
```

The separate `session-db` service is now **commented out** in `docker-compose.yml` since most users don't need it.

### If You Need Separate Database
Uncomment the `session-db` service in `docker-compose.yml`.

## Benefits

1. **Simpler Setup**: One less thing to configure
2. **Fewer ENV Variables**: Less confusion about what to set
3. **Easier Docker**: No need for separate database container in most cases
4. **Better Defaults**: Works out of the box with existing PostgreSQL
5. **Still Flexible**: Can override if needed for advanced use cases

## Migration from Old Config

If you already have `SESSION_POSTGRES_*` set in your `.env`:

**Option 1: Simplify (Recommended)**
```bash
# Remove these lines from .env
# SESSION_POSTGRES_HOST=...
# SESSION_POSTGRES_PORT=...
# SESSION_POSTGRES_USER=...
# SESSION_POSTGRES_PASSWORD=...
# SESSION_POSTGRES_DATABASE=...

# Keep only this
SESSION_HISTORY_ENABLED=true
```

**Option 2: Keep Separate Database**
```bash
# Keep your SESSION_POSTGRES_* settings if you need separate database
SESSION_HISTORY_ENABLED=true
SESSION_POSTGRES_HOST=other-host
# ... other settings
```

## Database Tables

Whether you use shared or separate PostgreSQL, these tables are created:

| Table | Purpose |
|-------|---------|
| `lightrag_chat_sessions_history` | Chat sessions |
| `lightrag_chat_messages_history` | Individual messages |
| `lightrag_message_citations_history` | Source citations |

## Why This Makes Sense

1. **Most users have ONE PostgreSQL instance** - No need to run multiple
2. **Session data is not that large** - Doesn't need separate database
3. **Simpler is better** - Follows principle of least configuration
4. **Still allows separation** - When needed for production/security reasons

## Example Scenarios

### Scenario 1: Development/Testing
```bash
# .env
POSTGRES_HOST=localhost
POSTGRES_DATABASE=lightrag_dev
SESSION_HISTORY_ENABLED=true
```
✅ Everything in one database, easy to reset/cleanup

### Scenario 2: Production (Simple)
```bash
# .env
POSTGRES_HOST=prod-db.example.com
POSTGRES_DATABASE=lightrag_prod
SESSION_HISTORY_ENABLED=true
```
✅ Production database with both LightRAG and session data

### Scenario 3: Production (Separated)
```bash
# .env
POSTGRES_HOST=prod-db.example.com
POSTGRES_DATABASE=lightrag_data

SESSION_POSTGRES_HOST=sessions-db.example.com
SESSION_POSTGRES_DATABASE=sessions
```
✅ Separate databases for data isolation (if required by architecture)

## Implementation Details

The fallback logic in `session_database.py`:

```python
# Uses 'or' instead of nested getenv for clarity
self.host = os.getenv("SESSION_POSTGRES_HOST") or os.getenv("POSTGRES_HOST", "localhost")
self.port = os.getenv("SESSION_POSTGRES_PORT") or os.getenv("POSTGRES_PORT", "5432")
# ... etc
```

This means:
- If `SESSION_POSTGRES_HOST` is set → use it
- If not set or empty → fallback to `POSTGRES_HOST`
- If that's also not set → use default "localhost"

## Logging

The system logs which configuration is being used:

```
INFO: Session database: shared with LightRAG at localhost:5432/lightrag_db
```
or
```
INFO: Session database: separate instance at sessions-host:5433/sessions_db
```
or
```
INFO: Session database: custom URL
```

## Conclusion

By defaulting to shared PostgreSQL configuration, we've made session history:
- ✅ Easier to set up
- ✅ Less confusing
- ✅ More intuitive
- ✅ Still flexible when needed

**Bottom line**: Just set `SESSION_HISTORY_ENABLED=true` and you're done! 🎉

