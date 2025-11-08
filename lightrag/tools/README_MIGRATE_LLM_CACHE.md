# LLM Cache Migration Tool - User Guide

## Overview

This tool migrates LightRAG's LLM response cache between different KV storage implementations. It specifically migrates caches generated during file extraction (mode `default`), including entity extraction and summary caches.

## Supported Storage Types

1. **JsonKVStorage** - File-based JSON storage
2. **RedisKVStorage** - Redis database storage
3. **PGKVStorage** - PostgreSQL database storage
4. **MongoKVStorage** - MongoDB database storage

## Cache Types

The tool migrates the following cache types:
- `default:extract:*` - Entity and relationship extraction caches
- `default:summary:*` - Entity and relationship summary caches

**Note**: Query caches (modes like `local`, `global`, etc.) are NOT migrated.

## Prerequisites

### 1. Environment Variable Configuration

Ensure the relevant storage environment variables are configured in your `.env` file:

#### Workspace Configuration (Optional)
```bash
# Generic workspace (shared by all storages)
WORKSPACE=space1

# Or configure independent workspace for specific storage
POSTGRES_WORKSPACE=pg_space
MONGODB_WORKSPACE=mongo_space
REDIS_WORKSPACE=redis_space
```

**Workspace Priority**: Storage-specific > Generic WORKSPACE > Empty string

#### JsonKVStorage
```bash
WORKING_DIR=./rag_storage
```

#### RedisKVStorage
```bash
REDIS_URI=redis://localhost:6379
```

#### PGKVStorage
```bash
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=your_username
POSTGRES_PASSWORD=your_password
POSTGRES_DATABASE=your_database
```

#### MongoKVStorage
```bash
MONGO_URI=mongodb://root:root@localhost:27017/
MONGO_DATABASE=LightRAG
```

### 2. Install Dependencies

Ensure LightRAG and its dependencies are installed:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run from the LightRAG project root directory:

```bash
python -m lightrag.tools.migrate_llm_cache
# or
python lightrag/tools/migrate_llm_cache.py
```

### Interactive Workflow

The tool guides you through the following steps:

#### 1. Select Source Storage Type
```
Supported KV Storage Types:
[1] JsonKVStorage
[2] RedisKVStorage
[3] PGKVStorage
[4] MongoKVStorage

Select Source storage type (1-4): 1
```

#### 2. Source Storage Validation
The tool will:
- Check required environment variables
- Auto-detect workspace configuration
- Initialize and connect to storage
- Count cache records available for migration

```
Checking environment variables...
✓ All required environment variables are set

Initializing Source storage...
- Storage Type: JsonKVStorage
- Workspace: space1
- Connection Status: ✓ Success

Counting cache records...
- default:extract: 8,500 records
- default:summary: 234 records
- Total: 8,734 records
```

#### 3. Select Target Storage Type

Repeat steps 1-2 to select and validate the target storage.

#### 4. Confirm Migration

Review the migration summary and confirm:

```
Migration Confirmation
==================================================
Source: JsonKVStorage (workspace: space1) - 8,734 records
Target: MongoKVStorage (workspace: space1) - 0 records
Batch Size: 1,000 records/batch

⚠ Warning: Target storage already has 0 records
Migration will overwrite records with the same keys

Continue? (y/n): y
```

#### 5. Execute Migration

Observe migration progress:

```
=== Starting Migration ===
Batch 1/9: ████████░░ 1000/8734 (11%) - default:extract
Batch 2/9: ████████████████░░ 2000/8734 (23%) - default:extract
...
Batch 9/9: ████████████████████ 8734/8734 (100%) - default:summary

Persisting data to disk...
```

#### 6. Review Migration Report

The tool provides a comprehensive final report showing statistics and any errors encountered:

**Successful Migration:**
```
Migration Complete - Final Report

📊 Statistics:
  Total source records:    8,734
  Total batches:           9
  Successful batches:      9
  Failed batches:          0
  Successfully migrated:   8,734
  Failed to migrate:       0
  Success rate:            100.00%

✓ SUCCESS: All records migrated successfully!
```

**Migration with Errors:**
```
Migration Complete - Final Report

📊 Statistics:
  Total source records:    8,734
  Total batches:           9
  Successful batches:      8
  Failed batches:          1
  Successfully migrated:   7,734
  Failed to migrate:       1,000
  Success rate:            88.55%

⚠️  Errors encountered: 1

Error Details:
------------------------------------------------------------

Error Summary:
  - ConnectionError: 1 occurrence(s)

First 5 errors:

  1. Batch 2
     Type: ConnectionError
     Message: Connection timeout after 30s
     Records lost: 1,000

⚠️  WARNING: Migration completed with errors!
   Please review the error details above.
```

## Technical Details

### Workspace Handling

The tool retrieves workspace in the following priority order:

1. **Storage-specific workspace environment variables**
   - PGKVStorage: `POSTGRES_WORKSPACE`
   - MongoKVStorage: `MONGODB_WORKSPACE`
   - RedisKVStorage: `REDIS_WORKSPACE`

2. **Generic workspace environment variable**
   - `WORKSPACE`

3. **Default value**
   - Empty string (uses storage's default workspace)

### Batch Migration

- Default batch size: 1000 records/batch
- Avoids memory overflow from loading too much data at once
- Each batch is committed independently, supporting resume capability

### Memory-Efficient Pagination

For large datasets, the tool implements storage-specific pagination strategies:

- **JsonKVStorage**: Direct in-memory access (data already loaded in shared storage)
- **RedisKVStorage**: Cursor-based SCAN with pipeline batching (1000 keys/batch)
- **PGKVStorage**: SQL LIMIT/OFFSET pagination (1000 records/batch)
- **MongoKVStorage**: Cursor streaming with batch_size (1000 documents/batch)

This ensures the tool can handle millions of cache records without memory issues.

### Prefix Filtering Implementation

The tool uses optimized filtering methods for different storage types:

- **JsonKVStorage**: Direct dictionary iteration with lock protection
- **RedisKVStorage**: SCAN command with namespace-prefixed patterns + pipeline for bulk GET
- **PGKVStorage**: SQL LIKE queries with proper field mapping (id, return_value, etc.)
- **MongoKVStorage**: MongoDB regex queries on `_id` field with cursor streaming

## Error Handling & Resilience

The tool implements comprehensive error tracking to ensure transparent and resilient migrations:

### Batch-Level Error Tracking
- Each batch is independently error-checked
- Failed batches are logged but don't stop the migration
- Successful batches are committed even if later batches fail
- Real-time progress shows ✓ (success) or ✗ (failed) for each batch

### Error Reporting
After migration completes, a detailed report includes:
- **Statistics**: Total records, success/failure counts, success rate
- **Error Summary**: Grouped by error type with occurrence counts
- **Error Details**: Batch number, error type, message, and records lost
- **Recommendations**: Clear indication of success or need for review

### No Double Data Loading
- Unlike traditional verification approaches, the tool does NOT reload all target data
- Errors are detected during migration, not after
- This eliminates memory overhead and handles pre-existing target data correctly

## Important Notes

1. **Data Overwrite Warning**
   - Migration will overwrite records with the same keys in the target storage
   - Tool displays a warning if target storage already has data
   - Pre-existing data in target storage is handled correctly

2. **Workspace Consistency**
   - Recommended to use the same workspace for source and target
   - Cache data in different workspaces are completely isolated

3. **Interrupt and Resume**
   - Migration can be interrupted at any time (Ctrl+C)
   - Already migrated data will remain in target storage
   - Re-running will overwrite existing records
   - Failed batches can be manually retried

4. **Performance Considerations**
   - Large data migration may take considerable time
   - Recommend migrating during off-peak hours
   - Ensure stable network connection (for remote databases)
   - Memory usage stays constant regardless of dataset size

## Troubleshooting

### Missing Environment Variables
```
✗ Missing required environment variables: POSTGRES_USER, POSTGRES_PASSWORD
```
**Solution**: Add missing variables to your `.env` file

### Connection Failed
```
✗ Initialization failed: Connection refused
```
**Solutions**:
- Check if database service is running
- Verify connection parameters (host, port, credentials)
- Check firewall settings

**Solutions**:
- Check migration process for error logs
- Re-run migration tool
- Check target storage capacity and permissions

## Example Scenarios

### Scenario 1: JSON to MongoDB Migration

Use case: Migrating from single-machine development to production

```bash
# 1. Configure environment variables
WORKSPACE=production
MONGO_URI=mongodb://user:pass@prod-server:27017/
MONGO_DATABASE=LightRAG

# 2. Run tool
python -m lightrag.tools.migrate_llm_cache

# 3. Select: 1 (JsonKVStorage) -> 4 (MongoKVStorage)
```

### Scenario 2: PostgreSQL Database Switch

Use case: Database migration or upgrade

```bash
# 1. Configure old and new databases
POSTGRES_WORKSPACE=old_db  # Source
# ... Configure new database as default

# 2. Run tool and select same storage type
```

### Scenario 3: Redis to PostgreSQL

Use case: Migrating from cache storage to relational database

```bash
# 1. Ensure both databases are accessible
REDIS_URI=redis://old-redis:6379
POSTGRES_HOST=new-postgres-server
# ... Other PostgreSQL configs

# 2. Run tool
python -m lightrag.tools.migrate_llm_cache

# 3. Select: 2 (RedisKVStorage) -> 3 (PGKVStorage)
```

## Tool Limitations

1. **Only Default Mode Caches**
   - Only migrates `default:extract:*` and `default:summary:*`
   - Query caches are not included

2. **Workspace Isolation**
   - Different workspaces are treated as completely separate
   - Cross-workspace migration requires manual workspace reconfiguration

3. **Network Dependency**
   - Tool requires stable network connection for remote databases
   - Large datasets may fail if connection is interrupted

## Best Practices

1. **Backup Before Migration**
   - Always backup your data before migration
   - Test migration on non-production data first

2. **Verify Results**
   - Check the verification output after migration
   - Manually verify a few cache entries if needed

3. **Monitor Performance**
   - Watch database resource usage during migration
   - Consider migrating in smaller batches if needed

4. **Clean Old Data**
   - After successful migration, consider cleaning old cache data
   - Keep backups for a reasonable period before deletion

## Support

For issues or questions:
- Check LightRAG documentation
- Review error logs for detailed information
- Ensure all environment variables are correctly configured
