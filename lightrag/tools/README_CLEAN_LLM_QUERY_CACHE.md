# LLM Query Cache Cleanup Tool - User Guide

## Overview

This tool cleans up LightRAG's LLM query cache from KV storage implementations. It specifically targets query caches generated during RAG query operations (modes: `mix`, `hybrid`, `local`, `global`), including both query and keywords caches.

## Supported Storage Types

1. **JsonKVStorage** - File-based JSON storage
2. **RedisKVStorage** - Redis database storage
3. **PGKVStorage** - PostgreSQL database storage
4. **MongoKVStorage** - MongoDB database storage
5. **OpenSearchKVStorage** - OpenSearch index storage

## Cache Types

The tool cleans up the following query cache types:

### Query Cache Modes (4 types)
- `mix:*` - Mixed mode query caches
- `hybrid:*` - Hybrid mode query caches
- `local:*` - Local mode query caches
- `global:*` - Global mode query caches

### Cache Content Types (2 types)
- `*:query:*` - Query result caches
- `*:keywords:*` - Keywords extraction caches

### Cache Key Format
```
<mode>:<cache_type>:<hash>
```

Examples:
- `mix:query:5ce04d25e957c290216cee5bfe6344fa`
- `mix:keywords:fee77b98244a0b047ce95e21060de60e`
- `global:query:abc123def456...`
- `local:keywords:789xyz...`

**Important Note**: This tool does NOT clean extraction caches (`default:extract:*` and `default:summary:*`). Use the migration tool or manual deletion for those caches.

## Prerequisites

- The tool reads storage configuration from environment variables
- Ensure the target storage is properly configured and accessible
- Backup important data before running cleanup operations

## Usage

### Basic Usage

Run from the LightRAG project root directory:

```bash
python -m lightrag.tools.clean_llm_query_cache
# or
python lightrag/tools/clean_llm_query_cache.py
```

### Interactive Workflow

The tool guides you through the following steps:

#### 1. Select Storage Type
```
============================================================
LLM Query Cache Cleanup Tool - LightRAG
============================================================

=== Storage Setup ===

Supported KV Storage Types:
[1] JsonKVStorage
[2] RedisKVStorage
[3] PGKVStorage
[4] MongoKVStorage
[5] OpenSearchKVStorage

Select storage type (1-5) (Press Enter to exit): 1
```

**Note**: You can press Enter or type `0` at any prompt to exit gracefully.

#### 2. Storage Validation
The tool will:
- Check required environment variables
- Auto-detect workspace configuration
- Initialize and connect to storage
- Verify connection status

```
Checking configuration...
✓ All required environment variables are set

Initializing storage...
- Storage Type: JsonKVStorage
- Workspace: space1
- Connection Status: ✓ Success
```

#### 3. View Cache Statistics

The tool displays a detailed breakdown of query caches by mode and type:

```
Counting query cache records...

📊 Query Cache Statistics (Before Cleanup):
┌────────────┬────────────┬────────────┬────────────┐
│ Mode       │ Query      │ Keywords   │ Total      │
├────────────┼────────────┼────────────┼────────────┤
│ mix        │      1,234 │        567 │      1,801 │
│ hybrid     │        890 │        423 │      1,313 │
│ local      │      2,345 │      1,123 │      3,468 │
│ global     │        678 │        345 │      1,023 │
├────────────┼────────────┼────────────┼────────────┤
│ Total      │      5,147 │      2,458 │      7,605 │
└────────────┴────────────┴────────────┴────────────┘
```

#### 4. Select Cleanup Scope

Choose what type of caches to delete:

```
=== Cleanup Options ===
[1] Delete all query caches (both query and keywords)
[2] Delete query caches only (keep keywords)
[3] Delete keywords caches only (keep query)
[0] Cancel

Select cleanup option (0-3): 1
```

**Cleanup Types:**
- **Option 1 (all)**: Deletes both query and keywords caches across all modes
- **Option 2 (query)**: Deletes only query caches, preserves keywords caches
- **Option 3 (keywords)**: Deletes only keywords caches, preserves query caches

#### 5. Confirm Deletion

Review the cleanup plan and confirm:

```
============================================================
Cleanup Confirmation
============================================================
Storage: JsonKVStorage (workspace: space1)
Cleanup Type: all
Records to Delete: 7,605 / 7,605

⚠️  WARNING: This will delete ALL query caches across all modes!

Continue with deletion? (y/n): y
```

#### 6. Execute Cleanup

The tool deletes the selected cache types in one operation, then persists the change:

```
=== Starting Cleanup ===
💡 Deleting ['keywords', 'query'] caches from JsonKVStorage

✓ Deleted 7,605 records in 0.45s

Persisting changes to storage...
✓ Changes persisted successfully
```

#### 7. Review Cleanup Report

The tool provides a comprehensive final report:

**Successful Cleanup:**
```
============================================================
Cleanup Complete - Final Report
============================================================

📊 Statistics:
  Total records to delete:  7,605
  Total batches:            1
  Successful batches:       1
  Failed batches:           0
  Successfully deleted:     7,605
  Failed to delete:         0
  Success rate:             100.00%

📈 Before/After Comparison:
  Total caches before:      7,605
  Total caches after:       0
  Net reduction:            7,605

============================================================
✓ SUCCESS: All records cleaned up successfully!
============================================================

📊 Query Cache Statistics (After Cleanup):
┌────────────┬────────────┬────────────┬────────────┐
│ Mode       │ Query      │ Keywords   │ Total      │
├────────────┼────────────┼────────────┼────────────┤
│ mix        │          0 │          0 │          0 │
│ hybrid     │          0 │          0 │          0 │
│ local      │          0 │          0 │          0 │
│ global     │          0 │          0 │          0 │
├────────────┼────────────┼────────────┼────────────┤
│ Total      │          0 │          0 │          0 │
└────────────┴────────────┴────────────┴────────────┘
```

**Cleanup with Errors:**
```
============================================================
Cleanup Complete - Final Report
============================================================

📊 Statistics:
  Total records to delete:  7,605
  Total batches:            1
  Successful batches:       0
  Failed batches:           1
  Successfully deleted:     0
  Failed to delete:         7,605
  Success rate:             0.00%

📈 Before/After Comparison:
  Total caches before:      7,605
  Total caches after:       7,605
  Net reduction:            0

⚠️  Errors encountered: 1

Error Details:
------------------------------------------------------------

Error Summary:
  - ConnectionError: 1 occurrence(s)

First 5 errors:

  1. Batch 1
     Type: ConnectionError
     Message: Connection timeout after 30s
     Records lost: 7,605

============================================================
⚠️  WARNING: Cleanup completed with errors!
   Please review the error details above.
============================================================
```

## Technical Details

### Workspace Handling

The tool retrieves workspace in the following priority order:

1. **Storage-specific workspace environment variables**
   - PGKVStorage: `POSTGRES_WORKSPACE`
   - MongoKVStorage: `MONGODB_WORKSPACE`
   - RedisKVStorage: `REDIS_WORKSPACE`
   - OpenSearchKVStorage: `OPENSEARCH_WORKSPACE`

2. **Generic workspace environment variable**
   - `WORKSPACE`

3. **Default value**
   - Empty string (uses storage's default workspace)

### Deletion

The tool deletes through `BaseKVStorage.delete_by_cache_types`, the same
implementation the server's `DELETE /documents/query_cache` route uses, so the two
cannot disagree about what a query cache row is. A row matches when its key parses
as `{mode}:{cache_type}:{hash}` with `cache_type` in the selected types, in any mode.

| Backend | Strategy |
|---|---|
| JsonKVStorage | removes the matching keys from memory under the namespace lock; the tool then persists the file |
| RedisKVStorage | SCANs the namespace and pipelines `DEL` for the matching keys |
| PGKVStorage | one parameterized `DELETE` (`split_part(id, ':', 2) = ANY($2)`) scoped to the workspace |
| MongoKVStorage | one `delete_many` with an anchored `_id` regex |
| OpenSearchKVStorage | scans the index and buffers deletes; the tool then flushes them |

The deletion runs as one operation per backend rather than in client-side batches.
To clear the query cache while the server is running, use the route instead:
`curl -X DELETE "http://localhost:9621/documents/query_cache"`.

## Error Handling & Resilience

The tool implements comprehensive error tracking:

### Error Tracking
- The deletion is error-checked as a whole and reported as one batch
- A failure is logged with full details, and the verification count below
  shows how many rows remain
- The output shows ✓ (success) or ✗ (failed)

### Error Reporting
After cleanup completes, a detailed report includes:
- **Statistics**: Total records, success/failure counts, success rate
- **Before/After Comparison**: Net reduction in cache count
- **Error Summary**: Grouped by error type with occurrence counts
- **Error Details**: Batch number, error type, message, and records lost
- **Recommendations**: Clear indication of success or need for review

### Verification
- Post-cleanup count verification
- Before/after statistics comparison
- Identifies partial cleanup scenarios

## Important Notes

1. **Irreversible Operation**
   - Deleted caches cannot be recovered
   - Always backup important data before cleanup
   - Test on non-production data first

2. **Performance Impact**
   - Query performance may degrade temporarily after cleanup
   - Caches will rebuild on subsequent queries
   - Consider cleanup during off-peak hours

3. **Selective Cleanup**
   - Choose cleanup scope carefully
   - Keywords caches may be valuable for future queries
   - Query caches rebuild faster than keywords caches

4. **Workspace Isolation**
   - Cleanup only affects the selected workspace
   - Other workspaces remain untouched
   - Verify workspace before confirming cleanup

5. **Interrupt and Resume**
   - Cleanup can be interrupted at any time (Ctrl+C)
   - Already deleted records cannot be recovered
   - No automatic resume - must run tool again

## Storage Configuration

The tool supports multiple configuration methods with the following priority:

1. **Environment variables** (highest priority)
2. **Default values** (lowest priority)

### Environment Variable Configuration

Configure storage settings in your `.env` file:

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

#### OpenSearchKVStorage

```bash
OPENSEARCH_HOSTS=localhost:9200
OPENSEARCH_WORKSPACE=search_space
```

If environment variables are not provided, the tool falls back to built-in defaults where available.

## Troubleshooting

### Missing Environment Variables
```
⚠️  Warning: Missing environment variables: POSTGRES_USER, POSTGRES_PASSWORD
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
- Ensure network connectivity for remote databases

### No Caches Found
```
⚠️  No query caches found in storage
```
**Possible Reasons**:
- No queries have been run yet
- Caches were already cleaned
- Wrong workspace selected
- Different storage type was used for queries

### Partial Cleanup
```
⚠️  WARNING: Cleanup completed with errors!
```
**Solutions**:
- Check error details in the report
- Verify storage connection stability
- Re-run tool to clean remaining caches
- Check storage capacity and permissions

## Use Cases

### Use Case 1: Clean All Query Caches

**Scenario**: Free up storage space by removing all query caches

```bash
# Run tool
python -m lightrag.tools.clean_llm_query_cache

# Select: Storage type -> Option 1 (all) -> Confirm (y)
```

**Result**: All query and keywords caches deleted, maximum storage freed

### Use Case 2: Refresh Query Caches Only

**Scenario**: Force query cache rebuild while keeping keywords

```bash
# Run tool
python -m lightrag.tools.clean_llm_query_cache

# Select: Storage type -> Option 2 (query only) -> Confirm (y)
```

**Result**: Query caches deleted, keywords preserved for faster rebuild

### Use Case 3: Clean Stale Keywords

**Scenario**: Remove outdated keywords while keeping recent query results

```bash
# Run tool
python -m lightrag.tools.clean_llm_query_cache

# Select: Storage type -> Option 3 (keywords only) -> Confirm (y)
```

**Result**: Keywords deleted, query caches preserved

### Use Case 4: Workspace-Specific Cleanup

**Scenario**: Clean caches for a specific workspace

```bash
# Configure workspace
export WORKSPACE=development

# Run tool
python -m lightrag.tools.clean_llm_query_cache

# Select: Storage type -> Cleanup option -> Confirm (y)
```

**Result**: Only development workspace caches cleaned

## Best Practices

1. **Backup Before Cleanup**
   - Always backup your storage before major cleanup
   - Test cleanup on non-production data first
   - Document cleanup decisions

2. **Monitor Performance**
   - Watch storage metrics during cleanup
   - Monitor query performance after cleanup
   - Allow time for cache rebuild

3. **Scheduled Cleanup**
   - Clean caches periodically (weekly/monthly)
   - Automate cleanup for development environments
   - Keep production cleanup manual for safety

4. **Selective Deletion**
   - Consider cleanup scope based on needs
   - Keywords caches are harder to rebuild
   - Query caches rebuild automatically

5. **Storage Capacity**
   - Monitor storage usage trends
   - Clean caches before reaching capacity limits
   - Archive old data if needed

## Comparison with Migration Tool

| Feature | Cleanup Tool | Migration Tool |
|---------|-------------|----------------|
| **Purpose** | Delete query caches | Migrate extraction caches |
| **Cache Types** | mix/hybrid/local/global | default:extract/summary |
| **Modes** | query, keywords | extract, summary |
| **Operation** | Deletion | Copy between storages |
| **Reversible** | No | Yes (source unchanged) |
| **Use Case** | Free storage, refresh caches | Change storage backend |

## Limitations

1. **Single Storage Operation**
   - Can only clean one storage type at a time
   - To clean multiple storages, run tool multiple times

2. **No Dry Run Mode**
   - Deletion is immediate after confirmation
   - No preview-only mode available
   - Test on non-production first

3. **No Selective Mode Cleanup**
   - Cannot clean only specific modes (e.g., only `mix`)
   - Cleanup applies to all modes for selected cache type
   - All-or-nothing per cache type

4. **No Scheduled Cleanup**
   - Manual execution required
   - No built-in scheduling
   - Use cron/scheduler if automation needed

5. **Verification Limitations**
   - Post-cleanup verification may fail in error scenarios
   - Manual verification recommended for critical operations

## Future Enhancements

Potential improvements for future versions:

- Selective mode cleanup (e.g., clean only `mix` mode)
- Age-based cleanup (delete caches older than X days)
- Size-based cleanup (delete largest caches first)
- Dry run mode for safe preview
- Automated scheduling support
- Cache statistics export
- Incremental cleanup with pause/resume

## Support

For issues, questions, or feature requests:
- Check the error details in the cleanup report
- Review storage configuration
- Verify workspace settings
- Test with a small dataset first
- Report bugs through project issue tracker
