# PostgreSQL Cascade Delete Implementation for LightRAG

## Overview

This document describes the implementation of PostgreSQL cascade delete functionality for individual and batch document deletion in LightRAG. The implementation ensures complete cleanup of document data across all related tables while maintaining data integrity.

## Architecture

### Components

1. **API Endpoints**
   - `DELETE /documents/{doc_id}` - Individual document deletion
   - `DELETE /documents/batch` - Batch document deletion

2. **PostgreSQL Storage Detection**
   - Automatically detects PostgreSQL storage backends
   - Identifies storage classes starting with "PG" (e.g., `PGVectorStorage`, `PGKVStorage`, `PGDocStatusStorage`)

3. **Database Function**
   - `delete_lightrag_document_with_summary(doc_id, file_name)` - PostgreSQL stored function that handles cascade deletion

## Implementation Details

### Storage Detection Logic

The implementation searches through all available storage backends to find a PostgreSQL instance:

```python
storage_backends = [
    rag.chunk_entity_relation_graph,
    rag.entities_vdb,
    rag.relationships_vdb,
    rag.chunks_vdb,
    rag.text_chunks,
    rag.full_docs,
    rag.doc_status
]

for storage in storage_backends:
    if hasattr(storage, '__class__') and ('Postgres' in storage.__class__.__name__ or storage.__class__.__name__.startswith('PG')):
        if hasattr(storage, 'db') and hasattr(storage.db, 'pool'):
            postgres_storage = storage
            break
```

### Database Connection

The implementation uses the PostgreSQL connection pool from the storage backend:

```python
async with postgres_storage.db.pool.acquire() as conn:
    result = await conn.fetch(
        "SELECT * FROM delete_lightrag_document_with_summary($1, $2)",
        doc_id,
        file_name
    )
```

### Deletion Flow

1. **File Deletion**: First attempts to delete the physical file from the input directory
2. **Database Deletion**: 
   - If PostgreSQL is detected: Executes the cascade delete function
   - If PostgreSQL is not found or fails: Falls back to regular `adelete_by_doc_id()` method
3. **Result Reporting**: Returns detailed cleanup statistics from the PostgreSQL function

### Response Format

The deletion endpoints return detailed information about the cleanup operation:

```json
{
    "status": "success",
    "message": "Document 'doc_123' deleted successfully",
    "doc_id": "doc_123",
    "database_cleanup": {
        "entities_updated": 26,
        "entities_deleted": 9,
        "relations_deleted": 27,
        "chunks_deleted": 4,
        "doc_status_deleted": 1,
        "doc_full_deleted": 1
    }
}
```

## PostgreSQL Function

### Function Signature

```sql
CREATE OR REPLACE FUNCTION delete_lightrag_document_with_summary(
    p_doc_id TEXT,
    p_file_name TEXT
) RETURNS TABLE (
    operation TEXT,
    rows_affected INTEGER
)
```

### Function Implementation

CREATE OR REPLACE FUNCTION delete_lightrag_document_with_summary(
    p_doc_id VARCHAR,
    p_file_name VARCHAR
)
RETURNS TABLE (
    operation VARCHAR,
    rows_affected INTEGER
)
LANGUAGE plpgsql
AS $$
DECLARE
    v_chunks_deleted INTEGER;
    v_entity_updated INTEGER;
    v_entity_deleted INTEGER;
    v_relation_deleted INTEGER;
    v_doc_status_deleted INTEGER;
    v_doc_full_deleted INTEGER;
BEGIN
    -- 1. Update multi-file entities FIRST
    UPDATE public.tll_lightrag_vdb_entity 
    SET file_path = 
        CASE 
            WHEN file_path LIKE p_file_name || '<SEP>%' 
            THEN SUBSTRING(file_path FROM LENGTH(p_file_name || '<SEP>') + 1)
            
            WHEN file_path LIKE '%<SEP>' || p_file_name 
            THEN LEFT(file_path, LENGTH(file_path) - LENGTH('<SEP>' || p_file_name))
            
            WHEN file_path LIKE '%<SEP>' || p_file_name || '<SEP>%' 
            THEN REPLACE(file_path, '<SEP>' || p_file_name || '<SEP>', '<SEP>')
            
            ELSE file_path
        END
    WHERE file_path LIKE '%' || p_file_name || '%'
      AND file_path != p_file_name;
    
    GET DIAGNOSTICS v_entity_updated = ROW_COUNT;
    
    -- 2. Delete single-file entities
    DELETE FROM public.tll_lightrag_vdb_entity 
    WHERE file_path = p_file_name;
    
    GET DIAGNOSTICS v_entity_deleted = ROW_COUNT;
    
    -- 3. Delete from relation table
    DELETE FROM public.tll_lightrag_vdb_relation 
    WHERE file_path LIKE '%' || p_file_name || '%';
    
    GET DIAGNOSTICS v_relation_deleted = ROW_COUNT;
    
    -- 4. Delete from chunks table (MUST be before doc_full!)
    DELETE FROM public.tll_lightrag_doc_chunks 
    WHERE full_doc_id = p_doc_id;
    
    GET DIAGNOSTICS v_chunks_deleted = ROW_COUNT;
    
    -- 5. Delete from doc_status
    DELETE FROM public.tll_lightrag_doc_status 
    WHERE id = p_doc_id;
    
    GET DIAGNOSTICS v_doc_status_deleted = ROW_COUNT;
    
    -- 6. Delete from doc_full (LAST because chunks reference it)
    DELETE FROM public.tll_lightrag_doc_full 
    WHERE id = p_doc_id;
    
    GET DIAGNOSTICS v_doc_full_deleted = ROW_COUNT;
    
    -- Return summary
    RETURN QUERY
    SELECT 'entities_updated'::VARCHAR, v_entity_updated
    UNION ALL
    SELECT 'entities_deleted'::VARCHAR, v_entity_deleted
    UNION ALL
    SELECT 'relations_deleted'::VARCHAR, v_relation_deleted
    UNION ALL
    SELECT 'chunks_deleted'::VARCHAR, v_chunks_deleted
    UNION ALL
    SELECT 'doc_status_deleted'::VARCHAR, v_doc_status_deleted
    UNION ALL
    SELECT 'doc_full_deleted'::VARCHAR, v_doc_full_deleted;
    
EXCEPTION
    WHEN OTHERS THEN
        RAISE EXCEPTION 'Error during delete operation: %', SQLERRM;
END;
$$;

### What the Function Does

The `delete_lightrag_document_with_summary` function performs the following operations:

1. **Entity Management**
   - Updates entities that have multiple source documents (removes the deleted document's reference)
   - Deletes entities that only belong to the deleted document

2. **Relationship Management**
   - Deletes all relationships associated with the document's chunks

3. **Document Data Cleanup**
   - Deletes all chunks from `tll_document_chunks`
   - Deletes document status from `tll_document_status`
   - Deletes full document from `tll_document_full`

4. **Returns Summary**
   - Provides counts of all affected rows for each operation

## Error Handling

The implementation includes robust error handling:

1. **PostgreSQL Connection Failures**: Falls back to regular deletion method
2. **Function Execution Errors**: Logs warnings and falls back to regular deletion
3. **Pipeline Busy State**: Prevents deletion when the pipeline is actively processing

## Batch Deletion

The batch deletion endpoint processes multiple documents efficiently:

1. Finds PostgreSQL storage once for all deletions
2. Processes each document individually to provide detailed per-document results
3. Returns overall status and individual results for each document

### Batch Request Format

```json
{
    "documents": [
        {"doc_id": "doc_123", "file_name": "file1.pdf"},
        {"doc_id": "doc_456", "file_name": "file2.pdf"}
    ]
}
```

### Batch Response Format

```json
{
    "overall_status": "success",
    "message": "All 2 documents deleted successfully",
    "results": [...],
    "deleted_count": 2,
    "failed_count": 0
}
```

## Key Implementation Decisions

1. **Graceful Fallback**: Always falls back to regular deletion if PostgreSQL is unavailable
2. **No Double-Dipping**: Ensures deletion happens only once (either via PostgreSQL or regular method)
3. **Detailed Logging**: Provides comprehensive debug information for troubleshooting
4. **Backward Compatibility**: Works seamlessly with non-PostgreSQL storage backends

## Future Enhancements

1. **Neo4j Integration**: Similar cascade delete functionality for Neo4j graph storage
2. **Transaction Support**: Wrap multiple deletions in a single transaction
3. **Performance Optimization**: Batch PostgreSQL function calls for better performance

## Troubleshooting

### Common Issues

1. **Function Not Found**: Ensure the PostgreSQL function is created in the correct schema
2. **Permission Errors**: Verify the database user has EXECUTE permission on the function
3. **Connection Pool Issues**: Check PostgreSQL connection settings in `.env` file

### Debug Logging

The implementation includes debug logging to help troubleshoot issues:

```
INFO: DEBUG: Looking for PostgreSQL storage in 7 backends
INFO: DEBUG: Storage type: PGVectorStorage
INFO: DEBUG: Found PostgreSQL storage: PGVectorStorage
INFO: DEBUG: PostgreSQL storage has valid pool connection
INFO: PostgreSQL cascade delete completed for doc doc-123: {...}
```

## Configuration

Ensure your `.env` file contains the necessary PostgreSQL configuration:

```env
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=your_user
POSTGRES_PASSWORD=your_password
POSTGRES_DATABASE=your_database
```

## Testing

To test the implementation:

1. Upload a document to LightRAG
2. Note the document ID
3. Call the deletion endpoint with the document ID and filename
4. Verify the response includes `database_cleanup` statistics
5. Check the database to confirm all related data was removed
