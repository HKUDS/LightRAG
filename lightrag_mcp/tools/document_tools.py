"""
Document management tools for LightRAG MCP integration.

Implements document lifecycle management including text insertion,
file processing, document listing, deletion, and batch operations.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from ..client.api_client import get_api_client
from ..client.direct_client import LightRAGDirectClient
from ..config import get_config
from ..utils import Validator, MCPError, sanitize_filename, format_bytes

logger = logging.getLogger("lightrag-mcp.document_tools")


async def lightrag_insert_text(
    text: str,
    title: str = "",
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Insert text document directly into the knowledge base.
    
    Args:
        text: Document content to process
        title: Document title for identification (optional)
        metadata: Additional document metadata (optional)
            - source: Source identifier
            - author: Document author
            - created_date: Creation date (ISO format)
            - tags: List of tags
            - custom_fields: Custom metadata fields
    
    Returns:
        Document processing status and information
    """
    config = get_config()
    
    # Validate inputs
    if not text or not text.strip():
        raise MCPError("INVALID_PARAMETER", "Text content cannot be empty")
    
    if len(text) > 10_000_000:  # 10MB text limit
        raise MCPError("TEXT_TOO_LARGE", 
                      f"Text size {len(text)} characters exceeds 10M character limit")
    
    if title and len(title) > Validator.TITLE_MAX_LENGTH:
        raise MCPError("INVALID_PARAMETER", 
                      f"Title too long: {len(title)} > {Validator.TITLE_MAX_LENGTH}")
    
    # Validate metadata if provided
    if metadata:
        if not isinstance(metadata, dict):
            raise MCPError("INVALID_METADATA", "Metadata must be a dictionary")
        
        # Check for reserved fields
        reserved_fields = {"document_id", "status", "processing_info"}
        if any(field in metadata for field in reserved_fields):
            raise MCPError("INVALID_METADATA", 
                          f"Metadata cannot contain reserved fields: {reserved_fields}")
    
    logger.info(f"Inserting text document: {title or 'Untitled'} ({len(text)} chars)")
    
    try:
        # Execute insertion based on mode
        if config.enable_direct_mode:
            async with LightRAGDirectClient(config) as client:
                result = await client.insert_text(text, title, metadata)
        else:
            if not config.enable_document_upload:
                raise MCPError("FORBIDDEN", "Document upload is disabled")
            
            async with get_api_client(config) as client:
                result = await client.insert_text(text, title, metadata)
        
        # Enhance result
        result.update({
            "text_length": len(text),
            "title": title or "Untitled",
            "metadata_provided": bool(metadata),
            "mcp_server": config.mcp_server_name
        })
        
        logger.info(f"Text document inserted successfully: {result.get('document_id', 'unknown')}")
        return result
        
    except MCPError:
        raise
    except Exception as e:
        logger.error(f"Text insertion failed: {e}")
        raise MCPError("PROCESSING_FAILED", f"Text insertion failed: {e}")


async def lightrag_insert_file(
    file_path: str,
    processing_options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Process and index files from filesystem.
    
    Args:
        file_path: Path to file to process
        processing_options: Processing configuration (optional)
            - chunk_size: Override default chunk size
            - chunk_overlap: Override default chunk overlap
            - enable_entity_extraction: Enable/disable entity extraction
            - processing_priority: Priority level (low, normal, high)
    
    Returns:
        Document processing status and information
    """
    config = get_config()
    
    # Validate file path
    if not file_path:
        raise MCPError("INVALID_PARAMETER", "File path cannot be empty")
    
    path = Path(file_path)
    
    # Check if file exists
    if not path.exists():
        raise MCPError("FILE_NOT_FOUND", f"File not found: {file_path}")
    
    if not path.is_file():
        raise MCPError("INVALID_PARAMETER", f"Path is not a file: {file_path}")
    
    # Validate file type
    Validator.validate_file_type(file_path, config.allowed_file_types)
    
    # Check file size
    file_size = path.stat().st_size
    file_size_mb = file_size / (1024 * 1024)
    
    if file_size_mb > config.max_file_size_mb:
        raise MCPError("FILE_TOO_LARGE", 
                      f"File size {format_bytes(file_size)} exceeds limit {config.max_file_size_mb}MB")
    
    # Validate processing options
    if processing_options:
        if not isinstance(processing_options, dict):
            raise MCPError("INVALID_PARAMETER", "Processing options must be a dictionary")
        
        # Validate specific options
        if "chunk_size" in processing_options:
            chunk_size = processing_options["chunk_size"]
            if not isinstance(chunk_size, int) or chunk_size <= 0:
                raise MCPError("INVALID_PARAMETER", "chunk_size must be a positive integer")
        
        if "processing_priority" in processing_options:
            priority = processing_options["processing_priority"]
            if priority not in ["low", "normal", "high"]:
                raise MCPError("INVALID_PARAMETER", 
                              "processing_priority must be 'low', 'normal', or 'high'")
    
    logger.info(f"Inserting file: {file_path} ({format_bytes(file_size)})")
    
    try:
        # Execute file insertion based on mode
        if config.enable_direct_mode:
            async with LightRAGDirectClient(config) as client:
                result = await client.insert_file(file_path, **(processing_options or {}))
        else:
            if not config.enable_document_upload:
                raise MCPError("FORBIDDEN", "Document upload is disabled")
            
            async with get_api_client(config) as client:
                result = await client.insert_file(file_path, **(processing_options or {}))
        
        # Enhance result
        result.update({
            "file_path": str(path),
            "file_name": path.name,
            "file_size": file_size,
            "file_size_formatted": format_bytes(file_size),
            "file_type": path.suffix.lower(),
            "processing_options_used": processing_options or {},
            "mcp_server": config.mcp_server_name
        })
        
        logger.info(f"File inserted successfully: {result.get('document_id', 'unknown')}")
        return result
        
    except MCPError:
        raise
    except Exception as e:
        logger.error(f"File insertion failed: {e}")
        raise MCPError("PROCESSING_FAILED", f"File insertion failed: {e}")


async def lightrag_list_documents(
    status_filter: str = "",
    limit: int = 50,
    offset: int = 0,
    sort_by: str = "created_date",
    sort_order: str = "desc"
) -> Dict[str, Any]:
    """
    List documents with filtering, sorting, and pagination.
    
    Args:
        status_filter: Filter by document status (pending, processing, processed, failed)
        limit: Maximum documents to return (default: 50, max: 200)
        offset: Pagination offset (default: 0)
        sort_by: Field to sort by (created_date, title, status, processing_time)
        sort_order: Sort direction (asc, desc)
    
    Returns:
        List of documents with pagination and statistics
    """
    config = get_config()
    
    # Validate inputs
    Validator.validate_limit_offset(limit, offset, max_limit=200)
    
    valid_status_filters = ["", "pending", "processing", "processed", "failed"]
    if status_filter and status_filter not in valid_status_filters:
        raise MCPError("INVALID_PARAMETER", 
                      f"Invalid status filter. Valid options: {valid_status_filters}")
    
    valid_sort_fields = ["created_date", "title", "status", "processing_time"]
    if sort_by not in valid_sort_fields:
        raise MCPError("INVALID_PARAMETER", 
                      f"Invalid sort field. Valid options: {valid_sort_fields}")
    
    valid_sort_orders = ["asc", "desc"]
    if sort_order not in valid_sort_orders:
        raise MCPError("INVALID_PARAMETER", 
                      f"Invalid sort order. Valid options: {valid_sort_orders}")
    
    logger.debug(f"Listing documents: limit={limit}, offset={offset}, "
                f"status={status_filter}, sort={sort_by} {sort_order}")
    
    try:
        # Execute listing based on mode
        if config.enable_direct_mode:
            async with LightRAGDirectClient(config) as client:
                result = await client.list_documents(status_filter, limit, offset)
        else:
            async with get_api_client(config) as client:
                result = await client.list_documents(status_filter, limit, offset)
        
        # Enhance result with pagination info
        if "pagination" not in result:
            total = result.get("total", 0)
            current_page = (offset // limit) + 1
            total_pages = max(1, (total + limit - 1) // limit)
            
            result["pagination"] = {
                "current_page": current_page,
                "total_pages": total_pages,
                "has_next": offset + limit < total,
                "has_previous": offset > 0,
                "limit": limit,
                "offset": offset
            }
        
        result.update({
            "filters_applied": {
                "status": status_filter,
                "sort_by": sort_by,
                "sort_order": sort_order
            },
            "mcp_server": config.mcp_server_name
        })
        
        logger.debug(f"Listed {len(result.get('documents', []))} documents")
        return result
        
    except MCPError:
        raise
    except Exception as e:
        logger.error(f"Document listing failed: {e}")
        raise MCPError("INTERNAL_ERROR", f"Document listing failed: {e}")


async def lightrag_delete_documents(
    document_ids: List[str],
    cascade_delete: bool = True,
    create_backup: bool = False
) -> Dict[str, Any]:
    """
    Remove documents and associated data from knowledge base.
    
    Args:
        document_ids: List of document IDs to delete
        cascade_delete: Delete associated entities/relationships (default: True)
        create_backup: Create backup before deletion (default: False)
    
    Returns:
        Deletion results with success/failure details
    """
    config = get_config()
    
    # Validate inputs
    if not document_ids:
        raise MCPError("INVALID_PARAMETER", "Document IDs list cannot be empty")
    
    if len(document_ids) > config.max_documents_per_batch:
        raise MCPError("INVALID_PARAMETER", 
                      f"Too many documents: {len(document_ids)} > {config.max_documents_per_batch}")
    
    # Validate document ID formats
    for doc_id in document_ids:
        if not doc_id or not isinstance(doc_id, str):
            raise MCPError("INVALID_PARAMETER", f"Invalid document ID: {doc_id}")
    
    logger.info(f"Deleting {len(document_ids)} documents (cascade={cascade_delete})")
    
    try:
        # Execute deletion based on mode
        if config.enable_direct_mode:
            async with LightRAGDirectClient(config) as client:
                result = await client.delete_documents(document_ids)
        else:
            async with get_api_client(config) as client:
                result = await client.delete_documents(document_ids)
        
        # Enhance result
        result.update({
            "requested_deletions": len(document_ids),
            "cascade_delete": cascade_delete,
            "backup_created": create_backup,
            "mcp_server": config.mcp_server_name
        })
        
        deleted_count = len(result.get("deleted_documents", []))
        failed_count = len(result.get("failed_deletions", []))
        
        logger.info(f"Deletion completed: {deleted_count} successful, {failed_count} failed")
        return result
        
    except MCPError:
        raise
    except Exception as e:
        logger.error(f"Document deletion failed: {e}")
        raise MCPError("INTERNAL_ERROR", f"Document deletion failed: {e}")


async def lightrag_batch_process(
    items: List[Dict[str, Any]],
    max_concurrent: int = 5,
    stop_on_error: bool = False
) -> Dict[str, Any]:
    """
    Process multiple documents in batch with progress tracking.
    
    Args:
        items: List of items to process
            Each item should have:
            - item_type: "file", "text", or "url"
            - content: File path, text content, or URL
            - title: Optional title
            - metadata: Optional metadata
        max_concurrent: Maximum concurrent processing (default: 5)
        stop_on_error: Stop batch if error occurs (default: False)
    
    Returns:
        Batch processing results with progress information
    """
    config = get_config()
    
    # Validate inputs
    if not items:
        raise MCPError("INVALID_PARAMETER", "Items list cannot be empty")
    
    if len(items) > Validator.MAX_BATCH_SIZE:
        raise MCPError("INVALID_PARAMETER", 
                      f"Too many items: {len(items)} > {Validator.MAX_BATCH_SIZE}")
    
    if max_concurrent <= 0 or max_concurrent > 20:
        raise MCPError("INVALID_PARAMETER", "max_concurrent must be between 1 and 20")
    
    # Validate items
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise MCPError("INVALID_PARAMETER", f"Item {i} must be a dictionary")
        
        if "item_type" not in item:
            raise MCPError("INVALID_PARAMETER", f"Item {i} missing 'item_type'")
        
        if item["item_type"] not in ["file", "text", "url"]:
            raise MCPError("INVALID_PARAMETER", 
                          f"Item {i} has invalid type: {item['item_type']}")
        
        if "content" not in item:
            raise MCPError("INVALID_PARAMETER", f"Item {i} missing 'content'")
    
    batch_id = f"batch_{hash(str(items))[:8]}"
    logger.info(f"Starting batch processing: {batch_id} with {len(items)} items")
    
    # Initialize results
    results = []
    completed = 0
    failed = 0
    
    async def process_item(index: int, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single item."""
        nonlocal completed, failed
        
        try:
            item_type = item["item_type"]
            content = item["content"]
            title = item.get("title", "")
            metadata = item.get("metadata")
            
            start_time = asyncio.get_event_loop().time()
            
            if item_type == "text":
                result = await lightrag_insert_text(content, title, metadata)
            elif item_type == "file":
                result = await lightrag_insert_file(content)
            else:  # url
                raise MCPError("NOT_IMPLEMENTED", "URL processing not yet implemented")
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            completed += 1
            return {
                "item_index": index,
                "status": "completed",
                "document_id": result.get("document_id"),
                "processing_time": processing_time,
                "error_message": None
            }
            
        except Exception as e:
            failed += 1
            logger.error(f"Batch item {index} failed: {e}")
            
            return {
                "item_index": index,
                "status": "failed",
                "document_id": None,
                "processing_time": None,
                "error_message": str(e)
            }
    
    try:
        # Use semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(index: int, item: Dict[str, Any]):
            async with semaphore:
                result = await process_item(index, item)
                
                # Stop on error if requested
                if stop_on_error and result["status"] == "failed":
                    raise MCPError("BATCH_PROCESSING_STOPPED", 
                                  f"Batch stopped due to error in item {index}")
                
                return result
        
        # Process all items concurrently
        tasks = [
            process_with_semaphore(i, item)
            for i, item in enumerate(items)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions in results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed += 1
                processed_results.append({
                    "item_index": i,
                    "status": "failed",
                    "document_id": None,
                    "processing_time": None,
                    "error_message": str(result)
                })
            else:
                processed_results.append(result)
        
        # Calculate final statistics
        success_rate = (completed / len(items)) * 100 if items else 0
        
        batch_result = {
            "batch_id": batch_id,
            "total_items": len(items),
            "status": "completed",
            "progress": {
                "completed": completed,
                "failed": failed,
                "remaining": 0,
                "percentage": 100.0
            },
            "results": processed_results,
            "statistics": {
                "success_rate": success_rate,
                "total_processing_time": sum(
                    r.get("processing_time", 0) or 0 
                    for r in processed_results
                ),
                "average_processing_time": sum(
                    r.get("processing_time", 0) or 0 
                    for r in processed_results
                ) / len(processed_results) if processed_results else 0
            },
            "configuration": {
                "max_concurrent": max_concurrent,
                "stop_on_error": stop_on_error
            },
            "mcp_server": config.mcp_server_name
        }
        
        logger.info(f"Batch processing completed: {batch_id} "
                   f"({completed} successful, {failed} failed)")
        
        return batch_result
        
    except MCPError:
        raise
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise MCPError("INTERNAL_ERROR", f"Batch processing failed: {e}")


# Tool registration helpers
def get_document_tools() -> Dict[str, Any]:
    """Get document tools for MCP server registration."""
    return {
        "lightrag_insert_text": lightrag_insert_text,
        "lightrag_insert_file": lightrag_insert_file,
        "lightrag_list_documents": lightrag_list_documents,
        "lightrag_delete_documents": lightrag_delete_documents,
        "lightrag_batch_process": lightrag_batch_process
    }