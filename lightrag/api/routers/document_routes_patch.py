"""
Integration patch for document_routes.py to add Docling service support.

This file contains the enhanced functions that can replace the existing
document processing functions in document_routes.py.
"""

import logging
from pathlib import Path
from typing import Tuple
from lightrag import LightRAG
from lightrag.utils import logger, get_env_value
from .enhanced_pipeline import enhanced_pipeline_process_any_file


async def enhanced_pipeline_enqueue_file_with_compatibility(
    rag: LightRAG, file_path: Path, track_id: str = None
) -> Tuple[bool, str]:
    """
    Enhanced version that maintains compatibility with existing code.
    
    This function can be used as a drop-in replacement for the existing
    pipeline_enqueue_file function.
    """
    
    # Check if enhanced processing is enabled
    use_enhanced_processing = get_env_value("LIGHTRAG_ENHANCED_PROCESSING", True, bool)
    
    if use_enhanced_processing:
        logger.info(f"Using enhanced processing for {file_path.name}")
        return await enhanced_pipeline_process_any_file(rag, file_path, track_id)
    else:
        # Fall back to original processing logic
        logger.info(f"Using legacy processing for {file_path.name}")
        
        # Import the original function dynamically to avoid circular imports
        from . import document_routes
        return await document_routes.pipeline_enqueue_file(rag, file_path, track_id)


def patch_document_routes():
    """
    Monkey patch the document_routes module to use enhanced processing.
    
    This allows seamless integration without modifying the existing file.
    """
    
    try:
        from . import document_routes
        
        # Store original function
        document_routes._original_pipeline_enqueue_file = document_routes.pipeline_enqueue_file
        
        # Replace with enhanced version
        document_routes.pipeline_enqueue_file = enhanced_pipeline_enqueue_file_with_compatibility
        
        logger.info("Document routes patched with enhanced Docling processing")
        
    except Exception as e:
        logger.error(f"Failed to patch document routes: {e}")
        logger.warning("Continuing with original document processing")


def unpatch_document_routes():
    """Restore original document processing (for testing/debugging)."""
    
    try:
        from . import document_routes
        
        if hasattr(document_routes, '_original_pipeline_enqueue_file'):
            document_routes.pipeline_enqueue_file = document_routes._original_pipeline_enqueue_file
            delattr(document_routes, '_original_pipeline_enqueue_file')
            logger.info("Document routes unpatched - using original processing")
        
    except Exception as e:
        logger.error(f"Failed to unpatch document routes: {e}")


# Service integration status functions
async def get_docling_service_info() -> dict:
    """Get information about Docling service integration."""
    
    from lightrag.docling_client.service_discovery import service_discovery
    from .document_processing import docling_processor
    
    try:
        service_info = await service_discovery.get_service_info()
        
        return {
            "integration_enabled": True,
            "service_mode": docling_processor.service_mode,
            "service_configured": service_info["configured"],
            "service_url": service_info["url"],
            "service_available": service_info["available"],
            "service_config": service_info.get("config"),
            "fallback_enabled": get_env_value("DOCLING_FALLBACK_ENABLED", True, bool),
            "enhanced_processing_enabled": get_env_value("LIGHTRAG_ENHANCED_PROCESSING", True, bool),
            "error": service_info.get("error")
        }
        
    except Exception as e:
        return {
            "integration_enabled": False,
            "error": str(e)
        }