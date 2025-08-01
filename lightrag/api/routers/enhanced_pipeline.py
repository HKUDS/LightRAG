"""
Enhanced document processing pipeline with Docling service integration.
"""

import asyncio
import logging
from pathlib import Path
from typing import Tuple
import aiofiles

from lightrag import LightRAG
from lightrag.utils import logger
from .document_processing import docling_processor


async def enhanced_pipeline_enqueue_file(
    rag: LightRAG, file_path: Path, track_id: str = None
) -> Tuple[bool, str]:
    """
    Enhanced version of pipeline_enqueue_file with Docling service integration.
    
    This function provides intelligent document processing routing:
    1. Tries Docling service for supported formats (PDF, DOCX, PPTX, XLSX)
    2. Falls back to basic parsers if service unavailable
    3. Uses basic parsers directly for unsupported formats
    
    Args:
        rag: LightRAG instance
        file_path: Path to the saved file
        track_id: Optional tracking ID, if not provided will be generated
        
    Returns:
        tuple: (success: bool, track_id: str)
    """
    
    try:
        logger.info(f"Processing file {file_path.name} with enhanced pipeline")
        
        # Process document using intelligent routing
        processing_result = await docling_processor.process_document(file_path)
        
        if not processing_result["success"]:
            error_msg = processing_result.get("error", "Unknown processing error")
            logger.error(f"Failed to process {file_path.name}: {error_msg}")
            return False, track_id or ""
        
        content = processing_result["content"]
        metadata = processing_result.get("metadata", {})
        
        # Validate content
        if not content or len(content.strip()) == 0:
            logger.error(f"Empty content after processing file: {file_path.name}")
            return False, track_id or ""
        
        # Log processing details
        processor_used = metadata.get("processor", "unknown")
        docling_service_used = metadata.get("docling_service_used", False)
        cache_hit = metadata.get("cache_hit", False)
        processing_time = metadata.get("processing_time_seconds", 0)
        
        logger.info(f"Processed {file_path.name}: "
                   f"processor={processor_used}, "
                   f"service_used={docling_service_used}, "
                   f"cache_hit={cache_hit}, "
                   f"time={processing_time:.2f}s, "
                   f"content_length={len(content)}")
        
        # Insert content into RAG system
        if track_id:
            await rag.ainsert(content, track_id=track_id)
        else:
            track_id = await rag.ainsert(content)
        
        logger.info(f"Successfully indexed {file_path.name} with track_id: {track_id}")
        
        return True, track_id
        
    except Exception as e:
        logger.error(f"Error processing file {file_path.name}: {str(e)}", exc_info=True)
        return False, track_id or ""


async def enhanced_process_text_file(file_path: Path) -> str:
    """
    Enhanced text file processing that handles various text formats.
    
    This is used for text-based files that don't need Docling processing.
    """
    
    try:
        # Read file content
        async with aiofiles.open(file_path, "rb") as f:
            file_data = await f.read()
        
        # Try to decode as UTF-8
        try:
            content = file_data.decode("utf-8")
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                content = file_data.decode("latin1")
                logger.warning(f"File {file_path.name} decoded with latin1 encoding")
            except UnicodeDecodeError:
                logger.error(f"Cannot decode file {file_path.name} - unsupported encoding")
                return ""
        
        # Validate content
        if not content or len(content.strip()) == 0:
            logger.error(f"Empty content in file: {file_path.name}")
            return ""
        
        # Check if content looks like binary data string representation
        if content.startswith("b'") or content.startswith('b"'):
            logger.error(f"File {file_path.name} appears to contain binary data representation")
            return ""
        
        return content.strip()
        
    except Exception as e:
        logger.error(f"Error reading text file {file_path.name}: {e}")
        return ""


async def enhanced_pipeline_process_any_file(
    rag: LightRAG, file_path: Path, track_id: str = None
) -> Tuple[bool, str]:
    """
    Process any file type with appropriate handler.
    
    This function routes files to the appropriate processor:
    - Text-based files: Direct text processing
    - Docling-supported files: Enhanced docling processing
    - Other files: Attempt text processing as fallback
    
    Args:
        rag: LightRAG instance
        file_path: Path to the file
        track_id: Optional tracking ID
        
    Returns:
        tuple: (success: bool, track_id: str)
    """
    
    try:
        ext = file_path.suffix.lower()
        
        # Text-based files - process directly
        text_extensions = {
            ".txt", ".md", ".html", ".htm", ".tex", ".json", ".xml", 
            ".yaml", ".yml", ".rtf", ".odt", ".epub", ".csv", ".log",
            ".conf", ".ini", ".properties", ".sql", ".bat", ".sh",
            ".c", ".cpp", ".py", ".java", ".js", ".ts", ".swift", 
            ".go", ".rb", ".php", ".css", ".scss", ".less"
        }
        
        if ext in text_extensions:
            logger.info(f"Processing text file {file_path.name}")
            content = await enhanced_process_text_file(file_path)
            
            if not content:
                return False, track_id or ""
            
            # Insert into RAG
            if track_id:
                await rag.ainsert(content, track_id=track_id)
            else:
                track_id = await rag.ainsert(content)
            
            logger.info(f"Successfully processed text file {file_path.name}")
            return True, track_id
        
        # Docling-supported files or others - use enhanced pipeline
        else:
            return await enhanced_pipeline_enqueue_file(rag, file_path, track_id)
            
    except Exception as e:
        logger.error(f"Error in enhanced pipeline for {file_path.name}: {e}", exc_info=True)
        return False, track_id or ""