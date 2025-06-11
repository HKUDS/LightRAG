#!/usr/bin/env python3
"""
Process JSON Files Script with Enhanced Semantic Chunking and Webhook Integration

This script:
1. Finds JSON files in a specified folder
2. Processes each file using advanced semantic chunking based on markdown headers
3. Organizes data into a structured format optimized for RAG retrieval
4. Sends the processed data to a webhook (test or production)
5. Provides a webhook server for receiving callbacks
6. Integrates with n8n by logging webhook responses to a file

Usage:
    python process_json_files.py --folder <folder_name> [--test] [--file <specific_file>] [--image-urls <image_urls_json>]
    python process_json_files.py --webhook-only --port 5000
"""

import os
import sys
import json
import glob
import argparse
import logging
import requests
import traceback
import signal
import threading
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from werkzeug.serving import make_server

# Load environment variables
load_dotenv()
# Also load runtime environment if it exists
if os.path.exists(".env.runtime"):
    load_dotenv(".env.runtime", override=True)

# Ensure logs and data directories exist
Path("logs").mkdir(exist_ok=True, parents=True)
Path("data").mkdir(exist_ok=True, parents=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/process_json.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Get configuration from environment variables
TEST_WEBHOOK_URL = os.getenv("TEST_WEBHOOK_URL", "")
PRODUCTION_WEBHOOK_URL = os.getenv("PRODUCTION_WEBHOOK_URL", "")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))  # Default chunk size in characters
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))  # Default overlap in characters
TOKEN_CHUNK_SIZE = int(os.getenv("TOKEN_CHUNK_SIZE", "400"))  # Default chunk size in tokens
TOKEN_CHUNK_OVERLAP = int(os.getenv("TOKEN_CHUNK_OVERLAP", "80"))  # Increased overlap for better context
FRAME_BASE_DIR = os.getenv("FRAME_BASE_DIR", "")
IMAGE_URLS_JSON = os.getenv("IMAGE_URLS_JSON", "")
INCLUDE_IMAGE_URLS = os.getenv("INCLUDE_IMAGE_URLS", "false").lower() == "true"
WEBHOOK_LOG_FILE = os.environ.get('WEBHOOK_LOG_FILE', '/tmp/webhook_response.log')
ENVIRONMENT = os.environ.get('PROCESSOR_ENVIRONMENT', 'test')

# Flask app for webhook handling
app = Flask(__name__)
server = None

# Define constants for headers to ensure consistency
H1_FRAME_SUMMARY = "Frame Summary"
H2_CONTEXT = "Context"
H2_TECHNICAL_DETAILS = "Technical Details"
H2_SCREEN_CONTENT = "Screen Content (OCR)"
H2_SOURCE = "Source"

# Validate essential environment variables
if not FRAME_BASE_DIR:
    logger.error("FRAME_BASE_DIR not set in environment variables")
    sys.exit(1)

if not TEST_WEBHOOK_URL or not PRODUCTION_WEBHOOK_URL:
    logger.warning("TEST_WEBHOOK_URL or PRODUCTION_WEBHOOK_URL not set in environment")

# ==================== Webhook Server Functions ====================

class ServerThread(threading.Thread):
    """Thread class for running Flask in background"""
    def __init__(self, app, host='0.0.0.0', port=5000):
        threading.Thread.__init__(self)
        self.daemon = True
        self.server = make_server(host, port, app)
        self.ctx = app.app_context()
        self.ctx.push()

    def run(self):
        logger.info("Starting webhook server")
        self.server.serve_forever()

    def shutdown(self):
        logger.info("Shutting down webhook server")
        self.server.shutdown()

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    logger.info("Shutting down...")
    if server:
        server.shutdown()
    sys.exit(0)

def log_webhook_result(result):
    """Write results to the webhook log file for n8n to capture"""
    try:
        if isinstance(result, dict):
            output = json.dumps(result, indent=2)
        else:
            output = str(result)
            
        with open(WEBHOOK_LOG_FILE, 'w') as f:
            f.write(output)
            
        logger.info(f"Results written to webhook log: {WEBHOOK_LOG_FILE}")
    except Exception as e:
        logger.error(f"Failed to write to webhook log: {e}")

# ==================== JSON Processing Functions ====================

def normalize_folder_name(folder_name: str) -> str:
    """
    Remove '_frames' suffix from folder name if present.
    """
    if folder_name and folder_name.endswith('_frames'):
        return folder_name[:-7]  # Remove the '_frames' suffix
    return folder_name

def load_json_file(file_path: Path) -> Dict[str, Any]:
    """Load a JSON file and return its contents as a dictionary"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON from {file_path}: {e}")
        return {}

def structure_metadata(metadata: Dict[str, Any], frame_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Process raw metadata into a structured format for easier retrieval.
    
    Args:
        metadata: Raw metadata dictionary
        frame_path: Path to the frame image (optional)
        
    Returns:
        Structured metadata dictionary
    """
    structured = {}
    
    # Keep track of source information
    # Look for the record ID in various possible fields
    record_id = metadata.get("RecordID", "")
    if not record_id:
        record_id = metadata.get("id", "")
    structured["airtable_record_id"] = record_id
    
    structured["frame_path"] = str(frame_path) if frame_path else ""
    
    # Extract frame summary (use existing Summary field for now)
    structured["frame_summary"] = metadata.get("Summary", "")
    
    # Extract key fields with clean names
    field_mappings = {
        "Timestamp": "timestamp",
        "ToolsVisible": "tools_visible",
        "ActionsDetected": "actions_detected",
        "TechnicalDetails": "technical_details",
        "OCRData": "ocr_data",
        "StageOfWork": "stage_of_work",
        "FrameNumber": "frame_number",
        "FolderName": "folder_name",
        "RelationshipToPrevious": "context_relationship"
    }
    
    # Copy fields with cleaned names
    for original, clean in field_mappings.items():
        if original in metadata and metadata[original]:
            structured[clean] = metadata[original]
    
    return structured

def create_text_representation(structured_metadata: Dict[str, Any]) -> str:
    """
    Create a coherent text representation from structured metadata,
    only including sections that have non-empty content.
    
    Args:
        structured_metadata: Structured metadata dictionary
        
    Returns:
        Formatted text string optimized for semantic chunking
    """
    text = []
    
    # Add frame summary as a prominent first section
    frame_summary = structured_metadata.get("frame_summary", "").strip()
    if frame_summary:
        text.append(f"# {H1_FRAME_SUMMARY}\n{frame_summary}\n")
    
    # Add main contextual information
    context_section = []
    
    if "timestamp" in structured_metadata and structured_metadata["timestamp"].strip():
        context_section.append(f"Timestamp: {structured_metadata['timestamp']}")
    
    if "stage_of_work" in structured_metadata and structured_metadata["stage_of_work"].strip():
        context_section.append(f"Stage: {structured_metadata['stage_of_work']}")
    
    if "tools_visible" in structured_metadata and structured_metadata["tools_visible"].strip():
        context_section.append(f"Tools Visible: {structured_metadata['tools_visible']}")
    
    if "actions_detected" in structured_metadata and structured_metadata["actions_detected"].strip():
        context_section.append(f"Actions Detected: {structured_metadata['actions_detected']}")
    
    if "context_relationship" in structured_metadata and structured_metadata["context_relationship"].strip():
        context_section.append(f"Context: {structured_metadata['context_relationship']}")
    
    if context_section:
        text.append(f"## {H2_CONTEXT}\n" + "\n".join(context_section) + "\n")
    
    # Add technical details as a separate section
    technical_details = structured_metadata.get("technical_details", "").strip()
    if technical_details:
        text.append(f"## {H2_TECHNICAL_DETAILS}\n{technical_details}\n")
    
    # Add OCR data as a separate section
    ocr_data = structured_metadata.get("ocr_data", "").strip()
    if ocr_data:
        text.append(f"## {H2_SCREEN_CONTENT}\n{ocr_data}\n")
    
    # Add source information
    source_info = []
    if "airtable_record_id" in structured_metadata and structured_metadata["airtable_record_id"].strip():
        source_info.append(f"Airtable ID: {structured_metadata['airtable_record_id']}")
    
    if "frame_path" in structured_metadata and structured_metadata["frame_path"].strip():
        source_info.append(f"Frame Path: {structured_metadata['frame_path']}")
    
    if source_info:
        text.append(f"## {H2_SOURCE}\n" + "\n".join(source_info))
    
    return "\n\n".join(text)

def simple_chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Simple fallback chunker if advanced chunking fails."""
    if not text:
        return []
        
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        if end < text_length and end - start == chunk_size:
            # Find the last space in the chunk to avoid cutting words
            last_space = text[start:end].rfind(' ')
            if last_space != -1:
                end = start + last_space + 1
        
        chunks.append(text[start:end].strip())
        start = end - chunk_overlap if end - chunk_overlap > start else end
    
    return chunks

def semantic_chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[Dict[str, Any]]:
    """
    Split text based on Markdown headers, with enhanced context for sub-chunks.
    Returns chunks with metadata about which section they belong to.
    
    Args:
        text: Text to split into chunks
        chunk_size: Maximum chunk size in characters (for secondary splitting fallback)
        chunk_overlap: Amount of overlap between chunks (for secondary splitting fallback)
        
    Returns:
        List of dictionaries with chunk text and metadata
    """
    if not text:
        return []
    
    try:
        # Try to import required libraries
        try:
            from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
            has_langchain = True
        except ImportError:
            logger.critical(
                "CRITICAL: langchain not installed! Semantic chunking will not be used, "
                "resulting in significantly degraded chunk quality for RAG. "
                "Please install langchain: pip install langchain"
            )
            has_langchain = False
            
        # If we don't have langchain, use the simple chunker with strong warning
        if not has_langchain:
            logger.warning("⚠️ FALLING BACK TO SIMPLE CHUNKING: RAG performance will be significantly degraded ⚠️")
            return [{"text": chunk, "metadata": {}} for chunk in simple_chunk_text(text, chunk_size, chunk_overlap)]
        
        # Define headers to split on
        headers_to_split_on = [
            ("#", "header_1"),     # Frame Summary (H1)
            ("##", "header_2"),    # Context, Technical Details, etc. (H2)
        ]
        
        # Create markdown splitter
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        
        # Split by headers first
        header_splits = markdown_splitter.split_text(text)
        
        # For token-based secondary splitting
        try:
            import tiktoken
            has_tiktoken = True
            encoding = tiktoken.get_encoding("cl100k_base")  # Good general-purpose tokenizer
            
            # Function to count tokens
            def token_counter(text):
                return len(encoding.encode(text))
            
            # Create token-based splitter for oversized sections
            recursive_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name="cl100k_base",
                chunk_size=TOKEN_CHUNK_SIZE,
                chunk_overlap=TOKEN_CHUNK_OVERLAP,  # Using larger overlap for better context between sub-chunks
                separators=["\n\n", "\n", ". ", " ", ""]
            )
        except ImportError:
            # Fall back to character-based if tiktoken is not available
            logger.warning(
                "WARNING: tiktoken not available, falling back to character-based splitting. "
                "This may result in less optimal chunk sizes for embedding models. "
                "Please install tiktoken: pip install tiktoken"
            )
            has_tiktoken = False
            recursive_splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", ". ", " ", ""],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len
            )
        
        # Process each header section
        final_chunks = []
        for chunk_doc in header_splits:
            # Extract header information
            section_content = chunk_doc.page_content
            metadata = chunk_doc.metadata
            header_1 = metadata.get("header_1", "")
            header_2 = metadata.get("header_2", "")
            
            # Skip empty sections
            if not section_content.strip():
                continue
            
            # Determine section title and level
            section_title = header_2 if header_2 else header_1
            header_level = 2 if header_2 else 1
            header_prefix = "#" * header_level
            
            # Check if secondary splitting is needed
            needs_splitting = False
            if has_tiktoken:
                if token_counter(section_content) > TOKEN_CHUNK_SIZE:
                    needs_splitting = True
            else:
                # Fall back to character length
                if len(section_content) > chunk_size:
                    needs_splitting = True
            
            if not needs_splitting:
                # If the section is small enough, keep it as one chunk
                section_text = f"{header_prefix} {section_title}\n{section_content}"
                final_chunks.append({
                    "text": section_text.strip(),
                    "metadata": {
                        "header_1": header_1,
                        "header_2": header_2,
                        "is_subsection": False
                    }
                })
            else:
                # Section is too large, apply secondary splitting
                logger.info(f"Applying secondary splitting to large section: {section_title}")
                
                # Get a prefix from the section to include for context
                # Extract first 100 characters (or fewer if section is shorter)
                section_prefix = section_content[:min(100, len(section_content))].strip()
                if len(section_prefix) < len(section_content) and len(section_prefix) > 0:
                    section_prefix += "..."
                
                # Split content (not including header)
                sub_chunks = recursive_splitter.split_text(section_content)
                
                for i, sub_chunk in enumerate(sub_chunks):
                    # Enhanced header with section prefix for better standalone context
                    enhanced_header = f"{header_prefix} {section_title}"
                    if section_prefix:
                        enhanced_header += f" - Full Section Start: {section_prefix}"
                    enhanced_header += f" (Part {i+1}/{len(sub_chunks)})"
                    
                    # Include the enhanced header with each sub-chunk
                    sub_section_text = f"{enhanced_header}\n{sub_chunk}"
                    final_chunks.append({
                        "text": sub_section_text.strip(),
                        "metadata": {
                            "header_1": header_1,
                            "header_2": header_2,
                            "is_subsection": True,
                            "subsection_index": i,
                            "subsection_total": len(sub_chunks),
                            "section_prefix": section_prefix  # Store the prefix for potential use
                        }
                    })
        
        # Filter out any empty chunks
        final_chunks = [chunk for chunk in final_chunks if chunk["text"].strip()]
        
        return final_chunks
    
    except Exception as e:
        logger.error(f"Error in semantic_chunk_text: {str(e)}")
        logger.error(traceback.format_exc())
        # Fall back to simple chunking
        logger.warning("⚠️ ERROR OCCURRED: Falling back to simple chunking, RAG performance will be degraded ⚠️")
        return [{"text": chunk, "metadata": {}} for chunk in simple_chunk_text(text, chunk_size, chunk_overlap)]

def get_frame_image_path(folder_name: str, frame_number: str) -> Optional[Path]:
    """
    Get the path to the frame image in the base directory.
    
    Args:
        folder_name: Name of the folder containing the frame
        frame_number: Frame number/name
        
    Returns:
        Path to the frame image or None if not found
    """
    # Normalize folder name
    folder_name = normalize_folder_name(folder_name)
    
    # Construct base folder path
    base_folder_path = Path(FRAME_BASE_DIR) / folder_name
    
    if not base_folder_path.exists():
        logger.error(f"Frame folder not found: {base_folder_path}")
        return None
    
    logger.info(f"Looking for frame in location: {base_folder_path}")
    
    # Try different extensions for the frame image
    extensions = ['.jpg', '.jpeg', '.png']
    
    for ext in extensions:
        # Try exact match first
        frame_path = base_folder_path / f"{frame_number}{ext}"
        if frame_path.exists():
            logger.info(f"Found frame image at: {frame_path}")
            return frame_path
        
        # Try with common frame name patterns
        patterns = [
            f"{frame_number}{ext}",
            f"frame_{frame_number}{ext}",
            f"frame_{frame_number.zfill(6)}{ext}"
        ]
        
        for pattern in patterns:
            for file_path in base_folder_path.glob(pattern):
                logger.info(f"Found frame image at: {file_path}")
                return file_path
    
    logger.error(f"Frame image not found for {folder_name}/{frame_number}")
    return None

def process_json_file(file_path: Path, test_mode: bool = False) -> Dict[str, Any]:
    """
    Process a JSON file:
    1. Extract metadata
    2. Structure the metadata
    3. Create a semantic text representation
    4. Chunk the text using header-based semantic chunking
    5. Structure the output with nested arrays and section metadata
    
    Args:
        file_path: Path to the JSON file
        test_mode: Whether to run in test mode
        
    Returns:
        Dictionary with processed data including chunks
    """
    data = load_json_file(file_path)
    if not data:
        logger.error(f"Failed to load JSON data from {file_path}")
        return {}
    
    # Extract relevant fields
    folder_path = data.get("folder_path", "")
    file_name = data.get("file_name", "")
    metadata = data.get("metadata", {})
    content = data.get("content", "")
    
    # Extract folder and frame names
    folder_name = Path(folder_path).name if folder_path else ""
    frame_number = Path(file_name).stem if file_name else ""
    
    # Normalize folder name
    folder_name = normalize_folder_name(folder_name)
    
    # Check if we have necessary information
    if not folder_name or not frame_number:
        logger.error(f"Missing folder_name or frame_number in {file_path}")
        return {}
    
    logger.info(f"Processing frame: {folder_name}/{frame_number}")
    
    # Get the frame image path for multimodal processing
    frame_image_path = get_frame_image_path(folder_name, frame_number)
    
    # Structure the metadata
    logger.info("Structuring metadata")
    structured_metadata = structure_metadata(metadata, frame_image_path)
    
    # Create semantic text representation
    logger.info("Creating text representation")
    text_to_chunk = create_text_representation(structured_metadata)
    
    # Chunk the text using our enhanced semantic chunking
    logger.info("Chunking text using enhanced semantic header-based chunking")
    chunks_with_metadata = semantic_chunk_text(text_to_chunk, CHUNK_SIZE, CHUNK_OVERLAP)
    logger.info(f"Created {len(chunks_with_metadata)} semantically meaningful chunks from {file_path}")
    
    # Process each chunk
    all_chunks = []
    for i, chunk_info in enumerate(chunks_with_metadata):
        chunk_text = chunk_info["text"]
        chunk_metadata = chunk_info["metadata"]
        
        all_chunks.append({
            "id": f"{folder_name}_{frame_number}_chunk_{i}",
            "chunk_index": i,
            "text": chunk_text,
            "section_metadata": chunk_metadata  # Include section metadata for retrieval context
        })
    
    # Create the frame result with the new structure
    frame_result = {
        "id": f"{folder_name}_{frame_number}",
        "folder_name": folder_name,
        "file_name": file_name,
        "frame_number": frame_number,
        "frame_image_path": str(frame_image_path) if frame_image_path else "",
        "metadata": structured_metadata,  # Use the structured metadata
        "raw_metadata": metadata,  # Keep the original metadata for reference
        "content": content,
        "chunks": all_chunks,
        "processed_at": datetime.now().isoformat(),
        "test_mode": test_mode
    }
    
    logger.info(f"Completed processing of {file_path}")
    return frame_result

def load_image_urls(image_urls_file: str) -> Dict[str, Any]:
    """
    Load image URLs from a JSON file
    
    Args:
        image_urls_file: Path to the JSON file containing image URLs
        
    Returns:
        Dictionary with image URLs data
    """
    if not image_urls_file or not os.path.exists(image_urls_file):
        logger.warning(f"Image URLs file not found or not specified: {image_urls_file}")
        return {}
        
    try:
        with open(image_urls_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading image URLs from {image_urls_file}: {e}")
        return {}

def send_to_webhook(data: Dict[str, Any], test_mode: bool = False, image_urls: Dict[str, Any] = None) -> bool:
    """
    Send processed data to the appropriate webhook (test or production)
    
    Args:
        data: Data to send
        test_mode: Whether to use the test webhook
        image_urls: Optional image URLs to include in the payload
        
    Returns:
        True if successful, False otherwise
    """
    webhook_url = TEST_WEBHOOK_URL if test_mode else PRODUCTION_WEBHOOK_URL
    
    if not webhook_url:
        logger.error(f"{'TEST_' if test_mode else 'PRODUCTION_'}WEBHOOK_URL not set")
        log_webhook_result(None)  # Log failure
        return False
    
    # Add image URLs to the payload if available
    if image_urls and INCLUDE_IMAGE_URLS:
        logger.info(f"Including {len(image_urls)} image URLs in webhook payload")
        data["image_urls"] = image_urls
    
    try:
        logger.info(f"Sending data to {'test' if test_mode else 'production'} webhook: {webhook_url}")
        response = requests.post(webhook_url, json=data)
        
        if response.status_code >= 200 and response.status_code < 300:
            logger.info(f"Successfully sent data to webhook: {response.status_code}")
            log_webhook_result(data)  # Log success with minimal info
            return True
        else:
            logger.error(f"Error sending data to webhook: {response.status_code} - {response.text}")
            log_webhook_result(None)  # Log failure
            return False
    except Exception as e:
        logger.error(f"Exception when sending data to webhook: {e}")
        log_webhook_result(None)  # Log failure
        return False

def find_json_files(folder_name: str) -> List[Path]:
    """
    Find JSON files for a specific folder in the data directory.
    
    Args:
        folder_name: Folder name to filter by
        
    Returns:
        List of paths to JSON files
    """
    # Normalize folder name
    normalized_name = normalize_folder_name(folder_name)
    
    # Define possible folder paths
    possible_folders = [
        Path("data") / normalized_name,  # Without '_frames'
        Path("data") / f"{normalized_name}_frames"  # With '_frames'
    ]
    
    files = []
    for folder_path in possible_folders:
        if folder_path.exists():
            logger.info(f"Looking for JSON files in {folder_path}")
            pattern = str(folder_path / "*.json")
            folder_files = [Path(p) for p in glob.glob(pattern)]
            files.extend(folder_files)
            
    if not files:
        logger.error(f"No JSON files found for folder {folder_name}")
        
    logger.info(f"Found {len(files)} JSON files for folder {folder_name}")
    return files

def process_folder(folder_name: str, test_mode: bool = False, image_urls_file: str = None) -> Dict[str, Any]:
    """
    Process all JSON files in a folder
    
    Args:
        folder_name: Name of the folder to process
        test_mode: Whether to use test mode
        image_urls_file: Path to the JSON file containing image URLs
        
    Returns:
        Dictionary with processing results
    """
    start_time = datetime.now()
    logger.info(f"Processing folder: {folder_name}")
    
    # Get image URLs if specified
    image_urls = load_image_urls(image_urls_file) if image_urls_file else {}
    if image_urls:
        logger.info(f"Loaded image URLs for {len(image_urls)} frames")
    
    # Get JSON files for the folder
    files = find_json_files(folder_name)
    
    if not files:
        result = {
            "status": "error",
            "message": f"No files found for folder {folder_name}",
            "folder_name": folder_name,
            "timestamp": datetime.now().isoformat()
        }
        log_webhook_result(result)
        return result
    
    # Initialize result arrays - frames
    frames = []
    
    # Process each file
    processed_count = 0
    error_count = 0
    
    for file_path in files:
        try:
            logger.info(f"Processing file {file_path}")
            
            # Process the file
            result = process_json_file(file_path, test_mode)
            
            if not result:
                logger.error(f"Failed to process {file_path}")
                error_count += 1
                continue
                
            # Add the frame to our frames array
            frames.append(result)
            
            processed_count += 1
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            logger.error(traceback.format_exc())
            error_count += 1
    
    # Create the final output structure
    output = {
        "folder_name": folder_name,
        "processed_at": datetime.now().isoformat(),
        "test_mode": test_mode,
        "frames": frames,
        "stats": {
            "total_files": len(files),
            "processed": processed_count,
            "errors": error_count
        },
        "webhookUrl": TEST_WEBHOOK_URL if test_mode else PRODUCTION_WEBHOOK_URL,
        "executionMode": "test" if test_mode else "production"
    }
    
    # Send to webhook
    send_to_webhook(output, test_mode, image_urls)
    
    # Also ensure result is available to n8n
    log_webhook_result(output)
    
    return output

def process_specific_file(folder_name: str, file_path: str, test_mode: bool = False, image_urls_file: str = None) -> Dict[str, Any]:
    """
    Process a specific JSON file
    
    Args:
        folder_name: Name of the folder containing the file
        file_path: Path to the specific file to process
        test_mode: Whether to use test mode
        image_urls_file: Path to the JSON file containing image URLs
        
    Returns:
        Dictionary with processing results
    """
    # Check if file exists
    path = Path(file_path)
    if not path.exists():
        result = {
            "status": "error",
            "message": f"File not found: {file_path}",
            "folder_name": folder_name,
            "timestamp": datetime.now().isoformat()
        }
        log_webhook_result(result)
        return result
    
    try:
        logger.info(f"Processing specific file: {file_path}")
        
        # Process the file
        result = process_json_file(path, test_mode)
        
        if not result:
            error_result = {
                "status": "error",
                "message": f"Failed to process {file_path}",
                "folder_name": folder_name,
                "timestamp": datetime.now().isoformat()
            }
            log_webhook_result(error_result)
            return error_result
        
        # Get image URLs if specified
        image_urls = load_image_urls(image_urls_file) if image_urls_file else {}
        if image_urls:
            logger.info(f"Loaded image URLs for {len(image_urls)} frames")
        
        # Create the output with just this frame
        output = {
            "folder_name": folder_name,
            "processed_at": datetime.now().isoformat(),
            "test_mode": test_mode,
            "frames": [result],
            "stats": {
                "total_files": 1,
                "processed": 1,
                "errors": 0
            },
            "webhookUrl": TEST_WEBHOOK_URL if test_mode else PRODUCTION_WEBHOOK_URL,
            "executionMode": "test" if test_mode else "production"
        }
        
        # Send to webhook
        send_to_webhook(output, test_mode, image_urls)
        
        # Also ensure result is available to n8n
        log_webhook_result(output)
        
        return output
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        logger.error(traceback.format_exc())
        error_result = {
            "status": "error",
            "message": f"Error processing {file_path}: {str(e)}",
            "folder_name": folder_name,
            "timestamp": datetime.now().isoformat()
        }
        log_webhook_result(error_result)
        return error_result

# ==================== Webhook Server Routes ====================

@app.route('/webhook', methods=['POST'])
def webhook_handler():
    """Handle webhook requests"""
    try:
        data = request.json
        logger.info(f"Received webhook data: {data}")
        
        folder = data.get('folder', '')
        env = data.get('environment', ENVIRONMENT)
        file_path = data.get('file_path', '')
        test_mode = env.lower() == 'test'
        
        if not folder:
            response = {
                "status": "error", 
                "message": "Missing folder parameter",
                "environment": env
            }
            log_webhook_result(response)
            return jsonify(response), 400
                
        # Process either a specific file or the entire folder
        if file_path:
            result = process_specific_file(folder, file_path, test_mode)
        else:
            result = process_folder(folder, test_mode)
        
        # Return the result
        return jsonify(result)
            
    except Exception as e:
        logger.exception("Webhook error")
        error = {
            "status": "error", 
            "message": f"Webhook error: {str(e)}",
            "environment": ENVIRONMENT
        }
        log_webhook_result(error)
        return jsonify(error), 500

@app.route('/status', methods=['GET'])
def status():
    """Status endpoint"""
    return jsonify({
        "status": "running",
        "environment": ENVIRONMENT,
        "timestamp": time.time()
    })

# ==================== Main Entry Point ====================

def main():
    """
    Main entry point for the script
    """
    global server
    
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process JSON files with semantic chunking and webhook support')
    parser.add_argument('--folder', help='Name of the folder to process')
    parser.add_argument('--test', action='store_true', help='Use test mode')
    parser.add_argument('--file', help='Path to a specific file to process')
    parser.add_argument('--image-urls', help='Path to JSON file with image URLs')
    parser.add_argument('--webhook-only', action='store_true', help='Run only the webhook server')
    parser.add_argument('--port', type=int, default=5000, help='Port for webhook server')
    
    args = parser.parse_args()
    
    # Start webhook server
    server = ServerThread(app, port=args.port)
    server.start()
    logger.info(f"Webhook server running at http://0.0.0.0:{args.port}/webhook")
    
    # Initial message to webhook log
    log_webhook_result({
        "status": "starting",
        "environment": "test" if args.test else "production",
        "webhook_server": f"http://0.0.0.0:{args.port}/webhook",
        "timestamp": datetime.now().isoformat()
    })
    
    # If webhook-only mode, just keep the server running
    if args.webhook_only:
        logger.info("Running in webhook-only mode")
        try:
            # Keep the main thread alive
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down")
        return
    
    try:
        # Process either a specific file or the entire folder
        if args.folder:
            if args.file:
                logger.info(f"Processing specific file: {args.file}")
                stats = process_specific_file(args.folder, args.file, args.test, args.image_urls)
            else:
                logger.info(f"Processing folder: {args.folder}")
                stats = process_folder(args.folder, args.test, args.image_urls)
            
            logger.info(f"Processing complete: {stats}")
        else:
            logger.info("No folder specified, running in webhook server mode")
            # Keep the main thread alive
            while True:
                time.sleep(1)
                
    except Exception as e:
        logger.error(f"Error processing: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()