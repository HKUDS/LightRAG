"""
Data Validation Module for LightRAG
Part of Phase 3: Data Validation implementation.

This module provides comprehensive validation for:
- Input documents and content
- Extracted entities and relationships
- Database schema compliance
- Content sanitization and security
"""

import re
import html
import unicodedata
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from . import utils

utils.setup_logger("lightrag.validation")
logger = logging.getLogger("lightrag.validation")


@dataclass
class ValidationError:
    """Container for validation error information"""
    field: str
    message: str
    severity: str = "error"  # error, warning, info
    context: Optional[Dict[str, Any]] = None


@dataclass
class ValidationResult:
    """Result of a validation operation"""
    is_valid: bool
    errors: List[ValidationError]
    warnings: List[ValidationError]
    sanitized_data: Optional[Dict[str, Any]] = None
    
    def has_errors(self) -> bool:
        return len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0
    
    def add_error(self, field: str, message: str, context: Optional[Dict] = None):
        self.errors.append(ValidationError(field, message, "error", context))
        self.is_valid = False
    
    def add_warning(self, field: str, message: str, context: Optional[Dict] = None):
        self.warnings.append(ValidationError(field, message, "warning", context))


class ContentSanitizer:
    """Sanitize and clean content for safe processing"""
    
    @staticmethod
    def sanitize_text(text: str, max_length: int = 100000) -> str:
        """Sanitize text content for safe processing"""
        if not isinstance(text, str):
            return str(text) if text is not None else ""
        
        # Remove or escape HTML/XML tags
        text = html.escape(text)
        
        # Normalize unicode characters
        text = unicodedata.normalize('NFKC', text)
        
        # Remove control characters except common whitespace
        text = ''.join(char for char in text if unicodedata.category(char) != 'Cc' or char in '\n\r\t ')
        
        # Limit length
        if len(text) > max_length:
            text = text[:max_length] + "..."
            logger.warning(f"Text truncated to {max_length} characters")
        
        return text.strip()
    
    @staticmethod
    def sanitize_entity_name(name: str) -> str:
        """Sanitize entity names for consistency"""
        if not name:
            return ""
        
        name = ContentSanitizer.sanitize_text(name, max_length=500)
        
        # Remove excessive whitespace
        name = re.sub(r'\s+', ' ', name)
        
        # Remove leading/trailing quotes and special characters
        name = name.strip('"\'()[]{}')
        
        return name.strip()
    
    @staticmethod
    def sanitize_file_path(path: str) -> str:
        """Sanitize file paths for security"""
        if not path:
            return "unknown_source"
        
        # Remove path traversal attempts
        path = path.replace('..', '').replace('//', '/')
        
        # Sanitize as text
        path = ContentSanitizer.sanitize_text(path, max_length=1000)
        
        return path if path else "unknown_source"


class DocumentValidator:
    """Validate input documents and content"""
    
    @staticmethod
    def validate_content(content: str, file_path: str = None) -> ValidationResult:
        """Validate document content"""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        # Check if content exists
        if not content:
            result.add_error("content", "Content is empty or None")
            return result
        
        if not isinstance(content, str):
            result.add_warning("content", f"Content is not string type: {type(content)}")
            content = str(content)
        
        # Check content length
        if len(content) < 10:
            result.add_warning("content", f"Content is very short ({len(content)} chars)")
        elif len(content) > 1000000:  # 1MB limit
            result.add_warning("content", f"Content is very large ({len(content)} chars)")
        
        # Check for suspicious patterns
        suspicious_patterns = [
            r'<script[^>]*>.*?</script>',  # Script tags
            r'javascript:',               # JavaScript URLs
            r'data:.*base64',            # Base64 data URLs
            r'\\x[0-9a-f]{2}',          # Hex encoded characters
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, content, re.IGNORECASE | re.DOTALL):
                result.add_warning("content", f"Suspicious pattern detected: {pattern}")
        
        # Sanitize content
        sanitized_content = ContentSanitizer.sanitize_text(content)
        sanitized_file_path = ContentSanitizer.sanitize_file_path(file_path) if file_path else "unknown_source"
        
        result.sanitized_data = {
            "content": sanitized_content,
            "file_path": sanitized_file_path,
            "original_length": len(content),
            "sanitized_length": len(sanitized_content)
        }
        
        return result
    
    @staticmethod
    def validate_chunk(chunk_data: Dict[str, Any]) -> ValidationResult:
        """Validate text chunk data"""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        # Required fields
        required_fields = ["content"]
        for field in required_fields:
            if field not in chunk_data:
                result.add_error(field, f"Required field '{field}' is missing")
        
        # Validate content if present
        if "content" in chunk_data:
            content_result = DocumentValidator.validate_content(
                chunk_data["content"], 
                chunk_data.get("file_path")
            )
            result.errors.extend(content_result.errors)
            result.warnings.extend(content_result.warnings)
            
            if content_result.sanitized_data:
                result.sanitized_data = {
                    "content": content_result.sanitized_data["content"],
                    "file_path": content_result.sanitized_data["file_path"],
                    "tokens": chunk_data.get("tokens", 0),
                    "chunk_order_index": chunk_data.get("chunk_order_index", 0),
                    "full_doc_id": chunk_data.get("full_doc_id", "unknown")
                }
        
        # Validate tokens field
        if "tokens" in chunk_data:
            if not isinstance(chunk_data["tokens"], int) or chunk_data["tokens"] < 0:
                result.add_warning("tokens", "Invalid token count")
        
        return result


class EntityValidator:
    """Validate extracted entities"""
    
    @staticmethod
    def validate_entity(entity_data: Dict[str, Any]) -> ValidationResult:
        """Validate entity data structure and content"""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        # Required fields
        required_fields = ["entity_name", "entity_type", "description"]
        for field in required_fields:
            if field not in entity_data:
                result.add_error(field, f"Required field '{field}' is missing")
        
        # Validate entity_name
        if "entity_name" in entity_data:
            name = entity_data["entity_name"]
            if not name or not str(name).strip():
                result.add_error("entity_name", "Entity name is empty")
            elif len(str(name)) > 500:
                result.add_warning("entity_name", f"Entity name is very long ({len(str(name))} chars)")
        
        # Validate entity_type
        if "entity_type" in entity_data:
            entity_type = entity_data["entity_type"]
            if not entity_type or not str(entity_type).strip():
                result.add_error("entity_type", "Entity type is empty")
            elif entity_type == "UNKNOWN":
                result.add_warning("entity_type", "Entity type is UNKNOWN")
        
        # Validate description
        if "description" in entity_data:
            description = entity_data["description"]
            if not description or not str(description).strip():
                result.add_error("description", "Description is empty")
            elif len(str(description)) < 10:
                result.add_warning("description", f"Description is very short ({len(str(description))} chars)")
        
        # Sanitize data
        if not result.has_errors():
            sanitized_data = {}
            
            if "entity_name" in entity_data:
                sanitized_data["entity_name"] = ContentSanitizer.sanitize_entity_name(str(entity_data["entity_name"]))
            
            if "entity_type" in entity_data:
                sanitized_data["entity_type"] = ContentSanitizer.sanitize_text(str(entity_data["entity_type"]), 100)
            
            if "description" in entity_data:
                sanitized_data["description"] = ContentSanitizer.sanitize_text(str(entity_data["description"]))
            
            # Copy other fields with sanitization
            for field in ["source_id", "file_path"]:
                if field in entity_data:
                    if field == "file_path":
                        sanitized_data[field] = ContentSanitizer.sanitize_file_path(str(entity_data[field]))
                    else:
                        sanitized_data[field] = ContentSanitizer.sanitize_text(str(entity_data[field]))
            
            # Add timestamp if missing
            if "created_at" not in entity_data:
                sanitized_data["created_at"] = int(datetime.now().timestamp())
            else:
                sanitized_data["created_at"] = entity_data["created_at"]
            
            result.sanitized_data = sanitized_data
        
        return result
    
    @staticmethod
    def validate_entity_batch(entities: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[ValidationError]]:
        """Validate a batch of entities and return valid ones"""
        valid_entities = []
        all_errors = []
        
        for i, entity in enumerate(entities):
            validation_result = EntityValidator.validate_entity(entity)
            
            if validation_result.is_valid and validation_result.sanitized_data:
                valid_entities.append(validation_result.sanitized_data)
            else:
                # Add context to errors
                for error in validation_result.errors:
                    error.context = {"entity_index": i, "entity_data": entity}
                all_errors.extend(validation_result.errors)
        
        return valid_entities, all_errors


class RelationshipValidator:
    """Validate extracted relationships"""
    
    @staticmethod
    def validate_relationship(relationship_data: Dict[str, Any]) -> ValidationResult:
        """Validate relationship data structure and content"""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        # Required fields
        required_fields = ["src_id", "tgt_id", "description"]
        for field in required_fields:
            if field not in relationship_data:
                result.add_error(field, f"Required field '{field}' is missing")
        
        # Validate source and target entities
        if "src_id" in relationship_data and "tgt_id" in relationship_data:
            src_id = relationship_data["src_id"]
            tgt_id = relationship_data["tgt_id"]
            
            if not src_id or not str(src_id).strip():
                result.add_error("src_id", "Source entity ID is empty")
            if not tgt_id or not str(tgt_id).strip():
                result.add_error("tgt_id", "Target entity ID is empty")
            
            # Check for self-loops
            if src_id == tgt_id:
                result.add_warning("relationship", "Self-loop relationship detected")
        
        # Validate weight
        if "weight" in relationship_data:
            weight = relationship_data["weight"]
            try:
                weight_float = float(weight)
                if weight_float < 0:
                    result.add_warning("weight", "Negative weight detected")
                elif weight_float > 10:
                    result.add_warning("weight", f"Very high weight detected: {weight_float}")
            except (ValueError, TypeError):
                result.add_error("weight", f"Invalid weight value: {weight}")
        
        # Validate description
        if "description" in relationship_data:
            description = relationship_data["description"]
            if not description or not str(description).strip():
                result.add_error("description", "Relationship description is empty")
        
        # Sanitize data
        if not result.has_errors():
            sanitized_data = {}
            
            for field in ["src_id", "tgt_id"]:
                if field in relationship_data:
                    sanitized_data[field] = ContentSanitizer.sanitize_entity_name(str(relationship_data[field]))
            
            if "description" in relationship_data:
                sanitized_data["description"] = ContentSanitizer.sanitize_text(str(relationship_data["description"]))
            
            if "keywords" in relationship_data:
                sanitized_data["keywords"] = ContentSanitizer.sanitize_text(str(relationship_data["keywords"]))
            
            # Handle weight
            if "weight" in relationship_data:
                try:
                    sanitized_data["weight"] = float(relationship_data["weight"])
                except (ValueError, TypeError):
                    sanitized_data["weight"] = 1.0  # Default weight
            else:
                sanitized_data["weight"] = 1.0
            
            # **CRITICAL FIX**: Preserve relationship type fields
            # These are essential for maintaining LLM-extracted relationship semantics
            type_fields = ["relationship_type", "original_type", "neo4j_type", "rel_type"]
            for field in type_fields:
                if field in relationship_data:
                    sanitized_data[field] = str(relationship_data[field]) if relationship_data[field] is not None else None
            
            # Copy other fields with sanitization
            for field in ["source_id", "file_path"]:
                if field in relationship_data:
                    if field == "file_path":
                        sanitized_data[field] = ContentSanitizer.sanitize_file_path(str(relationship_data[field]))
                    else:
                        sanitized_data[field] = ContentSanitizer.sanitize_text(str(relationship_data[field]))
            
            # Add timestamp if missing
            if "created_at" not in relationship_data:
                sanitized_data["created_at"] = int(datetime.now().timestamp())
            else:
                sanitized_data["created_at"] = relationship_data["created_at"]
            
            result.sanitized_data = sanitized_data
        
        return result
    
    @staticmethod
    def validate_relationship_batch(relationships: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[ValidationError]]:
        """Validate a batch of relationships and return valid ones"""
        valid_relationships = []
        all_errors = []
        
        for i, relationship in enumerate(relationships):
            validation_result = RelationshipValidator.validate_relationship(relationship)
            
            if validation_result.is_valid and validation_result.sanitized_data:
                valid_relationships.append(validation_result.sanitized_data)
            else:
                # Add context to errors
                for error in validation_result.errors:
                    error.context = {"relationship_index": i, "relationship_data": relationship}
                all_errors.extend(validation_result.errors)
        
        return valid_relationships, all_errors


class DatabaseValidator:
    """Validate data for database operations"""
    
    @staticmethod
    def validate_node_data(node_data: Dict[str, Any]) -> ValidationResult:
        """Validate node data before database upsert"""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        # Required fields for Neo4j nodes
        required_fields = ["entity_id", "entity_type"]
        for field in required_fields:
            if field not in node_data:
                result.add_error(field, f"Required field '{field}' is missing for node")
        
        # Validate entity_id (used as primary key)
        if "entity_id" in node_data:
            entity_id = node_data["entity_id"]
            if not entity_id or not str(entity_id).strip():
                result.add_error("entity_id", "Entity ID cannot be empty")
            elif len(str(entity_id)) > 1000:
                result.add_error("entity_id", f"Entity ID is too long ({len(str(entity_id))} chars)")
        
        # Validate all string fields
        string_fields = ["entity_type", "description", "source_id", "file_path"]
        for field in string_fields:
            if field in node_data and node_data[field] is not None:
                if len(str(node_data[field])) > 50000:  # Reasonable limit for Neo4j
                    result.add_warning(field, f"Field '{field}' is very large ({len(str(node_data[field]))} chars)")
        
        return result
    
    @staticmethod
    def validate_edge_data(edge_data: Dict[str, Any]) -> ValidationResult:
        """Validate edge data before database upsert"""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        # Check for weight field and validate
        if "weight" in edge_data:
            try:
                weight = float(edge_data["weight"])
                if weight < 0:
                    result.add_warning("weight", "Negative edge weight")
                elif weight > 100:
                    result.add_warning("weight", f"Very high edge weight: {weight}")
            except (ValueError, TypeError):
                result.add_error("weight", f"Invalid weight value: {edge_data['weight']}")
        
        return result


def validate_extraction_results(
    entities: List[Dict[str, Any]], 
    relationships: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[ValidationError]]:
    """
    Validate and sanitize extraction results
    
    Returns:
        Tuple of (valid_entities, valid_relationships, all_errors)
    """
    all_errors = []
    
    # Validate entities
    valid_entities, entity_errors = EntityValidator.validate_entity_batch(entities)
    all_errors.extend(entity_errors)
    
    # Validate relationships
    valid_relationships, relationship_errors = RelationshipValidator.validate_relationship_batch(relationships)
    all_errors.extend(relationship_errors)
    
    # Log validation summary
    if all_errors:
        logger.warning(f"Validation completed with {len(all_errors)} errors")
        logger.debug(f"Valid entities: {len(valid_entities)}/{len(entities)}")
        logger.debug(f"Valid relationships: {len(valid_relationships)}/{len(relationships)}")
    else:
        logger.debug(f"Validation successful: {len(valid_entities)} entities, {len(valid_relationships)} relationships")
    
    return valid_entities, valid_relationships, all_errors


def log_validation_errors(errors: List[ValidationError], context: str = ""):
    """Log validation errors with appropriate severity levels"""
    if not errors:
        return
    
    context_prefix = f"[{context}] " if context else ""
    
    error_count = len([e for e in errors if e.severity == "error"])
    warning_count = len([e for e in errors if e.severity == "warning"])
    
    if error_count > 0:
        logger.error(f"{context_prefix}Validation found {error_count} errors, {warning_count} warnings")
    elif warning_count > 0:
        logger.warning(f"{context_prefix}Validation found {warning_count} warnings")
    
    # Log individual errors (limit to prevent spam)
    for i, error in enumerate(errors[:10]):  # Limit to first 10 errors
        level = logger.error if error.severity == "error" else logger.warning
        level(f"{context_prefix}{error.field}: {error.message}")
    
    if len(errors) > 10:
        logger.info(f"{context_prefix}... and {len(errors) - 10} more validation issues") 