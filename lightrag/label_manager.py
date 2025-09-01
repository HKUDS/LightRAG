"""
LabelManager for handling document labels in LightRAG
"""
from __future__ import annotations

import os
import json
import shutil
import hashlib
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import asyncio
from lightrag.utils import logger


@dataclass
class DocumentLabel:
    """Represents a document label with metadata"""
    name: str
    description: str = ""
    color: str = "#0066cc"  # Default blue color
    created_at: str = ""
    document_count: int = 0


@dataclass
class LabeledDocument:
    """Represents a document with its labels"""
    doc_id: str
    labels: Set[str]
    file_path: str = ""
    original_path: str = ""


class LabelManager:
    """Manages document labels and directory organization"""
    
    def __init__(self, working_dir: str, auto_create_directories: bool = True):
        self.working_dir = Path(working_dir)
        self.labels_dir = self.working_dir / "labels"
        self.labels_file = self.working_dir / "labels_config.json"
        self.document_labels_file = self.working_dir / "document_labels.json"
        self.auto_create_directories = auto_create_directories
        
        # In-memory caches
        self._labels: Dict[str, DocumentLabel] = {}
        self._document_labels: Dict[str, Set[str]] = {}
        self._label_to_docs: Dict[str, Set[str]] = {}
        
        # Initialize directories synchronously
        self._ensure_directories_sync()
        
        # Data will be loaded lazily when first accessed
        self._data_loaded = False
    
    def _ensure_directories_sync(self):
        """Create necessary directories synchronously"""
        self.working_dir.mkdir(exist_ok=True)
        self.labels_dir.mkdir(exist_ok=True)
    
    async def _ensure_directories(self):
        """Create necessary directories (async version for compatibility)"""
        self._ensure_directories_sync()
    
    async def _ensure_data_loaded(self):
        """Ensure data is loaded from disk"""
        if not self._data_loaded:
            await self._load_data()
            self._data_loaded = True
    
    def _ensure_data_loaded_sync(self):
        """Ensure data is loaded from disk (synchronous version)"""
        if not self._data_loaded:
            self._load_data_sync()
            self._data_loaded = True
    
    def _load_data_sync(self):
        """Load labels and document mappings from disk (synchronous version)"""
        try:
            if self.labels_file.exists():
                with open(self.labels_file, 'r') as f:
                    data = json.load(f)
                    self._labels = {
                        name: DocumentLabel(**label_data) 
                        for name, label_data in data.items()
                    }
            
            if self.document_labels_file.exists():
                with open(self.document_labels_file, 'r') as f:
                    data = json.load(f)
                    self._document_labels = {
                        doc_id: set(labels) 
                        for doc_id, labels in data.items()
                    }
                    
                    # Build reverse mapping
                    self._label_to_docs = {}
                    for doc_id, labels in self._document_labels.items():
                        for label in labels:
                            if label not in self._label_to_docs:
                                self._label_to_docs[label] = set()
                            self._label_to_docs[label].add(doc_id)
                            
        except Exception as e:
            logger.warning(f"Failed to load label data: {e}")
    
    async def _load_data(self):
        """Load labels and document mappings from disk"""
        try:
            if self.labels_file.exists():
                with open(self.labels_file, 'r') as f:
                    data = json.load(f)
                    self._labels = {
                        name: DocumentLabel(**label_data) 
                        for name, label_data in data.items()
                    }
            
            if self.document_labels_file.exists():
                with open(self.document_labels_file, 'r') as f:
                    data = json.load(f)
                    self._document_labels = {
                        doc_id: set(labels) 
                        for doc_id, labels in data.items()
                    }
                    
                    # Build reverse mapping
                    self._label_to_docs = {}
                    for doc_id, labels in self._document_labels.items():
                        for label in labels:
                            if label not in self._label_to_docs:
                                self._label_to_docs[label] = set()
                            self._label_to_docs[label].add(doc_id)
                            
        except Exception as e:
            logger.warning(f"Failed to load label data: {e}")
    
    async def _save_data(self):
        """Save labels and document mappings to disk"""
        try:
            # Save labels
            labels_data = {
                name: {
                    "name": label.name,
                    "description": label.description,
                    "color": label.color,
                    "created_at": label.created_at,
                    "document_count": len(self._label_to_docs.get(name, set()))
                }
                for name, label in self._labels.items()
            }
            with open(self.labels_file, 'w') as f:
                json.dump(labels_data, f, indent=2)
            
            # Save document labels
            doc_labels_data = {
                doc_id: list(labels) 
                for doc_id, labels in self._document_labels.items()
            }
            with open(self.document_labels_file, 'w') as f:
                json.dump(doc_labels_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save label data: {e}")
    
    def validate_label_name(self, name: str) -> bool:
        """Validate label name format"""
        if not name or not isinstance(name, str):
            return False
        # Allow alphanumeric, hyphens, underscores, and spaces
        import re
        return bool(re.match(r'^[a-zA-Z0-9_\-\s]+$', name.strip()))
    
    async def create_label(
        self, 
        name: str, 
        description: str = "", 
        color: str = "#0066cc"
    ) -> bool:
        """Create a new label"""
        await self._ensure_data_loaded()
        
        name = name.strip()
        if not self.validate_label_name(name):
            raise ValueError(f"Invalid label name: {name}")
        
        if name in self._labels:
            logger.warning(f"Label '{name}' already exists")
            return False
        
        # Create label directory if auto_create is enabled
        if self.auto_create_directories:
            label_dir = self.labels_dir / self._sanitize_filename(name)
            label_dir.mkdir(exist_ok=True)
        
        # Add to in-memory cache
        from datetime import datetime
        self._labels[name] = DocumentLabel(
            name=name,
            description=description,
            color=color,
            created_at=datetime.now().isoformat(),
            document_count=0
        )
        
        await self._save_data()
        logger.info(f"Created label: {name}")
        return True
    
    async def assign_labels_to_document(
        self, 
        doc_id: str, 
        labels: List[str],
        file_path: str = ""
    ) -> bool:
        """Assign labels to a document"""
        if not doc_id or not labels:
            return False
        
        # Validate labels
        valid_labels = []
        for label in labels:
            label = label.strip()
            if not self.validate_label_name(label):
                logger.warning(f"Skipping invalid label: {label}")
                continue
            
            # Auto-create label if it doesn't exist
            if label not in self._labels:
                await self.create_label(label)
            
            valid_labels.append(label)
        
        if not valid_labels:
            return False
        
        # Remove document from old label mappings
        if doc_id in self._document_labels:
            old_labels = self._document_labels[doc_id]
            for old_label in old_labels:
                if old_label in self._label_to_docs:
                    self._label_to_docs[old_label].discard(doc_id)
        
        # Update document labels
        self._document_labels[doc_id] = set(valid_labels)
        
        # Update reverse mapping
        for label in valid_labels:
            if label not in self._label_to_docs:
                self._label_to_docs[label] = set()
            self._label_to_docs[label].add(doc_id)
        
        # Move/organize files if file_path is provided
        if file_path and self.auto_create_directories:
            await self._organize_document_file(doc_id, file_path, valid_labels)
        
        await self._save_data()
        logger.info(f"Assigned labels {valid_labels} to document {doc_id}")
        return True
    
    async def _organize_document_file(
        self, 
        doc_id: str, 
        file_path: str, 
        labels: List[str]
    ):
        """Organize document file into label directories"""
        try:
            source_path = Path(file_path)
            if not source_path.exists():
                return
            
            # For multiple labels, use the first one as primary
            primary_label = labels[0]
            label_dir = self.labels_dir / self._sanitize_filename(primary_label)
            label_dir.mkdir(exist_ok=True)
            
            # Copy file to label directory
            dest_path = label_dir / source_path.name
            if dest_path != source_path:
                shutil.copy2(source_path, dest_path)
                logger.debug(f"Copied {source_path} to {dest_path}")
            
            # Create symlinks for other labels
            for label in labels[1:]:
                label_dir = self.labels_dir / self._sanitize_filename(label)
                label_dir.mkdir(exist_ok=True)
                symlink_path = label_dir / source_path.name
                
                try:
                    if not symlink_path.exists():
                        symlink_path.symlink_to(dest_path)
                        logger.debug(f"Created symlink {symlink_path} -> {dest_path}")
                except OSError:
                    # Fallback to copy if symlinks not supported
                    if not symlink_path.exists():
                        shutil.copy2(dest_path, symlink_path)
                        
        except Exception as e:
            logger.warning(f"Failed to organize document file: {e}")
    
    def _sanitize_filename(self, name: str) -> str:
        """Sanitize label name for use as directory name"""
        import re
        # Replace invalid characters with underscores
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', name)
        # Remove leading/trailing whitespace and replace spaces with underscores
        sanitized = sanitized.strip().replace(' ', '_')
        return sanitized
    
    def get_documents_by_label(self, label_name: str) -> Set[str]:
        """Get all document IDs with a specific label"""
        self._ensure_data_loaded_sync()
        return self._label_to_docs.get(label_name, set()).copy()
    
    def get_document_labels(self, doc_id: str) -> Set[str]:
        """Get all labels for a specific document"""
        self._ensure_data_loaded_sync()
        return self._document_labels.get(doc_id, set()).copy()
    
    def get_all_labels(self) -> Dict[str, DocumentLabel]:
        """Get all available labels"""
        self._ensure_data_loaded_sync()
        return self._labels.copy()
    
    def get_label_statistics(self) -> Dict[str, Any]:
        """Get statistics about labels and documents"""
        return {
            "total_labels": len(self._labels),
            "total_labeled_documents": len(self._document_labels),
            "labels_with_counts": {
                label: len(self._label_to_docs.get(label, set()))
                for label in self._labels.keys()
            }
        }
    
    async def update_document_labels(self, doc_id: str, new_labels: List[str]) -> bool:
        """Update labels for a document"""
        return await self.assign_labels_to_document(doc_id, new_labels)
    
    async def remove_label_from_document(self, doc_id: str, label: str) -> bool:
        """Remove a specific label from a document"""
        if doc_id not in self._document_labels:
            return False
        
        if label in self._document_labels[doc_id]:
            self._document_labels[doc_id].remove(label)
            
            # Update reverse mapping
            if label in self._label_to_docs:
                self._label_to_docs[label].discard(doc_id)
            
            # Remove document entry if no labels left
            if not self._document_labels[doc_id]:
                del self._document_labels[doc_id]
            
            await self._save_data()
            logger.info(f"Removed label '{label}' from document {doc_id}")
            return True
        
        return False
    
    async def delete_label(self, label_name: str) -> bool:
        """Delete a label and remove it from all documents"""
        if label_name not in self._labels:
            return False
        
        # Remove from all documents
        docs_to_update = self._label_to_docs.get(label_name, set()).copy()
        for doc_id in docs_to_update:
            if doc_id in self._document_labels:
                self._document_labels[doc_id].discard(label_name)
                if not self._document_labels[doc_id]:
                    del self._document_labels[doc_id]
        
        # Remove label
        del self._labels[label_name]
        if label_name in self._label_to_docs:
            del self._label_to_docs[label_name]
        
        # Remove directory if it exists
        label_dir = self.labels_dir / self._sanitize_filename(label_name)
        try:
            if label_dir.exists():
                shutil.rmtree(label_dir)
        except Exception as e:
            logger.warning(f"Failed to remove label directory {label_dir}: {e}")
        
        await self._save_data()
        logger.info(f"Deleted label: {label_name}")
        return True
    
    async def cleanup_empty_directories(self):
        """Remove empty label directories"""
        try:
            for label_dir in self.labels_dir.iterdir():
                if label_dir.is_dir() and not any(label_dir.iterdir()):
                    label_dir.rmdir()
                    logger.debug(f"Removed empty directory: {label_dir}")
        except Exception as e:
            logger.warning(f"Failed to cleanup directories: {e}")
    
    def filter_documents_by_labels(
        self, 
        doc_ids: Set[str], 
        required_labels: List[str],
        match_all: bool = False
    ) -> Set[str]:
        """
        Filter document IDs by labels
        
        Args:
            doc_ids: Set of document IDs to filter
            required_labels: List of labels to filter by
            match_all: If True, document must have ALL labels. If False, ANY label.
        
        Returns:
            Set of filtered document IDs
        """
        if not required_labels:
            return doc_ids
        
        filtered_docs = set()
        
        for doc_id in doc_ids:
            doc_labels = self.get_document_labels(doc_id)
            
            if match_all:
                # Document must have all required labels
                if all(label in doc_labels for label in required_labels):
                    filtered_docs.add(doc_id)
            else:
                # Document must have at least one required label
                if any(label in doc_labels for label in required_labels):
                    filtered_docs.add(doc_id)
        
        return filtered_docs
    
    async def migrate_existing_documents(self, doc_metadata: Dict[str, Dict]) -> int:
        """Migrate existing documents to use labels based on metadata"""
        migrated_count = 0
        
        for doc_id, metadata in doc_metadata.items():
            # Extract potential labels from file path or metadata
            labels = []
            
            file_path = metadata.get('file_path', '')
            if file_path:
                # Use directory name as label
                path_parts = Path(file_path).parts
                if len(path_parts) > 1:
                    directory_label = path_parts[-2]  # Parent directory
                    if self.validate_label_name(directory_label):
                        labels.append(directory_label)
            
            # Check for existing labels in metadata
            if 'labels' in metadata:
                existing_labels = metadata['labels']
                if isinstance(existing_labels, list):
                    labels.extend(existing_labels)
                elif isinstance(existing_labels, str):
                    labels.append(existing_labels)
            
            # Default label if none found
            if not labels:
                labels = ['general']
            
            if await self.assign_labels_to_document(doc_id, labels, file_path):
                migrated_count += 1
        
        logger.info(f"Migrated {migrated_count} documents with labels")
        return migrated_count