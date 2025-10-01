import asyncio
import os
from typing import Any, final
from dataclasses import dataclass
import numpy as np
from lightrag.utils import logger, compute_mdhash_id
from ..base import BaseVectorStorage
from ..constants import DEFAULT_MAX_FILE_PATH_LENGTH
from ..kg.shared_storage import get_data_init_lock, get_storage_lock
import pipmaster as pm

if not pm.is_installed("pymilvus"):
    pm.install("pymilvus==2.5.2")

import configparser
from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema  # type: ignore

config = configparser.ConfigParser()
config.read("config.ini", "utf-8")


@final
@dataclass
class MilvusVectorDBStorage(BaseVectorStorage):
    def _create_schema_for_namespace(self) -> CollectionSchema:
        """Create schema based on the current instance's namespace"""

        # Get vector dimension from embedding_func
        dimension = self.embedding_func.embedding_dim

        # Base fields (common to all collections)
        base_fields = [
            FieldSchema(
                name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True
            ),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension),
            FieldSchema(name="created_at", dtype=DataType.INT64),
        ]

        # Determine specific fields based on namespace
        if self.namespace.endswith("entities"):
            specific_fields = [
                FieldSchema(
                    name="entity_name",
                    dtype=DataType.VARCHAR,
                    max_length=512,
                    nullable=True,
                ),
                FieldSchema(
                    name="file_path",
                    dtype=DataType.VARCHAR,
                    max_length=DEFAULT_MAX_FILE_PATH_LENGTH,
                    nullable=True,
                ),
            ]
            description = "LightRAG entities vector storage"

        elif self.namespace.endswith("relationships"):
            specific_fields = [
                FieldSchema(
                    name="src_id", dtype=DataType.VARCHAR, max_length=512, nullable=True
                ),
                FieldSchema(
                    name="tgt_id", dtype=DataType.VARCHAR, max_length=512, nullable=True
                ),
                FieldSchema(
                    name="file_path",
                    dtype=DataType.VARCHAR,
                    max_length=DEFAULT_MAX_FILE_PATH_LENGTH,
                    nullable=True,
                ),
            ]
            description = "LightRAG relationships vector storage"

        elif self.namespace.endswith("chunks"):
            specific_fields = [
                FieldSchema(
                    name="full_doc_id",
                    dtype=DataType.VARCHAR,
                    max_length=64,
                    nullable=True,
                ),
                FieldSchema(
                    name="file_path",
                    dtype=DataType.VARCHAR,
                    max_length=DEFAULT_MAX_FILE_PATH_LENGTH,
                    nullable=True,
                ),
            ]
            description = "LightRAG chunks vector storage"

        else:
            # Default generic schema (backward compatibility)
            specific_fields = [
                FieldSchema(
                    name="file_path",
                    dtype=DataType.VARCHAR,
                    max_length=DEFAULT_MAX_FILE_PATH_LENGTH,
                    nullable=True,
                ),
            ]
            description = "LightRAG generic vector storage"

        # Merge all fields
        all_fields = base_fields + specific_fields

        return CollectionSchema(
            fields=all_fields,
            description=description,
            enable_dynamic_field=True,  # Support dynamic fields
        )

    def _get_index_params(self):
        """Get IndexParams in a version-compatible way"""
        try:
            # Try to use client's prepare_index_params method (most common)
            if hasattr(self._client, "prepare_index_params"):
                return self._client.prepare_index_params()
        except Exception:
            pass

        try:
            # Try to import IndexParams from different possible locations
            from pymilvus.client.prepare import IndexParams

            return IndexParams()
        except ImportError:
            pass

        try:
            from pymilvus.client.types import IndexParams

            return IndexParams()
        except ImportError:
            pass

        try:
            from pymilvus import IndexParams

            return IndexParams()
        except ImportError:
            pass

        # If all else fails, return None to use fallback method
        return None

    def _create_vector_index_fallback(self):
        """Fallback method to create vector index using direct API"""
        try:
            self._client.create_index(
                collection_name=self.final_namespace,
                field_name="vector",
                index_params={
                    "index_type": "HNSW",
                    "metric_type": "COSINE",
                    "params": {"M": 16, "efConstruction": 256},
                },
            )
            logger.debug(
                f"[{self.workspace}] Created vector index using fallback method"
            )
        except Exception as e:
            logger.warning(
                f"[{self.workspace}] Failed to create vector index using fallback method: {e}"
            )

    def _create_scalar_index_fallback(self, field_name: str, index_type: str):
        """Fallback method to create scalar index using direct API"""
        # Skip unsupported index types
        if index_type == "SORTED":
            logger.info(
                f"[{self.workspace}] Skipping SORTED index for {field_name} (not supported in this Milvus version)"
            )
            return

        try:
            self._client.create_index(
                collection_name=self.final_namespace,
                field_name=field_name,
                index_params={"index_type": index_type},
            )
            logger.debug(
                f"[{self.workspace}] Created {field_name} index using fallback method"
            )
        except Exception as e:
            logger.info(
                f"[{self.workspace}] Could not create {field_name} index using fallback method: {e}"
            )

    def _create_indexes_after_collection(self):
        """Create indexes after collection is created"""
        try:
            # Try to get IndexParams in a version-compatible way
            IndexParamsClass = self._get_index_params()

            if IndexParamsClass is not None:
                # Use IndexParams approach if available
                try:
                    # Create vector index first (required for most operations)
                    vector_index = IndexParamsClass
                    vector_index.add_index(
                        field_name="vector",
                        index_type="HNSW",
                        metric_type="COSINE",
                        params={"M": 16, "efConstruction": 256},
                    )
                    self._client.create_index(
                        collection_name=self.final_namespace, index_params=vector_index
                    )
                    logger.debug(
                        f"[{self.workspace}] Created vector index using IndexParams"
                    )
                except Exception as e:
                    logger.debug(
                        f"[{self.workspace}] IndexParams method failed for vector index: {e}"
                    )
                    self._create_vector_index_fallback()

                # Create scalar indexes based on namespace
                if self.namespace.endswith("entities"):
                    # Create indexes for entity fields
                    try:
                        entity_name_index = self._get_index_params()
                        entity_name_index.add_index(
                            field_name="entity_name", index_type="INVERTED"
                        )
                        self._client.create_index(
                            collection_name=self.final_namespace,
                            index_params=entity_name_index,
                        )
                    except Exception as e:
                        logger.debug(
                            f"[{self.workspace}] IndexParams method failed for entity_name: {e}"
                        )
                        self._create_scalar_index_fallback("entity_name", "INVERTED")

                elif self.namespace.endswith("relationships"):
                    # Create indexes for relationship fields
                    try:
                        src_id_index = self._get_index_params()
                        src_id_index.add_index(
                            field_name="src_id", index_type="INVERTED"
                        )
                        self._client.create_index(
                            collection_name=self.final_namespace,
                            index_params=src_id_index,
                        )
                    except Exception as e:
                        logger.debug(
                            f"[{self.workspace}] IndexParams method failed for src_id: {e}"
                        )
                        self._create_scalar_index_fallback("src_id", "INVERTED")

                    try:
                        tgt_id_index = self._get_index_params()
                        tgt_id_index.add_index(
                            field_name="tgt_id", index_type="INVERTED"
                        )
                        self._client.create_index(
                            collection_name=self.final_namespace,
                            index_params=tgt_id_index,
                        )
                    except Exception as e:
                        logger.debug(
                            f"[{self.workspace}] IndexParams method failed for tgt_id: {e}"
                        )
                        self._create_scalar_index_fallback("tgt_id", "INVERTED")

                elif self.namespace.endswith("chunks"):
                    # Create indexes for chunk fields
                    try:
                        doc_id_index = self._get_index_params()
                        doc_id_index.add_index(
                            field_name="full_doc_id", index_type="INVERTED"
                        )
                        self._client.create_index(
                            collection_name=self.final_namespace,
                            index_params=doc_id_index,
                        )
                    except Exception as e:
                        logger.debug(
                            f"[{self.workspace}] IndexParams method failed for full_doc_id: {e}"
                        )
                        self._create_scalar_index_fallback("full_doc_id", "INVERTED")

                # No common indexes needed

            else:
                # Fallback to direct API calls if IndexParams is not available
                logger.info(
                    f"[{self.workspace}] IndexParams not available, using fallback methods for {self.namespace}"
                )

                # Create vector index using fallback
                self._create_vector_index_fallback()

                # Create scalar indexes using fallback
                if self.namespace.endswith("entities"):
                    self._create_scalar_index_fallback("entity_name", "INVERTED")
                elif self.namespace.endswith("relationships"):
                    self._create_scalar_index_fallback("src_id", "INVERTED")
                    self._create_scalar_index_fallback("tgt_id", "INVERTED")
                elif self.namespace.endswith("chunks"):
                    self._create_scalar_index_fallback("full_doc_id", "INVERTED")

            logger.info(
                f"[{self.workspace}] Created indexes for collection: {self.namespace}"
            )

        except Exception as e:
            logger.warning(
                f"[{self.workspace}] Failed to create some indexes for {self.namespace}: {e}"
            )

    def _get_required_fields_for_namespace(self) -> dict:
        """Get required core field definitions for current namespace"""

        # Base fields (common to all types)
        base_fields = {
            "id": {"type": "VarChar", "is_primary": True},
            "vector": {"type": "FloatVector"},
            "created_at": {"type": "Int64"},
        }

        # Add specific fields based on namespace
        if self.namespace.endswith("entities"):
            specific_fields = {
                "entity_name": {"type": "VarChar"},
                "file_path": {"type": "VarChar"},
            }
        elif self.namespace.endswith("relationships"):
            specific_fields = {
                "src_id": {"type": "VarChar"},
                "tgt_id": {"type": "VarChar"},
                "file_path": {"type": "VarChar"},
            }
        elif self.namespace.endswith("chunks"):
            specific_fields = {
                "full_doc_id": {"type": "VarChar"},
                "file_path": {"type": "VarChar"},
            }
        else:
            specific_fields = {
                "file_path": {"type": "VarChar"},
            }

        return {**base_fields, **specific_fields}

    def _is_field_compatible(self, existing_field: dict, expected_config: dict) -> bool:
        """Check compatibility of a single field"""
        field_name = existing_field.get("name", "unknown")
        existing_type = existing_field.get("type")
        expected_type = expected_config.get("type")

        logger.debug(
            f"[{self.workspace}] Checking field '{field_name}': existing_type={existing_type} (type={type(existing_type)}), expected_type={expected_type}"
        )

        # Convert DataType enum values to string names if needed
        original_existing_type = existing_type
        if hasattr(existing_type, "name"):
            existing_type = existing_type.name
            logger.debug(
                f"[{self.workspace}] Converted enum to name: {original_existing_type} -> {existing_type}"
            )
        elif isinstance(existing_type, int):
            # Map common Milvus internal type codes to type names for backward compatibility
            type_mapping = {
                21: "VarChar",
                101: "FloatVector",
                5: "Int64",
                9: "Double",
            }
            mapped_type = type_mapping.get(existing_type, str(existing_type))
            logger.debug(
                f"[{self.workspace}] Mapped numeric type: {existing_type} -> {mapped_type}"
            )
            existing_type = mapped_type

        # Normalize type names for comparison
        type_aliases = {
            "VARCHAR": "VarChar",
            "String": "VarChar",
            "FLOAT_VECTOR": "FloatVector",
            "INT64": "Int64",
            "BigInt": "Int64",
            "DOUBLE": "Double",
            "Float": "Double",
        }

        original_existing = existing_type
        original_expected = expected_type
        existing_type = type_aliases.get(existing_type, existing_type)
        expected_type = type_aliases.get(expected_type, expected_type)

        if original_existing != existing_type or original_expected != expected_type:
            logger.debug(
                f"[{self.workspace}] Applied aliases: {original_existing} -> {existing_type}, {original_expected} -> {expected_type}"
            )

        # Basic type compatibility check
        type_compatible = existing_type == expected_type
        logger.debug(
            f"[{self.workspace}] Type compatibility for '{field_name}': {existing_type} == {expected_type} -> {type_compatible}"
        )

        if not type_compatible:
            logger.warning(
                f"[{self.workspace}] Type mismatch for field '{field_name}': expected {expected_type}, got {existing_type}"
            )
            return False

        # Primary key check - be more flexible about primary key detection
        if expected_config.get("is_primary"):
            # Check multiple possible field names for primary key status
            is_primary = (
                existing_field.get("is_primary_key", False)
                or existing_field.get("is_primary", False)
                or existing_field.get("primary_key", False)
            )
            logger.debug(
                f"[{self.workspace}] Primary key check for '{field_name}': expected=True, actual={is_primary}"
            )
            logger.debug(
                f"[{self.workspace}] Raw field data for '{field_name}': {existing_field}"
            )

            # For ID field, be more lenient - if it's the ID field, assume it should be primary
            if field_name == "id" and not is_primary:
                logger.info(
                    f"[{self.workspace}] ID field '{field_name}' not marked as primary in existing collection, but treating as compatible"
                )
                # Don't fail for ID field primary key mismatch
            elif not is_primary:
                logger.warning(
                    f"[{self.workspace}] Primary key mismatch for field '{field_name}': expected primary key, but field is not primary"
                )
                return False

        logger.debug(f"[{self.workspace}] Field '{field_name}' is compatible")
        return True

    def _check_vector_dimension(self, collection_info: dict):
        """Check vector dimension compatibility"""
        current_dimension = self.embedding_func.embedding_dim

        # Find vector field dimension
        for field in collection_info.get("fields", []):
            if field.get("name") == "vector":
                field_type = field.get("type")

                # Extract type name from DataType enum or string
                type_name = None
                if hasattr(field_type, "name"):
                    type_name = field_type.name
                elif isinstance(field_type, str):
                    type_name = field_type
                else:
                    type_name = str(field_type)

                # Check if it's a vector type (supports multiple formats)
                if type_name in ["FloatVector", "FLOAT_VECTOR"]:
                    existing_dimension = field.get("params", {}).get("dim")

                    if existing_dimension != current_dimension:
                        raise ValueError(
                            f"Vector dimension mismatch for collection '{self.final_namespace}': "
                            f"existing={existing_dimension}, current={current_dimension}"
                        )

                    logger.debug(
                        f"[{self.workspace}] Vector dimension check passed: {current_dimension}"
                    )
                    return

        # If no vector field found, this might be an old collection created with simple schema
        logger.warning(
            f"[{self.workspace}] Vector field not found in collection '{self.namespace}'. This might be an old collection created with simple schema."
        )
        logger.warning(
            f"[{self.workspace}] Consider recreating the collection for optimal performance."
        )
        return

    def _check_file_path_length_restriction(self, collection_info: dict) -> bool:
        """Check if collection has file_path length restrictions that need migration

        Returns:
            bool: True if migration is needed, False otherwise
        """
        existing_fields = {
            field["name"]: field for field in collection_info.get("fields", [])
        }

        # Check if file_path field exists and has length restrictions
        if "file_path" in existing_fields:
            file_path_field = existing_fields["file_path"]
            # Get max_length from field params
            max_length = file_path_field.get("params", {}).get("max_length")

            if max_length and max_length < DEFAULT_MAX_FILE_PATH_LENGTH:
                logger.info(
                    f"[{self.workspace}] Collection {self.namespace} has file_path max_length={max_length}, "
                    f"needs migration to {DEFAULT_MAX_FILE_PATH_LENGTH}"
                )
                return True

        return False

    def _check_schema_compatibility(self, collection_info: dict):
        """Check schema field compatibility and detect migration needs"""
        existing_fields = {
            field["name"]: field for field in collection_info.get("fields", [])
        }

        # Check if this is an old collection created with simple schema
        has_vector_field = any(
            field.get("name") == "vector" for field in collection_info.get("fields", [])
        )

        if not has_vector_field:
            logger.warning(
                f"[{self.workspace}] Collection {self.namespace} appears to be created with old simple schema (no vector field)"
            )
            logger.warning(
                f"[{self.workspace}] This collection will work but may have suboptimal performance"
            )
            logger.warning(
                f"[{self.workspace}] Consider recreating the collection for optimal performance"
            )
            return

        # Check if migration is needed for file_path length restrictions
        if self._check_file_path_length_restriction(collection_info):
            logger.info(
                f"[{self.workspace}] Starting automatic migration for collection {self.namespace}"
            )
            self._migrate_collection_schema()
            return

        # For collections with vector field, check basic compatibility
        # Only check for critical incompatibilities, not missing optional fields
        critical_fields = {"id": {"type": "VarChar", "is_primary": True}}

        incompatible_fields = []

        for field_name, expected_config in critical_fields.items():
            if field_name in existing_fields:
                existing_field = existing_fields[field_name]
                if not self._is_field_compatible(existing_field, expected_config):
                    incompatible_fields.append(
                        f"{field_name}: expected {expected_config['type']}, "
                        f"got {existing_field.get('type')}"
                    )

        if incompatible_fields:
            raise ValueError(
                f"Critical schema incompatibility in collection '{self.final_namespace}': {incompatible_fields}"
            )

        # Get all expected fields for informational purposes
        expected_fields = self._get_required_fields_for_namespace()
        missing_fields = [
            field for field in expected_fields if field not in existing_fields
        ]

        if missing_fields:
            logger.info(
                f"[{self.workspace}] Collection {self.namespace} missing optional fields: {missing_fields}"
            )
            logger.info(
                "These fields would be available in a newly created collection for better performance"
            )

        logger.debug(
            f"[{self.workspace}] Schema compatibility check passed for {self.namespace}"
        )

    def _migrate_collection_schema(self):
        """Migrate collection schema using query_iterator - completely solves query window limitations"""
        original_collection_name = self.final_namespace
        temp_collection_name = f"{self.final_namespace}_temp"
        iterator = None

        try:
            logger.info(
                f"[{self.workspace}] Starting iterator-based schema migration for {self.namespace}"
            )

            # Step 1: Create temporary collection with new schema
            logger.info(
                f"[{self.workspace}] Step 1: Creating temporary collection: {temp_collection_name}"
            )
            # Temporarily update final_namespace for index creation
            self.final_namespace = temp_collection_name
            new_schema = self._create_schema_for_namespace()
            self._client.create_collection(
                collection_name=temp_collection_name, schema=new_schema
            )
            try:
                self._create_indexes_after_collection()
            except Exception as index_error:
                logger.warning(
                    f"[{self.workspace}] Failed to create indexes for new collection: {index_error}"
                )
                # Continue with migration even if index creation fails

            # Load the new collection
            self._client.load_collection(temp_collection_name)

            # Step 2: Copy data using query_iterator (solves query window limitation)
            logger.info(
                f"[{self.workspace}] Step 2: Copying data using query_iterator from: {original_collection_name}"
            )

            # Create query iterator
            try:
                iterator = self._client.query_iterator(
                    collection_name=original_collection_name,
                    batch_size=2000,  # Adjustable batch size for optimal performance
                    output_fields=["*"],  # Get all fields
                )
                logger.debug(f"[{self.workspace}] Query iterator created successfully")
            except Exception as iterator_error:
                logger.error(
                    f"[{self.workspace}] Failed to create query iterator: {iterator_error}"
                )
                raise

            # Iterate through all data
            total_migrated = 0
            batch_number = 1

            while True:
                try:
                    batch_data = iterator.next()
                    if not batch_data:
                        # No more data available
                        break

                    # Insert batch data to new collection
                    try:
                        self._client.insert(
                            collection_name=temp_collection_name, data=batch_data
                        )
                        total_migrated += len(batch_data)

                        logger.info(
                            f"[{self.workspace}] Iterator batch {batch_number}: "
                            f"processed {len(batch_data)} records, total migrated: {total_migrated}"
                        )
                        batch_number += 1

                    except Exception as batch_error:
                        logger.error(
                            f"[{self.workspace}] Failed to insert iterator batch {batch_number}: {batch_error}"
                        )
                        raise

                except Exception as next_error:
                    logger.error(
                        f"[{self.workspace}] Iterator next() failed at batch {batch_number}: {next_error}"
                    )
                    raise

            if total_migrated > 0:
                logger.info(
                    f"[{self.workspace}] Successfully migrated {total_migrated} records using iterator"
                )
            else:
                logger.info(
                    f"[{self.workspace}] No data found in original collection, migration completed"
                )

            # Step 3: Rename origin collection (keep for safety)
            logger.info(
                f"[{self.workspace}] Step 3: Rename origin collection to {original_collection_name}_old"
            )
            try:
                self._client.rename_collection(
                    original_collection_name, f"{original_collection_name}_old"
                )
            except Exception as rename_error:
                try:
                    logger.warning(
                        f"[{self.workspace}] Try to drop origin collection instead"
                    )
                    self._client.drop_collection(original_collection_name)
                except Exception as e:
                    logger.error(
                        f"[{self.workspace}] Rename operation failed: {rename_error}"
                    )
                    raise e

            # Step 4: Rename temporary collection to original name
            logger.info(
                f"[{self.workspace}] Step 4: Renaming collection {temp_collection_name} -> {original_collection_name}"
            )
            try:
                self._client.rename_collection(
                    temp_collection_name, original_collection_name
                )
                logger.info(f"[{self.workspace}] Rename operation completed")
            except Exception as rename_error:
                logger.error(
                    f"[{self.workspace}] Rename operation failed: {rename_error}"
                )
                raise RuntimeError(
                    f"Failed to rename collection: {rename_error}"
                ) from rename_error

            # Restore final_namespace
            self.final_namespace = original_collection_name

        except Exception as e:
            logger.error(
                f"[{self.workspace}] Iterator-based migration failed for {self.namespace}: {e}"
            )

            # Attempt cleanup of temporary collection if it exists
            try:
                if self._client and self._client.has_collection(temp_collection_name):
                    logger.info(
                        f"[{self.workspace}] Cleaning up failed migration temporary collection"
                    )
                    self._client.drop_collection(temp_collection_name)
            except Exception as cleanup_error:
                logger.warning(
                    f"[{self.workspace}] Failed to cleanup temporary collection: {cleanup_error}"
                )

            # Re-raise the original error
            raise RuntimeError(
                f"Iterator-based migration failed for collection {self.namespace}: {e}"
            ) from e

        finally:
            # Ensure iterator is properly closed
            if iterator:
                try:
                    iterator.close()
                    logger.debug(
                        f"[{self.workspace}] Query iterator closed successfully"
                    )
                except Exception as close_error:
                    logger.warning(
                        f"[{self.workspace}] Failed to close query iterator: {close_error}"
                    )

    def _validate_collection_compatibility(self):
        """Validate existing collection's dimension and schema compatibility"""
        try:
            collection_info = self._client.describe_collection(self.final_namespace)

            # 1. Check vector dimension
            self._check_vector_dimension(collection_info)

            # 2. Check schema compatibility
            self._check_schema_compatibility(collection_info)

            logger.info(
                f"[{self.workspace}] VectorDB Collection '{self.namespace}' compatibility validation passed"
            )

        except Exception as e:
            logger.error(
                f"[{self.workspace}] Collection compatibility validation failed for {self.namespace}: {e}"
            )
            raise

    def _ensure_collection_loaded(self):
        """Ensure the collection is loaded into memory for search operations"""
        try:
            # Check if collection exists first
            if not self._client.has_collection(self.final_namespace):
                logger.error(
                    f"[{self.workspace}] Collection {self.namespace} does not exist"
                )
                raise ValueError(f"Collection {self.final_namespace} does not exist")

            # Load the collection if it's not already loaded
            # In Milvus, collections need to be loaded before they can be searched
            self._client.load_collection(self.final_namespace)
            # logger.debug(f"[{self.workspace}] Collection {self.namespace} loaded successfully")

        except Exception as e:
            logger.error(
                f"[{self.workspace}] Failed to load collection {self.namespace}: {e}"
            )
            raise

    def _create_collection_if_not_exist(self):
        """Create collection if not exists and check existing collection compatibility"""

        try:
            # Check if our specific collection exists
            collection_exists = self._client.has_collection(self.final_namespace)
            logger.info(
                f"[{self.workspace}] VectorDB collection '{self.namespace}' exists check: {collection_exists}"
            )

            if collection_exists:
                # Double-check by trying to describe the collection
                try:
                    self._client.describe_collection(self.final_namespace)
                    self._validate_collection_compatibility()
                    # Ensure the collection is loaded after validation
                    self._ensure_collection_loaded()
                    return
                except Exception as validation_error:
                    # CRITICAL: Collection exists but validation failed
                    # This indicates potential data migration failure or incompatible schema
                    # Stop execution to prevent data loss and require manual intervention
                    logger.error(
                        f"[{self.workspace}] CRITICAL ERROR: Collection '{self.namespace}' exists but validation failed!"
                    )
                    logger.error(
                        f"[{self.workspace}] This indicates potential data migration failure or schema incompatibility."
                    )
                    logger.error(
                        f"[{self.workspace}] Validation error: {validation_error}"
                    )
                    logger.error(f"[{self.workspace}] MANUAL INTERVENTION REQUIRED:")
                    logger.error(
                        f"[{self.workspace}] 1. Check the existing collection schema and data integrity"
                    )
                    logger.error(
                        f"[{self.workspace}] 2. Backup existing data if needed"
                    )
                    logger.error(
                        f"[{self.workspace}] 3. Manually resolve schema compatibility issues"
                    )
                    logger.error(
                        f"[{self.workspace}] 4. Consider dropping and recreating the collection if data is not critical"
                    )
                    logger.error(
                        f"[{self.workspace}] Program execution stopped to prevent potential data loss."
                    )

                    # Raise a specific exception to stop execution
                    raise RuntimeError(
                        f"Collection validation failed for '{self.final_namespace}'. "
                        f"Data migration failure detected. Manual intervention required to prevent data loss. "
                        f"Original error: {validation_error}"
                    )

            # Collection doesn't exist, create new collection
            logger.info(f"[{self.workspace}] Creating new collection: {self.namespace}")
            schema = self._create_schema_for_namespace()

            # Create collection with schema only first
            self._client.create_collection(
                collection_name=self.final_namespace, schema=schema
            )

            # Then create indexes
            self._create_indexes_after_collection()

            # Load the newly created collection
            self._ensure_collection_loaded()

            logger.info(
                f"[{self.workspace}] Successfully created Milvus collection: {self.namespace}"
            )

        except RuntimeError:
            # Re-raise RuntimeError (validation failures) without modification
            # These are critical errors that should stop execution
            raise

        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error in _create_collection_if_not_exist for {self.namespace}: {e}"
            )

            # If there's any error (other than validation failure), try to force create the collection
            logger.info(
                f"[{self.workspace}] Attempting to force create collection {self.namespace}..."
            )
            try:
                # Try to drop the collection first if it exists in a bad state
                try:
                    if self._client.has_collection(self.final_namespace):
                        logger.info(
                            f"[{self.workspace}] Dropping potentially corrupted collection {self.namespace}"
                        )
                        self._client.drop_collection(self.final_namespace)
                except Exception as drop_error:
                    logger.warning(
                        f"[{self.workspace}] Could not drop collection {self.namespace}: {drop_error}"
                    )

                # Create fresh collection
                schema = self._create_schema_for_namespace()
                self._client.create_collection(
                    collection_name=self.final_namespace, schema=schema
                )
                self._create_indexes_after_collection()

                # Load the newly created collection
                self._ensure_collection_loaded()

                logger.info(
                    f"[{self.workspace}] Successfully force-created collection {self.namespace}"
                )

            except Exception as create_error:
                logger.error(
                    f"[{self.workspace}] Failed to force-create collection {self.namespace}: {create_error}"
                )
                raise

    def __post_init__(self):
        # Check for MILVUS_WORKSPACE environment variable first (higher priority)
        # This allows administrators to force a specific workspace for all Milvus storage instances
        milvus_workspace = os.environ.get("MILVUS_WORKSPACE")
        if milvus_workspace and milvus_workspace.strip():
            # Use environment variable value, overriding the passed workspace parameter
            effective_workspace = milvus_workspace.strip()
            logger.info(
                f"Using MILVUS_WORKSPACE environment variable: '{effective_workspace}' (overriding passed workspace: '{self.workspace}')"
            )
        else:
            # Use the workspace parameter passed during initialization
            effective_workspace = self.workspace
            if effective_workspace:
                logger.debug(
                    f"Using passed workspace parameter: '{effective_workspace}'"
                )

        # Build final_namespace with workspace prefix for data isolation
        # Keep original namespace unchanged for type detection logic
        if effective_workspace:
            self.final_namespace = f"{effective_workspace}_{self.namespace}"
            logger.debug(
                f"Final namespace with workspace prefix: '{self.final_namespace}'"
            )
        else:
            # When workspace is empty, final_namespace equals original namespace
            self.final_namespace = self.namespace
            logger.debug(f"Final namespace (no workspace): '{self.final_namespace}'")
            self.workspace = "_"

        kwargs = self.global_config.get("vector_db_storage_cls_kwargs", {})
        cosine_threshold = kwargs.get("cosine_better_than_threshold")
        if cosine_threshold is None:
            raise ValueError(
                "cosine_better_than_threshold must be specified in vector_db_storage_cls_kwargs"
            )
        self.cosine_better_than_threshold = cosine_threshold

        # Ensure created_at is in meta_fields
        if "created_at" not in self.meta_fields:
            self.meta_fields.add("created_at")

        # Initialize client as None - will be created in initialize() method
        self._client = None
        self._max_batch_size = self.global_config["embedding_batch_num"]
        self._initialized = False

    async def initialize(self):
        """Initialize Milvus collection"""
        async with get_data_init_lock(enable_logging=True):
            if self._initialized:
                return

            try:
                # Create MilvusClient if not already created
                if self._client is None:
                    self._client = MilvusClient(
                        uri=os.environ.get(
                            "MILVUS_URI",
                            config.get(
                                "milvus",
                                "uri",
                                fallback=os.path.join(
                                    self.global_config["working_dir"], "milvus_lite.db"
                                ),
                            ),
                        ),
                        user=os.environ.get(
                            "MILVUS_USER", config.get("milvus", "user", fallback=None)
                        ),
                        password=os.environ.get(
                            "MILVUS_PASSWORD",
                            config.get("milvus", "password", fallback=None),
                        ),
                        token=os.environ.get(
                            "MILVUS_TOKEN", config.get("milvus", "token", fallback=None)
                        ),
                        db_name=os.environ.get(
                            "MILVUS_DB_NAME",
                            config.get("milvus", "db_name", fallback=None),
                        ),
                    )
                    logger.debug(
                        f"[{self.workspace}] MilvusClient created successfully"
                    )

                # Create collection and check compatibility
                self._create_collection_if_not_exist()
                self._initialized = True
                logger.info(
                    f"[{self.workspace}] Milvus collection '{self.namespace}' initialized successfully"
                )
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Failed to initialize Milvus collection '{self.namespace}': {e}"
                )
                raise

    async def upsert(self, data: dict[str, dict[str, Any]]) -> None:
        # logger.debug(f"[{self.workspace}] Inserting {len(data)} to {self.namespace}")
        if not data:
            return

        # Ensure collection is loaded before upserting
        self._ensure_collection_loaded()

        import time

        current_time = int(time.time())

        list_data: list[dict[str, Any]] = [
            {
                "id": k,
                "created_at": current_time,
                **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
            }
            for k, v in data.items()
        ]
        contents = [v["content"] for v in data.values()]
        batches = [
            contents[i : i + self._max_batch_size]
            for i in range(0, len(contents), self._max_batch_size)
        ]

        embedding_tasks = [self.embedding_func(batch) for batch in batches]
        embeddings_list = await asyncio.gather(*embedding_tasks)

        embeddings = np.concatenate(embeddings_list)
        for i, d in enumerate(list_data):
            d["vector"] = embeddings[i]
        results = self._client.upsert(
            collection_name=self.final_namespace, data=list_data
        )
        return results

    async def query(
        self, query: str, top_k: int, query_embedding: list[float] = None
    ) -> list[dict[str, Any]]:
        # Ensure collection is loaded before querying
        self._ensure_collection_loaded()

        # Use provided embedding or compute it
        if query_embedding is not None:
            embedding = [query_embedding]  # Milvus expects a list of embeddings
        else:
            embedding = await self.embedding_func(
                [query], _priority=5
            )  # higher priority for query

        # Include all meta_fields (created_at is now always included)
        output_fields = list(self.meta_fields)

        results = self._client.search(
            collection_name=self.final_namespace,
            data=embedding,
            limit=top_k,
            output_fields=output_fields,
            search_params={
                "metric_type": "COSINE",
                "params": {"radius": self.cosine_better_than_threshold},
            },
        )
        return [
            {
                **dp["entity"],
                "id": dp["id"],
                "distance": dp["distance"],
                "created_at": dp.get("created_at"),
            }
            for dp in results[0]
        ]

    async def index_done_callback(self) -> None:
        # Milvus handles persistence automatically
        pass

    async def delete_entity(self, entity_name: str) -> None:
        """Delete an entity from the vector database

        Args:
            entity_name: The name of the entity to delete
        """
        try:
            # Compute entity ID from name
            entity_id = compute_mdhash_id(entity_name, prefix="ent-")
            logger.debug(
                f"[{self.workspace}] Attempting to delete entity {entity_name} with ID {entity_id}"
            )

            # Delete the entity from Milvus collection
            result = self._client.delete(
                collection_name=self.final_namespace, pks=[entity_id]
            )

            if result and result.get("delete_count", 0) > 0:
                logger.debug(
                    f"[{self.workspace}] Successfully deleted entity {entity_name}"
                )
            else:
                logger.debug(
                    f"[{self.workspace}] Entity {entity_name} not found in storage"
                )

        except Exception as e:
            logger.error(f"[{self.workspace}] Error deleting entity {entity_name}: {e}")

    async def delete_entity_relation(self, entity_name: str) -> None:
        """Delete all relations associated with an entity

        Args:
            entity_name: The name of the entity whose relations should be deleted
        """
        try:
            # Ensure collection is loaded before querying
            self._ensure_collection_loaded()

            # Search for relations where entity is either source or target
            expr = f'src_id == "{entity_name}" or tgt_id == "{entity_name}"'

            # Find all relations involving this entity
            results = self._client.query(
                collection_name=self.final_namespace, filter=expr, output_fields=["id"]
            )

            if not results or len(results) == 0:
                logger.debug(
                    f"[{self.workspace}] No relations found for entity {entity_name}"
                )
                return

            # Extract IDs of relations to delete
            relation_ids = [item["id"] for item in results]
            logger.debug(
                f"[{self.workspace}] Found {len(relation_ids)} relations for entity {entity_name}"
            )

            # Delete the relations
            if relation_ids:
                delete_result = self._client.delete(
                    collection_name=self.final_namespace, pks=relation_ids
                )

                logger.debug(
                    f"[{self.workspace}] Deleted {delete_result.get('delete_count', 0)} relations for {entity_name}"
                )

        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error deleting relations for {entity_name}: {e}"
            )

    async def delete(self, ids: list[str]) -> None:
        """Delete vectors with specified IDs

        Args:
            ids: List of vector IDs to be deleted
        """
        try:
            # Ensure collection is loaded before deleting
            self._ensure_collection_loaded()

            # Delete vectors by IDs
            result = self._client.delete(collection_name=self.final_namespace, pks=ids)

            if result and result.get("delete_count", 0) > 0:
                logger.debug(
                    f"[{self.workspace}] Successfully deleted {result.get('delete_count', 0)} vectors from {self.namespace}"
                )
            else:
                logger.debug(
                    f"[{self.workspace}] No vectors were deleted from {self.namespace}"
                )

        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error while deleting vectors from {self.namespace}: {e}"
            )

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        """Get vector data by its ID

        Args:
            id: The unique identifier of the vector

        Returns:
            The vector data if found, or None if not found
        """
        try:
            # Ensure collection is loaded before querying
            self._ensure_collection_loaded()

            # Include all meta_fields (created_at is now always included) plus id
            output_fields = list(self.meta_fields) + ["id"]

            # Query Milvus for a specific ID
            result = self._client.query(
                collection_name=self.final_namespace,
                filter=f'id == "{id}"',
                output_fields=output_fields,
            )

            if not result or len(result) == 0:
                return None

            return result[0]
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error retrieving vector data for ID {id}: {e}"
            )
            return None

    async def get_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        """Get multiple vector data by their IDs

        Args:
            ids: List of unique identifiers

        Returns:
            List of vector data objects that were found
        """
        if not ids:
            return []

        try:
            # Ensure collection is loaded before querying
            self._ensure_collection_loaded()

            # Include all meta_fields (created_at is now always included) plus id
            output_fields = list(self.meta_fields) + ["id"]

            # Prepare the ID filter expression
            id_list = '", "'.join(ids)
            filter_expr = f'id in ["{id_list}"]'

            # Query Milvus with the filter
            result = self._client.query(
                collection_name=self.final_namespace,
                filter=filter_expr,
                output_fields=output_fields,
            )

            return result or []
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error retrieving vector data for IDs {ids}: {e}"
            )
            return []

    async def get_vectors_by_ids(self, ids: list[str]) -> dict[str, list[float]]:
        """Get vectors by their IDs, returning only ID and vector data for efficiency

        Args:
            ids: List of unique identifiers

        Returns:
            Dictionary mapping IDs to their vector embeddings
            Format: {id: [vector_values], ...}
        """
        if not ids:
            return {}

        try:
            # Ensure collection is loaded before querying
            self._ensure_collection_loaded()

            # Prepare the ID filter expression
            id_list = '", "'.join(ids)
            filter_expr = f'id in ["{id_list}"]'

            # Query Milvus with the filter, requesting only vector field
            result = self._client.query(
                collection_name=self.final_namespace,
                filter=filter_expr,
                output_fields=["vector"],
            )

            vectors_dict = {}
            for item in result:
                if item and "vector" in item and "id" in item:
                    # Convert numpy array to list if needed
                    vector_data = item["vector"]
                    if isinstance(vector_data, np.ndarray):
                        vector_data = vector_data.tolist()
                    vectors_dict[item["id"]] = vector_data

            return vectors_dict
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error retrieving vectors by IDs from {self.namespace}: {e}"
            )
            return {}

    async def drop(self) -> dict[str, str]:
        """Drop all vector data from storage and clean up resources

        This method will delete all data from the Milvus collection.

        Returns:
            dict[str, str]: Operation status and message
            - On success: {"status": "success", "message": "data dropped"}
            - On failure: {"status": "error", "message": "<error details>"}
        """
        async with get_storage_lock():
            try:
                # Drop the collection and recreate it
                if self._client.has_collection(self.final_namespace):
                    self._client.drop_collection(self.final_namespace)

                # Recreate the collection
                self._create_collection_if_not_exist()

                logger.info(
                    f"[{self.workspace}] Process {os.getpid()} drop Milvus collection {self.namespace}"
                )
                return {"status": "success", "message": "data dropped"}
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Error dropping Milvus collection {self.namespace}: {e}"
                )
                return {"status": "error", "message": str(e)}
