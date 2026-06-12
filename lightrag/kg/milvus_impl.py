import asyncio
import json
import os
from typing import Any, final, Optional, Dict
from dataclasses import dataclass, fields
import numpy as np
from lightrag.utils import (
    logger,
    compute_mdhash_id,
    _cooperative_yield,
    validate_workspace,
)
from ..base import BaseVectorStorage
from ..constants import (
    DEFAULT_MAX_FILE_PATH_LENGTH,
    DEFAULT_QUERY_PRIORITY,
    GRAPH_FIELD_SEP,
)
from ..kg.shared_storage import get_data_init_lock, get_namespace_lock
import pipmaster as pm

if not pm.is_installed("pymilvus"):
    pm.install("pymilvus>=2.6.2")

import configparser
from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema  # type: ignore
from packaging import version

config = configparser.ConfigParser()
config.read("config.ini", "utf-8")


@dataclass
class _PendingVectorDoc:
    """Buffered vector upsert waiting for embedding and/or bulk flush."""

    source: dict[str, Any]
    content: str
    vector: list[float] | None = None


# Flush-time batching limits. Milvus' server-side proxy rejects any single
# gRPC message larger than ~64MB (grpc.serverMaxRecvSize); the client library
# cannot raise that ceiling, so large flushes must be split client-side.
# The payload-byte budget is the primary limiter; the record-count caps are a
# secondary guard that only binds when individual records are small.
# Upsert and delete have separate count caps on purpose: upsert records each
# carry a full embedding vector and are far heavier than delete pks, so the
# upsert batch count is kept much smaller than the delete one.
DEFAULT_MILVUS_UPSERT_MAX_PAYLOAD_BYTES = (
    32 * 1024 * 1024
)  # 32MB, well below the 64MB gRPC ceiling
DEFAULT_MILVUS_UPSERT_MAX_RECORDS_PER_BATCH = 128
DEFAULT_MILVUS_DELETE_MAX_RECORDS_PER_BATCH = 1000
MILVUS_MAX_VARCHAR_BYTES = 65535
# The Milvus primary key. Truncating it would let two distinct ids collapse to
# the same key (silent overwrite) and make the row unreachable by its real id
# via get_by_id/delete, so it must never be truncated under any circumstance.
MILVUS_PRIMARY_KEY_FIELDS = frozenset({"id"})
# Non-primary identity fields. They are not the Milvus primary key, so the row
# stays uniquely keyed by `id` even if these collide after truncation (no
# storage-level overwrite). On the live upsert path we still reject oversize
# values so callers fix their input; during migration of pre-existing data we
# truncate-and-warn instead, so a single pathological legacy value cannot abort
# the whole collection migration.
MILVUS_IDENTITY_VARCHAR_FIELDS = frozenset(
    {"id", "entity_name", "full_doc_id", "src_id", "tgt_id"}
)
# Fields whose value is a GRAPH_FIELD_SEP-joined list of ids (chunk ids / file
# paths). When such a value overflows we truncate on the last separator that
# fits rather than mid-id, so we drop whole ids instead of leaving a dangling
# partial id that resolves to nothing.
MILVUS_SEPARATOR_JOINED_FIELDS = frozenset({"source_id", "file_path"})

# Supported index types
SUPPORTED_INDEX_TYPES = {
    "AUTOINDEX",
    "HNSW",
    "HNSW_SQ",
    "HNSW_PQ",
    "HNSW_PRQ",
    "IVF_FLAT",
    "IVF_SQ8",
    "IVF_PQ",
    "DISKANN",
    "SCANN",
}

# Supported metric types
SUPPORTED_METRIC_TYPES = {"COSINE", "L2", "IP"}

# HNSW_SQ quantization types
SUPPORTED_SQ_TYPES = {"SQ4U", "SQ6", "SQ8", "BF16", "FP16"}
SUPPORTED_REFINE_TYPES = {"SQ6", "SQ8", "BF16", "FP16", "FP32"}

# Index type version requirements
# Important: HNSW_SQ was first introduced in Milvus 2.6.8 (not 2.5)
INDEX_VERSION_REQUIREMENTS = {
    "HNSW_SQ": "2.6.8",  # HNSW_SQ requires Milvus 2.6.8+ (supports sq_types such as SQ4U, SQ6, SQ8, BF16, FP16)
}


def _get_env_bool(key: str, default: bool = False) -> bool:
    """Parse environment variable as boolean"""
    val = os.environ.get(key, "").lower()
    if val in ("true", "1", "yes", "on"):
        return True
    elif val in ("false", "0", "no", "off"):
        return False
    return default


def _get_env_int(key: str, default: int) -> int:
    """Parse environment variable as integer"""
    val = os.environ.get(key, "")
    if val:
        try:
            return int(val)
        except ValueError:
            logger.warning(
                f"Invalid integer value for {key}: {val}, using default {default}"
            )
    return default


@dataclass
class MilvusIndexConfig:
    """
    Milvus vector index configuration class

    Supports configuration via environment variables or initialization parameters.
    Initialization parameters take precedence over environment variables.
    """

    # Base configuration
    index_type: Optional[str] = None
    metric_type: Optional[str] = None

    # HNSW series parameters
    hnsw_m: Optional[int] = None
    hnsw_ef_construction: Optional[int] = None
    hnsw_ef: Optional[int] = None

    # HNSW_SQ specific parameters
    sq_type: Optional[str] = None
    sq_refine: Optional[bool] = None
    sq_refine_type: Optional[str] = None
    sq_refine_k: Optional[int] = None

    # IVF series parameters
    ivf_nlist: Optional[int] = None
    ivf_nprobe: Optional[int] = None

    def __post_init__(self):
        """Load configuration from environment variables (init parameters take precedence)"""
        # Index type
        self.index_type = (
            self.index_type or os.environ.get("MILVUS_INDEX_TYPE", "AUTOINDEX")
        ).upper()

        # Metric type
        self.metric_type = (
            self.metric_type or os.environ.get("MILVUS_METRIC_TYPE", "COSINE")
        ).upper()

        # HNSW parameters
        # Defaults aligned with Milvus 2.4+ official documentation
        if self.hnsw_m is None:
            self.hnsw_m = _get_env_int("MILVUS_HNSW_M", 16)
        if self.hnsw_ef_construction is None:
            self.hnsw_ef_construction = _get_env_int("MILVUS_HNSW_EF_CONSTRUCTION", 360)
        if self.hnsw_ef is None:
            self.hnsw_ef = _get_env_int("MILVUS_HNSW_EF", 200)

        # HNSW_SQ parameters
        if self.sq_type is None:
            self.sq_type = os.environ.get("MILVUS_HNSW_SQ_TYPE", "SQ8").upper()
        if self.sq_refine is None:
            self.sq_refine = _get_env_bool("MILVUS_HNSW_SQ_REFINE", False)
        if self.sq_refine_type is None:
            self.sq_refine_type = os.environ.get(
                "MILVUS_HNSW_SQ_REFINE_TYPE", "FP32"
            ).upper()
        if self.sq_refine_k is None:
            self.sq_refine_k = _get_env_int("MILVUS_HNSW_SQ_REFINE_K", 10)

        # IVF parameters
        if self.ivf_nlist is None:
            self.ivf_nlist = _get_env_int("MILVUS_IVF_NLIST", 1024)
        if self.ivf_nprobe is None:
            self.ivf_nprobe = _get_env_int("MILVUS_IVF_NPROBE", 16)

        # Validate configuration
        self._validate()

    def _validate(self):
        """Validate configuration validity"""
        if self.index_type not in SUPPORTED_INDEX_TYPES:
            raise ValueError(
                f"Unsupported index type: {self.index_type}. "
                f"Supported: {SUPPORTED_INDEX_TYPES}"
            )

        if self.metric_type not in SUPPORTED_METRIC_TYPES:
            raise ValueError(
                f"Unsupported metric type: {self.metric_type}. "
                f"Supported: {SUPPORTED_METRIC_TYPES}"
            )

        if self.index_type == "HNSW_SQ":
            if self.sq_type not in SUPPORTED_SQ_TYPES:
                raise ValueError(
                    f"Unsupported sq_type: {self.sq_type}. "
                    f"Supported: {SUPPORTED_SQ_TYPES}"
                )
            if self.sq_refine and self.sq_refine_type not in SUPPORTED_REFINE_TYPES:
                raise ValueError(
                    f"Unsupported refine_type: {self.sq_refine_type}. "
                    f"Supported: {SUPPORTED_REFINE_TYPES}"
                )

        # Parameter range validation
        if not (2 <= self.hnsw_m <= 2048):
            raise ValueError(f"hnsw_m must be in [2, 2048], got {self.hnsw_m}")
        if self.hnsw_ef_construction < 1:
            raise ValueError(
                f"hnsw_ef_construction must be >= 1, got {self.hnsw_ef_construction}"
            )
        if self.ivf_nlist < 1 or self.ivf_nlist > 65536:
            raise ValueError(f"ivf_nlist must be in [1, 65536], got {self.ivf_nlist}")

    def validate_milvus_version(self, server_version: str) -> None:
        """
        Validate Milvus server version supports the configured index type

        Args:
            server_version: Milvus server version string (e.g., "2.6.9")

        Raises:
            ValueError: Version does not meet index type requirements
        """
        current_ver = version.parse(
            server_version.split("-")[0]
        )  # Handle "2.6.9-dev" format

        # Check HNSW_SQ index type version requirements (requires 2.6.8+)
        if self.index_type == "HNSW_SQ":
            required = INDEX_VERSION_REQUIREMENTS["HNSW_SQ"]
            if current_ver < version.parse(required):
                raise ValueError(
                    f"HNSW_SQ requires Milvus {required}+, "
                    f"current version: {server_version}"
                )

        logger.info(
            f"Milvus version {server_version} validated for index type "
            f"{self.index_type}"
            + (f" with sq_type {self.sq_type}" if self.index_type == "HNSW_SQ" else "")
        )

    def build_index_params(self, index_params, field_name: str = "vector"):
        """
        Build pymilvus index parameters

        Args:
            index_params: IndexParams instance (from compatibility helper or client.prepare_index_params())
            field_name: Vector field name

        Returns:
            IndexParams object, or a dict fallback when direct API creation is needed.
        """
        if index_params is None:
            if self.index_type == "AUTOINDEX":
                logger.info(
                    "Using AUTOINDEX with direct API fallback because IndexParams is unavailable"
                )
                return {
                    "field_name": field_name,
                    "index_type": self.index_type,
                    "metric_type": self.metric_type,
                    "params": {},
                }
            raise RuntimeError(
                f"IndexParams not available but required for index type "
                f"'{self.index_type}'. Ensure pymilvus is installed correctly."
            )

        params: Dict[str, Any] = {}

        # HNSW series indexes
        if self.index_type in ("HNSW", "HNSW_SQ", "HNSW_PQ", "HNSW_PRQ"):
            params["M"] = self.hnsw_m
            params["efConstruction"] = self.hnsw_ef_construction

            # HNSW_SQ specific parameters
            if self.index_type == "HNSW_SQ":
                params["sq_type"] = self.sq_type
                if self.sq_refine:
                    params["refine"] = True
                    params["refine_type"] = self.sq_refine_type

        # IVF series indexes
        elif self.index_type in ("IVF_FLAT", "IVF_SQ8", "IVF_PQ"):
            params["nlist"] = self.ivf_nlist

        # DISKANN / SCANN have no additional params

        index_params.add_index(
            field_name=field_name,
            index_type=self.index_type,
            metric_type=self.metric_type,
            params=params,
        )

        logger.info(
            f"Milvus index configured: type={self.index_type}, "
            f"metric={self.metric_type}, params={params}"
        )

        return index_params

    def build_search_params(self) -> Dict[str, Any]:
        """
        Build search parameters

        Returns:
            Search parameters dictionary
        """
        search_params: Dict[str, Any] = {}

        if self.index_type in ("HNSW", "HNSW_SQ", "HNSW_PQ", "HNSW_PRQ"):
            search_params["ef"] = self.hnsw_ef
            if self.index_type == "HNSW_SQ" and self.sq_refine:
                search_params["refine_k"] = self.sq_refine_k

        elif self.index_type in ("IVF_FLAT", "IVF_SQ8", "IVF_PQ"):
            search_params["nprobe"] = self.ivf_nprobe

        return {"params": search_params} if search_params else {}

    @classmethod
    def get_config_field_names(cls) -> set:
        """Get all configuration field names from the dataclass.

        This method provides a single source of truth for configuration parameter names,
        eliminating the need to maintain duplicate hardcoded lists elsewhere.

        Returns:
            Set of field names that can be used to extract configuration from kwargs
        """
        return {f.name for f in fields(cls)}

    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary (for logging/debugging)"""
        return {
            "index_type": self.index_type,
            "metric_type": self.metric_type,
            "hnsw_m": self.hnsw_m,
            "hnsw_ef_construction": self.hnsw_ef_construction,
            "hnsw_ef": self.hnsw_ef,
            "sq_type": self.sq_type if self.index_type == "HNSW_SQ" else None,
            "sq_refine": self.sq_refine if self.index_type == "HNSW_SQ" else None,
            "sq_refine_type": (
                self.sq_refine_type
                if self.index_type == "HNSW_SQ" and self.sq_refine
                else None
            ),
            "sq_refine_k": (
                self.sq_refine_k
                if self.index_type == "HNSW_SQ" and self.sq_refine
                else None
            ),
            "ivf_nlist": (
                self.ivf_nlist if self.index_type.startswith("IVF") else None
            ),
            "ivf_nprobe": (
                self.ivf_nprobe if self.index_type.startswith("IVF") else None
            ),
        }


@final
@dataclass
class MilvusVectorDBStorage(BaseVectorStorage):
    def _get_milvus_connection_kwargs(self, include_db_name: bool = True) -> dict:
        """Build Milvus connection kwargs from env/config."""
        connection_kwargs = {
            "uri": os.environ.get(
                "MILVUS_URI",
                config.get(
                    "milvus",
                    "uri",
                    fallback=os.path.join(
                        self.global_config["working_dir"], "milvus_lite.db"
                    ),
                ),
            ),
            "user": os.environ.get(
                "MILVUS_USER", config.get("milvus", "user", fallback=None)
            ),
            "password": os.environ.get(
                "MILVUS_PASSWORD",
                config.get("milvus", "password", fallback=None),
            ),
            "token": os.environ.get(
                "MILVUS_TOKEN", config.get("milvus", "token", fallback=None)
            ),
        }

        db_name = os.environ.get(
            "MILVUS_DB_NAME",
            config.get("milvus", "db_name", fallback=None),
        )
        if include_db_name and db_name:
            connection_kwargs["db_name"] = db_name

        return connection_kwargs

    def _get_milvus_db_name(self) -> Optional[str]:
        """Return the configured Milvus database name, if any."""
        db_name = self._get_milvus_connection_kwargs(include_db_name=True).get(
            "db_name"
        )
        if db_name is None:
            return None

        normalized_name = str(db_name).strip()
        return normalized_name or None

    def _create_milvus_client(self) -> MilvusClient:
        """Create a Milvus client and ensure the configured database exists."""
        client = MilvusClient(
            **self._get_milvus_connection_kwargs(include_db_name=False)
        )
        db_name = self._get_milvus_db_name()

        if not db_name:
            return client

        existing_databases = set(client.list_databases())
        if db_name not in existing_databases:
            logger.warning(
                f"[{self.workspace}] Milvus database '{db_name}' not found, creating it"
            )
            client.create_database(db_name)

        use_database = getattr(client, "use_database", None) or getattr(
            client, "using_database", None
        )
        if callable(use_database):
            use_database(db_name)
            logger.debug(
                f"[{self.workspace}] Using Milvus database '{db_name}' for namespace '{self.namespace}'"
            )
            return client

        return MilvusClient(**self._get_milvus_connection_kwargs(include_db_name=True))

    def _create_schema_for_namespace(self) -> CollectionSchema:
        """Create schema based on the current instance's namespace"""

        # Get vector dimension from embedding_func
        dimension = self.embedding_func.embedding_dim
        varchar_limits = self._get_varchar_field_limits_for_namespace()

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
                    max_length=varchar_limits["entity_name"],
                    nullable=True,
                ),
                FieldSchema(
                    name="content",
                    dtype=DataType.VARCHAR,
                    max_length=varchar_limits["content"],
                    nullable=True,
                ),
                FieldSchema(
                    name="source_id",
                    dtype=DataType.VARCHAR,
                    max_length=varchar_limits["source_id"],
                    nullable=True,
                ),
                FieldSchema(
                    name="file_path",
                    dtype=DataType.VARCHAR,
                    max_length=varchar_limits["file_path"],
                    nullable=True,
                ),
            ]
            description = "LightRAG entities vector storage"

        elif self.namespace.endswith("relationships"):
            specific_fields = [
                FieldSchema(
                    name="src_id",
                    dtype=DataType.VARCHAR,
                    max_length=varchar_limits["src_id"],
                    nullable=True,
                ),
                FieldSchema(
                    name="tgt_id",
                    dtype=DataType.VARCHAR,
                    max_length=varchar_limits["tgt_id"],
                    nullable=True,
                ),
                FieldSchema(
                    name="content",
                    dtype=DataType.VARCHAR,
                    max_length=varchar_limits["content"],
                    nullable=True,
                ),
                FieldSchema(
                    name="source_id",
                    dtype=DataType.VARCHAR,
                    max_length=varchar_limits["source_id"],
                    nullable=True,
                ),
                FieldSchema(
                    name="file_path",
                    dtype=DataType.VARCHAR,
                    max_length=varchar_limits["file_path"],
                    nullable=True,
                ),
            ]
            description = "LightRAG relationships vector storage"

        elif self.namespace.endswith("chunks"):
            specific_fields = [
                FieldSchema(
                    name="full_doc_id",
                    dtype=DataType.VARCHAR,
                    max_length=varchar_limits["full_doc_id"],
                    nullable=True,
                ),
                FieldSchema(
                    name="content",
                    dtype=DataType.VARCHAR,
                    max_length=varchar_limits["content"],
                    nullable=True,
                ),
                FieldSchema(
                    name="file_path",
                    dtype=DataType.VARCHAR,
                    max_length=varchar_limits["file_path"],
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
                    max_length=varchar_limits["file_path"],
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

    def _get_varchar_field_limits_for_namespace(self) -> dict[str, int]:
        base_fields = {
            "id": 64,
            "content": MILVUS_MAX_VARCHAR_BYTES,
            "file_path": DEFAULT_MAX_FILE_PATH_LENGTH,
        }
        if self.namespace.endswith("entities"):
            return {
                **base_fields,
                "entity_name": 512,
                "source_id": MILVUS_MAX_VARCHAR_BYTES,
            }
        if self.namespace.endswith("relationships"):
            return {
                **base_fields,
                "src_id": 512,
                "tgt_id": 512,
                "source_id": MILVUS_MAX_VARCHAR_BYTES,
            }
        if self.namespace.endswith("chunks"):
            return {**base_fields, "full_doc_id": 64}
        return base_fields

    def _get_migrated_metadata_field_limits(self) -> dict[str, int]:
        if self.namespace.endswith("entities"):
            return {
                "content": MILVUS_MAX_VARCHAR_BYTES,
                "source_id": MILVUS_MAX_VARCHAR_BYTES,
            }
        if self.namespace.endswith("relationships"):
            return {
                "content": MILVUS_MAX_VARCHAR_BYTES,
                "source_id": MILVUS_MAX_VARCHAR_BYTES,
            }
        if self.namespace.endswith("chunks"):
            return {"content": MILVUS_MAX_VARCHAR_BYTES}
        return {}

    @staticmethod
    def _field_max_length(field: dict) -> int | None:
        max_length = field.get("params", {}).get("max_length")
        if max_length is None:
            return None
        try:
            return int(max_length)
        except (TypeError, ValueError):
            return None

    def _truncate_varchar_value(
        self,
        field_name: str,
        value: Any,
        record_id: str | None = None,
        allow_identity_truncation: bool = False,
    ) -> Any:
        limit = self._varchar_field_limits.get(field_name)
        if limit is None or not isinstance(value, str):
            return value

        encoded = value.encode("utf-8")
        if len(encoded) <= limit:
            return value

        # The primary key is never truncated: collapsing two ids into one would
        # silently overwrite a row and orphan it from get_by_id/delete.
        if field_name in MILVUS_PRIMARY_KEY_FIELDS:
            raise ValueError(
                f"[{self.workspace}] Milvus primary key '{field_name}' for record "
                f"'{record_id or '<unknown>'}' exceeds {limit} bytes "
                f"({len(encoded)} bytes); primary keys cannot be truncated"
            )

        # Other identity fields: reject on the live upsert path, but allow
        # truncate-and-warn during migration so legacy data can be carried over
        # without aborting the whole collection.
        if (
            field_name in MILVUS_IDENTITY_VARCHAR_FIELDS
            and not allow_identity_truncation
        ):
            raise ValueError(
                f"[{self.workspace}] Milvus field '{field_name}' for record "
                f"'{record_id or '<unknown>'}' exceeds {limit} bytes "
                f"({len(encoded)} bytes); identity fields cannot be truncated"
            )

        # Cut to the byte budget on a valid UTF-8 boundary first.
        truncated = encoded[:limit].decode("utf-8", errors="ignore")
        # For separator-joined id lists, back off to the last separator that
        # fits so we never persist a half id. Fall back to the raw byte cut when
        # no separator fits (e.g. a single id longer than the limit).
        if field_name in MILVUS_SEPARATOR_JOINED_FIELDS:
            boundary = truncated.rfind(GRAPH_FIELD_SEP)
            if boundary > 0:
                truncated = truncated[:boundary]
        logger.warning(
            "[%s] Milvus field '%s' for record '%s' truncated from %d to %d bytes",
            self.workspace,
            field_name,
            record_id or "<unknown>",
            len(encoded),
            len(truncated.encode("utf-8")),
        )
        return truncated

    def _sanitize_varchar_fields(
        self, row: dict[str, Any], allow_identity_truncation: bool = False
    ) -> dict[str, Any]:
        record_id = str(row.get("id", "")) or None
        return {
            field_name: self._truncate_varchar_value(
                field_name,
                value,
                record_id,
                allow_identity_truncation=allow_identity_truncation,
            )
            for field_name, value in row.items()
        }

    def _normalize_migration_row(self, row: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(row)
        metadata = normalized.pop("$meta", None)
        if isinstance(metadata, dict):
            for field_name, value in metadata.items():
                # Explicit nullable fields can hold None while the real value
                # still lives in $meta (schema-drift rows), so backfill on None
                # rather than mere key presence.
                if normalized.get(field_name) is None:
                    normalized[field_name] = value
        # Migration carries pre-existing rows: non-primary identity fields are
        # truncated-and-warned rather than rejected (see _truncate_varchar_value).
        return self._sanitize_varchar_fields(normalized, allow_identity_truncation=True)

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
            from pymilvus.client.prepare import IndexParams  # type: ignore

            return IndexParams()
        except ImportError:
            pass

        try:
            from pymilvus.client.types import IndexParams  # type: ignore

            return IndexParams()
        except ImportError:
            pass

        try:
            from pymilvus import IndexParams  # type: ignore

            return IndexParams()
        except ImportError:
            pass

        # If all else fails, return None to use fallback method
        return None

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
        # Build vector index using index configuration
        # Use compatibility helper to get IndexParams
        index_params_for_vector = self._get_index_params()

        vector_index_params = self.index_config.build_index_params(
            index_params_for_vector, field_name="vector"
        )

        # Re-raise exceptions to surface vector index creation failures
        if isinstance(vector_index_params, dict):
            self._client.create_index(
                collection_name=self.final_namespace,
                field_name=vector_index_params["field_name"],
                index_params={
                    "index_type": vector_index_params["index_type"],
                    "metric_type": vector_index_params["metric_type"],
                    "params": vector_index_params["params"],
                },
            )
        else:
            self._client.create_index(
                collection_name=self.final_namespace,
                index_params=vector_index_params,
            )

        logger.debug(
            f"[{self.workspace}] Created vector index with config: {self.index_config.to_dict()}"
        )

        # Create scalar indexes based on namespace
        # Wrap scalar index creation in try-except to allow graceful degradation
        try:
            # Try to get IndexParams in a version-compatible way
            scalar_index_params = self._get_index_params()

            if scalar_index_params is not None:
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

            else:
                # Fallback to direct API calls if IndexParams is not available
                logger.info(
                    f"[{self.workspace}] IndexParams not available, using fallback methods for {self.namespace}"
                )

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
            # Scalar index failures are logged as warnings (not critical)
            logger.warning(
                f"[{self.workspace}] Failed to create some scalar indexes for {self.namespace}: {e}"
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
                "content": {"type": "VarChar"},
                "source_id": {"type": "VarChar"},
                "file_path": {"type": "VarChar"},
            }
        elif self.namespace.endswith("relationships"):
            specific_fields = {
                "src_id": {"type": "VarChar"},
                "tgt_id": {"type": "VarChar"},
                "content": {"type": "VarChar"},
                "source_id": {"type": "VarChar"},
                "file_path": {"type": "VarChar"},
            }
        elif self.namespace.endswith("chunks"):
            specific_fields = {
                "full_doc_id": {"type": "VarChar"},
                "content": {"type": "VarChar"},
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

                    # Convert both to int for comparison to handle type mismatches
                    # (Milvus API may return string "1024" vs int 1024)
                    try:
                        existing_dim_int = (
                            int(existing_dimension)
                            if existing_dimension is not None
                            else None
                        )
                        current_dim_int = (
                            int(current_dimension)
                            if current_dimension is not None
                            else None
                        )
                    except (TypeError, ValueError) as e:
                        logger.error(
                            f"[{self.workspace}] Failed to parse dimensions: existing={existing_dimension} (type={type(existing_dimension)}), "
                            f"current={current_dimension} (type={type(current_dimension)}), error={e}"
                        )
                        raise ValueError(
                            f"Invalid dimension values for collection '{self.final_namespace}': "
                            f"existing={existing_dimension}, current={current_dimension}"
                        ) from e

                    if existing_dim_int != current_dim_int:
                        raise ValueError(
                            f"Vector dimension mismatch for collection '{self.final_namespace}': "
                            f"existing={existing_dim_int}, current={current_dim_int}"
                        )

                    logger.debug(
                        f"[{self.workspace}] Vector dimension check passed: {current_dim_int}"
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

    @staticmethod
    def _has_vector_field(collection_info: dict) -> bool:
        """Return True when the collection exposes a 'vector' field.

        Old simple-schema collections may lack a vector field entirely. Their
        rows therefore carry no vector data, and copying them into the new
        schema (whose vector field is required) would fail at insert time, so
        callers use this to skip migration for such collections.
        """
        return any(
            field.get("name") == "vector" for field in collection_info.get("fields", [])
        )

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

    def _check_metadata_schema_migration_needed(self, collection_info: dict) -> bool:
        existing_fields = {
            field["name"]: field for field in collection_info.get("fields", [])
        }

        for (
            field_name,
            expected_max_length,
        ) in self._get_migrated_metadata_field_limits().items():
            existing_field = existing_fields.get(field_name)
            if existing_field is None:
                logger.info(
                    f"[{self.workspace}] Collection {self.namespace} missing explicit Milvus field '{field_name}', needs migration"
                )
                return True

            if not self._is_field_compatible(existing_field, {"type": "VarChar"}):
                logger.info(
                    f"[{self.workspace}] Collection {self.namespace} has incompatible Milvus field '{field_name}', needs migration"
                )
                return True

            max_length = self._field_max_length(existing_field)
            if max_length is not None and max_length < expected_max_length:
                logger.info(
                    f"[{self.workspace}] Collection {self.namespace} has {field_name} max_length={max_length}, "
                    f"needs migration to {expected_max_length}"
                )
                return True

        return False

    def _check_schema_compatibility(self, collection_info: dict):
        """Check schema field compatibility and detect migration needs"""
        existing_fields = {
            field["name"]: field for field in collection_info.get("fields", [])
        }

        # Check if this is an old collection created with simple schema
        has_vector_field = self._has_vector_field(collection_info)

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

        if self._check_file_path_length_restriction(
            collection_info
        ) or self._check_metadata_schema_migration_needed(collection_info):
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

    def _create_collection_with_schema(
        self, collection_name: str, ignore_index_errors: bool = False
    ) -> None:
        original_final_namespace = self.final_namespace
        try:
            self.final_namespace = collection_name
            schema = self._create_schema_for_namespace()
            self._client.create_collection(
                collection_name=collection_name, schema=schema
            )
            try:
                self._create_indexes_after_collection()
            except Exception as index_error:
                if not ignore_index_errors:
                    raise
                logger.warning(
                    f"[{self.workspace}] Failed to create indexes for new collection: {index_error}"
                )
        finally:
            self.final_namespace = original_final_namespace

    def _migrate_collection_schema(
        self,
        source_collection_name: str | None = None,
        target_collection_name: str | None = None,
    ):
        source_collection_name = source_collection_name or self.final_namespace
        target_collection_name = target_collection_name or self.final_namespace
        temp_collection_name = f"{target_collection_name}_temp"
        original_final_namespace = self.final_namespace
        iterator = None

        try:
            logger.info(
                f"[{self.workspace}] Starting iterator-based schema migration for {self.namespace}: "
                f"{source_collection_name} -> {target_collection_name}"
            )

            logger.info(
                f"[{self.workspace}] Step 1: Creating temporary collection: {temp_collection_name}"
            )
            if self._client.has_collection(temp_collection_name):
                self._client.drop_collection(temp_collection_name)
            self._create_collection_with_schema(
                temp_collection_name, ignore_index_errors=True
            )

            self._client.load_collection(temp_collection_name)

            logger.info(
                f"[{self.workspace}] Step 2: Copying data using query_iterator from: {source_collection_name}"
            )

            try:
                iterator = self._client.query_iterator(
                    collection_name=source_collection_name,
                    batch_size=2000,
                    output_fields=["*"],
                )
                logger.debug(f"[{self.workspace}] Query iterator created successfully")
            except Exception as iterator_error:
                logger.error(
                    f"[{self.workspace}] Failed to create query iterator: {iterator_error}"
                )
                raise

            total_migrated = 0
            batch_number = 1

            while True:
                try:
                    batch_data = iterator.next()
                    if not batch_data:
                        # No more data available
                        break

                    sanitized_batch_data = [
                        self._normalize_migration_row(row) for row in batch_data
                    ]
                    insert_batches = self._build_upsert_batches(
                        sanitized_batch_data,
                        max_payload_bytes=self._max_upsert_payload_bytes,
                        max_records_per_batch=self._max_upsert_records_per_batch,
                    )
                    try:
                        for insert_batch_number, (
                            records_batch,
                            estimated_bytes,
                        ) in enumerate(insert_batches, 1):
                            logger.debug(
                                f"[{self.workspace}] Milvus migration insert batch "
                                f"{batch_number}.{insert_batch_number}/{len(insert_batches)}: "
                                f"records={len(records_batch)}, estimated_payload_bytes={estimated_bytes}"
                            )
                            self._client.insert(
                                collection_name=temp_collection_name,
                                data=records_batch,
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

            if source_collection_name == target_collection_name:
                logger.info(
                    f"[{self.workspace}] Step 3: Rename origin collection to {source_collection_name}_old"
                )
                try:
                    self._client.rename_collection(
                        source_collection_name, f"{source_collection_name}_old"
                    )
                except Exception as rename_error:
                    try:
                        logger.warning(
                            f"[{self.workspace}] Try to drop origin collection instead"
                        )
                        self._client.drop_collection(source_collection_name)
                    except Exception as e:
                        logger.error(
                            f"[{self.workspace}] Rename operation failed: {rename_error}"
                        )
                        raise e
            elif self._client.has_collection(target_collection_name):
                raise RuntimeError(
                    f"Target collection already exists: {target_collection_name}"
                )

            logger.info(
                f"[{self.workspace}] Step 4: Renaming collection {temp_collection_name} -> {target_collection_name}"
            )
            try:
                self._client.rename_collection(
                    temp_collection_name, target_collection_name
                )
                logger.info(f"[{self.workspace}] Rename operation completed")
            except Exception as rename_error:
                if source_collection_name == target_collection_name:
                    logger.error(
                        f"[{self.workspace}] Rename operation failed: {rename_error}"
                    )
                else:
                    logger.error(
                        f"[{self.workspace}] Target rename operation failed: {rename_error}"
                    )
                raise RuntimeError(
                    f"Failed to rename collection: {rename_error}"
                ) from rename_error

            self.final_namespace = target_collection_name

        except Exception as e:
            self.final_namespace = original_final_namespace
            logger.error(
                f"[{self.workspace}] Iterator-based migration failed for {self.namespace}: {e}"
            )

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

            raise RuntimeError(
                f"Iterator-based migration failed for collection {self.namespace}: {e}"
            ) from e

        finally:
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

    def _validate_collection_and_load(self) -> None:
        try:
            self._client.describe_collection(self.final_namespace)
            self._validate_collection_compatibility()
        except Exception as validation_error:
            logger.error(
                f"[{self.workspace}] CRITICAL ERROR: Collection '{self.namespace}' exists but validation failed!"
            )
            logger.error(
                f"[{self.workspace}] This indicates potential data migration failure or schema incompatibility."
            )
            logger.error(f"[{self.workspace}] Validation error: {validation_error}")
            logger.error(f"[{self.workspace}] MANUAL INTERVENTION REQUIRED:")
            logger.error(
                f"[{self.workspace}] 1. Check the existing collection schema and data integrity"
            )
            logger.error(f"[{self.workspace}] 2. Backup existing data if needed")
            logger.error(
                f"[{self.workspace}] 3. Manually resolve schema compatibility issues"
            )
            logger.error(
                f"[{self.workspace}] 4. Consider recreating the collection using: lightrag-rebuild-vdb "
            )
            logger.error(
                f"[{self.workspace}] Program execution stopped to prevent potential data loss."
            )
            raise RuntimeError(
                f"Collection validation failed for '{self.final_namespace}'. "
                f"Data migration failure detected. Manual intervention required to prevent data loss. "
                f"Original error: {validation_error}"
            )

        try:
            self._ensure_collection_loaded()
        except Exception as load_error:
            if not self._is_missing_vector_index_error(load_error):
                raise

            try:
                self._repair_missing_vector_index()
                self._ensure_collection_loaded()
                logger.info(
                    f"[{self.workspace}] Repaired missing vector index for existing collection '{self.namespace}'"
                )
            except Exception as repair_error:
                raise RuntimeError(
                    f"Index repair failed for collection '{self.final_namespace}'. "
                    f"Original error: {repair_error}"
                ) from repair_error

    @staticmethod
    def _is_missing_vector_index_error(error: Exception) -> bool:
        """Return True when the error indicates the collection lacks a vector index."""
        error_message = str(error).lower()
        return (
            "no vector index" in error_message
            or "please create index firstly" in error_message
        )

    def _repair_missing_vector_index(self):
        """Create indexes for an existing collection that is missing its vector index."""
        logger.warning(
            f"[{self.workspace}] Collection '{self.namespace}' is missing a vector index, attempting repair"
        )
        self._create_indexes_after_collection()

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
            collection_exists = self._client.has_collection(self.final_namespace)
            logger.info(
                f"[{self.workspace}] VectorDB collection '{self.namespace}' exists check: {collection_exists}"
            )

            if collection_exists:
                self._validate_collection_and_load()
                return

            legacy_collection_exists = (
                self.legacy_namespace != self.final_namespace
                and self._client.has_collection(self.legacy_namespace)
            )
            if legacy_collection_exists:
                legacy_collection_info = self._client.describe_collection(
                    self.legacy_namespace
                )
                if not self._has_vector_field(legacy_collection_info):
                    # Old simple-schema collection with no vector field: its rows
                    # carry no vectors, so migrating them into the required-vector
                    # schema would fail at insert and block startup. Skip the
                    # migration and create a fresh suffixed collection instead.
                    logger.warning(
                        f"[{self.workspace}] Legacy collection '{self.legacy_namespace}' "
                        f"has no vector field (old simple schema); cannot migrate its rows "
                        f"into '{self.final_namespace}'. Creating a new collection instead."
                    )
                else:
                    try:
                        self._check_vector_dimension(legacy_collection_info)
                    except ValueError as legacy_error:
                        logger.warning(
                            f"[{self.workspace}] Legacy collection '{self.legacy_namespace}' "
                            f"is not compatible with '{self.final_namespace}': {legacy_error}. "
                            f"Creating a new collection without migrating legacy vectors."
                        )
                    else:
                        self._migrate_collection_schema(
                            source_collection_name=self.legacy_namespace,
                            target_collection_name=self.final_namespace,
                        )
                        self._ensure_collection_loaded()
                        return

            # Collection doesn't exist, create new collection
            logger.info(f"[{self.workspace}] Creating new collection: {self.namespace}")
            self._create_collection_with_schema(self.final_namespace)
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
                self._create_collection_with_schema(self.final_namespace)

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
        validate_workspace(self.workspace)
        self._validate_embedding_func()

        # Extract MilvusIndexConfig parameters from vector_db_storage_cls_kwargs
        #
        # IMPORTANT: This approach allows Milvus index configuration via vector_db_storage_cls_kwargs,
        # which is the RECOMMENDED method for framework integration (e.g., RAGAnything).
        #
        # All 11 index configuration parameters can be passed through vector_db_storage_cls_kwargs:
        #   - index_type, metric_type
        #   - hnsw_m, hnsw_ef_construction, hnsw_ef
        #   - sq_type, sq_refine, sq_refine_type, sq_refine_k
        #   - ivf_nlist, ivf_nprobe
        #
        # Example:
        #   LightRAG(
        #       vector_storage="MilvusVectorDBStorage",
        #       vector_db_storage_cls_kwargs={
        #           "cosine_better_than_threshold": 0.2,
        #           "index_type": "HNSW",
        #           "metric_type": "COSINE",
        #           "hnsw_m": 32,
        #           "hnsw_ef_construction": 256,
        #       }
        #   )
        #
        # Use MilvusIndexConfig.get_config_field_names() to dynamically extract valid parameters.
        # This ensures we always stay in sync with the MilvusIndexConfig dataclass definition.
        kwargs = self.global_config.get("vector_db_storage_cls_kwargs", {})
        index_config_keys = MilvusIndexConfig.get_config_field_names()
        index_config_params = {
            k: v for k, v in kwargs.items() if k in index_config_keys
        }

        # Initialize index configuration (if not already set)
        # Configuration priority: init params from kwargs > environment variables > defaults
        if not hasattr(self, "index_config") or self.index_config is None:
            self.index_config = MilvusIndexConfig(**index_config_params)

        # Check for MILVUS_WORKSPACE environment variable first (higher priority)
        # This allows administrators to force a specific workspace for all Milvus storage instances
        milvus_workspace = os.environ.get("MILVUS_WORKSPACE")
        if milvus_workspace and milvus_workspace.strip():
            # Use environment variable value, overriding the passed workspace parameter
            effective_workspace = milvus_workspace.strip()
            logger.info(
                f"Using MILVUS_WORKSPACE environment variable: '{effective_workspace}' (overriding '{self.workspace}/{self.namespace}')"
            )
        else:
            # Use the workspace parameter passed during initialization
            effective_workspace = self.workspace
            if effective_workspace:
                logger.debug(
                    f"Using passed workspace parameter: '{effective_workspace}'"
                )

        self.workspace = effective_workspace or ""
        self.model_suffix = self._generate_collection_suffix()
        if self.workspace:
            self.legacy_namespace = f"{self.workspace}_{self.namespace}"
            logger.debug(
                f"Legacy namespace with workspace prefix: '{self.legacy_namespace}'"
            )
        else:
            self.legacy_namespace = self.namespace
            logger.debug(f"Legacy namespace (no workspace): '{self.legacy_namespace}'")
        if self.model_suffix:
            self.final_namespace = f"{self.legacy_namespace}_{self.model_suffix}"
            logger.info(f"Milvus collection: {self.final_namespace}")
        else:
            self.final_namespace = self.legacy_namespace
            logger.warning(
                f"Milvus collection: {self.final_namespace} missing suffix. Please add model_name to embedding_func for proper model-based data isolation."
            )
        cosine_threshold = kwargs.get("cosine_better_than_threshold")
        if cosine_threshold is None:
            raise ValueError(
                "cosine_better_than_threshold must be specified in vector_db_storage_cls_kwargs"
            )
        self.cosine_better_than_threshold = cosine_threshold

        # Ensure created_at is in meta_fields
        if "created_at" not in self.meta_fields:
            self.meta_fields.add("created_at")
        self._varchar_field_limits = self._get_varchar_field_limits_for_namespace()

        # Initialize client as None - will be created in initialize() method
        self._client = None
        self._max_batch_size = self.global_config["embedding_batch_num"]

        # Flush-time batching limits (see module-level DEFAULT_MILVUS_* constants).
        # A non-positive value disables that splitting dimension.
        self._max_upsert_payload_bytes = int(
            os.getenv(
                "MILVUS_UPSERT_MAX_PAYLOAD_BYTES",
                str(DEFAULT_MILVUS_UPSERT_MAX_PAYLOAD_BYTES),
            )
        )
        self._max_upsert_records_per_batch = int(
            os.getenv(
                "MILVUS_UPSERT_MAX_RECORDS_PER_BATCH",
                str(DEFAULT_MILVUS_UPSERT_MAX_RECORDS_PER_BATCH),
            )
        )
        self._max_delete_records_per_batch = int(
            os.getenv(
                "MILVUS_DELETE_MAX_RECORDS_PER_BATCH",
                str(DEFAULT_MILVUS_DELETE_MAX_RECORDS_PER_BATCH),
            )
        )
        if self._max_upsert_payload_bytes <= 0:
            logger.warning(
                f"MILVUS_UPSERT_MAX_PAYLOAD_BYTES={self._max_upsert_payload_bytes} is non-positive, disable payload-size splitting"
            )
        if self._max_upsert_records_per_batch <= 0:
            logger.warning(
                f"MILVUS_UPSERT_MAX_RECORDS_PER_BATCH={self._max_upsert_records_per_batch} is non-positive, disable upsert record-count splitting"
            )
        if self._max_delete_records_per_batch <= 0:
            logger.warning(
                f"MILVUS_DELETE_MAX_RECORDS_PER_BATCH={self._max_delete_records_per_batch} is non-positive, disable delete record-count splitting"
            )
        self._initialized = False

        # Deferred-embedding buffers and the per-namespace flush lock.
        # The lock keys on final_namespace so two instances pointing at the
        # same Milvus collection (e.g. when MILVUS_WORKSPACE env override is
        # used) share a single writer lock. We construct it here in
        # __post_init__ — not in initialize() — so any code path that
        # touches the buffer before initialize() still has a valid lock.
        self._pending_vector_docs: dict[str, _PendingVectorDoc] = {}
        self._pending_vector_deletes: set[str] = set()
        self._flush_lock = get_namespace_lock(
            namespace=self.final_namespace, workspace=""
        )

    async def initialize(self):
        """Initialize Milvus collection"""
        async with get_data_init_lock():
            if self._initialized:
                return

            try:
                # Create MilvusClient if not already created
                if self._client is None:
                    self._client = self._create_milvus_client()
                    logger.debug(
                        f"[{self.workspace}] MilvusClient created successfully"
                    )

                # Validate Milvus version compatibility with configured index
                if self.index_config.index_type in INDEX_VERSION_REQUIREMENTS:
                    try:
                        server_version = self._client.get_server_version()
                        self.index_config.validate_milvus_version(server_version)
                    except Exception as version_error:
                        logger.error(
                            f"[{self.workspace}] Milvus version validation failed: {version_error}"
                        )
                        raise

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
        """Buffer vector docs for embedding and batched flush.

        Embedding deliberately does NOT happen here: repeated upserts of the
        same id, or many small batches, collapse into a single flush-time
        embedding pass. Reads (`get_by_id`/`get_by_ids`/`get_vectors_by_ids`)
        observe pending docs via the same lock for read-your-writes.
        """
        if not data:
            return

        import time

        current_time = int(time.time())

        pending_docs: list[tuple[str, _PendingVectorDoc]] = []
        for i, (k, v) in enumerate(data.items(), start=1):
            # _sanitize_varchar_fields already byte-truncates the stored
            # `content` when it is a meta field; the pending doc keeps the full
            # untruncated text so the embedding sees the complete chunk.
            source = self._sanitize_varchar_fields(
                {
                    "id": k,
                    "created_at": current_time,
                    **{k1: v1 for k1, v1 in v.items() if k1 in self.meta_fields},
                }
            )
            pending_docs.append(
                (
                    k,
                    _PendingVectorDoc(source=source, content=v["content"]),
                )
            )
            await _cooperative_yield(i)

        # An upsert overrides any pending delete on the same id; installing
        # a fresh _PendingVectorDoc instance invalidates any vector cached
        # by a prior get_vectors_by_ids() call on a stale revision.
        async with self._flush_lock:
            for doc_id, pdoc in pending_docs:
                self._pending_vector_deletes.discard(doc_id)
                self._pending_vector_docs[doc_id] = pdoc

    async def query(
        self, query: str, top_k: int, query_embedding: list[float] = None
    ) -> list[dict[str, Any]]:
        """Similarity search against the persisted Milvus collection.

        Note: buffered-but-unflushed upserts are NOT visible to this method —
        they exist only in `_pending_vector_docs` until `index_done_callback()`
        embeds and writes them. Callers that need read-after-write visibility
        for similarity search must run an explicit flush first.
        """
        # Ensure collection is loaded before querying
        self._ensure_collection_loaded()

        # Use provided embedding or compute it
        if query_embedding is not None:
            embedding = [query_embedding]  # Milvus expects a list of embeddings
        else:
            embedding = await self.embedding_func(
                [query], context="query", _priority=DEFAULT_QUERY_PRIORITY
            )  # higher priority for query

        # Include all meta_fields (created_at is now always included)
        output_fields = list(self.meta_fields)

        # Build search params from index config
        search_params_base = self.index_config.build_search_params()

        # Merge with metric type and radius threshold
        search_params = {
            "metric_type": self.index_config.metric_type,
            "params": {
                **search_params_base.get("params", {}),
                "radius": self.cosine_better_than_threshold,
            },
        }

        results = self._client.search(
            collection_name=self.final_namespace,
            data=embedding,
            limit=top_k,
            output_fields=output_fields,
            search_params=search_params,
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

    @staticmethod
    def _build_upsert_batches(
        records: list[dict[str, Any]],
        max_payload_bytes: int,
        max_records_per_batch: int,
    ) -> list[tuple[list[dict[str, Any]], int]]:
        """Split upsert records into batches by estimated payload size and count.

        The byte budget is the primary limiter: records accumulate until adding
        the next one would exceed ``max_payload_bytes``, then a new batch starts.
        Size is estimated by JSON-serializing each record; this overestimates the
        actual gRPC protobuf size (a JSON float string is far longer than the 4
        protobuf bytes it encodes), so the split stays conservatively below the
        server limit and never underestimates.

        A single record larger than the byte budget is emitted as its own batch
        rather than raising: JSON overestimation means such a record's real
        protobuf size is often still under Milvus' 64MB ceiling, so we let the
        server be the final arbiter instead of failing client-side. Returns a
        list of ``(batch, estimated_bytes)`` tuples (estimate used for logging).
        """
        if not records:
            return []

        payload_limit = max_payload_bytes if max_payload_bytes > 0 else float("inf")
        records_limit = (
            max_records_per_batch if max_records_per_batch > 0 else float("inf")
        )

        batches: list[tuple[list[dict[str, Any]], int]] = []
        current_batch: list[dict[str, Any]] = []
        # JSON array overhead ("[]")
        current_estimated_bytes = 2

        for record in records:
            record_size = len(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    default=str,
                ).encode("utf-8")
            )

            # If current batch not empty, a comma is needed before next element.
            separator_overhead = 1 if current_batch else 0
            next_batch_size = current_estimated_bytes + separator_overhead + record_size

            if current_batch and (
                len(current_batch) >= records_limit or next_batch_size > payload_limit
            ):
                batches.append((current_batch, current_estimated_bytes))
                current_batch = []
                current_estimated_bytes = 2
                next_batch_size = current_estimated_bytes + record_size

            current_batch.append(record)
            current_estimated_bytes = next_batch_size

        if current_batch:
            batches.append((current_batch, current_estimated_bytes))

        return batches

    async def index_done_callback(self) -> None:
        """Flush all buffered vector ops to Milvus before returning.

        Contract: on a successful return, every previously buffered upsert
        has been embedded and committed to the collection, and every buffered
        delete has been issued — i.e. all pending vectors are durable in
        Milvus (which persists automatically once written). On any embed-
        or server-side failure this method raises and leaves both buffers
        intact for the next callback to retry; the caller MUST NOT assume
        clean persistence in that case.
        """
        await self._flush_pending_vector_ops()

    async def drop_pending_index_ops(self) -> None:
        """Discard buffered upserts/deletes (pipeline aborting on error)."""
        async with self._flush_lock:
            self._pending_vector_docs.clear()
            self._pending_vector_deletes.clear()

    async def _flush_pending_vector_ops(self) -> None:
        """Flush buffered vector upserts and deletes to Milvus.

        Embedding runs *inside* this lock (not in `upsert` or lock-free):
        it makes deferred embedding and bulk indexing atomic against
        concurrent upserts and destructive mutations. Any failure (embed
        or server write) raises and leaves both buffers intact; the next
        `index_done_callback` retries automatically.
        """
        async with self._flush_lock:
            if not self._pending_vector_docs and not self._pending_vector_deletes:
                return
            if self._client is None:
                return

            # Milvus requires the collection to be loaded before upsert/delete.
            self._ensure_collection_loaded()

            pending_docs = self._pending_vector_docs
            pending_deletes = self._pending_vector_deletes

            docs_to_embed: list[tuple[str, _PendingVectorDoc]] = [
                (doc_id, pdoc)
                for doc_id, pdoc in pending_docs.items()
                if pdoc.vector is None
            ]

            if docs_to_embed:
                contents = [pdoc.content for _, pdoc in docs_to_embed]
                batches = [
                    contents[i : i + self._max_batch_size]
                    for i in range(0, len(contents), self._max_batch_size)
                ]
                logger.info(
                    f"[{self.workspace}] {self.namespace} flush: embedding "
                    f"{len(docs_to_embed)} vectors in {len(batches)} batch(es) "
                    f"(batch_num={self._max_batch_size})"
                )
                try:
                    embeddings_list = await asyncio.gather(
                        *[
                            self.embedding_func(batch, context="document")
                            for batch in batches
                        ]
                    )
                except Exception as e:
                    logger.error(
                        f"[{self.workspace}] Error embedding pending vector ops "
                        f"(upserts={len(docs_to_embed)}): {e}"
                    )
                    raise

                embeddings = np.concatenate(embeddings_list)
                if len(embeddings) != len(docs_to_embed):
                    raise RuntimeError(
                        f"[{self.workspace}] Embedding count mismatch: expected "
                        f"{len(docs_to_embed)}, got {len(embeddings)}"
                    )
                for i, ((_, pdoc), embedding) in enumerate(
                    zip(docs_to_embed, embeddings), start=1
                ):
                    # Cache as float32 so a second flush after a server-side
                    # error doesn't re-embed, and so the upsert JSON payload
                    # stays compact (float32 serializes to a shorter string
                    # than float64, and Milvus stores FLOAT_VECTOR as float32
                    # anyway, so the cast is lossless).
                    pdoc.vector = np.array(embedding, dtype=np.float32).tolist()
                    await _cooperative_yield(i)

            # Assemble final upsert payload. After the embed loop above every
            # pending doc has a non-None vector (count-mismatch was checked),
            # so we can iterate without re-guarding.
            committed_ids: list[str] = list(pending_docs.keys())
            # source was already byte-truncated in upsert(); no need to
            # re-sanitize here (vector is not a VarChar field).
            list_data: list[dict[str, Any]] = [
                {
                    **pending_docs[doc_id].source,
                    "vector": pending_docs[doc_id].vector,
                }
                for doc_id in committed_ids
            ]

            try:
                if list_data:
                    # Split the upsert into batches that stay under the server-side
                    # 64MB gRPC message limit. Fail-fast: any batch failure raises
                    # immediately and the full buffer is retained for the next flush.
                    upsert_batches = self._build_upsert_batches(
                        list_data,
                        max_payload_bytes=self._max_upsert_payload_bytes,
                        max_records_per_batch=self._max_upsert_records_per_batch,
                    )
                    if len(upsert_batches) > 1:
                        logger.info(
                            f"[{self.workspace}] {self.namespace} flush: upsert split into "
                            f"{len(upsert_batches)} batches for {len(list_data)} records "
                            f"(max_payload={self._max_upsert_payload_bytes} batch={self._max_upsert_records_per_batch})"
                        )
                    for batch_index, (records_batch, estimated_bytes) in enumerate(
                        upsert_batches, 1
                    ):
                        if (
                            len(records_batch) == 1
                            and self._max_upsert_payload_bytes > 0
                            and estimated_bytes > self._max_upsert_payload_bytes
                        ):
                            logger.warning(
                                f"[{self.workspace}] {self.namespace} flush: single record "
                                f"id={records_batch[0].get('id')} estimated {estimated_bytes} bytes "
                                f"exceeds {self._max_upsert_payload_bytes}"
                            )
                        logger.debug(
                            f"[{self.workspace}] Milvus upsert batch {batch_index}/{len(upsert_batches)}: "
                            f"records={len(records_batch)}, estimated_payload_bytes={estimated_bytes}"
                        )
                        self._client.upsert(
                            collection_name=self.final_namespace, data=records_batch
                        )
                if pending_deletes:
                    # Chunk deletes by record count; pks are short strings so a
                    # count cap is enough to stay under the gRPC message limit.
                    delete_ids = list(pending_deletes)
                    delete_chunk = (
                        self._max_delete_records_per_batch
                        if self._max_delete_records_per_batch > 0
                        else len(delete_ids)
                    )
                    for i in range(0, len(delete_ids), delete_chunk):
                        self._client.delete(
                            collection_name=self.final_namespace,
                            pks=delete_ids[i : i + delete_chunk],
                        )
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Error flushing vector ops "
                    f"(upserts={len(pending_docs)}, "
                    f"deletes={len(pending_deletes)}): {e}"
                )
                raise

            # On success, clear the buffers in-place so external references
            # (e.g. drop()) see the cleared state.
            for doc_id in committed_ids:
                pending_docs.pop(doc_id, None)
            pending_deletes.clear()

    async def delete_entity(self, entity_name: str) -> None:
        """Buffer an entity vector delete by computing its hash ID."""
        entity_id = compute_mdhash_id(entity_name, prefix="ent-")
        async with self._flush_lock:
            self._pending_vector_docs.pop(entity_id, None)
            self._pending_vector_deletes.add(entity_id)
        logger.debug(
            f"[{self.workspace}] Buffered delete for entity {entity_name} (id={entity_id})"
        )

    async def delete_entity_relation(self, entity_name: str) -> None:
        """Delete all relation vectors where entity appears as src or tgt.

        The whole method runs under ``_flush_lock`` so the server-side query
        + delete cannot interleave with an in-flight bulk upsert.
        Server-side failures are re-raised (no log-and-swallow): the caller
        decides whether to retry.

        Buffer semantics — post-prune with caller short-circuit contract:
            Matching pending upserts in ``_pending_vector_docs`` are
            pruned **only after** the server-side query + delete
            succeeds. On failure the pending buffer stays intact and
            the exception propagates so the caller (``adelete_by_entity``
            in ``utils_graph.py``) can short-circuit before
            ``_persist_graph_updates`` flushes a half-cleaned buffer.

        Semantic note (deferred-buffer ↔ persisted divergence): pruning only
        consults the *current* buffered ``src_id`` / ``tgt_id`` view; we do
        not re-read the persisted row a buffered upsert is about to
        overwrite. So if a pending upsert is rewriting an already-persisted
        ``rel-X-Y`` so that its new ``src_id`` / ``tgt_id`` matches
        ``entity_name`` while the persisted row's do not (or vice versa),
        the persisted row will not be deleted by the server-side filter and
        the pending overwrite is dropped — i.e. the final state can diverge
        from the eager-flush ordering (upsert → flush → delete). Callers
        that require eager-equivalent semantics should call
        ``index_done_callback()`` before ``delete_entity_relation``.
        """

        def _prune_pending() -> None:
            for doc_id in [
                k
                for k, v in self._pending_vector_docs.items()
                if v.source.get("src_id") == entity_name
                or v.source.get("tgt_id") == entity_name
            ]:
                self._pending_vector_docs.pop(doc_id, None)

        async with self._flush_lock:
            if self._client is None:
                # No server state to mutate; buffer prune is the only
                # delete intent we can record.
                _prune_pending()
                return

            self._ensure_collection_loaded()

            expr = f'src_id == "{entity_name}" or tgt_id == "{entity_name}"'
            results = self._client.query(
                collection_name=self.final_namespace,
                filter=expr,
                output_fields=["id"],
            )

            if not results:
                # No server rows to delete — still safe to prune any
                # pending upserts so they can't re-create the relation.
                _prune_pending()
                logger.debug(
                    f"[{self.workspace}] No relations found for entity {entity_name}"
                )
                return

            relation_ids = [item["id"] for item in results]
            self._client.delete(collection_name=self.final_namespace, pks=relation_ids)
            # Server-side delete succeeded — safe to prune the pending
            # buffer so subsequent flushes don't re-upsert the deleted
            # relations.
            _prune_pending()
            logger.debug(
                f"[{self.workspace}] Deleted {len(relation_ids)} relations for {entity_name}"
            )

    async def delete(self, ids: list[str]) -> None:
        """Buffer vector deletes for batched flush."""
        if not ids:
            return
        if isinstance(ids, set):
            ids = list(ids)
        async with self._flush_lock:
            for doc_id in ids:
                self._pending_vector_docs.pop(doc_id, None)
                self._pending_vector_deletes.add(doc_id)
        logger.debug(
            f"[{self.workspace}] Buffered delete for {len(ids)} vectors in {self.namespace}"
        )

    async def get_by_id(self, id: str) -> dict[str, Any] | None:
        """Get vector data by its ID, with read-your-writes against the buffer."""
        async with self._flush_lock:
            if id in self._pending_vector_deletes:
                return None
            pending = self._pending_vector_docs.get(id)
            if pending is not None:
                doc = dict(pending.source)
                doc["id"] = id
                return doc

        try:
            # Ensure collection is loaded before querying
            self._ensure_collection_loaded()

            # Include all meta_fields (created_at is now always included) plus id
            output_fields = list(self.meta_fields) + ["id"]

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
        """Get multiple vector data by their IDs (read-your-writes), preserving order."""
        if not ids:
            return []

        buffered: dict[str, dict[str, Any] | None] = {}
        remaining: list[str] = []
        async with self._flush_lock:
            for doc_id in ids:
                if doc_id in self._pending_vector_deletes:
                    buffered[doc_id] = None
                    continue
                pending = self._pending_vector_docs.get(doc_id)
                if pending is not None:
                    doc = dict(pending.source)
                    doc["id"] = doc_id
                    buffered[doc_id] = doc
                    continue
                remaining.append(doc_id)

        result_map: dict[str, dict[str, Any]] = {}
        if remaining:
            try:
                # Ensure collection is loaded before querying
                self._ensure_collection_loaded()

                # Include all meta_fields (created_at is now always included) plus id
                output_fields = list(self.meta_fields) + ["id"]

                id_list = '", "'.join(remaining)
                filter_expr = f'id in ["{id_list}"]'

                result = self._client.query(
                    collection_name=self.final_namespace,
                    filter=filter_expr,
                    output_fields=output_fields,
                )

                if result:
                    for row in result:
                        if not row:
                            continue
                        row_id = row.get("id")
                        if row_id is not None:
                            result_map[str(row_id)] = row
            except Exception as e:
                logger.error(
                    f"[{self.workspace}] Error retrieving vector data for IDs {remaining}: {e}"
                )
                return []

        return [
            buffered[doc_id] if doc_id in buffered else result_map.get(str(doc_id))
            for doc_id in ids
        ]

    async def get_vectors_by_ids(self, ids: list[str]) -> dict[str, list[float]]:
        """Get vector embeddings for given IDs, with read-your-writes.

        Pending docs with `vector is None` trigger a lazy embed inside the
        lock; the resulting vector is cached on the buffered `_PendingVectorDoc`
        so the next flush won't re-embed the same content.
        """
        if not ids:
            return {}

        result: dict[str, list[float]] = {}
        remaining: list[str] = []
        async with self._flush_lock:
            docs_to_embed: list[tuple[str, _PendingVectorDoc]] = []
            for doc_id in ids:
                if doc_id in self._pending_vector_deletes:
                    continue
                pending = self._pending_vector_docs.get(doc_id)
                if pending is not None:
                    if pending.vector is None:
                        docs_to_embed.append((doc_id, pending))
                    else:
                        result[doc_id] = pending.vector
                    continue
                remaining.append(doc_id)

            if docs_to_embed:
                contents = [pdoc.content for _, pdoc in docs_to_embed]
                batches = [
                    contents[i : i + self._max_batch_size]
                    for i in range(0, len(contents), self._max_batch_size)
                ]
                try:
                    embeddings_list = await asyncio.gather(
                        *[
                            self.embedding_func(batch, context="document")
                            for batch in batches
                        ]
                    )
                except Exception as e:
                    logger.error(
                        f"[{self.workspace}] Error lazily embedding pending vectors "
                        f"(upserts={len(docs_to_embed)}): {e}"
                    )
                    raise
                embeddings = np.concatenate(embeddings_list)
                if len(embeddings) != len(docs_to_embed):
                    raise RuntimeError(
                        f"[{self.workspace}] Embedding count mismatch: expected "
                        f"{len(docs_to_embed)}, got {len(embeddings)}"
                    )
                for i, ((doc_id, pdoc), embedding) in enumerate(
                    zip(docs_to_embed, embeddings), start=1
                ):
                    # Cache float32 to match the flush path so the buffered
                    # vector dtype is uniform regardless of which path embedded.
                    pdoc.vector = np.array(embedding, dtype=np.float32).tolist()
                    result[doc_id] = pdoc.vector
                    await _cooperative_yield(i)

        if not remaining:
            return result

        try:
            self._ensure_collection_loaded()

            id_list = '", "'.join(remaining)
            filter_expr = f'id in ["{id_list}"]'

            rows = self._client.query(
                collection_name=self.final_namespace,
                filter=filter_expr,
                output_fields=["id", "vector"],
            )

            for item in rows or []:
                if item and "vector" in item and "id" in item:
                    vector_data = item["vector"]
                    if isinstance(vector_data, np.ndarray):
                        vector_data = vector_data.tolist()
                    # Match get_by_ids: stringify the server-returned id so
                    # callers can index the dict by the original requested id.
                    result[str(item["id"])] = vector_data
            return result
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error retrieving vectors by IDs from {self.namespace}: {e}"
            )
            return result

    async def finalize(self):
        """Flush pending vector ops; surface unflushed data as RuntimeError.

        Milvus has no client connection to release (the MilvusClient is
        stateless from the storage layer's perspective), but we still need
        to fail loudly when a transient bulk error left writes buffered —
        the caller must not believe storage finalized cleanly.
        """
        flush_error: Exception | None = None
        try:
            await self._flush_pending_vector_ops()
        except Exception as e:
            flush_error = e

        # Read the residual buffer sizes under the flush lock so the
        # snapshot is consistent with any racing late-arriving mutator
        # (cancellation paths can land an upsert/delete between the flush
        # above and the post-mortem check below).
        async with self._flush_lock:
            pending_docs = len(self._pending_vector_docs)
            pending_deletes = len(self._pending_vector_deletes)

        if flush_error is not None:
            raise RuntimeError(
                f"[{self.workspace}] MilvusVectorDBStorage.finalize() flush raised; "
                f"{pending_docs} pending upserts and {pending_deletes} pending "
                f"deletes were left buffered (data lost)"
            ) from flush_error
        if pending_docs or pending_deletes:
            raise RuntimeError(
                f"[{self.workspace}] MilvusVectorDBStorage.finalize() left "
                f"{pending_docs} pending upserts and {pending_deletes} pending "
                f"deletes buffered after final flush attempt (these writes have been lost)"
            )

    async def drop(self) -> dict[str, str]:
        """Drop all data from the Milvus collection. Destructive.

        MUST only be called when ``pipeline_status`` is idle (see the
        Pipeline concurrency contract in ``AGENTS.md``); the only
        in-tree caller ``clear_documents`` enforces this.

        Caveat — only this instance's buffers are cleared. Other
        ``MilvusVectorDBStorage`` instances aliased onto the same
        ``final_namespace`` (multi-worker processes, or distinct
        workspaces collapsed by ``MILVUS_WORKSPACE``) keep their own
        buffers; a sibling whose prior flush failed and left buffers
        intact will, on its next flush, upsert those stale rows into
        the freshly recreated collection. Direct callers bypassing the
        idle precondition MUST flush every aliased instance first.

        Returns:
            dict[str, str]: ``{"status": "success"|"error", "message": str}``
        """
        try:
            async with self._flush_lock:
                # Discard any buffered writes before the collection is gone;
                # a concurrent flush would otherwise resurrect them.
                self._pending_vector_docs.clear()
                self._pending_vector_deletes.clear()

                # Drop the collection and recreate it empty.
                if self._client.has_collection(self.final_namespace):
                    self._client.drop_collection(self.final_namespace)

                # Recreate an EMPTY collection. Do NOT route through
                # _create_collection_if_not_exist here: with the suffixed
                # collection now gone it would see the intentionally-kept legacy
                # collection and re-run the legacy->suffixed migration, pulling
                # the just-dropped rows back in. That makes drop() non-empty
                # (clear_documents would leave stale legacy data behind) and
                # forces a needless full migration on every rebuild/clear.
                self._create_collection_with_schema(self.final_namespace)
                self._ensure_collection_loaded()

            logger.info(
                f"[{self.workspace}] Process {os.getpid()} drop Milvus collection {self.namespace}"
            )
            return {"status": "success", "message": "data dropped"}
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error dropping Milvus collection {self.namespace}: {e}"
            )
            return {"status": "error", "message": str(e)}
