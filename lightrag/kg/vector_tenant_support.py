# Vector Database Multi-Tenant Support Module
# Supports: Qdrant, Milvus, FAISS, Nano Vector DB

from typing import Any, Dict, List


class VectorTenantHelper:
    """Helper class for vector DB multi-tenant operations"""

    @staticmethod
    def add_tenant_metadata(
        payload: Dict[str, Any], tenant_id: str, kb_id: str
    ) -> Dict[str, Any]:
        """Add tenant_id and kb_id to vector payload/metadata"""
        payload["tenant_id"] = tenant_id
        payload["kb_id"] = kb_id
        return payload

    @staticmethod
    def get_tenant_filter(tenant_id: str, kb_id: str) -> Dict[str, Any]:
        """Create a filter for tenant isolation in vector DB queries"""
        return {
            "must": [
                {"key": "tenant_id", "match": {"value": tenant_id}},
                {"key": "kb_id", "match": {"value": kb_id}},
            ]
        }

    @staticmethod
    def get_tenant_filter_milvus(tenant_id: str, kb_id: str) -> str:
        """Create a Milvus WHERE clause for tenant isolation"""
        return f'tenant_id == "{tenant_id}" && kb_id == "{kb_id}"'

    @staticmethod
    def make_tenant_id(tenant_id: str, kb_id: str, original_id: str) -> str:
        """Create a tenant-scoped vector ID"""
        return f"{tenant_id}:{kb_id}:{original_id}"

    @staticmethod
    def parse_tenant_id(tenant_id_str: str) -> Dict[str, str]:
        """Parse a tenant-scoped vector ID"""
        parts = tenant_id_str.split(":", 2)
        if len(parts) == 3:
            return {"tenant_id": parts[0], "kb_id": parts[1], "original_id": parts[2]}
        return {"original_id": tenant_id_str}

    @staticmethod
    def create_tenant_collection_name(
        base_name: str, tenant_id: str, kb_id: str
    ) -> str:
        """Create a tenant-scoped collection name"""
        return f"{base_name}_{tenant_id}_{kb_id}".replace("-", "_")

    @staticmethod
    def get_tenant_collection_pattern(
        base_name: str, tenant_id: str, kb_id: str
    ) -> str:
        """Get a pattern for finding tenant-specific collections"""
        return f"{base_name}_{tenant_id}_{kb_id}*"


class QdrantTenantHelper(VectorTenantHelper):
    """Qdrant-specific tenant helper"""

    @staticmethod
    def build_qdrant_filter(
        tenant_id: str, kb_id: str, additional_filter: Dict = None
    ) -> Dict[str, Any]:
        """Build a Qdrant filter for tenant isolation"""
        must_conditions = [
            {"key": "tenant_id", "match": {"value": tenant_id}},
            {"key": "kb_id", "match": {"value": kb_id}},
        ]

        if additional_filter:
            if "must" in additional_filter:
                must_conditions.extend(additional_filter["must"])

            result = {"must": must_conditions}

            # Copy over any other filter conditions
            for key in ["should", "must_not"]:
                if key in additional_filter:
                    result[key] = additional_filter[key]

            return result

        return {"must": must_conditions}

    @staticmethod
    def update_qdrant_payload(
        payload: Dict[str, Any], tenant_id: str, kb_id: str
    ) -> Dict[str, Any]:
        """Ensure Qdrant payload includes tenant metadata"""
        return VectorTenantHelper.add_tenant_metadata(payload, tenant_id, kb_id)


class MilvusTenantHelper(VectorTenantHelper):
    """Milvus-specific tenant helper"""

    @staticmethod
    def build_milvus_expr(
        tenant_id: str, kb_id: str, additional_expr: str = None
    ) -> str:
        """Build a Milvus WHERE expression for tenant isolation"""
        expr = f'tenant_id == "{tenant_id}" && kb_id == "{kb_id}"'

        if additional_expr:
            expr += f" && ({additional_expr})"

        return expr

    @staticmethod
    def insert_with_tenant(
        collection, data: List[Dict[str, Any]], tenant_id: str, kb_id: str
    ):
        """Insert data with tenant metadata into Milvus"""
        for item in data:
            item["tenant_id"] = tenant_id
            item["kb_id"] = kb_id
        return collection.insert(data)


class FAISSTenantHelper(VectorTenantHelper):
    """FAISS-specific tenant helper"""

    @staticmethod
    def create_tenant_index_name(base_name: str, tenant_id: str, kb_id: str) -> str:
        """Create a tenant-scoped FAISS index name"""
        sanitized_tenant = tenant_id.replace("-", "_").replace(":", "_")
        sanitized_kb = kb_id.replace("-", "_").replace(":", "_")
        return f"{base_name}_{sanitized_tenant}_{sanitized_kb}"

    @staticmethod
    def create_tenant_metadata_list(
        num_vectors: int, tenant_id: str, kb_id: str, base_metadata: List[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Create metadata list for FAISS vectors with tenant info"""
        metadata_list = []

        for i in range(num_vectors):
            metadata = {"tenant_id": tenant_id, "kb_id": kb_id, "index": i}

            if base_metadata and i < len(base_metadata):
                metadata.update(base_metadata[i])

            metadata_list.append(metadata)

        return metadata_list

    @staticmethod
    def filter_metadata_by_tenant(
        metadata_list: List[Dict[str, Any]], tenant_id: str, kb_id: str
    ) -> List[int]:
        """Filter metadata list and return matching indices"""
        matching_indices = []

        for i, metadata in enumerate(metadata_list):
            if (
                metadata.get("tenant_id") == tenant_id
                and metadata.get("kb_id") == kb_id
            ):
                matching_indices.append(i)

        return matching_indices


class NanoVectorTenantHelper(VectorTenantHelper):
    """Nano Vector DB-specific tenant helper"""

    @staticmethod
    def build_nano_filter(tenant_id: str, kb_id: str) -> Dict[str, Any]:
        """Build a Nano Vector DB filter for tenant isolation"""
        return {"tenant_id": tenant_id, "kb_id": kb_id}

    @staticmethod
    def update_nano_document(
        doc: Dict[str, Any], tenant_id: str, kb_id: str
    ) -> Dict[str, Any]:
        """Update document to include tenant metadata"""
        if "metadata" not in doc:
            doc["metadata"] = {}

        doc["metadata"]["tenant_id"] = tenant_id
        doc["metadata"]["kb_id"] = kb_id

        return doc
