# MongoDB Multi-Tenant Support Module

from typing import Any, Dict, List, Tuple


class MongoTenantHelper:
    """Helper class for MongoDB multi-tenant operations"""

    @staticmethod
    def add_tenant_fields(
        data: Dict[str, Any], tenant_id: str, kb_id: str
    ) -> Dict[str, Any]:
        """Add tenant_id and kb_id fields to a document"""
        data["tenant_id"] = tenant_id
        data["kb_id"] = kb_id
        return data

    @staticmethod
    def get_tenant_filter(
        tenant_id: str, kb_id: str, additional_filter: Dict = None
    ) -> Dict:
        """Build a MongoDB filter for tenant and KB"""
        filter_dict = {"tenant_id": tenant_id, "kb_id": kb_id}
        if additional_filter:
            filter_dict.update(additional_filter)
        return filter_dict

    @staticmethod
    def create_tenant_indexes(collection_name: str) -> List[Dict[str, str]]:
        """Get recommended indexes for tenant isolation"""
        return [
            {
                "name": f"idx_{collection_name}_tenant_kb",
                "keys": [("tenant_id", 1), ("kb_id", 1)],
            },
            {
                "name": f"idx_{collection_name}_tenant_kb_id",
                "keys": [("tenant_id", 1), ("kb_id", 1), ("_id", 1)],
            },
        ]

    @staticmethod
    def build_upsert_with_tenant(
        filter_dict: Dict, update_dict: Dict, tenant_id: str, kb_id: str
    ) -> Tuple[Dict, Dict]:
        """Build filter and update dictionaries for upsert operations"""
        filter_dict["tenant_id"] = tenant_id
        filter_dict["kb_id"] = kb_id

        update_dict["$set"]["tenant_id"] = tenant_id
        update_dict["$set"]["kb_id"] = kb_id

        return filter_dict, update_dict


# ============================================================================
# MIGRATION HELPER FOR EXISTING MONGODB COLLECTIONS
# ============================================================================


async def add_tenant_fields_to_collection(
    db,
    collection_name: str,
    default_tenant_id: str = "default",
    default_kb_id: str = "default",
):
    """
    Add tenant_id and kb_id fields to existing MongoDB collection.

    This is a one-time migration operation that adds the new fields
    to all documents in the collection with default values.
    """
    collection = db[collection_name]

    # Check if any document already has tenant_id field
    sample = await collection.find_one({"tenant_id": {"$exists": True}})

    if not sample:
        # Perform bulk update to add tenant fields to all documents
        result = await collection.update_many(
            {"tenant_id": {"$exists": False}},
            {"$set": {"tenant_id": default_tenant_id, "kb_id": default_kb_id}},
        )
        return {"modified": result.modified_count, "status": "migration_completed"}
    else:
        return {"status": "already_migrated"}


async def create_tenant_indexes_on_collection(db, collection_name: str):
    """Create multi-tenant indexes on a MongoDB collection"""
    collection = db[collection_name]

    # Create indexes
    indexes = MongoTenantHelper.create_tenant_indexes(collection_name)

    for index in indexes:
        try:
            await collection.create_index(
                index["keys"], name=index["name"], background=True
            )
        except Exception as e:
            print(f"Warning: Could not create index {index['name']}: {e}")
