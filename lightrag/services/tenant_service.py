"""Service for managing tenants and knowledge bases."""

from typing import Optional, Dict, Any
import logging
from datetime import datetime

from lightrag.models.tenant import Tenant, KnowledgeBase, TenantConfig, KBConfig
from lightrag.base import BaseKVStorage

logger = logging.getLogger(__name__)


class TenantService:
    """Service for managing tenants and knowledge bases."""

    def __init__(self, kv_storage: BaseKVStorage):
        """Initialize tenant service with KV storage backend.

        Args:
            kv_storage: Backend storage for tenant/KB metadata
        """
        self.kv_storage = kv_storage
        self.tenant_namespace = "__tenants__"
        self.kb_namespace = "__knowledge_bases__"

    async def create_tenant(
        self,
        tenant_name: str,
        description: Optional[str] = None,
        config: Optional[TenantConfig] = None,
        created_by: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tenant:
        """Create a new tenant.

        Args:
            tenant_name: Display name for the tenant
            description: Optional description
            config: Optional tenant configuration
            created_by: User ID that created the tenant
            metadata: Optional metadata dictionary

        Returns:
            Created Tenant object
        """
        import json

        tenant = Tenant(
            tenant_name=tenant_name,
            description=description,
            config=config or TenantConfig(),
            created_by=created_by,
            metadata=metadata or {},
        )

        # Store tenant in PostgreSQL tenants table for FK integrity
        if hasattr(self.kv_storage, "db") and self.kv_storage.db is not None:
            try:
                metadata_json = json.dumps(tenant.metadata) if tenant.metadata else "{}"
                # Use query method with RETURNING to insert tenant
                await self.kv_storage.db.query(
                    """
                    INSERT INTO tenants (tenant_id, name, description, metadata, created_at, updated_at)
                    VALUES ($1, $2, $3, $4::jsonb, NOW(), NOW())
                    ON CONFLICT (tenant_id) DO UPDATE SET
                        name = EXCLUDED.name,
                        description = EXCLUDED.description,
                        metadata = EXCLUDED.metadata,
                        updated_at = NOW()
                    RETURNING tenant_id
                    """,
                    [tenant.tenant_id, tenant_name, description or "", metadata_json],
                )
                logger.debug(
                    f"Inserted tenant {tenant.tenant_id} into PostgreSQL tenants table"
                )
            except Exception as e:
                logger.error(f"Failed to insert tenant into PostgreSQL: {e}")
                raise
        else:
            # Fallback: Store tenant metadata in KV storage only if no PostgreSQL DB
            # Note: PGKVStorage doesn't support custom namespaces like __tenants__
            # so we skip this when PostgreSQL is available (data is already in tenants table)
            try:
                tenant_data = tenant.to_dict()
                await self.kv_storage.upsert(
                    {f"{self.tenant_namespace}:{tenant.tenant_id}": tenant_data}
                )
            except Exception as e:
                logger.warning(
                    f"Could not store tenant in KV storage (non-critical): {e}"
                )

        logger.info(f"Created tenant: {tenant.tenant_id} ({tenant_name})")
        return tenant

    async def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Retrieve a tenant by ID.

        Queries both PostgreSQL tenants table and KV storage to ensure
        tenants created via database initialization scripts are also available.

        Args:
            tenant_id: Tenant identifier

        Returns:
            Tenant object if found, None otherwise
        """
        # First, try to get tenant from PostgreSQL database if available
        if hasattr(self.kv_storage, "db") and self.kv_storage.db is not None:
            try:
                logger.debug(
                    f"Attempting to query tenant {tenant_id} from PostgreSQL database"
                )
                # Query the tenants table directly (use $1 for PostgreSQL parameter)
                row = await self.kv_storage.db.query(
                    "SELECT tenant_id, name, description, created_at, updated_at FROM tenants WHERE tenant_id = $1",
                    [tenant_id],
                )

                if row:
                    # Create a Tenant object from the database row
                    tenant = Tenant(
                        tenant_id=row["tenant_id"],
                        tenant_name=row["name"],
                        description=row.get("description", ""),
                        created_by=None,  # Not tracked in basic schema
                        metadata={
                            "is_public": True
                        },  # Allow all users to access demo tenants
                    )
                    # Override timestamps from database
                    if "created_at" in row:
                        tenant.created_at = row["created_at"]
                    if "updated_at" in row:
                        tenant.updated_at = row["updated_at"]

                    logger.debug(
                        f"Retrieved tenant {tenant_id} from PostgreSQL database"
                    )
                    return tenant
            except Exception as e:
                logger.debug(f"Could not query tenant from PostgreSQL database: {e}")
                # Fall through to KV storage

        # Try KV storage as fallback
        data = await self.kv_storage.get_by_id(f"{self.tenant_namespace}:{tenant_id}")
        if not data:
            return None
        return self._deserialize_tenant(data)

    async def verify_user_access(
        self, user_id: str, tenant_id: str, required_role: str = "viewer"
    ) -> bool:
        """Verify that a user has required role for a specific tenant.

        This is a CRITICAL security function that prevents unauthorized
        cross-tenant data access. Checks user-tenant membership table
        with role-based access control.

        Args:
            user_id: User identifier from JWT token
            tenant_id: Requested tenant ID
            required_role: Minimum required role (viewer, editor, admin, owner)

        Returns:
            True if user has access with required role, False otherwise
        """
        if not user_id or not tenant_id:
            logger.warning("verify_user_access called with empty user_id or tenant_id")
            return False

        # SEC-002 FIX: Check for super-admin users from configuration instead of hardcoded "admin"
        # Super-admins are configured via LIGHTRAG_SUPER_ADMIN_USERS environment variable
        super_admins_list = []
        try:
            from lightrag.api.config import SUPER_ADMIN_USERS

            if SUPER_ADMIN_USERS:
                super_admins_list = [
                    u.strip().lower() for u in SUPER_ADMIN_USERS.split(",") if u.strip()
                ]
        except ImportError:
            pass  # Config not available

        # Fallback: If no super admins configured, default to "admin" for backward compatibility
        # This ensures the default admin user always has access unless explicitly disabled
        if not super_admins_list:
            import os

            env_super_admins = os.environ.get("LIGHTRAG_SUPER_ADMIN_USERS")
            # If env var is not set, default to "admin". If set to empty string, it means "no super admins"
            if env_super_admins is None:
                super_admins_list = ["admin"]
            elif env_super_admins.strip():
                super_admins_list = [
                    u.strip().lower() for u in env_super_admins.split(",") if u.strip()
                ]

        if user_id.lower() in super_admins_list:
            logger.debug(
                f"Access granted: super-admin user {user_id} has access to all tenants"
            )
            return True

        # Check membership table using PostgreSQL function
        if hasattr(self.kv_storage, "db") and self.kv_storage.db:
            try:
                result = await self.kv_storage.db.query(
                    "SELECT has_tenant_access($1, $2, $3) as has_access",
                    [user_id, tenant_id, required_role],
                )
                # result is an asyncpg Record object (not a list when multirows=False)
                # Access using dict-style: result['has_access']
                if result is not None:
                    # Handle both dict-like Record and raw boolean result
                    if hasattr(result, "__getitem__"):
                        # Try dict-style access first (asyncpg Record)
                        try:
                            has_access = result["has_access"]
                        except (KeyError, TypeError):
                            # Fall back to index access if key doesn't work
                            has_access = result[0] if len(result) > 0 else False
                    else:
                        # Direct boolean result
                        has_access = bool(result)

                    if has_access:
                        logger.debug(
                            f"Access granted: user {user_id} has {required_role}+ role for tenant {tenant_id}"
                        )
                        return True
                    else:
                        logger.warning(
                            f"Access denied: user {user_id} lacks {required_role} role for tenant {tenant_id}"
                        )
                        return False
            except Exception as e:
                # Function might not exist if migration hasn't run - use legacy fallback
                error_msg = str(e)
                if "has_tenant_access" in error_msg and "does not exist" in error_msg:
                    logger.debug(
                        "has_tenant_access function not found, using legacy access check"
                    )
                else:
                    logger.warning(f"Error checking user access: {e}")
                # Fall through to legacy check

        # Legacy fallback: Check if tenant is public or user is creator
        tenant = await self.get_tenant(tenant_id)
        if not tenant:
            logger.debug(f"Tenant {tenant_id} not found during access check")
            return False

        # Check if tenant is public
        if tenant.metadata.get("is_public", False):
            logger.debug(f"Access granted: tenant {tenant_id} is public")
            return True

        # Check if user is the creator
        if tenant.created_by == user_id:
            logger.debug(
                f"Access granted: user {user_id} is creator of tenant {tenant_id}"
            )
            return True

        logger.warning(
            f"Access denied: user {user_id} has no access to tenant {tenant_id}"
        )
        return False

    async def add_user_to_tenant(
        self, user_id: str, tenant_id: str, role: str, created_by: str
    ) -> dict:
        """Add a user to a tenant with specified role.

        Args:
            user_id: User identifier to add
            tenant_id: Tenant identifier
            role: User role (owner, admin, editor, viewer)
            created_by: User who is adding this member

        Returns:
            Dictionary with membership information

        Raises:
            ValueError: If tenant doesn't exist or role is invalid
        """
        # Validate role
        valid_roles = ["owner", "admin", "editor", "viewer"]
        if role not in valid_roles:
            raise ValueError(f"Invalid role: {role}. Must be one of {valid_roles}")

        # Verify tenant exists
        tenant = await self.get_tenant(tenant_id)
        if not tenant:
            raise ValueError(f"Tenant {tenant_id} not found")

        if not self.kv_storage.db:
            raise RuntimeError("PostgreSQL database required for membership management")

        try:
            # Insert membership - use multirows=True to get a list of Records
            results = await self.kv_storage.db.query(
                """
                INSERT INTO user_tenant_memberships (user_id, tenant_id, role, created_by)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (user_id, tenant_id)
                DO UPDATE SET role = $3, updated_at = NOW()
                RETURNING id, user_id, tenant_id, role, created_at, created_by, updated_at
                """,
                [user_id, tenant_id, role, created_by],
                multirows=True,
            )

            if results and len(results) > 0:
                membership = results[0]
                logger.info(
                    f"Added user {user_id} to tenant {tenant_id} with role {role}"
                )
                return {
                    "id": str(membership["id"]),
                    "user_id": str(membership["user_id"]),
                    "tenant_id": str(membership["tenant_id"]),
                    "role": str(membership["role"]),
                    "created_at": membership["created_at"].isoformat()
                    if hasattr(membership["created_at"], "isoformat")
                    else str(membership["created_at"]),
                    "created_by": str(membership["created_by"])
                    if membership["created_by"]
                    else None,
                    "updated_at": membership["updated_at"].isoformat()
                    if hasattr(membership["updated_at"], "isoformat")
                    else str(membership["updated_at"]),
                }
            else:
                raise RuntimeError("Failed to add user to tenant")

        except Exception as e:
            logger.error(f"Error adding user to tenant: {e}")
            raise

    async def remove_user_from_tenant(self, user_id: str, tenant_id: str) -> bool:
        """Remove a user from a tenant.

        Args:
            user_id: User identifier to remove
            tenant_id: Tenant identifier

        Returns:
            True if removed, False if membership didn't exist
        """
        if not self.kv_storage.db:
            raise RuntimeError("PostgreSQL database required for membership management")

        try:
            results = await self.kv_storage.db.query(
                """
                DELETE FROM user_tenant_memberships
                WHERE user_id = $1 AND tenant_id = $2
                RETURNING id
                """,
                [user_id, tenant_id],
                multirows=True,
            )

            if results and len(results) > 0:
                logger.info(f"Removed user {user_id} from tenant {tenant_id}")
                return True
            else:
                logger.debug(
                    f"No membership found for user {user_id} in tenant {tenant_id}"
                )
                return False

        except Exception as e:
            logger.error(f"Error removing user from tenant: {e}")
            raise

    async def update_user_role(
        self, user_id: str, tenant_id: str, new_role: str
    ) -> dict:
        """Update a user's role in a tenant.

        Args:
            user_id: User identifier
            tenant_id: Tenant identifier
            new_role: New role to assign

        Returns:
            Updated membership information

        Raises:
            ValueError: If role is invalid or membership doesn't exist
        """
        # Validate role
        valid_roles = ["owner", "admin", "editor", "viewer"]
        if new_role not in valid_roles:
            raise ValueError(f"Invalid role: {new_role}. Must be one of {valid_roles}")

        if not self.kv_storage.db:
            raise RuntimeError("PostgreSQL database required for membership management")

        try:
            results = await self.kv_storage.db.query(
                """
                UPDATE user_tenant_memberships
                SET role = $1, updated_at = NOW()
                WHERE user_id = $2 AND tenant_id = $3
                RETURNING id, user_id, tenant_id, role, created_at, created_by, updated_at
                """,
                [new_role, user_id, tenant_id],
                multirows=True,
            )

            if results and len(results) > 0:
                membership = results[0]
                logger.info(
                    f"Updated role for user {user_id} in tenant {tenant_id} to {new_role}"
                )
                return {
                    "id": str(membership["id"]),
                    "user_id": str(membership["user_id"]),
                    "tenant_id": str(membership["tenant_id"]),
                    "role": str(membership["role"]),
                    "created_at": membership["created_at"].isoformat()
                    if hasattr(membership["created_at"], "isoformat")
                    else str(membership["created_at"]),
                    "created_by": str(membership["created_by"])
                    if membership["created_by"]
                    else None,
                    "updated_at": membership["updated_at"].isoformat()
                    if hasattr(membership["updated_at"], "isoformat")
                    else str(membership["updated_at"]),
                }
            else:
                raise ValueError(
                    f"No membership found for user {user_id} in tenant {tenant_id}"
                )

        except Exception as e:
            logger.error(f"Error updating user role: {e}")
            raise

    async def get_user_tenants(
        self, user_id: str, skip: int = 0, limit: int = 100
    ) -> dict:
        """Get all tenants a user has access to.

        Args:
            user_id: User identifier
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            Dictionary with items and total count
        """
        if not self.kv_storage.db:
            # Fallback: return all public tenants
            all_tenants = await self.list_tenants(skip=skip, limit=limit)
            public_tenants = [
                t for t in all_tenants["items"] if t.metadata.get("is_public", False)
            ]
            return {
                "items": public_tenants,
                "total": len(public_tenants),
                "skip": skip,
                "limit": limit,
            }

        try:
            # Get tenants with user's membership
            result = await self.kv_storage.db.query(
                """
                SELECT t.*, m.role, m.created_at as member_since
                FROM tenants t
                INNER JOIN user_tenant_memberships m ON t.tenant_id = m.tenant_id
                WHERE m.user_id = $1
                ORDER BY t.created_at DESC
                LIMIT $2 OFFSET $3
                """,
                [user_id, limit, skip],
                multirows=True,
            )

            # Get total count
            count_result = await self.kv_storage.db.query(
                """
                SELECT COUNT(*) as total
                FROM user_tenant_memberships
                WHERE user_id = $1
                """,
                [user_id],
            )

            # count_result is a single Record when multirows=False (default)
            total = count_result["total"] if count_result else 0

            tenants = []
            if result:
                for row in result:
                    tenant_dict = dict(row)
                    tenant_dict["user_role"] = tenant_dict.pop("role", None)
                    tenant_dict["member_since"] = tenant_dict.pop("member_since", None)
                    tenants.append(self._deserialize_tenant(tenant_dict))

            return {"items": tenants, "total": total, "skip": skip, "limit": limit}

        except Exception as e:
            logger.error(f"Error getting user tenants: {e}")
            raise

    async def get_tenant_members(
        self, tenant_id: str, skip: int = 0, limit: int = 100
    ) -> dict:
        """Get all members of a tenant.

        Args:
            tenant_id: Tenant identifier
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            Dictionary with items and total count
        """
        if not self.kv_storage.db:
            raise RuntimeError("PostgreSQL database required for membership management")

        try:
            result = await self.kv_storage.db.query(
                """
                SELECT user_id, role, created_at, created_by, updated_at
                FROM user_tenant_memberships
                WHERE tenant_id = $1
                ORDER BY created_at DESC
                LIMIT $2 OFFSET $3
                """,
                [tenant_id, limit, skip],
                multirows=True,
            )

            # Get total count
            count_result = await self.kv_storage.db.query(
                """
                SELECT COUNT(*) as total
                FROM user_tenant_memberships
                WHERE tenant_id = $1
                """,
                [tenant_id],
            )

            # count_result is a single Record when multirows=False (default)
            total = count_result["total"] if count_result else 0

            members = []
            if result:
                for row in result:
                    members.append(
                        {
                            "user_id": str(row["user_id"]),
                            "role": str(row["role"]),
                            "created_at": row["created_at"].isoformat()
                            if hasattr(row["created_at"], "isoformat")
                            else str(row["created_at"]),
                            "created_by": str(row["created_by"])
                            if row["created_by"]
                            else None,
                            "updated_at": row["updated_at"].isoformat()
                            if row["updated_at"]
                            and hasattr(row["updated_at"], "isoformat")
                            else (
                                str(row["updated_at"]) if row["updated_at"] else None
                            ),
                        }
                    )

            return {"items": members, "total": total, "skip": skip, "limit": limit}

        except Exception as e:
            logger.error(f"Error getting tenant members: {e}")
            raise

    async def update_tenant(
        self,
        tenant_id: str,
        **kwargs,
    ) -> Optional[Tenant]:
        """Update a tenant.

        Args:
            tenant_id: Tenant identifier
            **kwargs: Fields to update

        Returns:
            Updated Tenant object if found, None otherwise
        """
        tenant = await self.get_tenant(tenant_id)
        if not tenant:
            return None

        # Update fields
        for key, value in kwargs.items():
            if hasattr(tenant, key):
                setattr(tenant, key, value)

        tenant.updated_at = datetime.utcnow()

        # Update in PostgreSQL if available
        if hasattr(self.kv_storage, "db") and self.kv_storage.db is not None:
            try:
                import json

                metadata_json = json.dumps(tenant.metadata) if tenant.metadata else "{}"
                await self.kv_storage.db.query(
                    """
                    UPDATE tenants
                    SET name = $2, description = $3, metadata = $4::jsonb, updated_at = NOW()
                    WHERE tenant_id = $1
                    """,
                    [
                        tenant_id,
                        tenant.tenant_name,
                        tenant.description or "",
                        metadata_json,
                    ],
                )
                logger.debug(f"Updated tenant {tenant_id} in PostgreSQL tenants table")
            except Exception as e:
                logger.error(f"Failed to update tenant in PostgreSQL: {e}")
                raise
        else:
            # Fallback: Store updated tenant in KV storage
            try:
                tenant_data = tenant.to_dict()
                await self.kv_storage.upsert(
                    {f"{self.tenant_namespace}:{tenant_id}": tenant_data}
                )
            except Exception as e:
                logger.warning(
                    f"Could not update tenant in KV storage (non-critical): {e}"
                )

        logger.info(f"Updated tenant: {tenant_id}")
        return tenant

    async def list_tenants(
        self,
        skip: int = 0,
        limit: int = 100,
        search: Optional[str] = None,
        tenant_id_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List all tenants with pagination.

        Queries both KV storage and PostgreSQL database to ensure tenants
        created via database initialization scripts are also available.

        Args:
            skip: Number of tenants to skip (for pagination)
            limit: Maximum number of tenants to return
            search: Optional search string to filter by name or description
            tenant_id_filter: Optional tenant ID to filter by (for non-admin users)

        Returns:
            Dict with 'items' (list of tenants) and 'total' (count) keys
        """
        try:
            all_tenants = []

            # First, try to get tenants from PostgreSQL database if available
            if hasattr(self.kv_storage, "db") and self.kv_storage.db is not None:
                try:
                    logger.debug("Attempting to query tenants from PostgreSQL database")
                    # Query tenants with computed statistics using LEFT JOINs
                    # This ensures we get real kb_count, total_documents, and storage from the database
                    # Note: Documents are stored in lightrag_doc_full with workspace={tenant_id}:{kb_id}
                    stats_query = """
                        SELECT
                            t.tenant_id,
                            t.name,
                            t.description,
                            t.created_at,
                            t.updated_at,
                            COALESCE(kb_stats.kb_count, 0) as kb_count,
                            COALESCE(doc_stats.doc_count, 0) as total_documents,
                            COALESCE(doc_stats.total_size_bytes, 0) as total_size_bytes
                        FROM tenants t
                        LEFT JOIN (
                            SELECT tenant_id, COUNT(*) as kb_count
                            FROM knowledge_bases
                            GROUP BY tenant_id
                        ) kb_stats ON t.tenant_id = kb_stats.tenant_id
                        LEFT JOIN (
                            SELECT
                                SPLIT_PART(workspace, ':', 1) as tenant_id,
                                COUNT(*) as doc_count,
                                COALESCE(SUM(LENGTH(content)), 0) as total_size_bytes
                            FROM lightrag_doc_full
                            GROUP BY SPLIT_PART(workspace, ':', 1)
                        ) doc_stats ON t.tenant_id = doc_stats.tenant_id
                        ORDER BY t.created_at DESC
                    """
                    rows = await self.kv_storage.db.query(stats_query, multirows=True)

                    if rows:
                        for row in rows:
                            try:
                                # Convert bytes to MB for storage
                                total_size_bytes = row.get("total_size_bytes", 0) or 0
                                total_storage_mb = total_size_bytes / (1024 * 1024)

                                # Create a Tenant object from the database row with computed statistics
                                tenant = Tenant(
                                    tenant_id=row["tenant_id"],
                                    tenant_name=row["name"],
                                    description=row.get("description", ""),
                                    created_by=None,  # Not tracked in basic schema
                                    metadata={},
                                    kb_count=row.get("kb_count", 0) or 0,
                                    total_documents=row.get("total_documents", 0) or 0,
                                    total_storage_mb=total_storage_mb,
                                )
                                # Override timestamps from database
                                if "created_at" in row:
                                    tenant.created_at = row["created_at"]
                                if "updated_at" in row:
                                    tenant.updated_at = row["updated_at"]

                                all_tenants.append(tenant)
                            except Exception as e:
                                logger.error(f"Error processing tenant row: {e}")
                                continue

                    logger.info(
                        f"Retrieved {len(all_tenants)} tenants from PostgreSQL database"
                    )
                except Exception as e:
                    logger.debug(
                        f"Could not query tenants from PostgreSQL database: {e}"
                    )
                    # Fall through to KV storage

            # If no tenants from database, try KV storage
            if not all_tenants:
                logger.debug("Querying tenants from KV storage")
                tenant_keys = []
                if hasattr(self.kv_storage, "get_by_prefix"):
                    # For storages that support prefix search
                    tenant_keys = await self.kv_storage.get_by_prefix(
                        self.tenant_namespace
                    )
                elif hasattr(self.kv_storage, "get_all"):
                    # For storages like JsonKVStorage that have get_all
                    all_data = await self.kv_storage.get_all()
                    tenant_keys = [
                        key
                        for key in all_data.keys()
                        if key.startswith(f"{self.tenant_namespace}:")
                    ]

                # Filter and deserialize tenants from KV storage
                for key in tenant_keys:
                    if not key.startswith(f"{self.tenant_namespace}:"):
                        continue
                    try:
                        data = await self.kv_storage.get_by_id(key)
                        if data:
                            tenant = self._deserialize_tenant(data)

                            # Skip invalid tenants
                            if not tenant.tenant_id:
                                logger.warning(
                                    f"Skipping tenant with empty ID from key {key}"
                                )
                                continue

                            all_tenants.append(tenant)
                    except Exception as e:
                        logger.error(f"Error deserializing tenant from key {key}: {e}")
                        continue

            # Apply filters
            filtered_tenants = []
            for tenant in all_tenants:
                # Apply tenant ID filter
                if tenant_id_filter and tenant.tenant_id != tenant_id_filter:
                    continue
                # Apply search filter
                if search:
                    search_lower = search.lower()
                    if not (
                        search_lower in tenant.tenant_name.lower()
                        or search_lower in (tenant.description or "").lower()
                    ):
                        continue
                filtered_tenants.append(tenant)

            # Sort by created_at descending
            filtered_tenants.sort(key=lambda t: t.created_at, reverse=True)

            # Apply pagination
            total = len(filtered_tenants)
            paginated_tenants = filtered_tenants[skip : skip + limit]

            logger.info(
                f"Listed {len(paginated_tenants)} tenants out of {total} (skip={skip}, limit={limit})"
            )
            return {"items": paginated_tenants, "total": total}
        except Exception as e:
            logger.error(f"Error listing tenants: {e}")
            return {"items": [], "total": 0}

    async def delete_tenant(self, tenant_id: str) -> bool:
        """Delete a tenant and all associated data.

        This method performs cascade delete:
        1. Deletes all knowledge bases (which cascade delete their LIGHTRAG_* data)
        2. Deletes user-tenant memberships
        3. Deletes tenant metadata from PostgreSQL and KV storage

        Args:
            tenant_id: Tenant identifier

        Returns:
            True if deleted, False if not found
        """
        tenant = await self.get_tenant(tenant_id)
        if not tenant:
            return False

        # Delete all KBs associated with tenant (includes cascade delete of LIGHTRAG_* data)
        kbs_result = await self.list_knowledge_bases(tenant_id)
        kbs_list = kbs_result.get("items", [])
        for kb in kbs_list:
            await self.delete_knowledge_base(tenant_id, kb.kb_id)

        # Delete user-tenant memberships from PostgreSQL
        if hasattr(self.kv_storage, "db") and self.kv_storage.db is not None:
            try:
                await self.kv_storage.db.execute(
                    "DELETE FROM user_tenant_memberships WHERE tenant_id = $1",
                    [tenant_id],
                )
                logger.debug(f"Deleted user memberships for tenant {tenant_id}")
            except Exception as e:
                logger.debug(f"Could not delete user memberships: {e}")

            # Delete from tenants table (FK cascade should handle knowledge_bases)
            try:
                await self.kv_storage.db.execute(
                    "DELETE FROM tenants WHERE tenant_id = $1", [tenant_id]
                )
                logger.debug(f"Deleted tenant {tenant_id} from PostgreSQL")
            except Exception as e:
                logger.debug(f"Could not delete from tenants table: {e}")

        # Delete tenant metadata from KV storage
        await self.kv_storage.delete([f"{self.tenant_namespace}:{tenant_id}"])

        logger.info(f"Deleted tenant: {tenant_id} (with cascade delete)")
        return True

    async def create_knowledge_base(
        self,
        tenant_id: str,
        kb_name: str,
        description: Optional[str] = None,
        config: Optional[KBConfig] = None,
        created_by: Optional[str] = None,
    ) -> KnowledgeBase:
        """Create a new knowledge base for a tenant.

        Args:
            tenant_id: Parent tenant ID
            kb_name: Display name for KB
            description: Optional description
            config: Optional KB configuration
            created_by: User ID that created the KB

        Returns:
            Created KnowledgeBase object

        Raises:
            ValueError: If tenant not found
        """
        # Verify tenant exists
        tenant = await self.get_tenant(tenant_id)
        if not tenant:
            raise ValueError(f"Tenant {tenant_id} not found")

        kb = KnowledgeBase(
            tenant_id=tenant_id,
            kb_name=kb_name,
            description=description,
            config=config,
            created_by=created_by,
        )

        # Store KB in PostgreSQL if available
        if hasattr(self.kv_storage, "db") and self.kv_storage.db is not None:
            try:
                await self.kv_storage.db.query(
                    """
                    INSERT INTO knowledge_bases (kb_id, tenant_id, name, description, created_at, updated_at)
                    VALUES ($1, $2, $3, $4, NOW(), NOW())
                    ON CONFLICT (tenant_id, kb_id) DO UPDATE SET
                        name = EXCLUDED.name,
                        description = EXCLUDED.description,
                        updated_at = NOW()
                    RETURNING kb_id
                    """,
                    [kb.kb_id, tenant_id, kb_name, description or ""],
                )
                logger.debug(
                    f"Inserted KB {kb.kb_id} into PostgreSQL knowledge_bases table"
                )
            except Exception as e:
                logger.error(f"Failed to insert KB into PostgreSQL: {e}")
                raise
        else:
            # Fallback: Store KB metadata in KV storage
            try:
                kb_data = kb.to_dict()
                await self.kv_storage.upsert(
                    {f"{self.kb_namespace}:{tenant_id}:{kb.kb_id}": kb_data}
                )
            except Exception as e:
                logger.warning(f"Could not store KB in KV storage (non-critical): {e}")

        # Update tenant KB count
        tenant.kb_count += 1
        await self.update_tenant(tenant_id, kb_count=tenant.kb_count)

        logger.info(f"Created KB: {kb.kb_id} ({kb_name}) for tenant {tenant_id}")
        return kb

    async def get_knowledge_base(
        self,
        tenant_id: str,
        kb_id: str,
    ) -> Optional[KnowledgeBase]:
        """Retrieve a knowledge base with computed statistics.

        Args:
            tenant_id: Parent tenant ID
            kb_id: Knowledge base ID

        Returns:
            KnowledgeBase object if found, None otherwise
        """
        # First try PostgreSQL if available
        if hasattr(self.kv_storage, "db") and self.kv_storage.db is not None:
            try:
                # Query knowledge base with computed statistics using LEFT JOINs
                # Documents are stored in lightrag_doc_full with workspace={tenant_id}:{kb_id}
                workspace = f"{tenant_id}:{kb_id}"
                kb_stats_query = """
                    SELECT
                        kb.kb_id,
                        kb.tenant_id,
                        kb.name,
                        kb.description,
                        kb.created_at,
                        kb.updated_at,
                        COALESCE((SELECT COUNT(*) FROM lightrag_doc_full WHERE workspace = $1), 0) as document_count,
                        COALESCE((SELECT COUNT(*) FROM lightrag_vdb_entity WHERE workspace = $1), 0) as entity_count,
                        COALESCE((SELECT COUNT(*) FROM lightrag_vdb_relation WHERE workspace = $1), 0) as relationship_count
                    FROM knowledge_bases kb
                    WHERE kb.tenant_id = $2 AND kb.kb_id = $3
                """
                row = await self.kv_storage.db.query(
                    kb_stats_query,
                    params=[workspace, tenant_id, kb_id],
                    multirows=False,
                )

                if row:
                    kb = KnowledgeBase(
                        kb_id=row["kb_id"],
                        tenant_id=row["tenant_id"],
                        kb_name=row["name"],
                        description=row.get("description", ""),
                        document_count=row.get("document_count", 0) or 0,
                        entity_count=row.get("entity_count", 0) or 0,
                        relationship_count=row.get("relationship_count", 0) or 0,
                    )
                    if "created_at" in row:
                        kb.created_at = row["created_at"]
                    if "updated_at" in row:
                        kb.updated_at = row["updated_at"]
                    return kb
            except Exception as e:
                logger.warning(f"Could not query KB from PostgreSQL: {e}")
                # Fall through to KV storage

        # Fallback to KV storage
        data = await self.kv_storage.get_by_id(
            f"{self.kb_namespace}:{tenant_id}:{kb_id}"
        )
        if not data:
            return None
        return self._deserialize_kb(data)

    async def update_knowledge_base(
        self,
        tenant_id: str,
        kb_id: str,
        **kwargs,
    ) -> Optional[KnowledgeBase]:
        """Update a knowledge base.

        Args:
            tenant_id: Parent tenant ID
            kb_id: Knowledge base ID
            **kwargs: Fields to update

        Returns:
            Updated KnowledgeBase object if found, None otherwise
        """
        kb = await self.get_knowledge_base(tenant_id, kb_id)
        if not kb:
            return None

        # Update fields
        for key, value in kwargs.items():
            if hasattr(kb, key):
                setattr(kb, key, value)

        kb.updated_at = datetime.utcnow()

        # Update in PostgreSQL if available
        if hasattr(self.kv_storage, "db") and self.kv_storage.db is not None:
            try:
                await self.kv_storage.db.query(
                    """
                    UPDATE knowledge_bases
                    SET name = $3, description = $4, updated_at = NOW()
                    WHERE tenant_id = $1 AND kb_id = $2
                    """,
                    [tenant_id, kb_id, kb.kb_name, kb.description or ""],
                )
                logger.debug(f"Updated KB {kb_id} in PostgreSQL knowledge_bases table")
            except Exception as e:
                logger.error(f"Failed to update KB in PostgreSQL: {e}")
                raise
        else:
            # Fallback: Store updated KB in KV storage
            try:
                kb_data = kb.to_dict()
                await self.kv_storage.upsert(
                    {f"{self.kb_namespace}:{tenant_id}:{kb_id}": kb_data}
                )
            except Exception as e:
                logger.warning(f"Could not update KB in KV storage (non-critical): {e}")

        logger.info(f"Updated KB: {kb_id} for tenant {tenant_id}")
        return kb

    async def list_knowledge_bases(
        self,
        tenant_id: str,
        skip: int = 0,
        limit: int = 100,
        search: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List all knowledge bases for a tenant with pagination.

        Queries both KV storage and PostgreSQL database to ensure KBs
        created via database initialization scripts are also available.

        Args:
            tenant_id: Parent tenant ID
            skip: Number of KBs to skip (for pagination)
            limit: Maximum number of KBs to return
            search: Optional search string to filter by name or description

        Returns:
            Dict with 'items' (list of KBs) and 'total' (count) keys
        """
        try:
            all_kbs = []

            # First, try to get KBs from PostgreSQL database if available
            db_queried = False
            if hasattr(self.kv_storage, "db"):
                logger.info(
                    f"PGKVStorage.db exists, attempting to query KBs for tenant {tenant_id}"
                )
                if self.kv_storage.db is not None:
                    try:
                        logger.info(
                            f"Querying knowledge bases from PostgreSQL for tenant {tenant_id}"
                        )
                        # Query knowledge bases with computed statistics using LEFT JOINs
                        # Documents are stored in lightrag_doc_full with workspace={tenant_id}:{kb_id}
                        # Use subqueries that check for table existence to avoid errors
                        kb_stats_query = """
                            SELECT
                                kb.kb_id,
                                kb.tenant_id,
                                kb.name,
                                kb.description,
                                kb.created_at,
                                kb.updated_at,
                                COALESCE(doc_stats.doc_count, 0) as document_count,
                                COALESCE(entity_stats.entity_count, 0) as entity_count,
                                COALESCE(rel_stats.relationship_count, 0) as relationship_count
                            FROM knowledge_bases kb
                            LEFT JOIN (
                                SELECT
                                    workspace,
                                    COUNT(*) as doc_count
                                FROM lightrag_doc_full
                                WHERE workspace LIKE $1 || ':%'
                                GROUP BY workspace
                            ) doc_stats ON doc_stats.workspace = $1 || ':' || kb.kb_id
                            LEFT JOIN (
                                SELECT
                                    workspace,
                                    COUNT(*) as entity_count
                                FROM lightrag_vdb_entity
                                WHERE workspace LIKE $1 || ':%'
                                GROUP BY workspace
                            ) entity_stats ON entity_stats.workspace = $1 || ':' || kb.kb_id
                            LEFT JOIN (
                                SELECT
                                    workspace,
                                    COUNT(*) as relationship_count
                                FROM lightrag_vdb_relation
                                WHERE workspace LIKE $1 || ':%'
                                GROUP BY workspace
                            ) rel_stats ON rel_stats.workspace = $1 || ':' || kb.kb_id
                            WHERE kb.tenant_id = $1
                            ORDER BY kb.created_at DESC
                        """
                        rows = await self.kv_storage.db.query(
                            kb_stats_query, params=[tenant_id], multirows=True
                        )

                        if rows:
                            for row in rows:
                                try:
                                    # Create a KnowledgeBase object from the database row with computed statistics
                                    kb = KnowledgeBase(
                                        kb_id=row["kb_id"],
                                        tenant_id=row["tenant_id"],
                                        kb_name=row["name"],
                                        description=row.get("description", ""),
                                        document_count=row.get("document_count", 0)
                                        or 0,
                                        entity_count=row.get("entity_count", 0) or 0,
                                        relationship_count=row.get(
                                            "relationship_count", 0
                                        )
                                        or 0,
                                    )
                                    # Override timestamps from database
                                    if "created_at" in row:
                                        kb.created_at = row["created_at"]
                                    if "updated_at" in row:
                                        kb.updated_at = row["updated_at"]

                                    all_kbs.append(kb)
                                except Exception as e:
                                    logger.error(f"Error processing KB row: {e}")
                                    continue

                        logger.info(
                            f"Retrieved {len(all_kbs)} knowledge bases from PostgreSQL for tenant {tenant_id}"
                        )
                        db_queried = True
                    except Exception as e:
                        logger.warning(
                            f"Could not query knowledge bases from PostgreSQL: {e}"
                        )
                        # Fall through to KV storage
                else:
                    logger.debug(f"PGKVStorage.db is None for tenant {tenant_id}")
            else:
                logger.debug(
                    f"Storage doesn't have 'db' attribute (type: {type(self.kv_storage).__name__})"
                )

            # If no KBs from database, try KV storage
            if not all_kbs and not db_queried:
                logger.info(
                    f"Querying knowledge bases from KV storage for tenant {tenant_id}"
                )
                kb_keys = []
                if hasattr(self.kv_storage, "get_by_prefix"):
                    # For storages that support prefix search
                    tenant_prefix = f"{self.kb_namespace}:{tenant_id}:"
                    kb_keys = await self.kv_storage.get_by_prefix(tenant_prefix)
                elif hasattr(self.kv_storage, "get_all"):
                    # For storages like JsonKVStorage that have get_all
                    all_data = await self.kv_storage.get_all()
                    kb_keys = [
                        key
                        for key in all_data.keys()
                        if key.startswith(f"{self.kb_namespace}:{tenant_id}:")
                    ]

                # Filter and deserialize KBs from KV storage
                for key in kb_keys:
                    if not key.startswith(f"{self.kb_namespace}:{tenant_id}:"):
                        continue
                    try:
                        data = await self.kv_storage.get_by_id(key)
                        if data:
                            kb = self._deserialize_kb(data)
                            all_kbs.append(kb)
                    except Exception as e:
                        logger.error(f"Error deserializing KB from key {key}: {e}")
                        continue

            # Apply search filter
            filtered_kbs = []
            for kb in all_kbs:
                if search:
                    search_lower = search.lower()
                    if not (
                        search_lower in kb.kb_name.lower()
                        or search_lower in (kb.description or "").lower()
                    ):
                        continue
                filtered_kbs.append(kb)

            # Sort by created_at descending
            filtered_kbs.sort(key=lambda k: k.created_at, reverse=True)

            # Apply pagination
            total = len(filtered_kbs)
            paginated_kbs = filtered_kbs[skip : skip + limit]

            logger.info(
                f"Listed {len(paginated_kbs)} KBs out of {total} for tenant {tenant_id} (skip={skip}, limit={limit})"
            )
            return {"items": paginated_kbs, "total": total}
        except Exception as e:
            logger.error(f"Error listing KBs for tenant {tenant_id}: {e}")
            return {"items": [], "total": 0}

    async def delete_knowledge_base(
        self,
        tenant_id: str,
        kb_id: str,
    ) -> bool:
        """Delete a knowledge base and all associated data.

        This method performs cascade delete:
        1. Deletes all LIGHTRAG_* table data for this workspace
        2. Deletes KB metadata from KV storage
        3. Updates tenant KB count

        Args:
            tenant_id: Parent tenant ID
            kb_id: Knowledge base ID

        Returns:
            True if deleted, False if not found
        """
        kb = await self.get_knowledge_base(tenant_id, kb_id)
        if not kb:
            return False

        # Cascade delete: Clean up LIGHTRAG_* tables for this workspace
        workspace = f"{tenant_id}:{kb_id}"
        if hasattr(self.kv_storage, "db") and self.kv_storage.db is not None:
            try:
                # List of all LIGHTRAG tables that use workspace
                lightrag_tables = [
                    "LIGHTRAG_DOC_FULL",
                    "LIGHTRAG_DOC_CHUNKS",
                    "LIGHTRAG_DOC_STATUS",
                    "LIGHTRAG_VDB_CHUNKS",
                    "LIGHTRAG_VDB_ENTITY",
                    "LIGHTRAG_VDB_RELATION",
                    "LIGHTRAG_LLM_CACHE",
                    "LIGHTRAG_FULL_ENTITIES",
                    "LIGHTRAG_FULL_RELATIONS",
                    "LIGHTRAG_ENTITY_CHUNKS",
                    "LIGHTRAG_RELATION_CHUNKS",
                ]

                total_deleted = 0
                for table in lightrag_tables:
                    try:
                        result = await self.kv_storage.db.execute(
                            f"DELETE FROM {table} WHERE workspace = $1", [workspace]
                        )
                        # Log if rows were deleted (result may be row count or None)
                        if result:
                            logger.debug(
                                f"Deleted rows from {table} for workspace {workspace}"
                            )
                            total_deleted += 1
                    except Exception as table_error:
                        # Table might not exist, log and continue
                        logger.debug(f"Could not delete from {table}: {table_error}")

                logger.info(
                    f"Cascade delete: cleaned up LIGHTRAG tables for workspace {workspace}"
                )
            except Exception as e:
                logger.warning(f"Error during cascade delete for KB {kb_id}: {e}")
                # Continue with KB deletion even if cascade fails

        # Delete KB metadata from KV storage
        await self.kv_storage.delete([f"{self.kb_namespace}:{tenant_id}:{kb_id}"])

        # Delete KB from knowledge_bases table if using PostgreSQL
        if hasattr(self.kv_storage, "db") and self.kv_storage.db is not None:
            try:
                await self.kv_storage.db.execute(
                    "DELETE FROM knowledge_bases WHERE tenant_id = $1 AND kb_id = $2",
                    [tenant_id, kb_id],
                )
            except Exception as e:
                logger.debug(f"Could not delete from knowledge_bases table: {e}")

        # Update tenant KB count
        tenant = await self.get_tenant(tenant_id)
        if tenant:
            tenant.kb_count = max(0, tenant.kb_count - 1)
            await self.update_tenant(tenant_id, kb_count=tenant.kb_count)

        logger.info(f"Deleted KB: {kb_id} for tenant {tenant_id} (with cascade delete)")
        return True

    def _deserialize_tenant(self, data: Dict[str, Any]) -> Tenant:
        """Convert stored data to Tenant object."""
        import json

        # Handle PGKVStorage wrapping
        if "data" in data:
            inner_data = data["data"]
            if isinstance(inner_data, str):
                try:
                    inner_data = json.loads(inner_data)
                except json.JSONDecodeError:
                    logger.warning(
                        f"Failed to decode JSON from tenant data: {inner_data}"
                    )

            if isinstance(inner_data, dict) and "tenant_id" in inner_data:
                data = inner_data

        if not data.get("tenant_id"):
            logger.warning(
                f"Deserializing tenant with missing ID. Data keys: {list(data.keys())}"
            )

        config_data = data.get("config", {})
        data.get("quota", {})

        config = TenantConfig(
            llm_model=config_data.get("llm_model", "gpt-4o-mini"),
            embedding_model=config_data.get("embedding_model", "bge-m3:latest"),
            rerank_model=config_data.get("rerank_model"),
            chunk_size=config_data.get("chunk_size", 1200),
            chunk_overlap=config_data.get("chunk_overlap", 100),
            top_k=config_data.get("top_k", 40),
            cosine_threshold=config_data.get("cosine_threshold", 0.2),
            enable_llm_cache=config_data.get("enable_llm_cache", True),
            custom_metadata=config_data.get("custom_metadata", {}),
        )

        # Helper to parse datetime that might be string or datetime object
        def parse_datetime(val, default=None):
            if val is None:
                return default or datetime.utcnow()
            if isinstance(val, datetime):
                return val
            if isinstance(val, str):
                return datetime.fromisoformat(val)
            return default or datetime.utcnow()

        # Create and return tenant
        tenant = Tenant(
            tenant_id=data.get("tenant_id", ""),
            tenant_name=data.get("tenant_name", ""),
            description=data.get("description"),
            config=config,
            is_active=data.get("is_active", True),
            created_at=parse_datetime(data.get("created_at")),
            updated_at=parse_datetime(data.get("updated_at")),
            created_by=data.get("created_by"),
            updated_by=data.get("updated_by"),
            metadata=data.get("metadata", {}),
            kb_count=data.get("kb_count", 0),
            total_documents=data.get("total_documents", 0),
            total_storage_mb=data.get("total_storage_mb", 0.0),
        )
        return tenant

    def _deserialize_kb(self, data: Dict[str, Any]) -> KnowledgeBase:
        """Convert stored data to KnowledgeBase object."""
        import json

        # Handle PGKVStorage wrapping
        if "data" in data:
            inner_data = data["data"]
            if isinstance(inner_data, str):
                try:
                    inner_data = json.loads(inner_data)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to decode JSON from KB data: {inner_data}")

            if isinstance(inner_data, dict) and "kb_id" in inner_data:
                data = inner_data

        config_data = data.get("config")
        config = KBConfig(**config_data) if config_data else None

        # Helper to parse datetime that might be string or datetime object
        def parse_datetime(val, default=None):
            if val is None:
                return default
            if isinstance(val, datetime):
                return val
            if isinstance(val, str):
                return datetime.fromisoformat(val)
            return default

        kb = KnowledgeBase(
            kb_id=data.get("kb_id", ""),
            tenant_id=data.get("tenant_id", ""),
            kb_name=data.get("kb_name", ""),
            description=data.get("description"),
            is_active=data.get("is_active", True),
            status=data.get("status", "ready"),
            document_count=data.get("document_count", 0),
            entity_count=data.get("entity_count", 0),
            relationship_count=data.get("relationship_count", 0),
            chunk_count=data.get("chunk_count", 0),
            storage_used_mb=data.get("storage_used_mb", 0.0),
            last_indexed_at=parse_datetime(data.get("last_indexed_at")),
            index_version=data.get("index_version", 1),
            config=config,
            created_at=parse_datetime(data.get("created_at"), datetime.utcnow()),
            updated_at=parse_datetime(data.get("updated_at"), datetime.utcnow()),
            created_by=data.get("created_by"),
            updated_by=data.get("updated_by"),
            metadata=data.get("metadata", {}),
        )
        return kb
