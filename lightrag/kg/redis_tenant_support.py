# Redis Multi-Tenant Support Module

from typing import Any, Dict, List, Optional


class RedisTenantHelper:
    """Helper class for Redis multi-tenant key management"""
    
    SEPARATOR = ":"
    
    @staticmethod
    def make_tenant_key(tenant_id: str, kb_id: str, original_key: str) -> str:
        """
        Create a tenant-scoped Redis key.
        Format: tenant_id:kb_id:original_key
        """
        return f"{tenant_id}{RedisTenantHelper.SEPARATOR}{kb_id}{RedisTenantHelper.SEPARATOR}{original_key}"
    
    @staticmethod
    def parse_tenant_key(tenant_key: str) -> Dict[str, str]:
        """
        Parse a tenant-scoped key to extract components.
        
        Returns:
            Dict with tenant_id, kb_id, and original_key
        """
        parts = tenant_key.split(RedisTenantHelper.SEPARATOR, 2)
        if len(parts) == 3:
            return {
                "tenant_id": parts[0],
                "kb_id": parts[1],
                "original_key": parts[2]
            }
        return {"original_key": tenant_key}
    
    @staticmethod
    def get_tenant_key_pattern(tenant_id: str, kb_id: str, pattern: str = "*") -> str:
        """
        Get a key pattern for scanning tenant-specific keys.
        Useful for operations like SCAN, KEYS.
        
        Format: tenant_id:kb_id:*
        """
        return f"{tenant_id}{RedisTenantHelper.SEPARATOR}{kb_id}{RedisTenantHelper.SEPARATOR}{pattern}"
    
    @staticmethod
    def extract_original_key(tenant_key: str) -> str:
        """Extract the original key from a tenant-scoped key"""
        parts = tenant_key.split(RedisTenantHelper.SEPARATOR, 2)
        return parts[2] if len(parts) == 3 else tenant_key
    
    @staticmethod
    def batch_make_tenant_keys(
        tenant_id: str,
        kb_id: str,
        keys: List[str]
    ) -> List[str]:
        """Create multiple tenant-scoped keys at once"""
        return [
            RedisTenantHelper.make_tenant_key(tenant_id, kb_id, key)
            for key in keys
        ]
    
    @staticmethod
    def batch_parse_tenant_keys(tenant_keys: List[str]) -> List[Dict[str, str]]:
        return [RedisTenantHelper.parse_tenant_key(key) for key in tenant_keys]


class RedisTenantNamespace:
    """Context manager for tenant-scoped Redis operations"""
    
    def __init__(self, redis_client, tenant_id: str, kb_id: str):
        self.redis = redis_client
        self.tenant_id = tenant_id
        self.kb_id = kb_id
        self._prefix = RedisTenantHelper.get_tenant_key_pattern(tenant_id, kb_id, "")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get a value with tenant scoping"""
        tenant_key = RedisTenantHelper.make_tenant_key(self.tenant_id, self.kb_id, key)
        return await self.redis.get(tenant_key)
    
    async def set(self, key: str, value: Any, ex: Optional[int] = None) -> bool:
        """Set a value with tenant scoping"""
        tenant_key = RedisTenantHelper.make_tenant_key(self.tenant_id, self.kb_id, key)
        if ex:
            return await self.redis.setex(tenant_key, ex, value)
        return await self.redis.set(tenant_key, value)
    
    async def delete(self, keys: List[str]) -> int:
        """Delete values with tenant scoping"""
        tenant_keys = RedisTenantHelper.batch_make_tenant_keys(
            self.tenant_id, self.kb_id, keys
        )
        return await self.redis.delete(*tenant_keys)
    
    async def exists(self, key: str) -> bool:
        """Check if key exists with tenant scoping"""
        tenant_key = RedisTenantHelper.make_tenant_key(self.tenant_id, self.kb_id, key)
        return await self.redis.exists(tenant_key) > 0
    
    async def scan_keys(self, pattern: str = "*") -> List[str]:
        """Scan keys for this tenant"""
        tenant_pattern = self._prefix + pattern
        keys = []
        cursor = 0
        while True:
            cursor, batch = await self.redis.scan(cursor, match=tenant_pattern, count=100)
            keys.extend([
                RedisTenantHelper.extract_original_key(k.decode() if isinstance(k, bytes) else k)
                for k in batch
            ])
            if cursor == 0:
                break
        return keys
    
    async def delete_all(self) -> int:
        """Delete all keys for this tenant"""
        keys = await self.scan_keys()
        if keys:
            return await self.delete(keys)
        return 0


# ============================================================================
# MIGRATION HELPER FOR EXISTING REDIS DATA
# ============================================================================

async def migrate_redis_to_tenant(
    redis_client,
    old_key_pattern: str = "*",
    default_tenant_id: str = "default",
    default_kb_id: str = "default",
    dry_run: bool = True
) -> Dict[str, int]:
    """
    Migrate existing Redis keys to tenant-scoped format.
    
    This operation:
    1. Scans all keys matching the pattern
    2. For each key, creates a new tenant-scoped key
    3. Optionally deletes the old key (if not dry_run)
    
    Args:
        redis_client: Redis async client
        old_key_pattern: Pattern to match old keys (default: all keys)
        default_tenant_id: Tenant ID to use for migration
        default_kb_id: KB ID to use for migration
        dry_run: If True, don't actually delete old keys
    
    Returns:
        Migration statistics
    """
    stats = {
        "migrated": 0,
        "failed": 0,
        "skipped": 0
    }
    
    cursor = 0
    while True:
        cursor, keys = await redis_client.scan(cursor, match=old_key_pattern, count=100)
        
        for key_bytes in keys:
            key = key_bytes.decode() if isinstance(key_bytes, bytes) else key_bytes
            
            # Check if key is already tenant-scoped (skip if yes)
            parts = key.split(":", 2)
            if len(parts) == 3:
                stats["skipped"] += 1
                continue
            
            try:
                # Get the value
                value = await redis_client.get(key)
                if value is None:
                    stats["skipped"] += 1
                    continue
                
                # Create tenant-scoped key
                tenant_key = RedisTenantHelper.make_tenant_key(
                    default_tenant_id,
                    default_kb_id,
                    key
                )
                
                # Set the new key
                await redis_client.set(tenant_key, value)
                
                # Delete old key (unless dry_run)
                if not dry_run:
                    await redis_client.delete(key)
                
                stats["migrated"] += 1
            
            except Exception as e:
                print(f"Error migrating key {key}: {e}")
                stats["failed"] += 1
        
        if cursor == 0:
            break
    
    return stats
